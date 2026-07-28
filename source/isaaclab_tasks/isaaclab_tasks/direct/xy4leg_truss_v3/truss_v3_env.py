from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .crossing_logic import (
    curriculum_crossing_success,
    curriculum_landing_condition,
    crossing_stage,
    gripper_open_state,
    normalized_crossing_progress,
    safe_monotonic_progress_gain,
    update_foot_crossing_state,
    update_curriculum_levels,
)
from .truss_v3_env_cfg import XY4LegTrussV3TeacherEnvCfg


def _quat_wxyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1.0e-8)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            torch.stack((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)), dim=-1),
            torch.stack((2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)), dim=-1),
            torch.stack((2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)), dim=-1),
        ),
        dim=-2,
    )


class XY4LegTrussV3TeacherEnv(DirectRLEnv):
    cfg: XY4LegTrussV3TeacherEnvCfg

    _FOOT_NAMES = ("RF_Link5", "LF_Link5", "RB_Link5", "LB_Link5")
    _LEG_NAMES = ("RF", "LF", "RB", "LB")

    def __init__(self, cfg: XY4LegTrussV3TeacherEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._main_joint_ids, self._main_joint_names = self.robot.find_joints(
            [f"{leg}_Joint[1-5]" for leg in self._LEG_NAMES], preserve_order=True
        )
        if len(self._main_joint_ids) != 20:
            raise RuntimeError(f"Expected 20 policy joints, found {self._main_joint_names}")

        gripper_ids: list[int] = []
        for leg in self._LEG_NAMES:
            ids, _ = self.robot.find_joints([f"{leg}_1out_Joint", f"{leg}_2out_Joint"], preserve_order=True)
            if len(ids) != 2:
                raise RuntimeError(f"Expected two gripper joints for {leg}")
            gripper_ids.extend(ids)
        self._gripper_joint_ids = gripper_ids
        self._learn_gripper_actions = bool(self.cfg.learn_gripper_actions)
        self._policy_action_dim = 24 if self._learn_gripper_actions else 20
        if int(self.cfg.action_space) != self._policy_action_dim:
            raise RuntimeError(
                f"Expected action_space={self._policy_action_dim} for "
                f"learn_gripper_actions={self._learn_gripper_actions}, got {self.cfg.action_space}"
            )

        self._foot_body_ids, _ = self.robot.find_bodies(self._FOOT_NAMES, preserve_order=True)
        foot_contact_body_ids: list[list[int]] = []
        foot_contact_sensor_ids: list[list[int]] = []
        for leg in self._LEG_NAMES:
            foot_contact_patterns = [f"{leg}_Link5", f"{leg}_1out_Link", f"{leg}_2out_Link"]
            body_ids, body_names = self.robot.find_bodies(foot_contact_patterns, preserve_order=True)
            sensor_ids, sensor_names = self.contact_sensor.find_bodies(foot_contact_patterns, preserve_order=True)
            if len(sensor_ids) != 3:
                raise RuntimeError(f"Expected three contact bodies for {leg}, found {sensor_names}")
            if body_names != sensor_names:
                raise RuntimeError(f"Robot/contact-sensor body ordering differs for foot contacts on {leg}")
            foot_contact_body_ids.append(body_ids)
            foot_contact_sensor_ids.append(sensor_ids)
        self._foot_contact_body_ids = torch.tensor(
            foot_contact_body_ids, dtype=torch.long, device=self.device
        )
        self._foot_contact_sensor_ids = torch.tensor(
            foot_contact_sensor_ids, dtype=torch.long, device=self.device
        )
        forbidden_patterns = ["base_link"] + [f"{leg}_Link[1-4]" for leg in self._LEG_NAMES]
        self._forbidden_body_ids, forbidden_names = self.robot.find_bodies(
            forbidden_patterns, preserve_order=True
        )
        self._forbidden_sensor_ids, sensor_names = self.contact_sensor.find_bodies(
            forbidden_patterns, preserve_order=True
        )
        if forbidden_names != sensor_names:
            raise RuntimeError("Robot/contact-sensor body ordering differs for forbidden contacts")

        limits = self.robot.data.soft_joint_pos_limits[0, self._main_joint_ids]
        self._action_offset = 0.5 * (limits[:, 0] + limits[:, 1])
        self._action_scale = 0.5 * (limits[:, 1] - limits[:, 0])
        self._joint_lower = limits[:, 0]
        self._joint_upper = limits[:, 1]
        default_joint_pos = self.robot.data.default_joint_pos[0, self._main_joint_ids]
        self._nominal_action = (
            (default_joint_pos - self._action_offset)
            / self._action_scale.clamp_min(1.0e-6)
        ).clamp(-1.0, 1.0)
        if self._learn_gripper_actions:
            self._nominal_action = torch.cat(
                (self._nominal_action, torch.zeros(4, device=self.device)), dim=0
            )

        self._raw_actions = torch.zeros((self.num_envs, self._policy_action_dim), device=self.device)
        self._previous_actions = torch.zeros_like(self._raw_actions)
        self._previous_previous_actions = torch.zeros_like(self._raw_actions)
        self._applied_actions = torch.zeros_like(self._raw_actions)
        self._action_history = torch.zeros(
            (self.num_envs, self.cfg.max_action_delay_steps + 1, self._policy_action_dim), device=self.device
        )
        self._action_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._action_history_index = 0
        self._all_env_ids = torch.arange(self.num_envs, device=self.device)

        self._gait_phase_origin = torch.rand(self.num_envs, device=self.device)
        self._phase_offsets = torch.tensor(
            self.cfg.gait_phase_offsets, dtype=torch.float32, device=self.device
        )
        self._gripper_open = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)
        self._gripper_target = torch.full(
            (self.num_envs, 8), float(self.cfg.gripper_closed_angle), device=self.device
        )
        self._gripper_latched_actions = torch.full(
            (self.num_envs, 4),
            float(self.cfg.gripper_action_latch_initial_action),
            device=self.device,
        )
        self._gripper_latch_steps_remaining = torch.zeros(
            (self.num_envs, 4), dtype=torch.long, device=self.device
        )

        self._curriculum_level = torch.full(
            (self.num_envs,),
            int(self.cfg.initial_curriculum_level),
            dtype=torch.long,
            device=self.device,
        )
        self._curriculum_success_ema = torch.zeros(self.num_envs, device=self.device)
        self._curriculum_episode_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._failure_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._bar_x_w = torch.zeros(self.num_envs, device=self.device)
        self._root_signed_distance = torch.zeros(self.num_envs, device=self.device)
        self._foot_signed_distance = torch.zeros((self.num_envs, 4), device=self.device)
        bar_history_len = max(int(self.cfg.bar_estimate_max_delay_steps) + 1, 1)
        self._bar_estimate_history = torch.zeros(
            (self.num_envs, bar_history_len, 3), device=self.device
        )
        self._bar_estimate_history_index = 0
        self._bar_estimate_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._bar_estimate_bias = torch.zeros((self.num_envs, 3), device=self.device)
        self._bar_estimate_filtered = torch.zeros((self.num_envs, 3), device=self.device)
        self._bar_estimate_visible = torch.ones((self.num_envs, 1), device=self.device)
        self._bar_estimate_last_update_step = -1
        self._bar_estimate_visibility_log_step = -1
        self._stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._foot_contacts = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)
        self._foot_contact_force = torch.zeros((self.num_envs, 4), device=self.device)
        self._foot_bar_contact = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)
        self._foot_bar_contact_force = torch.zeros((self.num_envs, 4), device=self.device)
        self._previous_foot_contacts = torch.zeros_like(self._foot_contacts)
        self._foot_touchdown = torch.zeros_like(self._foot_contacts)
        self._foot_touchdown_air_time = torch.zeros(
            (self.num_envs, 4), device=self.device
        )
        self._foot_pos_b = torch.zeros((self.num_envs, 4, 3), device=self.device)
        self._foot_air_time = torch.zeros((self.num_envs, 4), device=self.device)
        self._foot_clearance = torch.zeros((self.num_envs, 4), device=self.device)
        self._foot_rail_y_error = torch.zeros((self.num_envs, 4), device=self.device)
        self._foot_rail_valid = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)
        self._foot_crossing_state = torch.zeros(
            (self.num_envs, 4), dtype=torch.long, device=self.device
        )
        self._crossing_state_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._forbidden_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._forbidden_contact_force = torch.zeros(self.num_envs, device=self.device)
        self._forbidden_contact_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._stable_contact_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._max_stable_contact_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._stable_contact_progress_gain = torch.zeros(self.num_envs, device=self.device)
        self._landing_condition = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self._max_foot_progress = torch.zeros((self.num_envs, 4), device=self.device)
        self._max_root_progress = torch.zeros(self.num_envs, device=self.device)
        self._foot_progress_gain = torch.zeros((self.num_envs, 4), device=self.device)
        self._root_progress_gain = torch.zeros(self.num_envs, device=self.device)
        self._progress_anchor = torch.zeros(self.num_envs, device=self.device)
        self._episode_start_root_x = torch.zeros(self.num_envs, device=self.device)
        self._no_progress_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._previous_foot_x = torch.zeros((self.num_envs, 4), device=self.device)
        self._foot_forward_delta = torch.zeros((self.num_envs, 4), device=self.device)
        self._progress_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self._front_complete = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hind_complete = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._root_complete = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._front_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._early_front_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hind_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._root_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._no_progress_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._invalid_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._height_low_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._height_high_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._tilt_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._yaw_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lateral_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._angular_velocity_failure = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._forbidden_failure = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._nonfinite_reward_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._tracker_step = -1
        self._depth_bar_estimator = self._load_depth_bar_estimator()
        self._depth_bar_camera_ready = False
        self._depth_bar_input_cache: torch.Tensor | None = None
        self._depth_bar_input_cache_step = -1
        self._depth_bar_pixel_grids: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

        self._joint_position_bias = torch.zeros((self.num_envs, 20), device=self.device)
        self._level_distance_ranges = torch.tensor(
            self.cfg.level_bar_distance_ranges, dtype=torch.float32, device=self.device
        )
        self._balanced_curriculum_level_weights = torch.tensor(
            self.cfg.balanced_curriculum_level_weights,
            dtype=torch.float32,
            device=self.device,
        )
        expected_level_count = int(self.cfg.curriculum_max_level) + 1
        if self.cfg.balanced_curriculum_enabled:
            if self._balanced_curriculum_level_weights.numel() != expected_level_count:
                raise RuntimeError(
                    "balanced_curriculum_level_weights must have "
                    f"{expected_level_count} entries, got {self._balanced_curriculum_level_weights.numel()}"
                )
            if torch.any(self._balanced_curriculum_level_weights < 0.0):
                raise RuntimeError("balanced_curriculum_level_weights must be non-negative")
            weight_sum = torch.sum(self._balanced_curriculum_level_weights)
            if weight_sum <= 0.0:
                raise RuntimeError(
                    "balanced_curriculum_level_weights must contain at least one positive weight"
                )
            self._balanced_curriculum_level_weights = (
                self._balanced_curriculum_level_weights / weight_sum
            )
        else:
            self._balanced_curriculum_level_weights = torch.ones(
                expected_level_count, dtype=torch.float32, device=self.device
            ) / float(expected_level_count)
        self._level_no_progress_grace = torch.tensor(
            self.cfg.no_progress_grace_s, dtype=torch.float32, device=self.device
        )
        self._level_no_progress_patience = torch.tensor(
            self.cfg.no_progress_patience_s, dtype=torch.float32, device=self.device
        )
        self._walk_foothold_ranges = torch.tensor(
            self.cfg.walk_foothold_ranges,
            dtype=torch.float32,
            device=self.device,
        )
        half_gap = 0.5 * self.cfg.rail_center_distance
        self._rail_y_offsets = torch.tensor(
            (-half_gap, half_gap, -half_gap, half_gap),
            dtype=torch.float32,
            device=self.device,
        )
        self._bar_dimensions = torch.tensor(
            (self.cfg.crossbar_width, self.cfg.crossbar_height),
            dtype=torch.float32,
            device=self.device,
        ).expand(self.num_envs, -1)
        self._bar_estimate_noise_std = torch.tensor(
            self.cfg.bar_estimate_gaussian_noise_std,
            dtype=torch.float32,
            device=self.device,
        )
        self._bar_estimate_bias_range = torch.tensor(
            self.cfg.bar_estimate_episode_bias_range,
            dtype=torch.float32,
            device=self.device,
        )
        self._bar_estimate_fov_x_range = torch.tensor(
            self.cfg.bar_estimate_fov_x_range,
            dtype=torch.float32,
            device=self.device,
        )
        self._bar_estimate_fov_y_range = torch.tensor(
            self.cfg.bar_estimate_fov_y_range,
            dtype=torch.float32,
            device=self.device,
        )
        self._bar_estimate_fov_z_range = torch.tensor(
            self.cfg.bar_estimate_fov_z_range,
            dtype=torch.float32,
            device=self.device,
        )
        self._forward_axis = torch.tensor(
            (1.0, 0.0, 0.0), dtype=torch.float32, device=self.device
        ).expand(self.num_envs, -1)
        self._reset_motion = self._load_reset_motion()
        self._episode_contact_steps = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._episode_foot_bar_contact_steps = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._episode_low_bar_clearance_steps = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._episode_low_bar_clearance_deficit_sum = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._episode_low_bar_clearance_samples = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._episode_rail_valid_steps = torch.zeros_like(self._episode_contact_steps)
        self._episode_rail_y_error_sum = torch.zeros_like(self._episode_contact_steps)
        self._episode_landing_condition_steps = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._episode_stance_contact_score = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._episode_swing_clear_score = torch.zeros_like(
            self._episode_stance_contact_score
        )
        self._episode_valid_touchdowns = torch.zeros_like(
            self._episode_stance_contact_score
        )
        self._episode_swing_clearance_sum = torch.zeros_like(
            self._episode_stance_contact_score
        )
        self._episode_swing_clearance_samples = torch.zeros_like(
            self._episode_stance_contact_score
        )
        self._episode_bar_visible_steps = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._episode_bar_visible_samples = torch.zeros_like(
            self._episode_bar_visible_steps
        )

        reward_names = (
            "forward_velocity",
            "upright",
            "heading",
            "support",
            "gait_contact",
            "walk_rail_center",
            "walk_contact",
            "walk_swing_clearance_error",
            "walk_foothold",
            "touchdown_impact",
            "walk_base_height",
            "base_motion",
            "forward_speed_limit",
            "gripper_phase",
            "front_progress",
            "hind_progress",
            "root_progress",
            "clearance",
            "landing_rail_center",
            "landing_contact",
            "landing_stability",
            "stable_contact_progress",
            "front_event",
            "hind_event",
            "root_event",
            "success_event",
            "action_magnitude",
            "action_soft_limit",
            "action_rate",
            "action_accel",
            "joint_limit",
            "joint_velocity",
            "energy",
            "slip",
            "long_air_no_progress",
            "forbidden_contact",
            "foot_bar_contact",
            "low_bar_clearance",
            "termination",
        )
        self._episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for name in reward_names
        }

    def _setup_scene(self):
        rail_cfg = sim_utils.CuboidCfg(
            size=(self.cfg.rail_length, self.cfg.rail_width, self.cfg.rail_height),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.20, 0.22)),
        )
        rail_z = self.cfg.rail_top_height - 0.5 * self.cfg.rail_height
        half_gap = 0.5 * self.cfg.rail_center_distance
        rail_cfg.func(
            "/World/envs/env_0/LeftRail",
            rail_cfg,
            translation=(0.0, half_gap, rail_z),
        )
        rail_cfg.func(
            "/World/envs/env_0/RightRail",
            rail_cfg,
            translation=(0.0, -half_gap, rail_z),
        )

        if os.path.isfile(self.cfg.default_ground_usd):
            spawn_ground_plane(
                prim_path="/World/GroundPlane",
                cfg=GroundPlaneCfg(
                    usd_path=self.cfg.default_ground_usd,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8,
                        dynamic_friction=0.8,
                        restitution=0.0,
                    ),
                ),
            )
        else:
            ground_cfg = sim_utils.CuboidCfg(
                size=(500.0, 500.0, 0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.8,
                    dynamic_friction=0.8,
                    restitution=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.18, 0.20, 0.22)
                ),
            )
            ground_cfg.func(
                "/World/GroundPlane",
                ground_cfg,
                translation=(0.0, 0.0, -0.025),
            )

        self.robot = Articulation(self.cfg.robot)
        self.crossbar = RigidObject(self.cfg.crossbar)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/GroundPlane"])
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["crossbar"] = self.crossbar

        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        if self.cfg.enable_depth_bar_camera:
            self.front_tiled_camera = TiledCamera(self.cfg.tiled_camera)
            self.rear_tiled_camera = TiledCamera(self.cfg.rear_tiled_camera)
            self.tiled_camera = self.front_tiled_camera
            self.scene.sensors["front_tiled_camera"] = self.front_tiled_camera
            self.scene.sensors["rear_tiled_camera"] = self.rear_tiled_camera

        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _load_depth_bar_estimator(self):
        path = self.cfg.depth_bar_estimator_checkpoint
        if not path:
            return None
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Depth bar estimator checkpoint does not exist: {path}")
        estimator = torch.jit.load(path, map_location=self.device)
        estimator.eval()
        return estimator

    def _load_reset_motion(self) -> dict[str, torch.Tensor] | None:
        path = self.cfg.reset_motion_file
        if not path or not os.path.isfile(path):
            return None

        data = np.load(path)
        start = max(int(self.cfg.hind_reset_frame_start), 0)
        end = min(int(self.cfg.hind_reset_frame_end), data["dof_positions"].shape[0])
        if start >= end:
            return None

        dof_names = data["dof_names"].tolist()
        body_names = data["body_names"].tolist()
        try:
            joint_indices = [dof_names.index(name) for name in self.robot.data.joint_names]
            base_index = body_names.index("base_link")
        except ValueError:
            return None

        return {
            "joint_pos": torch.as_tensor(
                data["dof_positions"][start:end, joint_indices],
                dtype=torch.float32,
                device=self.device,
            ),
            "root_z": torch.as_tensor(
                data["body_positions"][start:end, base_index, 2],
                dtype=torch.float32,
                device=self.device,
            ),
            "root_quat": torch.as_tensor(
                data["body_rotations"][start:end, base_index],
                dtype=torch.float32,
                device=self.device,
            ),
        }

    def _pre_physics_step(self, actions: torch.Tensor):
        # DirectRLEnv reuses the extras dict across steps. Clear reset logs here
        # so RSL-RL does not append stale episode metrics on non-reset steps.
        self.extras.pop("log", None)
        self._previous_previous_actions.copy_(self._previous_actions)
        self._previous_actions.copy_(self._raw_actions)
        self._raw_actions.copy_(
            torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        )

        if self.cfg.max_action_delay_steps > 0:
            self._action_history[:, self._action_history_index] = self._raw_actions
            read_index = (
                self._action_history_index - self._action_delay
            ) % self._action_history.shape[1]
            self._applied_actions.copy_(
                self._action_history[self._all_env_ids, read_index]
            )
            self._action_history_index = (
                self._action_history_index + 1
            ) % self._action_history.shape[1]
        else:
            self._applied_actions.copy_(self._raw_actions)

        if self._learn_gripper_actions:
            gripper_actions = self._apply_gripper_action_latch(self._applied_actions[:, 20:24])
            step = float(self.cfg.gripper_action_step)
            delta = torch.where(
                gripper_actions > 0.0,
                torch.full_like(gripper_actions, step),
                torch.where(
                    gripper_actions < 0.0,
                    torch.full_like(gripper_actions, -step),
                    torch.zeros_like(gripper_actions),
                ),
            )
            gripper_angles = self._gripper_target[:, 0::2] + delta
            gripper_angles = gripper_angles.clamp(
                float(self.cfg.gripper_closed_angle), float(self.cfg.gripper_open_angle)
            )
            self._gripper_target[:, 0::2] = gripper_angles
            self._gripper_target[:, 1::2] = gripper_angles
        else:
            phase = self._leg_phase()
            self._gripper_open = gripper_open_state(
                phase,
                self._gripper_open,
                self.cfg.gait_duty_factor,
                self.cfg.gripper_phase_hysteresis,
            )
            angles = torch.where(
                self._gripper_open,
                torch.full_like(phase, float(self.cfg.gripper_open_angle)),
                torch.full_like(phase, float(self.cfg.gripper_closed_angle)),
            )
            self._gripper_target[:, 0::2] = angles
            self._gripper_target[:, 1::2] = angles

    def _apply_gripper_action_latch(self, gripper_actions: torch.Tensor) -> torch.Tensor:
        if not self.cfg.gripper_action_latch_enabled:
            return gripper_actions

        deadband = float(self.cfg.gripper_action_latch_deadband)
        candidate = torch.where(
            gripper_actions > deadband,
            torch.ones_like(gripper_actions),
            torch.where(
                gripper_actions < -deadband,
                -torch.ones_like(gripper_actions),
                self._gripper_latched_actions,
            ),
        )
        remaining_after_tick = torch.clamp(self._gripper_latch_steps_remaining - 1, min=0)
        can_update = self._gripper_latch_steps_remaining <= 0
        should_update = can_update & (candidate != self._gripper_latched_actions)
        self._gripper_latched_actions.copy_(
            torch.where(should_update, candidate, self._gripper_latched_actions)
        )
        min_hold_steps = max(int(math.ceil(float(self.cfg.gripper_action_latch_min_hold_s) / self.step_dt)), 1)
        hold_steps = torch.full_like(self._gripper_latch_steps_remaining, min_hold_steps)
        self._gripper_latch_steps_remaining.copy_(
            torch.where(should_update, hold_steps, remaining_after_tick)
        )
        return self._gripper_latched_actions

    def _apply_action(self):
        target = self._action_offset + self._action_scale * self._applied_actions[:, :20]
        target = target.clamp(self._joint_lower, self._joint_upper)
        self.robot.set_joint_position_target(target, joint_ids=self._main_joint_ids)
        self.robot.set_joint_position_target(self._gripper_target, joint_ids=self._gripper_joint_ids)

    def _leg_phase(self) -> torch.Tensor:
        time = self.episode_length_buf.float() * self.step_dt
        return (
            self._gait_phase_origin[:, None]
            + time[:, None] * self.cfg.gait_frequency
            + self._phase_offsets[None, :]
        ) % 1.0

    def _refresh_crossing_state(self, update_trackers: bool):
        feet_pos = torch.nan_to_num(
            self.robot.data.body_pos_w[:, self._foot_body_ids],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        root_pos = torch.nan_to_num(
            self.robot.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0
        )
        root_quat = torch.nan_to_num(
            self.robot.data.root_quat_w, nan=0.0, posinf=0.0, neginf=0.0
        )
        self._foot_pos_b.copy_(
            torch.nan_to_num(
                quat_apply_inverse(
                    root_quat[:, None, :].expand(-1, 4, -1).reshape(-1, 4),
                    (feet_pos - root_pos[:, None, :]).reshape(-1, 3),
                ),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).reshape(self.num_envs, 4, 3)
        )
        bar_x = self.crossbar.data.root_pos_w[:, 0]
        bar_x = torch.where(torch.isfinite(bar_x), bar_x, self._bar_x_w)
        self._bar_x_w.copy_(bar_x)
        self._root_signed_distance.copy_(
            torch.nan_to_num(
                root_pos[:, 0] - self._bar_x_w,
                nan=-self.cfg.progress_approach_distance,
                posinf=self.cfg.root_pass_margin,
                neginf=-self.cfg.progress_approach_distance,
            )
        )
        self._foot_signed_distance.copy_(
            torch.nan_to_num(
                feet_pos[:, :, 0] - self._bar_x_w[:, None],
                nan=-self.cfg.progress_approach_distance,
                posinf=self.cfg.pass_margin,
                neginf=-self.cfg.progress_approach_distance,
            )
        )
        contact_history = torch.nan_to_num(
            self.contact_sensor.data.net_forces_w_history,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        foot_forces = contact_history[:, :, self._foot_contact_sensor_ids]
        foot_force_magnitude = torch.linalg.vector_norm(foot_forces, dim=-1)
        self._foot_contact_force.copy_(torch.amax(foot_force_magnitude, dim=(1, 3)))
        self._foot_contacts.copy_(
            self._foot_contact_force > self.cfg.foot_contact_force
        )
        rail_top = self.scene.env_origins[:, 2, None] + self.cfg.rail_top_height
        self._foot_clearance.copy_(
            torch.nan_to_num(
                feet_pos[:, :, 2] - rail_top,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        )
        expected_rail_y = self.scene.env_origins[:, 1, None] + self._rail_y_offsets[None, :]
        self._foot_rail_y_error.copy_(feet_pos[:, :, 1] - expected_rail_y)
        self._foot_rail_valid.copy_(
            torch.abs(self._foot_rail_y_error) <= self.cfg.foot_rail_y_tolerance
        )
        crossing_initializing = ~self._crossing_state_initialized
        self._foot_crossing_state.copy_(
            update_foot_crossing_state(
                self._foot_crossing_state,
                self._foot_signed_distance,
                self._foot_rail_valid,
                self._foot_clearance,
                clearance_x_window=self.cfg.clearance_x_window,
                min_clearance=self.cfg.crossing_min_clearance,
                pass_margin=self.cfg.pass_margin,
                initializing=crossing_initializing,
            )
        )
        self._crossing_state_initialized.fill_(True)
        foot_passed = self._foot_crossing_state >= 2
        self._stage.copy_(
            crossing_stage(
                self._root_signed_distance,
                self._foot_signed_distance,
                self.cfg.pass_margin,
                self.cfg.attempt_distance,
                foot_passed=foot_passed,
            )
        )

        forbidden_forces = contact_history[:, :, self._forbidden_sensor_ids]
        forbidden_magnitude = torch.amax(torch.linalg.vector_norm(forbidden_forces, dim=-1), dim=1)
        forbidden_pos = torch.nan_to_num(
            self.robot.data.body_pos_w[:, self._forbidden_body_ids],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        bar_pos = torch.nan_to_num(
            self.crossbar.data.root_pos_w,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        foot_contact_pos = torch.nan_to_num(
            self.robot.data.body_pos_w[:, self._foot_contact_body_ids.reshape(-1)],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).reshape(self.num_envs, 4, 3, 3)
        foot_force_by_body = torch.amax(foot_force_magnitude, dim=1)
        foot_near_bar_x = (
            torch.abs(foot_contact_pos[..., 0] - self._bar_x_w[:, None, None])
            < self.cfg.foot_bar_contact_x_window
        )
        foot_near_bar_y = (
            torch.abs(foot_contact_pos[..., 1] - bar_pos[:, None, None, 1])
            < 0.5 * self.cfg.crossbar_length + self.cfg.foot_bar_contact_y_margin
        )
        foot_near_bar_z = (
            torch.abs(foot_contact_pos[..., 2] - bar_pos[:, None, None, 2])
            < 0.5 * self.cfg.crossbar_height + self.cfg.foot_bar_contact_z_margin
        )
        foot_near_bar = foot_near_bar_x & foot_near_bar_y & foot_near_bar_z
        foot_bar_force = torch.where(
            foot_near_bar, foot_force_by_body, torch.zeros_like(foot_force_by_body)
        )
        self._foot_bar_contact_force.copy_(torch.amax(foot_bar_force, dim=2))
        self._foot_bar_contact.copy_(
            self._foot_bar_contact_force > self.cfg.foot_bar_contact_force
        )
        near_bar_x = (
            torch.abs(forbidden_pos[:, :, 0] - self._bar_x_w[:, None])
            < self.cfg.forbidden_contact_x_window
        )
        near_bar_y = (
            torch.abs(forbidden_pos[:, :, 1] - bar_pos[:, None, 1])
            < 0.5 * self.cfg.crossbar_length + self.cfg.forbidden_contact_y_margin
        )
        near_bar_z = (
            torch.abs(forbidden_pos[:, :, 2] - bar_pos[:, None, 2])
            < 0.5 * self.cfg.crossbar_height + self.cfg.forbidden_contact_z_margin
        )
        near_bar = near_bar_x & near_bar_y & near_bar_z
        near_bar_force = torch.where(
            near_bar, forbidden_magnitude, torch.zeros_like(forbidden_magnitude)
        )
        self._forbidden_contact_force.copy_(torch.amax(near_bar_force, dim=1))
        self._forbidden_contact.copy_(
            self._forbidden_contact_force > self.cfg.forbidden_contact_force
        )

        if not update_trackers or self._tracker_step == self.common_step_counter:
            return
        self._tracker_step = self.common_step_counter
        self._foot_touchdown.copy_(
            self._foot_contacts & ~self._previous_foot_contacts
        )
        self._foot_touchdown_air_time.copy_(
            torch.where(
                self._foot_touchdown,
                self._foot_air_time,
                torch.zeros_like(self._foot_air_time),
            )
        )
        self._previous_foot_contacts.copy_(self._foot_contacts)
        initializing = ~self._progress_initialized

        foot_progress = normalized_crossing_progress(
            self._foot_signed_distance,
            self.cfg.progress_approach_distance,
            self.cfg.pass_margin,
        )
        root_progress = normalized_crossing_progress(
            self._root_signed_distance,
            self.cfg.progress_approach_distance,
            self.cfg.root_pass_margin,
        )
        projected_gravity = torch.nan_to_num(
            self.robot.data.projected_gravity_b,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        upright = torch.linalg.vector_norm(projected_gravity[:, :2], dim=1) < math.sin(
            self.cfg.max_tilt_angle
        )
        local_root_z = root_pos[:, 2] - self.scene.env_origins[:, 2]
        safe_posture = (
            upright
            & (local_root_z >= self.cfg.termination_height_low)
            & (local_root_z <= self.cfg.termination_height_high)
            & ~self._forbidden_contact
        )
        before_bar_or_cleared = (
            self._foot_signed_distance <= -0.5 * self.cfg.crossbar_width
        ) | (self._foot_crossing_state >= 1)
        foot_progress_valid = (
            self._foot_rail_valid & before_bar_or_cleared & safe_posture[:, None]
        )
        root_centered = (
            torch.abs(root_pos[:, 1] - self.scene.env_origins[:, 1])
            <= self.cfg.root_center_y_tolerance
        )
        root_progress_valid = safe_posture & root_centered
        foot_gain, new_foot_max = safe_monotonic_progress_gain(
            foot_progress, self._max_foot_progress, foot_progress_valid
        )
        root_gain, new_root_max = safe_monotonic_progress_gain(
            root_progress, self._max_root_progress, root_progress_valid
        )
        foot_gain[initializing] = 0.0
        root_gain[initializing] = 0.0
        new_foot_max[initializing] = torch.where(
            foot_progress_valid[initializing],
            foot_progress[initializing],
            torch.zeros_like(foot_progress[initializing]),
        )
        new_root_max[initializing] = torch.where(
            root_progress_valid[initializing],
            root_progress[initializing],
            torch.zeros_like(root_progress[initializing]),
        )
        self._foot_progress_gain.copy_(foot_gain)
        self._root_progress_gain.copy_(root_gain)
        self._max_foot_progress.copy_(new_foot_max)
        self._max_root_progress.copy_(new_root_max)

        self._foot_forward_delta.copy_((feet_pos[:, :, 0] - self._previous_foot_x).clamp_min(0.0))
        self._foot_forward_delta[initializing] = 0.0
        self._previous_foot_x.copy_(feet_pos[:, :, 0])

        front_now = torch.all(foot_passed[:, :2], dim=1)
        hind_now = torch.all(foot_passed[:, 2:], dim=1)
        root_now = (
            (self._root_signed_distance > self.cfg.root_pass_margin)
            & root_centered
            & safe_posture
        )
        self._front_event.copy_(front_now & ~self._front_complete)
        self._hind_event.copy_(hind_now & ~self._hind_complete)
        self._root_event.copy_(root_now & ~self._root_complete)
        self._front_event[initializing] = False
        self._hind_event[initializing] = False
        self._root_event[initializing] = False
        self._early_front_event |= (
            self._front_event
            & (self._root_signed_distance < -self.cfg.early_front_event_root_distance)
        )
        self._front_complete |= front_now
        self._hind_complete |= hind_now
        self._root_complete |= root_now

        self._landing_condition.copy_(
            curriculum_landing_condition(
                self._curriculum_level,
                foot_passed,
                self._foot_rail_valid,
                self._foot_contacts,
                final_level=self.cfg.curriculum_max_level,
            )
        )
        stable_now = self._landing_condition & root_now
        self._stable_contact_steps.copy_(
            torch.where(stable_now, self._stable_contact_steps + 1, torch.zeros_like(self._stable_contact_steps))
        )
        required_steps = math.ceil(self.cfg.stable_landing_s / self.step_dt)
        new_stable_max = torch.maximum(
            self._max_stable_contact_steps, self._stable_contact_steps
        )
        self._stable_contact_progress_gain.copy_(
            (new_stable_max - self._max_stable_contact_steps).float()
            / max(required_steps, 1)
        )
        self._max_stable_contact_steps.copy_(new_stable_max)
        self._forbidden_contact_steps.copy_(
            torch.where(
                self._forbidden_contact,
                self._forbidden_contact_steps + 1,
                torch.zeros_like(self._forbidden_contact_steps),
            )
        )

        if self.cfg.walk_only:
            total_progress = (
                root_pos[:, 0] - self._episode_start_root_x
            ).clamp_min(0.0)
        else:
            total_progress = torch.sum(self._max_foot_progress, dim=1) + self._max_root_progress
        made_progress = (total_progress - self._progress_anchor) >= self.cfg.no_progress_epsilon
        self._progress_anchor[initializing] = total_progress[initializing]
        made_progress[initializing] = True
        self._progress_anchor.copy_(torch.where(made_progress, total_progress, self._progress_anchor))
        grace = self._level_no_progress_grace[self._curriculum_level]
        after_grace = self.episode_length_buf.float() * self.step_dt >= grace
        self._no_progress_steps.copy_(
            torch.where(
                made_progress | ~after_grace,
                torch.zeros_like(self._no_progress_steps),
                self._no_progress_steps + 1,
            )
        )

        self._success.copy_(
            curriculum_crossing_success(
                root_now,
                self._landing_condition,
                self._stable_contact_steps,
                upright,
                self._forbidden_contact,
                required_steps,
            )
        )
        if self.cfg.walk_only:
            self._success.fill_(False)
        self._foot_air_time.copy_(
            torch.where(
                self._foot_contacts,
                torch.zeros_like(self._foot_air_time),
                self._foot_air_time + self.step_dt,
            )
        )
        self._episode_contact_steps += self._foot_contacts.float()
        self._episode_foot_bar_contact_steps += torch.any(self._foot_bar_contact, dim=1).float()
        self._episode_rail_valid_steps += self._foot_rail_valid.float()
        self._episode_rail_y_error_sum += torch.abs(self._foot_rail_y_error)
        self._episode_landing_condition_steps += self._landing_condition.float()
        self._progress_initialized.fill_(True)

    def _empty_depth_bar_input(self) -> torch.Tensor:
        grid = int(self.cfg.depth_bar_heightmap_size)
        return torch.zeros((self.num_envs, 2, grid, grid), dtype=torch.float32, device=self.device)

    def _camera_depth_m(self, camera: TiledCamera, camera_cfg) -> tuple[torch.Tensor, torch.Tensor]:
        height = int(camera_cfg.height)
        width = int(camera_cfg.width)
        if "depth" not in camera.data.output:
            depth = torch.zeros((self.num_envs, height, width), dtype=torch.float32, device=self.device)
            return depth, torch.zeros_like(depth, dtype=torch.bool)

        depth = camera.data.output["depth"]
        while depth.ndim > 4:
            depth = depth[:, 0]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        depth = depth.to(dtype=torch.float32)

        near = float(camera_cfg.spawn.clipping_range[0])
        far = float(camera_cfg.spawn.clipping_range[1])
        max_depth = min(float(self.cfg.depth_bar_max_depth), far)
        valid = torch.isfinite(depth) & (depth > near + 1.0e-4) & (depth < max_depth - 1.0e-4)
        depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=0.0)
        depth = depth.clamp(0.0, max_depth)
        return depth, valid

    def _camera_pixel_grid(self, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (height, width)
        if key not in self._depth_bar_pixel_grids:
            v, u = torch.meshgrid(
                torch.arange(height, dtype=torch.float32, device=self.device),
                torch.arange(width, dtype=torch.float32, device=self.device),
                indexing="ij",
            )
            self._depth_bar_pixel_grids[key] = (u, v)
        return self._depth_bar_pixel_grids[key]

    def _depth_to_base_points(self, depth_m: torch.Tensor, camera_cfg) -> torch.Tensor:
        batch_size, height, width = depth_m.shape
        u, v = self._camera_pixel_grid(height, width)
        focal_length = float(camera_cfg.spawn.focal_length)
        horizontal_aperture = float(camera_cfg.spawn.horizontal_aperture)
        fx = focal_length / horizontal_aperture * float(width)
        fy = focal_length / (horizontal_aperture * float(height) / float(width)) * float(height)
        cx = 0.5 * float(width - 1)
        cy = 0.5 * float(height - 1)

        z = depth_m
        x = (u[None, :, :] - cx) * z / fx
        y = (v[None, :, :] - cy) * z / fy
        points_cam = torch.stack((x, y, z), dim=-1).reshape(batch_size, -1, 3)

        rotation = torch.tensor(camera_cfg.offset.rot, dtype=torch.float32, device=self.device)
        translation = torch.tensor(camera_cfg.offset.pos, dtype=torch.float32, device=self.device)
        rot_base_cam = _quat_wxyz_to_rotmat(rotation).to(dtype=points_cam.dtype)
        return points_cam @ rot_base_cam.T + translation.view(1, 1, 3)

    def _rasterize_heightmap(self, points_b: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, _ = points_b.shape
        grid = int(self.cfg.depth_bar_heightmap_size)
        grid_cells = grid * grid
        x_min, x_max = self.cfg.depth_bar_heightmap_x_range
        y_min, y_max = self.cfg.depth_bar_heightmap_y_range
        z_min, z_max = self.cfg.depth_bar_heightmap_z_range

        x = points_b[..., 0]
        y = points_b[..., 1]
        z = points_b[..., 2].clamp(float(z_min), float(z_max))
        in_grid = (
            valid
            & (x >= float(x_min))
            & (x <= float(x_max))
            & (y >= float(y_min))
            & (y <= float(y_max))
            & (points_b[..., 2] >= float(z_min))
            & (points_b[..., 2] <= float(z_max))
        )

        col = (((x - float(x_min)) / max(float(x_max - x_min), 1.0e-6)) * float(grid)).long()
        row = (((y - float(y_min)) / max(float(y_max - y_min), 1.0e-6)) * float(grid)).long()
        col = col.clamp(0, grid - 1)
        row = row.clamp(0, grid - 1)
        cell_index = row * grid + col
        batch_offsets = torch.arange(batch_size, device=self.device, dtype=torch.long).view(batch_size, 1) * grid_cells
        flat_index = (cell_index + batch_offsets).masked_select(in_grid)
        flat_z = z.masked_select(in_grid)

        height_flat = torch.full(
            (batch_size * grid_cells,),
            float(z_min),
            dtype=torch.float32,
            device=self.device,
        )
        valid_flat = torch.zeros((batch_size * grid_cells,), dtype=torch.bool, device=self.device)
        if flat_index.numel() > 0:
            height_flat.scatter_reduce_(0, flat_index, flat_z, reduce="amax", include_self=True)
            valid_flat[flat_index] = True

        height = height_flat.view(batch_size, 1, grid, grid)
        valid_map = valid_flat.view(batch_size, 1, grid, grid)
        height_norm = (height - float(z_min)) / max(float(z_max - z_min), 1.0e-6)
        height_norm = torch.where(valid_map, height_norm.clamp(0.0, 1.0), torch.zeros_like(height_norm))
        return torch.cat((height_norm, valid_map.float()), dim=1)

    def _depth_bar_input_observation(self) -> torch.Tensor:
        if self._depth_bar_input_cache_step == self.common_step_counter and self._depth_bar_input_cache is not None:
            return self._depth_bar_input_cache
        if (
            not self.cfg.enable_depth_bar_camera
            or not hasattr(self, "front_tiled_camera")
            or not hasattr(self, "rear_tiled_camera")
        ):
            self._depth_bar_input_cache = self._empty_depth_bar_input()
            self._depth_bar_input_cache_step = self.common_step_counter
            return self._depth_bar_input_cache

        point_chunks = []
        valid_chunks = []
        for camera, camera_cfg in (
            (self.front_tiled_camera, self.cfg.tiled_camera),
            (self.rear_tiled_camera, self.cfg.rear_tiled_camera),
        ):
            depth_m, valid = self._camera_depth_m(camera, camera_cfg)
            point_chunks.append(self._depth_to_base_points(depth_m, camera_cfg))
            valid_chunks.append(valid.reshape(self.num_envs, -1))

        points_b = torch.cat(point_chunks, dim=1)
        valid = torch.cat(valid_chunks, dim=1)
        self._depth_bar_input_cache = self._rasterize_heightmap(points_b, valid)
        self._depth_bar_input_cache_step = self.common_step_counter
        return self._depth_bar_input_cache

    def _record_bar_estimate_visibility(self, visible: torch.Tensor) -> None:
        if self._bar_estimate_visibility_log_step == self.common_step_counter:
            return
        self._episode_bar_visible_steps += visible.squeeze(1).float()
        self._episode_bar_visible_samples += 1.0
        self._bar_estimate_visibility_log_step = self.common_step_counter

    def _bar_estimate_observation(self, bar_delta_b: torch.Tensor) -> torch.Tensor:
        if self.cfg.enable_depth_bar_camera and self._depth_bar_estimator is not None:
            depth_obs = self._depth_bar_input_observation()
            with torch.inference_mode():
                estimate_b = self._depth_bar_estimator(depth_obs)
            estimate_b = torch.nan_to_num(
                estimate_b[:, :3].to(dtype=torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            visible = torch.full_like(
                estimate_b[:, 0:1],
                float(self.cfg.depth_bar_visible_value),
            )
            self._record_bar_estimate_visibility(visible)
            return torch.cat(
                (
                    (estimate_b[:, 0:1] / self.cfg.bar_estimate_dx_scale).clamp(-2.0, 2.0),
                    (estimate_b[:, 1:2] / self.cfg.bar_estimate_dy_scale).clamp(-2.0, 2.0),
                    (estimate_b[:, 2:3] / self.cfg.bar_estimate_dz_scale).clamp(-2.0, 2.0),
                    visible,
                ),
                dim=1,
            )

        if not self.cfg.bar_estimate_noise_enabled:
            estimate_b = bar_delta_b
            visible = torch.ones_like(bar_delta_b[:, 0:1])
        else:
            if self._bar_estimate_last_update_step != self.common_step_counter:
                self._bar_estimate_history[:, self._bar_estimate_history_index] = bar_delta_b
                read_index = (
                    self._bar_estimate_history_index - self._bar_estimate_delay
                ) % self._bar_estimate_history.shape[1]
                delayed_b = self._bar_estimate_history[self._all_env_ids, read_index]

                noise = torch.randn_like(delayed_b) * self._bar_estimate_noise_std
                measured_b = delayed_b + self._bar_estimate_bias + noise
                in_fov = (
                    (delayed_b[:, 0] >= self._bar_estimate_fov_x_range[0])
                    & (delayed_b[:, 0] <= self._bar_estimate_fov_x_range[1])
                    & (delayed_b[:, 1] >= self._bar_estimate_fov_y_range[0])
                    & (delayed_b[:, 1] <= self._bar_estimate_fov_y_range[1])
                    & (delayed_b[:, 2] >= self._bar_estimate_fov_z_range[0])
                    & (delayed_b[:, 2] <= self._bar_estimate_fov_z_range[1])
                )
                dropout = (
                    torch.rand(self.num_envs, device=self.device)
                    < self.cfg.bar_estimate_dropout_prob
                )
                visible_bool = in_fov & ~dropout

                alpha = float(self.cfg.bar_estimate_lowpass_alpha)
                alpha = max(0.0, min(alpha, 1.0))
                filtered_b = alpha * measured_b + (1.0 - alpha) * self._bar_estimate_filtered
                self._bar_estimate_filtered = torch.where(
                    visible_bool.unsqueeze(1), filtered_b, self._bar_estimate_filtered
                )
                self._bar_estimate_visible = visible_bool.float().unsqueeze(1)
                self._bar_estimate_history_index = (
                    self._bar_estimate_history_index + 1
                ) % self._bar_estimate_history.shape[1]
                self._bar_estimate_last_update_step = self.common_step_counter

            estimate_b = self._bar_estimate_filtered
            visible = self._bar_estimate_visible

        self._record_bar_estimate_visibility(visible)
        return torch.cat(
            (
                (estimate_b[:, 0:1] / self.cfg.bar_estimate_dx_scale).clamp(-2.0, 2.0),
                (estimate_b[:, 1:2] / self.cfg.bar_estimate_dy_scale).clamp(-2.0, 2.0),
                (estimate_b[:, 2:3] / self.cfg.bar_estimate_dz_scale).clamp(-2.0, 2.0),
                visible,
            ),
            dim=1,
        )

    def _reset_bar_estimate_state(
        self,
        env_ids: torch.Tensor,
        root_state: torch.Tensor,
        bar_pose: torch.Tensor,
    ) -> None:
        bar_delta_w = bar_pose[:, :3] - root_state[:, :3]
        bar_delta_b = torch.nan_to_num(
            quat_apply_inverse(root_state[:, 3:7], bar_delta_w),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self._bar_estimate_history[env_ids] = bar_delta_b[:, None, :].expand(
            -1, self._bar_estimate_history.shape[1], -1
        )
        self._bar_estimate_filtered[env_ids] = bar_delta_b
        self._bar_estimate_visible[env_ids] = 1.0
        if self.cfg.bar_estimate_max_delay_steps > 0:
            self._bar_estimate_delay[env_ids] = torch.randint(
                0,
                int(self.cfg.bar_estimate_max_delay_steps) + 1,
                (env_ids.numel(),),
                device=self.device,
            )
        else:
            self._bar_estimate_delay[env_ids] = 0

        if torch.any(self._bar_estimate_bias_range > 0.0):
            self._bar_estimate_bias[env_ids] = (
                2.0 * torch.rand((env_ids.numel(), 3), device=self.device) - 1.0
            ) * self._bar_estimate_bias_range
        else:
            self._bar_estimate_bias[env_ids] = 0.0

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._refresh_crossing_state(update_trackers=False)

        joint_pos = torch.nan_to_num(
            self.robot.data.joint_pos[:, self._main_joint_ids],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        joint_vel = torch.nan_to_num(
            self.robot.data.joint_vel[:, self._main_joint_ids],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if self.cfg.domain_randomization_enabled:
            joint_pos = joint_pos + self._joint_position_bias
            joint_pos = joint_pos + torch.randn_like(joint_pos) * self.cfg.joint_position_noise_std
            joint_vel = joint_vel + torch.randn_like(joint_vel) * self.cfg.joint_velocity_noise_std

        joint_pos_norm = ((joint_pos - self._action_offset) / self._action_scale.clamp_min(1.0e-6)).clamp(
            -2.0, 2.0
        )
        joint_vel_norm = (joint_vel / 10.0).clamp(-2.0, 2.0)
        projected_gravity = torch.nan_to_num(
            self.robot.data.projected_gravity_b,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        angular_velocity = torch.nan_to_num(
            self.robot.data.root_ang_vel_b / 5.0,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(-2.0, 2.0)
        phase = self._leg_phase()
        gait_clock = torch.stack(
            (
                torch.sin(2.0 * torch.pi * phase[:, 0]),
                torch.cos(2.0 * torch.pi * phase[:, 0]),
            ),
            dim=1,
        )
        deployable_terms = [
            joint_pos_norm,
            joint_vel_norm,
            projected_gravity,
            angular_velocity,
            gait_clock,
            self._raw_actions,
        ]
        if self.cfg.include_gripper_position_obs:
            gripper_angle = torch.nan_to_num(
                self.robot.data.joint_pos[:, self._gripper_joint_ids].view(self.num_envs, 4, 2).mean(dim=-1),
                nan=float(self.cfg.gripper_closed_angle),
                posinf=float(self.cfg.gripper_open_angle),
                neginf=float(self.cfg.gripper_closed_angle),
            )
            gripper_angle = gripper_angle.clamp(
                min(float(self.cfg.gripper_closed_angle), float(self.cfg.gripper_open_angle)),
                max(float(self.cfg.gripper_closed_angle), float(self.cfg.gripper_open_angle)),
            )
            deployable_terms.append(gripper_angle)
        deployable = torch.cat(tuple(deployable_terms), dim=1)

        root_pos = torch.nan_to_num(
            self.robot.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0
        )
        root_quat = torch.nan_to_num(
            self.robot.data.root_quat_w, nan=0.0, posinf=0.0, neginf=0.0
        )
        bar_pos = torch.nan_to_num(
            self.crossbar.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0
        )
        bar_delta_w = bar_pos - root_pos
        bar_delta_b = torch.nan_to_num(
            quat_apply_inverse(root_quat, bar_delta_w),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        bar_estimate = self._bar_estimate_observation(bar_delta_b)
        stage_one_hot = F.one_hot(self._stage, num_classes=4).float()
        teacher_privileged = torch.cat(
            (
                torch.nan_to_num(
                    self.robot.data.root_lin_vel_b / 0.2,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clamp(-3.0, 3.0),
                (bar_delta_b / 1.2).clamp(-2.0, 2.0),
                self._bar_dimensions / 0.1,
                (self._foot_signed_distance / 1.2).clamp(-2.0, 2.0),
                stage_one_hot,
            ),
            dim=1,
        )
        policy_obs = torch.cat((deployable, teacher_privileged), dim=1)

        feet_pos_b = self._foot_pos_b.reshape(self.num_envs, 12)
        base_clearance = (
            root_pos[:, 2] - self.scene.env_origins[:, 2] - self.cfg.rail_top_height
        ).unsqueeze(1)
        critic_extra = torch.cat(
            (
                feet_pos_b,
                self._foot_contacts.float(),
                self._foot_clearance,
                base_clearance,
                (self._forbidden_contact_force / 20.0).unsqueeze(1),
            ),
            dim=1,
        )
        policy_obs = torch.nan_to_num(policy_obs, nan=0.0, posinf=0.0, neginf=0.0)
        critic_obs = torch.nan_to_num(
            torch.cat((policy_obs, critic_extra), dim=1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        observations = {
            "policy": policy_obs,
            "critic": critic_obs,
            "deployable": deployable,
            "bar_estimate": bar_estimate,
            "bar_estimate_label": bar_delta_b,
        }
        if self.cfg.enable_depth_bar_camera:
            observations["heightmap"] = self._depth_bar_input_observation()
        return observations

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_crossing_state(update_trackers=True)

        local_root_z = self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        projected_gravity = torch.nan_to_num(
            self.robot.data.projected_gravity_b,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self._tilt_failure.copy_(
            torch.linalg.vector_norm(projected_gravity[:, :2], dim=1)
            > math.sin(self.cfg.max_tilt_angle)
        )
        root_quat = torch.nan_to_num(
            self.robot.data.root_quat_w,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        root_forward_w = torch.nan_to_num(
            quat_apply(root_quat, self._forward_axis),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        yaw_error = torch.abs(torch.atan2(root_forward_w[:, 1], root_forward_w[:, 0]))
        self._yaw_failure.copy_(yaw_error > self.cfg.max_yaw_angle)
        self._lateral_failure.copy_(
            torch.abs(self.robot.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1])
            > self.cfg.max_root_lateral_deviation
        )
        root_ang_vel = torch.nan_to_num(
            self.robot.data.root_ang_vel_b,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self._angular_velocity_failure.copy_(
            torch.linalg.vector_norm(root_ang_vel, dim=1)
            > self.cfg.max_root_angular_velocity
        )
        self._invalid_failure.copy_(
            ~torch.isfinite(self.robot.data.root_state_w).all(dim=1)
            | ~torch.isfinite(
                self.robot.data.joint_pos[:, self._main_joint_ids]
            ).all(dim=1)
            | ~torch.isfinite(
                self.robot.data.joint_vel[:, self._main_joint_ids]
            ).all(dim=1)
            | ~torch.isfinite(
                self.robot.data.body_pos_w[:, self._foot_body_ids]
            ).all(dim=(1, 2))
        )
        self._height_low_failure.copy_(local_root_z < self.cfg.termination_height_low)
        self._height_high_failure.copy_(local_root_z > self.cfg.termination_height_high)
        self._forbidden_failure.copy_(
            self._forbidden_contact_steps >= self.cfg.forbidden_contact_terminate_steps
        )
        patience = self._level_no_progress_patience[self._curriculum_level]
        self._no_progress_failure.copy_(
            self._no_progress_steps.float() * self.step_dt >= patience
        )
        self._failure.copy_(
            self._invalid_failure
            | self._tilt_failure
            | self._yaw_failure
            | self._lateral_failure
            | self._angular_velocity_failure
            | self._height_low_failure
            | self._height_high_failure
            | self._forbidden_failure
            | self._no_progress_failure
        )
        terminated = self._failure | self._success
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        root_vel_x = torch.nan_to_num(
            self.robot.data.root_lin_vel_b[:, 0],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        velocity_gate = torch.tanh((root_vel_x / 0.03).clamp_min(0.0))
        velocity_range_error = (
            (self.cfg.forward_velocity_min - root_vel_x).clamp_min(0.0)
            + (root_vel_x - self.cfg.forward_velocity_max).clamp_min(0.0)
        )
        forward_velocity = velocity_gate * torch.exp(
            -velocity_range_error / self.cfg.forward_velocity_sigma
        )
        forward_speed_limit = torch.square(
            (
                (root_vel_x - self.cfg.forward_speed_limit).clamp_min(0.0)
                / self.cfg.forward_speed_limit_sigma
            ).clamp_max(3.0)
        )

        projected_gravity = torch.nan_to_num(
            self.robot.data.projected_gravity_b,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        gravity_xy = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
        static_upright = torch.exp(-gravity_xy / 0.05)
        upright = static_upright * velocity_gate
        forward_axis_w = torch.nan_to_num(
            quat_apply(
                torch.nan_to_num(
                    self.robot.data.root_quat_w,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                self._forward_axis,
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        static_heading = torch.exp(
            -torch.sum(torch.square(forward_axis_w[:, 1:]), dim=1) / 0.05
        )
        heading = static_heading * velocity_gate
        expected_stance = self._leg_phase() < self.cfg.gait_duty_factor
        expected_swing = ~expected_stance
        if self._learn_gripper_actions:
            gripper_pos = torch.nan_to_num(
                self.robot.data.joint_pos[:, self._gripper_joint_ids].view(self.num_envs, 4, 2).mean(dim=-1),
                nan=float(self.cfg.gripper_closed_angle),
                posinf=float(self.cfg.gripper_open_angle),
                neginf=float(self.cfg.gripper_closed_angle),
            )
            gripper_open_fraction = (
                (gripper_pos - float(self.cfg.gripper_closed_angle))
                / max(float(self.cfg.gripper_open_angle - self.cfg.gripper_closed_angle), 1.0e-6)
            ).clamp(0.0, 1.0)
            desired_open_fraction = expected_swing.float()
            gripper_phase = torch.mean(
                torch.square(gripper_open_fraction - desired_open_fraction), dim=1
            )
        else:
            gripper_phase = torch.zeros(self.num_envs, device=self.device)
        stance_count = expected_stance.sum(dim=1).float().clamp_min(1.0)
        swing_count = expected_swing.sum(dim=1).float().clamp_min(1.0)
        contact_count = self._foot_contacts.sum(dim=1).float()
        expected_contact_count = expected_stance.sum(dim=1).float()
        support = torch.exp(-torch.square(contact_count - expected_contact_count))
        support *= velocity_gate

        stance_contact = (
            (self._foot_contacts & expected_stance).sum(dim=1).float()
            / stance_count
        )
        swing_clear = (
            ((~self._foot_contacts) & expected_swing).sum(dim=1).float()
            / swing_count
        )
        # Stance and swing each contribute half, so keeping every foot in
        # contact cannot collect most of the gait reward.
        gait_contact = 0.5 * (stance_contact + swing_clear) * velocity_gate

        rail_center_quality = torch.exp(
            -torch.abs(self._foot_rail_y_error)
            / self.cfg.landing_rail_center_sigma
        )
        stance_on_rail = expected_stance & self._foot_contacts & self._foot_rail_valid
        walk_rail_center = (
            torch.sum(rail_center_quality * stance_on_rail.float(), dim=1)
            / stance_count
        ) * velocity_gate
        walk_contact = (
            stance_on_rail.sum(dim=1).float() / stance_count
        ) * velocity_gate

        swing_clearance_error = torch.abs(
            self._foot_clearance - self.cfg.walk_swing_clearance_target
        ).clamp_max(self.cfg.walk_swing_clearance_error_max)
        walk_swing_clearance_error = (
            torch.sum(swing_clearance_error * expected_swing.float(), dim=1)
            / swing_count
        )

        foot_x = self._foot_pos_b[:, :, 0]
        foot_y = self._foot_pos_b[:, :, 1]
        foothold_x_low = self._walk_foothold_ranges[:, 0]
        foothold_x_high = self._walk_foothold_ranges[:, 1]
        foothold_y_low = self._walk_foothold_ranges[:, 2]
        foothold_y_high = self._walk_foothold_ranges[:, 3]
        foothold_dx = (
            (foothold_x_low[None, :] - foot_x).clamp_min(0.0)
            + (foot_x - foothold_x_high[None, :]).clamp_min(0.0)
        )
        foothold_dy = (
            (foothold_y_low[None, :] - foot_y).clamp_min(0.0)
            + (foot_y - foothold_y_high[None, :]).clamp_min(0.0)
        )
        foothold_quality = torch.exp(
            -torch.sqrt(torch.square(foothold_dx) + torch.square(foothold_dy))
            / self.cfg.foothold_distance_sigma
        )
        valid_touchdown = (
            self._foot_touchdown
            & (self._foot_touchdown_air_time >= self.cfg.foothold_min_air_time)
            & expected_stance
            & self._foot_rail_valid
        )
        touchdown_count = valid_touchdown.sum(dim=1).float()
        walk_foothold = torch.where(
            touchdown_count > 0,
            torch.sum(foothold_quality * valid_touchdown.float(), dim=1)
            / touchdown_count.clamp_min(1.0),
            torch.zeros_like(touchdown_count),
        )
        self._episode_stance_contact_score += stance_contact
        self._episode_swing_clear_score += swing_clear
        self._episode_valid_touchdowns += touchdown_count
        self._episode_swing_clearance_sum += torch.sum(
            self._foot_clearance * expected_swing.float(), dim=1
        )
        self._episode_swing_clearance_samples += swing_count

        local_root_height = (
            self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        )
        walk_base_height = torch.exp(
            -torch.square(
                (local_root_height - self.cfg.walk_base_height_target)
                / self.cfg.walk_base_height_sigma
            )
        ) * velocity_gate
        root_linear_velocity = torch.nan_to_num(
            self.robot.data.root_lin_vel_b,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(
            -self.cfg.reward_root_linear_velocity_max,
            self.cfg.reward_root_linear_velocity_max,
        )
        root_angular_velocity = torch.nan_to_num(
            self.robot.data.root_ang_vel_b,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(
            -self.cfg.reward_root_angular_velocity_max,
            self.cfg.reward_root_angular_velocity_max,
        )
        base_motion = (
            torch.square(root_linear_velocity[:, 1])
            + torch.square(root_linear_velocity[:, 2])
            + 0.1 * torch.sum(torch.square(root_angular_velocity[:, :2]), dim=1)
            + 0.05 * torch.square(root_angular_velocity[:, 2])
        ).clamp_max(self.cfg.reward_base_motion_max)

        near_bar = torch.abs(self._foot_signed_distance) < self.cfg.clearance_x_window
        advancing = self._foot_forward_delta > 1.0e-4
        swing = ~self._foot_contacts
        clearance_quality = torch.exp(
            -torch.abs(self._foot_clearance - self.cfg.clearance_target) / 0.04
        )
        clearance = torch.mean(
            clearance_quality
            * near_bar
            * advancing
            * swing
            * self._foot_rail_valid,
            dim=1,
        )
        low_bar_near = torch.abs(self._foot_signed_distance) < self.cfg.low_bar_clearance_x_window
        low_bar_advancing = self._foot_forward_delta > self.cfg.low_bar_clearance_min_forward_delta
        low_bar_active = low_bar_near & low_bar_advancing & expected_swing & self._foot_rail_valid
        low_bar_deficit = (self.cfg.low_bar_clearance_target - self._foot_clearance).clamp_min(0.0)
        low_bar_scaled_deficit = (
            low_bar_deficit / max(float(self.cfg.low_bar_clearance_sigma), 1.0e-6)
        ).clamp_max(3.0)
        low_bar_penalty_per_foot = torch.square(low_bar_scaled_deficit) * low_bar_active.float()
        low_bar_clearance = torch.mean(low_bar_penalty_per_foot, dim=1)
        low_bar_violation = low_bar_active & (low_bar_deficit > 0.0)
        self._episode_low_bar_clearance_steps += torch.any(low_bar_violation, dim=1).float()
        self._episode_low_bar_clearance_deficit_sum += torch.sum(
            low_bar_deficit * low_bar_violation.float(), dim=1
        )
        self._episode_low_bar_clearance_samples += low_bar_violation.sum(dim=1).float()
        level_feet_complete = torch.where(
            self._curriculum_level == 0,
            self._hind_complete,
            self._front_complete & self._hind_complete,
        )
        landing_active = level_feet_complete & (
            self._root_signed_distance > self.cfg.root_pass_margin
        )
        landing_rail_center = landing_active.float() * torch.mean(
            rail_center_quality, dim=1
        )
        landing_contact = landing_active.float() * torch.mean(
            rail_center_quality * self._foot_contacts.float(), dim=1
        )
        valid_support_fraction = (
            torch.sum(self._foot_contacts & self._foot_rail_valid, dim=1).float()
            / 3.0
        ).clamp(0.0, 1.0)
        root_speed = torch.linalg.vector_norm(
            torch.nan_to_num(
                self.robot.data.root_lin_vel_b,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            dim=1,
        )
        low_speed = torch.exp(-root_speed / self.cfg.landing_velocity_sigma)
        root_angular_speed = torch.linalg.vector_norm(root_angular_velocity, dim=1)
        low_angular_speed = torch.exp(
            -root_angular_speed / self.cfg.landing_angular_velocity_sigma
        )
        landing_stability = (
            landing_active.float()
            * static_upright
            * static_heading
            * low_speed
            * low_angular_speed
            * valid_support_fraction
        )

        joint_pos = torch.nan_to_num(
            self.robot.data.joint_pos[:, self._main_joint_ids],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        joint_vel = torch.nan_to_num(
            self.robot.data.joint_vel[:, self._main_joint_ids],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(
            -self.cfg.reward_joint_velocity_max,
            self.cfg.reward_joint_velocity_max,
        )
        normalized_joint_position = torch.abs(
            (joint_pos - self._action_offset) / self._action_scale.clamp_min(1.0e-6)
        ).clamp_max(self.cfg.reward_joint_position_norm_max)
        action_magnitude = torch.sum(
            torch.square(self._raw_actions - self._nominal_action), dim=1
        )
        action_soft_limit = torch.sum(
            torch.square(
                (
                    torch.abs(self._raw_actions)
                    - self.cfg.action_soft_limit
                ).clamp_min(0.0)
            ),
            dim=1,
        )
        action_rate = torch.sum(torch.square(self._raw_actions - self._previous_actions), dim=1)
        action_accel = torch.sum(
            torch.square(
                self._raw_actions
                - 2.0 * self._previous_actions
                + self._previous_previous_actions
            ),
            dim=1,
        )
        joint_limit = torch.sum((normalized_joint_position - 0.90).clamp_min(0.0), dim=1)
        joint_velocity = torch.sum(torch.square(joint_vel), dim=1)
        torque = torch.nan_to_num(
            self.robot.data.applied_torque[:, self._main_joint_ids],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(-self.cfg.reward_torque_max, self.cfg.reward_torque_max)
        energy = torch.sum(torch.abs(torque * joint_vel), dim=1)

        feet_velocity = torch.nan_to_num(
            self.robot.data.body_lin_vel_w[:, self._foot_body_ids],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(
            -self.cfg.reward_foot_velocity_max,
            self.cfg.reward_foot_velocity_max,
        )
        slip = torch.mean(
            torch.linalg.vector_norm(feet_velocity[:, :, :2], dim=-1)
            * self._foot_contacts,
            dim=1,
        )
        touchdown_speed = torch.linalg.vector_norm(
            feet_velocity.clamp(
                -self.cfg.reward_touchdown_velocity_max,
                self.cfg.reward_touchdown_velocity_max,
            ),
            dim=-1,
        )
        touchdown_count = self._foot_touchdown.sum(dim=1).float()
        touchdown_impact = torch.where(
            touchdown_count > 0.0,
            torch.sum(touchdown_speed * self._foot_touchdown.float(), dim=1)
            / touchdown_count.clamp_min(1.0),
            torch.zeros_like(touchdown_count),
        )
        long_air = ((self._foot_air_time - 1.2) / 1.8).clamp(0.0, 1.0).square()
        no_progress = (self._no_progress_steps.float() * self.step_dt > 0.5).float()
        long_air_no_progress = torch.mean(long_air, dim=1) * no_progress
        foot_bar_contact = torch.mean(self._foot_bar_contact.float(), dim=1)
        crossing_reward_valid = (~self._failure).float()

        rewards = {
            "forward_velocity": self.cfg.rew_forward_velocity * forward_velocity * self.step_dt,
            "upright": self.cfg.rew_upright * upright * self.step_dt,
            "heading": self.cfg.rew_heading * heading * self.step_dt,
            "support": self.cfg.rew_support * support * self.step_dt,
            "gait_contact": self.cfg.rew_gait_contact * gait_contact * self.step_dt,
            "walk_rail_center": self.cfg.rew_walk_rail_center
            * walk_rail_center
            * self.step_dt,
            "walk_contact": self.cfg.rew_walk_contact * walk_contact * self.step_dt,
            "walk_swing_clearance_error": self.cfg.rew_walk_swing_clearance_error
            * walk_swing_clearance_error
            * self.step_dt,
            # Touchdown shaping is an event reward and is not integrated by dt.
            "walk_foothold": self.cfg.rew_walk_foothold * walk_foothold,
            "touchdown_impact": self.cfg.rew_touchdown_impact * touchdown_impact,
            "walk_base_height": self.cfg.rew_walk_base_height
            * walk_base_height
            * self.step_dt,
            "base_motion": self.cfg.rew_base_motion * base_motion * self.step_dt,
            "forward_speed_limit": self.cfg.rew_forward_speed_limit
            * forward_speed_limit
            * self.step_dt,
            "gripper_phase": self.cfg.rew_gripper_phase
            * gripper_phase
            * self.step_dt,
            "front_progress": self.cfg.rew_front_progress
            * torch.sum(self._foot_progress_gain[:, :2], dim=1)
            * crossing_reward_valid,
            "hind_progress": self.cfg.rew_hind_progress
            * torch.sum(self._foot_progress_gain[:, 2:], dim=1)
            * crossing_reward_valid,
            "root_progress": self.cfg.rew_root_progress
            * self._root_progress_gain
            * crossing_reward_valid,
            "clearance": self.cfg.rew_clearance
            * clearance
            * crossing_reward_valid
            * self.step_dt,
            "landing_rail_center": self.cfg.rew_landing_rail_center
            * landing_rail_center
            * crossing_reward_valid
            * self.step_dt,
            "landing_contact": self.cfg.rew_landing_contact
            * landing_contact
            * crossing_reward_valid
            * self.step_dt,
            "landing_stability": self.cfg.rew_landing_stability
            * landing_stability
            * crossing_reward_valid
            * self.step_dt,
            "stable_contact_progress": self.cfg.rew_stable_contact_progress
            * self._stable_contact_progress_gain
            * crossing_reward_valid,
            "front_event": self.cfg.rew_front_event
            * self._front_event
            * crossing_reward_valid,
            "hind_event": self.cfg.rew_hind_event
            * self._hind_event
            * crossing_reward_valid,
            "root_event": self.cfg.rew_root_event
            * self._root_event
            * crossing_reward_valid,
            "success_event": self.cfg.rew_success_event * self._success,
            "action_magnitude": self.cfg.rew_action_magnitude
            * action_magnitude
            * self.step_dt,
            "action_soft_limit": self.cfg.rew_action_soft_limit
            * action_soft_limit
            * self.step_dt,
            "action_rate": self.cfg.rew_action_rate * action_rate * self.step_dt,
            "action_accel": self.cfg.rew_action_accel * action_accel * self.step_dt,
            "joint_limit": self.cfg.rew_joint_limit * joint_limit * self.step_dt,
            "joint_velocity": self.cfg.rew_joint_velocity * joint_velocity * self.step_dt,
            "energy": self.cfg.rew_energy * energy * self.step_dt,
            "slip": self.cfg.rew_slip * slip * self.step_dt,
            "long_air_no_progress": self.cfg.rew_long_air_no_progress
            * long_air_no_progress
            * self.step_dt,
            "forbidden_contact": self.cfg.rew_forbidden_contact
            * self._forbidden_contact.float()
            * self.step_dt,
            "foot_bar_contact": self.cfg.rew_foot_bar_contact
            * foot_bar_contact
            * crossing_reward_valid
            * self.step_dt,
            "low_bar_clearance": self.cfg.rew_low_bar_clearance
            * low_bar_clearance
            * crossing_reward_valid
            * self.step_dt,
            "termination": self.cfg.rew_termination * self._failure.float(),
        }
        nonfinite_reward = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        for name, value in rewards.items():
            nonfinite_reward |= ~torch.isfinite(value)
            rewards[name] = torch.nan_to_num(
                value, nan=0.0, posinf=0.0, neginf=0.0
            )
        self._nonfinite_reward_steps += nonfinite_reward.long()

        for name, value in rewards.items():
            self._episode_sums[name] += value
        return torch.stack(tuple(rewards.values()), dim=0).sum(dim=0)

    def _sample_balanced_curriculum_levels(self, count: int) -> torch.Tensor:
        return torch.multinomial(
            self._balanced_curriculum_level_weights,
            count,
            replacement=True,
        ).to(dtype=torch.long)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = env_ids.to(device=self.device, dtype=torch.long)

        episode_lengths = self.episode_length_buf[env_ids].clone()
        completed_episode = episode_lengths > 0
        old_levels = self._curriculum_level[env_ids].clone()
        episode_success = self._success[env_ids].float()

        active_ids = env_ids[completed_episode]
        active_success = episode_success[completed_episode]
        alpha = self.cfg.curriculum_ema_alpha
        self._curriculum_success_ema[active_ids] = (
            (1.0 - alpha) * self._curriculum_success_ema[active_ids]
            + alpha * active_success
        )
        self._curriculum_episode_count[active_ids] += 1
        self._failure_streak[active_ids] = torch.where(
            active_success > 0.5,
            torch.zeros_like(self._failure_streak[active_ids]),
            self._failure_streak[active_ids] + 1,
        )
        balanced_curriculum = bool(
            self.cfg.curriculum_enabled and self.cfg.balanced_curriculum_enabled
        )
        if self.cfg.curriculum_enabled and not balanced_curriculum:
            new_levels = update_curriculum_levels(
                self._curriculum_level[active_ids],
                self._curriculum_success_ema[active_ids],
                self._curriculum_episode_count[active_ids],
                self._failure_streak[active_ids],
                min_episodes=self.cfg.curriculum_min_episodes,
                promote_threshold=self.cfg.curriculum_promote_threshold,
                demote_threshold=self.cfg.curriculum_demote_threshold,
                max_level=self.cfg.curriculum_max_level,
                failure_streak_limit=self.cfg.curriculum_failure_streak,
            )
            changed = new_levels != self._curriculum_level[active_ids]
            self._curriculum_level[active_ids] = new_levels
            self._curriculum_episode_count[active_ids[changed]] = 0
            self._failure_streak[active_ids[changed]] = 0
            self._curriculum_success_ema[active_ids[changed]] = 0.5

        self._log_resets(env_ids, completed_episode, old_levels, episode_lengths)

        self.robot.reset(env_ids)
        self.crossbar.reset(env_ids)
        super()._reset_idx(env_ids)

        count = env_ids.numel()
        if balanced_curriculum:
            self._curriculum_level[env_ids] = self._sample_balanced_curriculum_levels(count)
        self._gait_phase_origin[env_ids] = torch.rand(count, device=self.device)
        self._gripper_open[env_ids] = False
        self._gripper_latched_actions[env_ids] = float(self.cfg.gripper_action_latch_initial_action)
        self._gripper_latch_steps_remaining[env_ids] = 0

        if self.cfg.max_action_delay_steps > 0:
            self._action_delay[env_ids] = torch.randint(
                0,
                self.cfg.max_action_delay_steps + 1,
                (count,),
                device=self.device,
            )
        else:
            self._action_delay[env_ids] = 0
        self._joint_position_bias[env_ids].uniform_(
            -self.cfg.joint_position_bias_range,
            self.cfg.joint_position_bias_range,
        )

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 0] = (
            self.scene.env_origins[env_ids, 0] + self.cfg.reset_root_x_offset
        )
        root_state[:, 1] = self.scene.env_origins[env_ids, 1]
        root_state[:, 7:] = 0.0
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel[env_ids])

        level_zero = self._curriculum_level[env_ids] == 0
        if self._reset_motion is not None:
            use_motion = level_zero & (
                torch.rand(count, device=self.device) < self.cfg.hind_reset_pose_probability
            )
            motion_rows = use_motion.nonzero(as_tuple=False).squeeze(-1)
            frame_ids = torch.randint(
                0,
                self._reset_motion["joint_pos"].shape[0],
                (count,),
                device=self.device,
            )
            selected_frames = frame_ids[motion_rows]
            joint_pos[motion_rows] = self._reset_motion["joint_pos"][selected_frames]
            root_state[motion_rows, 2] = (
                self.scene.env_origins[env_ids[motion_rows], 2]
                + self._reset_motion["root_z"][selected_frames]
            )
            root_state[motion_rows, 3:7] = self._reset_motion["root_quat"][selected_frames]

        if self.cfg.joint_reset_noise > 0.0:
            noise = torch.empty((count, 20), device=self.device).uniform_(
                -self.cfg.joint_reset_noise, self.cfg.joint_reset_noise
            )
            joint_pos[:, self._main_joint_ids] += noise
            all_limits = self.robot.data.soft_joint_pos_limits[env_ids]
            joint_pos = joint_pos.clamp(all_limits[:, :, 0], all_limits[:, :, 1])
        reset_actions = (
            (joint_pos[:, self._main_joint_ids] - self._action_offset)
            / self._action_scale.clamp_min(1.0e-6)
        ).clamp(-1.0, 1.0)
        policy_reset_actions = torch.zeros(
            (count, self._policy_action_dim), device=self.device
        )
        policy_reset_actions[:, :20] = reset_actions
        self._raw_actions[env_ids] = policy_reset_actions
        self._previous_actions[env_ids] = policy_reset_actions
        self._previous_previous_actions[env_ids] = policy_reset_actions
        self._applied_actions[env_ids] = policy_reset_actions
        self._action_history[env_ids] = policy_reset_actions[:, None, :]
        gripper_pos = joint_pos[:, self._gripper_joint_ids].view(count, 4, 2).mean(dim=-1)
        gripper_pos = gripper_pos.clamp(
            float(self.cfg.gripper_closed_angle), float(self.cfg.gripper_open_angle)
        )
        self._gripper_target[env_ids, 0::2] = gripper_pos
        self._gripper_target[env_ids, 1::2] = gripper_pos

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        selected_ranges = self._level_distance_ranges[self._curriculum_level[env_ids]]
        bar_distance = selected_ranges[:, 0] + torch.rand(count, device=self.device) * (
            selected_ranges[:, 1] - selected_ranges[:, 0]
        )
        bar_pose = torch.zeros((count, 7), device=self.device)
        bar_pose[:, 0] = root_state[:, 0] + bar_distance
        bar_pose[:, 1] = self.scene.env_origins[env_ids, 1]
        bar_pose[:, 2] = (
            self.scene.env_origins[env_ids, 2]
            + self.cfg.rail_top_height
            - 0.5 * self.cfg.crossbar_height
        )
        bar_pose[:, 3] = 1.0
        self.crossbar.write_root_pose_to_sim(bar_pose, env_ids)
        self._bar_x_w[env_ids] = bar_pose[:, 0]
        self._reset_bar_estimate_state(env_ids, root_state, bar_pose)

        self._max_foot_progress[env_ids] = 0.0
        self._max_root_progress[env_ids] = 0.0
        self._foot_progress_gain[env_ids] = 0.0
        self._root_progress_gain[env_ids] = 0.0
        self._progress_anchor[env_ids] = 0.0
        self._episode_start_root_x[env_ids] = root_state[:, 0]
        self._no_progress_steps[env_ids] = 0
        self._previous_foot_x[env_ids] = root_state[:, 0].unsqueeze(1)
        self._foot_forward_delta[env_ids] = 0.0
        self._foot_air_time[env_ids] = 0.0
        self._previous_foot_contacts[env_ids] = False
        self._foot_touchdown[env_ids] = False
        self._foot_touchdown_air_time[env_ids] = 0.0
        self._foot_bar_contact[env_ids] = False
        self._foot_bar_contact_force[env_ids] = 0.0
        self._foot_pos_b[env_ids] = 0.0
        self._foot_rail_valid[env_ids] = False
        self._foot_crossing_state[env_ids] = 0
        self._crossing_state_initialized[env_ids] = False
        self._progress_initialized[env_ids] = False
        self._front_complete[env_ids] = False
        self._hind_complete[env_ids] = False
        self._root_complete[env_ids] = False
        self._front_event[env_ids] = False
        self._early_front_event[env_ids] = False
        self._hind_event[env_ids] = False
        self._root_event[env_ids] = False
        self._stable_contact_steps[env_ids] = 0
        self._max_stable_contact_steps[env_ids] = 0
        self._stable_contact_progress_gain[env_ids] = 0.0
        self._landing_condition[env_ids] = False
        self._forbidden_contact_steps[env_ids] = 0
        self._success[env_ids] = False
        self._failure[env_ids] = False
        self._no_progress_failure[env_ids] = False
        self._invalid_failure[env_ids] = False
        self._height_low_failure[env_ids] = False
        self._height_high_failure[env_ids] = False
        self._tilt_failure[env_ids] = False
        self._yaw_failure[env_ids] = False
        self._lateral_failure[env_ids] = False
        self._angular_velocity_failure[env_ids] = False
        self._forbidden_failure[env_ids] = False
        self._nonfinite_reward_steps[env_ids] = 0
        self._episode_contact_steps[env_ids] = 0.0
        self._episode_foot_bar_contact_steps[env_ids] = 0.0
        self._episode_low_bar_clearance_steps[env_ids] = 0.0
        self._episode_low_bar_clearance_deficit_sum[env_ids] = 0.0
        self._episode_low_bar_clearance_samples[env_ids] = 0.0
        self._episode_rail_valid_steps[env_ids] = 0.0
        self._episode_rail_y_error_sum[env_ids] = 0.0
        self._episode_landing_condition_steps[env_ids] = 0.0
        self._episode_stance_contact_score[env_ids] = 0.0
        self._episode_swing_clear_score[env_ids] = 0.0
        self._episode_valid_touchdowns[env_ids] = 0.0
        self._episode_swing_clearance_sum[env_ids] = 0.0
        self._episode_swing_clearance_samples[env_ids] = 0.0
        self._episode_bar_visible_steps[env_ids] = 0.0
        self._episode_bar_visible_samples[env_ids] = 0.0

    def _log_resets(
        self,
        env_ids: torch.Tensor,
        completed_episode: torch.Tensor,
        old_levels: torch.Tensor,
        episode_lengths: torch.Tensor,
    ):
        self.extras["log"] = {}
        ids = env_ids[completed_episode]
        if ids.numel() == 0:
            for values in self._episode_sums.values():
                values[env_ids] = 0.0
            return

        episode_duration = (
            episode_lengths[completed_episode].float() * self.step_dt
        ).clamp_min(self.step_dt)
        episode_step_count = episode_lengths[completed_episode].float().clamp_min(1.0)
        for name, values in self._episode_sums.items():
            self.extras["log"][f"RewardRate/{name}"] = torch.mean(
                values[ids] / episode_duration
            )
            values[env_ids] = 0.0

        forward_distance = (
            self.robot.data.root_pos_w[ids, 0] - self._episode_start_root_x[ids]
        ).clamp_min(0.0)
        terminal_swing = ~(
            self._leg_phase()[ids] < self.cfg.gait_duty_factor
        )
        terminal_swing_clearance = self._foot_clearance[ids][terminal_swing]
        terminal_clearance_p95 = (
            torch.quantile(terminal_swing_clearance, 0.95)
            if terminal_swing_clearance.numel() > 0
            else torch.zeros((), device=self.device)
        )
        self.extras["log"].update(
            {
                "Curriculum/level_mean": torch.mean(old_levels[completed_episode].float()),
                "Curriculum/success_ema": torch.mean(self._curriculum_success_ema[ids]),
                "Crossing/success_rate": torch.mean(self._success[ids].float()),
                "Crossing/front_complete_rate": torch.mean(self._front_complete[ids].float()),
                "Crossing/early_front_event_rate": torch.mean(
                    self._early_front_event[ids].float()
                ),
                "Crossing/hind_complete_rate": torch.mean(self._hind_complete[ids].float()),
                "Crossing/root_complete_rate": torch.mean(self._root_complete[ids].float()),
                "Crossing/stage_mean": torch.mean(self._stage[ids].float()),
                "Crossing/rail_valid_rate": torch.mean(self._foot_rail_valid[ids].float()),
                "Crossing/clearance_state_rate": torch.mean(
                    (self._foot_crossing_state[ids] >= 1).float()
                ),
                "Walk/forward_distance": torch.mean(forward_distance),
                "Walk/mean_forward_speed": torch.mean(
                    forward_distance / episode_duration
                ),
                "Walk/episode_duration": torch.mean(episode_duration),
                "Gait/stance_contact_score": torch.mean(
                    self._episode_stance_contact_score[ids] / episode_step_count
                ),
                "Gait/swing_no_contact_score": torch.mean(
                    self._episode_swing_clear_score[ids] / episode_step_count
                ),
                "Gait/touchdown_frequency_hz": torch.mean(
                    self._episode_valid_touchdowns[ids] / episode_duration
                ),
                "Gait/swing_clearance_mean": torch.mean(
                    self._episode_swing_clearance_sum[ids]
                    / self._episode_swing_clearance_samples[ids].clamp_min(1.0)
                ),
                "Gait/swing_clearance_terminal_p95": terminal_clearance_p95,
                "Landing/condition_episode_rate": torch.mean(
                    self._episode_landing_condition_steps[ids] / episode_step_count
                ),
                "Landing/max_stable_fraction": torch.mean(
                    (
                        self._max_stable_contact_steps[ids].float()
                        / max(math.ceil(self.cfg.stable_landing_s / self.step_dt), 1)
                    ).clamp(0.0, 1.0)
                ),
                "Curriculum/promotion_rate": torch.mean(
                    (self._curriculum_level[ids] > old_levels[completed_episode]).float()
                ),
                "Curriculum/demotion_rate": torch.mean(
                    (self._curriculum_level[ids] < old_levels[completed_episode]).float()
                ),
                "Diagnostics/action_saturation": torch.mean(
                    (torch.abs(self._raw_actions[ids]) > 0.95).float()
                ),
                "Diagnostics/action_abs_gt_075": torch.mean(
                    (torch.abs(self._raw_actions[ids]) > 0.75).float()
                ),
                "BarEstimate/visible_rate": torch.mean(
                    self._episode_bar_visible_steps[ids]
                    / self._episode_bar_visible_samples[ids].clamp_min(1.0)
                ),
                "BarEstimate/delay_steps": torch.mean(
                    self._bar_estimate_delay[ids].float()
                ),
                "Diagnostics/forbidden_contact_rate": torch.mean(
                    self._forbidden_contact[ids].float()
                ),
                "Diagnostics/foot_bar_contact_rate": torch.mean(
                    self._episode_foot_bar_contact_steps[ids] / episode_step_count
                ),
                "Diagnostics/foot_bar_contact_terminal_rate": torch.mean(
                    torch.any(self._foot_bar_contact[ids], dim=1).float()
                ),
                "Diagnostics/foot_bar_contact_force": torch.mean(
                    torch.amax(self._foot_bar_contact_force[ids], dim=1)
                ),
                "Diagnostics/low_bar_clearance_rate": torch.mean(
                    self._episode_low_bar_clearance_steps[ids] / episode_step_count
                ),
                "Diagnostics/low_bar_clearance_deficit": torch.mean(
                    self._episode_low_bar_clearance_deficit_sum[ids]
                    / self._episode_low_bar_clearance_samples[ids].clamp_min(1.0)
                ),
                "Diagnostics/nonfinite_reward_steps": torch.mean(
                    self._nonfinite_reward_steps[ids].float()
                ),
                "ResetRate/fall_or_tilt": torch.mean(
                    (
                        self._failure[ids]
                        & ~self._no_progress_failure[ids]
                        & (
                            self._forbidden_contact_steps[ids]
                            < self.cfg.forbidden_contact_terminate_steps
                        )
                    ).float()
                ),
                "ResetRate/no_progress": torch.mean(self._no_progress_failure[ids].float()),
                "ResetRate/invalid_state": torch.mean(self._invalid_failure[ids].float()),
                "ResetRate/height_low": torch.mean(self._height_low_failure[ids].float()),
                "ResetRate/height_high": torch.mean(self._height_high_failure[ids].float()),
                "ResetRate/tilt": torch.mean(self._tilt_failure[ids].float()),
                "ResetRate/yaw": torch.mean(self._yaw_failure[ids].float()),
                "ResetRate/lateral": torch.mean(self._lateral_failure[ids].float()),
                "ResetRate/angular_velocity": torch.mean(
                    self._angular_velocity_failure[ids].float()
                ),
                "ResetRate/forbidden_contact": torch.mean(self._forbidden_failure[ids].float()),
                "ResetRate/success": torch.mean(self._success[ids].float()),
                "ResetRate/time_out": torch.mean(self.reset_time_outs[ids].float()),
            }
        )
        if self.cfg.enable_depth_bar_camera:
            heightmap = self._depth_bar_input_observation()
            self.extras["log"]["Heightmap/valid_rate"] = torch.mean(heightmap[ids, 1].float())
        for leg_index, leg_name in enumerate(self._LEG_NAMES):
            self.extras["log"][f"FootContact/{leg_name}_episode_rate"] = torch.mean(
                self._episode_contact_steps[ids, leg_index] / episode_step_count
            )
            self.extras["log"][f"FootContact/{leg_name}_terminal_rate"] = torch.mean(
                self._foot_contacts[ids, leg_index].float()
            )
            self.extras["log"][f"RailValid/{leg_name}_episode_rate"] = torch.mean(
                self._episode_rail_valid_steps[ids, leg_index] / episode_step_count
            )
            self.extras["log"][f"RailValid/{leg_name}_terminal_rate"] = torch.mean(
                self._foot_rail_valid[ids, leg_index].float()
            )
            self.extras["log"][f"RailValid/{leg_name}_mean_y_error"] = torch.mean(
                self._episode_rail_y_error_sum[ids, leg_index]
                / episode_step_count
            )
        for level in range(self.cfg.curriculum_max_level + 1):
            level_mask = old_levels[completed_episode] == level
            level_count = torch.sum(level_mask)
            level_success_count = torch.sum(self._success[ids].float() * level_mask.float())
            self.extras["log"][f"Curriculum/level_{level}_episodes"] = level_count.float()
            self.extras["log"][f"Curriculum/level_{level}_success_count"] = level_success_count
            if int(level_count.item()) > 0:
                self.extras["log"][f"Curriculum/level_{level}_success_rate"] = (
                    level_success_count / level_count
                )

    @property
    def nominal_action(self) -> torch.Tensor:
        """Normalized default pose used to initialize a new actor policy."""
        return self._nominal_action

    def get_curriculum_state(self) -> dict[str, torch.Tensor]:
        """Return checkpointable per-environment curriculum state."""
        return {
            "level": self._curriculum_level.detach().cpu(),
            "success_ema": self._curriculum_success_ema.detach().cpu(),
            "episode_count": self._curriculum_episode_count.detach().cpu(),
            "failure_streak": self._failure_streak.detach().cpu(),
        }

    def set_curriculum_state(self, state: dict[str, torch.Tensor]) -> None:
        """Restore curriculum state, repeating entries if the environment count changed."""
        targets = {
            "level": self._curriculum_level,
            "success_ema": self._curriculum_success_ema,
            "episode_count": self._curriculum_episode_count,
            "failure_streak": self._failure_streak,
        }
        for name, target in targets.items():
            source = state.get(name)
            if source is None or source.numel() == 0:
                continue
            source = source.to(device=self.device, dtype=target.dtype).flatten()
            target.copy_(source[self._all_env_ids % source.numel()])
