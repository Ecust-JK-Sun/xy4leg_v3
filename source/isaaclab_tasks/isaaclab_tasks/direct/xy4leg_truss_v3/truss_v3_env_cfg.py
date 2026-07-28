from __future__ import annotations

import os

import numpy as np

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.xy4legv2_amp.controllers_5dof_gripper import actuators_implicit


_V2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xy4legv2_amp")
_ROBOT_USD = os.path.abspath(os.path.join(_V2_DIR, "usd", "xy4legv5_gripper_truss_v2.usd"))
_RESET_MOTION = os.path.abspath(
    os.path.join(_V2_DIR, "data_v5", "sim_output_truss_0520_polynomial.npz")
)
_DEFAULT_GROUND_USD = (
    "/home/lab202/isaacsim_assets/Assets/Isaac/4.5/Isaac/Environments/Grid/default_environment.usd"
)


def _quat_multiply_xyzw(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _quat_from_axis_angle_xyzw(axis: str, angle_deg: float):
    angle_rad = np.deg2rad(angle_deg) * 0.5
    s = float(np.sin(angle_rad))
    c = float(np.cos(angle_rad))
    if axis == "x":
        return (s, 0.0, 0.0, c)
    if axis == "y":
        return (0.0, s, 0.0, c)
    if axis == "z":
        return (0.0, 0.0, s, c)
    raise ValueError(f"Unsupported axis: {axis}")


def _quat_from_euler_wxyz(seq: str, angles_deg: tuple[float, ...]):
    q = (0.0, 0.0, 0.0, 1.0)
    for axis, angle in zip(seq, angles_deg):
        q = _quat_multiply_xyzw(q, _quat_from_axis_angle_xyzw(axis, angle))
    x, y, z, w = q
    return (w, x, y, z)


@configclass
class TrussV3DomainRandomizationCfg:
    robot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.6, 1.2),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
        },
    )
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_Joint[1-5]"),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class TrussV3Sim2RealDomainRandomizationCfg:
    robot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.6, 1.4),
            "dynamic_friction_range": (0.5, 1.3),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
        },
    )
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.8, 0.8),
            "operation": "add",
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_Joint[1-5]"),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class XY4LegTrussV3TeacherEnvCfg(DirectRLEnvCfg):
    # Environment and spaces
    decimation = 2
    episode_length_s = 30.0
    action_space = 20
    observation_space = 84
    state_space = 106
    deployable_observation_dim = 68
    bar_estimate_observation_dim = 4
    viewer = ViewerCfg(
        eye=(-1.8, 1.5, 1.0),
        lookat=(0.3, 0.0, -0.2),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
    )

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 50.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -1.6),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=5,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_ROBOT_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.02),
            joint_pos={
                "LB_Joint1": 65.0 * np.pi / 180.0,
                "LB_Joint2": 45.0 * np.pi / 180.0,
                "LB_Joint3": -20.0 * np.pi / 180.0,
                "LB_Joint4": 24.0 * np.pi / 180.0,
                "LB_Joint5": 24.0 * np.pi / 180.0,
                "LF_Joint1": -65.0 * np.pi / 180.0,
                "LF_Joint2": -85.0 * np.pi / 180.0,
                "LF_Joint3": -20.0 * np.pi / 180.0,
                "LF_Joint4": 24.0 * np.pi / 180.0,
                "LF_Joint5": 24.0 * np.pi / 180.0,
                "RB_Joint1": -65.0 * np.pi / 180.0,
                "RB_Joint2": -85.0 * np.pi / 180.0,
                "RB_Joint3": -20.0 * np.pi / 180.0,
                "RB_Joint4": 24.0 * np.pi / 180.0,
                "RB_Joint5": 24.0 * np.pi / 180.0,
                "RF_Joint1": 65.0 * np.pi / 180.0,
                "RF_Joint2": 45.0 * np.pi / 180.0,
                "RF_Joint3": -20.0 * np.pi / 180.0,
                "RF_Joint4": 24.0 * np.pi / 180.0,
                "RF_Joint5": 24.0 * np.pi / 180.0,
                ".*_[12]out_Joint": -14.0 * np.pi / 180.0,
            },
        ),
        actuators=actuators_implicit,
        soft_joint_pos_limit_factor=0.95,
    )

    crossbar: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Crossbar",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.54, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.12, 0.08)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.48)),
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.01,
        track_air_time=True,
    )
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/front_depth_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.16),
            rot=_quat_from_euler_wxyz("yyz", (180.0, -30.0, -90.0)),
            convention="ros",
        ),
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=14.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        width=64,
        height=64,
        depth_clipping_behavior="max",
    )
    rear_tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/rear_depth_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.20, 0.0, -0.16),
            rot=_quat_from_euler_wxyz("yyz", (180.0, 30.0, 90.0)),
            convention="ros",
        ),
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=14.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        width=64,
        height=64,
        depth_clipping_behavior="max",
    )
    events = None

    # Truss geometry
    rail_length = 5.0
    rail_width = 0.04
    rail_height = 0.04
    rail_top_height = 0.50
    rail_center_distance = 0.296
    crossbar_width = 0.04
    crossbar_length = 0.54
    crossbar_height = 0.04
    default_ground_usd = _DEFAULT_GROUND_USD

    # Gait and deterministic grippers
    gait_frequency = 0.5
    gait_duty_factor = 0.82
    gait_phase_offsets = (0.0, 0.5, 0.75, 0.25)  # RF, LF, RB, LB
    walk_swing_clearance_target = 0.04
    walk_swing_clearance_error_max = 0.20
    walk_base_height_target = 1.02
    walk_base_height_sigma = 0.08
    action_soft_limit = 0.60
    foothold_distance_sigma = 0.04
    foothold_min_air_time = 0.20
    # Body-frame touchdown boxes from the V2 normal-walking expert data.
    # Order: RF, LF, RB, LB; values are [x_min, x_max, y_min, y_max].
    walk_foothold_ranges = (
        (0.19, 0.29, -0.20, -0.10),
        (0.22, 0.36, 0.10, 0.20),
        (-0.36, -0.22, -0.20, -0.10),
        (-0.29, -0.19, 0.10, 0.20),
    )
    gripper_phase_hysteresis = 0.06
    gripper_open_angle = -14.0 * np.pi / 180.0
    gripper_closed_angle = -30.0 * np.pi / 180.0
    learn_gripper_actions = False
    gripper_action_step = 1.0 * np.pi / 180.0

    # Crossing definition
    attempt_distance = 0.35
    progress_approach_distance = 0.60
    early_front_event_root_distance = 0.45
    bar_estimate_dx_scale = 1.2
    bar_estimate_dy_scale = 0.3
    bar_estimate_dz_scale = 0.5
    bar_estimate_noise_enabled = False
    bar_estimate_gaussian_noise_std = (0.0, 0.0, 0.0)
    bar_estimate_episode_bias_range = (0.0, 0.0, 0.0)
    bar_estimate_max_delay_steps = 0
    bar_estimate_lowpass_alpha = 1.0
    bar_estimate_dropout_prob = 0.0
    bar_estimate_fov_x_range = (-10.0, 10.0)
    bar_estimate_fov_y_range = (-10.0, 10.0)
    bar_estimate_fov_z_range = (-10.0, 10.0)
    enable_depth_bar_camera = False
    depth_bar_estimator_checkpoint = ""
    depth_bar_visible_value = 1.0
    depth_bar_max_depth = 10.0
    depth_bar_heightmap_size = 64
    depth_bar_heightmap_x_range = (-0.85, 1.45)
    depth_bar_heightmap_y_range = (-0.45, 0.45)
    depth_bar_heightmap_z_range = (-0.85, 0.25)
    pass_margin = 0.05
    root_pass_margin = 0.15
    stable_landing_s = 0.40
    max_tilt_angle = 25.0 * np.pi / 180.0
    max_yaw_angle = 25.0 * np.pi / 180.0
    max_root_lateral_deviation = 0.20
    max_root_angular_velocity = 8.0
    forbidden_contact_force = 8.0
    forbidden_contact_x_window = 0.08
    forbidden_contact_y_margin = 0.04
    forbidden_contact_z_margin = 0.08
    foot_bar_contact_force = 1.0
    foot_bar_contact_x_window = 0.06
    foot_bar_contact_y_margin = 0.02
    foot_bar_contact_z_margin = 0.08
    foot_contact_force = 1.0
    foot_rail_y_tolerance = 0.08
    root_center_y_tolerance = 0.12
    clearance_x_window = 0.14
    crossing_min_clearance = 0.04
    clearance_target = 0.10
    low_bar_clearance_x_window = 0.14
    low_bar_clearance_target = 0.08
    low_bar_clearance_sigma = 0.03
    low_bar_clearance_min_forward_delta = 0.002
    landing_rail_center_sigma = 0.04
    landing_velocity_sigma = 0.15
    landing_angular_velocity_sigma = 2.50
    forward_velocity_min = 0.03
    forward_velocity_max = 0.09
    forward_velocity_sigma = 0.025
    forward_speed_limit = 0.25
    forward_speed_limit_sigma = 0.10
    reset_root_x_offset = 0.0

    # Curriculum
    walk_only = False
    initial_curriculum_level = 0
    curriculum_max_level = 4
    curriculum_enabled = True
    curriculum_ema_alpha = 0.1
    curriculum_min_episodes = 10
    curriculum_promote_threshold = 0.70
    curriculum_demote_threshold = 0.30
    curriculum_failure_streak = 8
    balanced_curriculum_enabled = False
    balanced_curriculum_level_weights = (1.0, 1.0, 1.0, 1.0, 1.0)
    level_bar_distance_ranges = (
        (-0.02, 0.02),
        (0.15, 0.35),
        (0.35, 0.50),
        (0.50, 0.80),
        (0.80, 1.20),
    )
    no_progress_grace_s = (6.0, 8.0, 9.0, 10.0, 8.0)
    no_progress_patience_s = (8.0, 8.0, 9.0, 10.0, 8.0)
    no_progress_epsilon = 0.005

    # Optional reset-only use of the incomplete expert recording.
    reset_motion_file = _RESET_MOTION
    hind_reset_frame_start = 2250
    hind_reset_frame_end = 2500
    hind_reset_pose_probability = 0.5
    joint_reset_noise = 0.02

    # Reward scales; all dense terms are multiplied by step_dt
    rew_forward_velocity = 1.8
    rew_upright = 0.7
    rew_heading = 0.5
    rew_support = 0.3
    rew_gait_contact = 0.15
    rew_walk_rail_center = 0.25
    rew_walk_contact = 0.25
    rew_walk_swing_clearance_error = 0.0
    rew_walk_foothold = 0.0
    rew_touchdown_impact = -0.04
    rew_walk_base_height = 0.2
    rew_base_motion = -0.15
    rew_forward_speed_limit = 0.0
    rew_gripper_phase = 0.0
    rew_front_progress = 4.0
    rew_hind_progress = 12.0
    rew_root_progress = 4.0
    rew_clearance = 2.0
    rew_landing_rail_center = 1.0
    rew_landing_contact = 1.0
    rew_landing_stability = 3.0
    rew_stable_contact_progress = 4.0
    rew_front_event = 2.0
    rew_hind_event = 6.0
    rew_root_event = 2.0
    rew_success_event = 14.0
    rew_action_magnitude = -0.03
    rew_action_soft_limit = -0.6
    rew_action_rate = -0.05
    rew_action_accel = -0.03
    rew_joint_limit = -1.0
    rew_joint_velocity = -0.0025
    rew_energy = -0.00025
    rew_slip = -0.25
    rew_long_air_no_progress = -0.5
    rew_forbidden_contact = -2.0
    rew_foot_bar_contact = 0.0
    rew_low_bar_clearance = 0.0
    rew_termination = -8.0

    # Bounds for penalty inputs prevent a single unstable physics step from
    # dominating the value targets before the environment is reset.
    reward_joint_position_norm_max = 2.0
    reward_joint_velocity_max = 50.0
    reward_torque_max = 500.0
    reward_foot_velocity_max = 5.0
    reward_touchdown_velocity_max = 3.0
    reward_root_linear_velocity_max = 2.0
    reward_root_angular_velocity_max = 8.0
    reward_base_motion_max = 10.0

    # Failure conditions
    termination_height_low = 0.76
    termination_height_high = 1.25
    forbidden_contact_terminate_steps = 10

    # Teacher domain-randomization knobs (disabled in the nominal config)
    domain_randomization_enabled = False
    max_action_delay_steps = 0
    joint_position_bias_range = 0.0
    joint_position_noise_std = 0.0
    joint_velocity_noise_std = 0.0
    include_gripper_position_obs = False
    pretrained_actor_copy_obs_dim = None
    gripper_action_latch_enabled = False
    gripper_action_latch_min_hold_s = 0.0
    gripper_action_latch_deadband = 0.0
    gripper_action_latch_initial_action = -1.0


@configclass
class XY4LegGripperTrussV3TeacherEnvCfg(XY4LegTrussV3TeacherEnvCfg):
    """Crossing teacher with learned one-action-per-gripper control."""

    action_space = 24
    observation_space = 88
    state_space = 110
    deployable_observation_dim = 72
    learn_gripper_actions = True
    level_bar_distance_ranges = (
        (0.00, 0.15),
        (0.15, 0.30),
        (0.55, 0.70),
        (0.60, 0.80),
        (0.65, 0.90),
    )
    curriculum_promote_threshold = 0.65
    curriculum_demote_threshold = 0.25
    rew_forward_speed_limit = -0.35
    rew_gripper_phase = -0.20
    rew_touchdown_impact = -0.06
    rew_base_motion = -0.25
    rew_action_rate = -0.07
    rew_action_accel = -0.05
    rew_slip = -0.35
    rew_foot_bar_contact = 0.0
    rew_low_bar_clearance = -1.5
    rew_hind_progress = 10.0
    rew_success_event = 12.0


@configclass
class XY4LegTrussV3TeacherDREnvCfg(XY4LegTrussV3TeacherEnvCfg):
    events = TrussV3DomainRandomizationCfg()
    domain_randomization_enabled = True
    initial_curriculum_level = 4
    curriculum_enabled = False
    max_action_delay_steps = 2
    joint_position_bias_range = 0.015
    joint_position_noise_std = 0.01
    joint_velocity_noise_std = 0.05


@configclass
class XY4LegTrussV3StudentEnvCfg(XY4LegTrussV3TeacherEnvCfg):
    """Nominal distillation environment; no domain randomization."""

    level_bar_distance_ranges = (
        (0.00, 0.15),
        (0.15, 0.30),
        (0.55, 0.70),
        (0.60, 0.80),
        (0.65, 0.90),
    )
    curriculum_promote_threshold = 0.65
    curriculum_demote_threshold = 0.25


@configclass
class XY4LegTrussV3StudentOracleBarEnvCfg(XY4LegTrussV3StudentEnvCfg):
    """Student environment with perfect low-dimensional crossbar estimate."""


@configclass
class XY4LegTrussV3StudentNoisyBarEnvCfg(XY4LegTrussV3StudentOracleBarEnvCfg):
    """Student validation environment with noisy low-dimensional crossbar estimate."""

    bar_estimate_noise_enabled = True
    bar_estimate_gaussian_noise_std = (0.03, 0.015, 0.015)
    bar_estimate_episode_bias_range = (0.02, 0.01, 0.01)
    bar_estimate_max_delay_steps = 2
    bar_estimate_lowpass_alpha = 0.7
    bar_estimate_dropout_prob = 0.02
    bar_estimate_fov_x_range = (-0.15, 1.45)
    bar_estimate_fov_y_range = (-0.45, 0.45)
    bar_estimate_fov_z_range = (-0.85, 0.25)


@configclass
class XY4LegTrussV3StudentNoisyBarNoiseOnlyEnvCfg(XY4LegTrussV3StudentOracleBarEnvCfg):
    """Ablation: noisy bar position only; visible remains in-distribution."""

    bar_estimate_noise_enabled = True
    bar_estimate_gaussian_noise_std = (0.03, 0.015, 0.015)
    bar_estimate_episode_bias_range = (0.02, 0.01, 0.01)
    bar_estimate_max_delay_steps = 0
    bar_estimate_lowpass_alpha = 1.0
    bar_estimate_dropout_prob = 0.0
    bar_estimate_fov_x_range = (-10.0, 10.0)
    bar_estimate_fov_y_range = (-10.0, 10.0)
    bar_estimate_fov_z_range = (-10.0, 10.0)


@configclass
class XY4LegTrussV3StudentNoisyBarDelayedEnvCfg(XY4LegTrussV3StudentNoisyBarNoiseOnlyEnvCfg):
    """Ablation: noisy and delayed bar position; visible remains in-distribution."""

    bar_estimate_max_delay_steps = 2
    bar_estimate_lowpass_alpha = 0.7


@configclass
class XY4LegTrussV3StudentDepthBarEnvCfg(XY4LegTrussV3StudentOracleBarEnvCfg):
    """Student validation environment with depth camera and depth-derived bar-estimate hook."""

    enable_depth_bar_camera = True


@configclass
class XY4LegGripperTrussV3StudentOracleBarEnvCfg(XY4LegGripperTrussV3TeacherEnvCfg):
    """Learned-gripper student with perfect low-dimensional crossbar estimate."""

    balanced_curriculum_enabled = True
    balanced_curriculum_level_weights = (0.05, 0.05, 0.30, 0.30, 0.30)
    level_bar_distance_ranges = (
        (0.00, 0.15),
        (0.15, 0.30),
        (0.55, 0.65),
        (0.60, 0.70),
        (0.65, 0.75),
    )


@configclass
class XY4LegGripperTrussV3StudentDepthBarEnvCfg(XY4LegGripperTrussV3StudentOracleBarEnvCfg):
    """Learned-gripper student with depth-derived low-dimensional crossbar estimate."""

    enable_depth_bar_camera = True


@configclass
class XY4LegGripperTrussV3TeacherSim2RealEnvCfg(XY4LegGripperTrussV3TeacherEnvCfg):
    """Privileged crossing teacher with Sim2Real dynamics and gripper observations."""

    events = TrussV3Sim2RealDomainRandomizationCfg()
    observation_space = 92
    state_space = 114
    deployable_observation_dim = 76
    pretrained_actor_copy_obs_dim = 76
    include_gripper_position_obs = True
    domain_randomization_enabled = True
    max_action_delay_steps = 3
    joint_position_bias_range = 0.02
    joint_position_noise_std = 0.015
    joint_velocity_noise_std = 0.08

    # Keep the near/mid bar distribution used by the latest balanced DepthBar
    # work, but let PPO learn progressively from level 0.
    balanced_curriculum_enabled = False
    level_bar_distance_ranges = (
        (0.00, 0.15),
        (0.15, 0.30),
        (0.55, 0.65),
        (0.60, 0.70),
        (0.65, 0.75),
    )
    curriculum_promote_threshold = 0.65
    curriculum_demote_threshold = 0.25
    no_progress_grace_s = (8.0, 10.0, 12.0, 14.0, 14.0)
    no_progress_patience_s = (10.0, 10.0, 12.0, 14.0, 14.0)

    gait_frequency = 0.35
    forward_velocity_min = 0.025
    forward_velocity_max = 0.07
    forward_velocity_sigma = 0.02
    forward_speed_limit = 0.14
    forward_speed_limit_sigma = 0.06
    action_soft_limit = 0.55
    gripper_action_step = 0.4 * np.pi / 180.0
    gripper_action_latch_enabled = True
    gripper_action_latch_min_hold_s = 0.8
    gripper_action_latch_deadband = 0.10
    gripper_action_latch_initial_action = -1.0

    rew_forward_velocity = 1.2
    rew_forward_speed_limit = -0.20
    rew_gripper_phase = -0.15
    rew_touchdown_impact = -0.08
    rew_base_motion = -0.8
    rew_action_magnitude = -0.10
    rew_action_soft_limit = -2.5
    rew_action_rate = -0.14
    rew_action_accel = -0.10
    rew_joint_velocity = -0.008
    rew_energy = -0.0008
    rew_slip = -0.8
    rew_low_bar_clearance = -1.5
    rew_hind_progress = 10.0
    rew_success_event = 12.0


@configclass
class XY4LegGripperTrussV3StudentOracleBarSim2RealEnvCfg(
    XY4LegGripperTrussV3TeacherSim2RealEnvCfg
):
    """Deployable Sim2Real student with perfect low-dimensional crossbar estimate."""

    balanced_curriculum_enabled = True
    balanced_curriculum_level_weights = (0.05, 0.05, 0.30, 0.30, 0.30)


@configclass
class XY4LegGripperTrussV3StudentDepthBarSim2RealEnvCfg(
    XY4LegGripperTrussV3StudentOracleBarSim2RealEnvCfg
):
    """Deployable Sim2Real student with depth-derived low-dimensional crossbar estimate."""

    enable_depth_bar_camera = True


@configclass
class XY4LegTrussV3WalkEnvCfg(XY4LegTrussV3TeacherEnvCfg):
    """No-crossbar pretraining with checkpoint-compatible observations and actions."""

    events = TrussV3DomainRandomizationCfg()
    episode_length_s = 20.0
    walk_only = True
    domain_randomization_enabled = True
    max_action_delay_steps = 2
    joint_position_bias_range = 0.015
    joint_position_noise_std = 0.01
    joint_velocity_noise_std = 0.05
    curriculum_enabled = False
    curriculum_max_level = 0
    initial_curriculum_level = 0
    level_bar_distance_ranges = ((100.0, 100.0),)
    no_progress_grace_s = (5.0,)
    no_progress_patience_s = (8.0,)
    reset_motion_file = None
    hind_reset_pose_probability = 0.0
    reset_root_x_offset = -1.8

    rew_forward_velocity = 1.5
    rew_upright = 1.2
    rew_heading = 0.8
    rew_support = 1.0
    rew_gait_contact = 1.5
    rew_walk_rail_center = 0.8
    rew_walk_contact = 1.0
    rew_walk_swing_clearance_error = -6.0
    rew_walk_foothold = 0.2
    rew_touchdown_impact = -0.05
    rew_walk_base_height = 1.0
    rew_base_motion = -0.8
    rew_action_magnitude = -0.08
    rew_action_soft_limit = -2.0
    rew_action_rate = -0.12
    rew_action_accel = -0.08
    rew_joint_velocity = -0.006
    rew_energy = -0.0006
    rew_slip = -0.8

    rew_front_progress = 0.0
    rew_hind_progress = 0.0
    rew_root_progress = 0.0
    rew_clearance = 0.0
    rew_landing_rail_center = 0.0
    rew_landing_contact = 0.0
    rew_landing_stability = 0.0
    rew_stable_contact_progress = 0.0
    rew_front_event = 0.0
    rew_hind_event = 0.0
    rew_root_event = 0.0
    rew_success_event = 0.0


@configclass
class XY4LegGripperTrussV3WalkEnvCfg(XY4LegTrussV3WalkEnvCfg):
    """No-crossbar walk pretraining with learned one-action-per-gripper control."""

    action_space = 24
    observation_space = 88
    state_space = 110
    deployable_observation_dim = 72
    learn_gripper_actions = True
    rew_gripper_phase = -0.20


@configclass
class XY4LegGripperTrussV3WalkSim2RealEnvCfg(XY4LegGripperTrussV3WalkEnvCfg):
    """Walk sim2real pretraining with final deployable bar-estimate actor interface."""

    events = TrussV3Sim2RealDomainRandomizationCfg()
    observation_space = 92
    state_space = 114
    deployable_observation_dim = 76
    pretrained_actor_copy_obs_dim = 72
    include_gripper_position_obs = True
    domain_randomization_enabled = True
    max_action_delay_steps = 3
    joint_position_bias_range = 0.02
    joint_position_noise_std = 0.015
    joint_velocity_noise_std = 0.08

    # Keep the no-crossbar task, but expose the same low-dimensional bar input
    # that the final camera/depth student will consume.
    bar_estimate_noise_enabled = False
    bar_estimate_max_delay_steps = 0
    balanced_curriculum_enabled = False
    balanced_curriculum_level_weights = (1.0,)
    level_bar_distance_ranges = ((100.0, 100.0),)

    gait_frequency = 0.35
    forward_velocity_min = 0.025
    forward_velocity_max = 0.06
    forward_velocity_sigma = 0.018
    forward_speed_limit = 0.10
    forward_speed_limit_sigma = 0.05
    action_soft_limit = 0.55
    gripper_action_step = 0.4 * np.pi / 180.0
    gripper_action_latch_enabled = True
    gripper_action_latch_min_hold_s = 0.8
    gripper_action_latch_deadband = 0.10
    gripper_action_latch_initial_action = -1.0

    rew_forward_velocity = 1.2
    rew_forward_speed_limit = -0.15
    rew_touchdown_impact = -0.08
    rew_base_motion = -1.0
    rew_action_magnitude = -0.10
    rew_action_soft_limit = -2.5
    rew_action_rate = -0.16
    rew_action_accel = -0.12
    rew_joint_velocity = -0.008
    rew_energy = -0.0008
    rew_slip = -1.0
    rew_gripper_phase = -0.15
