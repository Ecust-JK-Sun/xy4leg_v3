# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the V3 learned-gripper policy on the real robot.

This script reuses the V2 serial/radar hardware interfaces, but loads a V3
RSL-RL checkpoint and builds the V3 deployable observation directly from real
sensors. For DepthBar crossing policies, pass ``--enable_camera`` and a
depth-bar estimator checkpoint to fill the 4D bar_estimate channel from a
RealSense depth crop.

It defaults to dry-run mode. Pass ``--enable_motors`` to send commands.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
TASK_DIR = THIS_DIR.parent
CAMERA_DIR = TASK_DIR / "camera"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(CAMERA_DIR) not in sys.path:
    sys.path.insert(0, str(CAMERA_DIR))

from isaaclab.app import AppLauncher

import cli_args_v3 as cli_args  # isort: skip


DEFAULT_TASK = "Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Student-DepthBar-Sim2Real-Direct-v0"
DEFAULT_CHECKPOINT = (
    "logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar_sim2real/"
    "2026-08-05_04-55-55_realcam_crop_noise_l2far_render12hz_from42000/model_68500.pt"
)
DEFAULT_DEPTH_BAR_ESTIMATOR = (
    "logs/rsl_rl/xy4leg_gripper_truss_v3_depth_crop_realcam/"
    "depth_crop_estimator_realcam_xmax060.pt"
)
DEFAULT_LOG_ROOT = str(Path(__file__).resolve().parents[6] / "logs" / "rsl_rl")


parser = argparse.ArgumentParser(description="Run a V3 learned-gripper policy on hardware.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="V3 task name.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="RL agent config entry point.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Must remain 1 for hardware play.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--loop_hz", type=float, default=None, help="Must match the checkpoint's training frequency.")
parser.add_argument("--feedback_poll_hz", type=float, default=200.0, help="Nonblocking feedback polling rate; independent of policy Hz.")
parser.add_argument("--torch_threads", type=int, default=1, help="CPU Torch threads for batch-one inference.")
parser.add_argument("--timing_interval_s", type=float, default=2.0, help="Print measured rates and latency at this interval.")
parser.add_argument("--timing_jsonl", type=str, default=None, help="Append timing reports to a JSONL file.")
parser.add_argument("--tx_mode", choices=("periodic", "on_update"), default="periodic",
                    help="Periodic resend, or one send per policy update without heartbeat resends.")
parser.add_argument("--training_config_dir", type=str, default=None, help="Saved params/ directory; inferred from checkpoint.")
parser.add_argument("--feedback_timeout_s", type=float, default=0.5, help="Maximum per-motor feedback age.")
parser.add_argument("--radar_timeout_s", type=float, default=0.5, help="Maximum radar receipt age.")
parser.add_argument("--command_timeout_s", type=float, default=0.5, help="Report expired policy commands; no shutdown action is sent.")
parser.add_argument("--sensor_ready_timeout_s", type=float, default=5.0, help="Wait for valid initial sensor feedback.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after N steps; 0 means run until interrupted.")
parser.add_argument("--print_interval", type=int, default=25, help="Print diagnostics every N control steps.")
parser.add_argument("--enable_motors", action="store_true", help="Actually enable motors and transmit commands.")
parser.add_argument("--motor_ports", nargs=4, default=None, metavar=("LB", "LF", "RB", "RF"),
                    help="Four serial paths in LB LF RB RF order; supports /dev/serial/by-id paths.")
parser.add_argument("--enable_radar", action="store_true", help="Read UDP radar pose/velocity.")
parser.add_argument("--radar_port", type=int, default=8888, help="Radar UDP port.")
parser.add_argument("--radar_source_ip", type=str, default=None, help="Optional expected radar sender IP.")
parser.add_argument(
    "--radar_velocity_world",
    action="store_true",
    help="Treat radar linear/angular velocities as world-frame and rotate them into body frame.",
)
parser.add_argument("--startup_hold_s", type=float, default=3.0, help="Hold initial command before policy loop.")
parser.add_argument(
    "--gripper_calibration",
    type=str,
    default=str(THIS_DIR / "gripper_calibration.yaml"),
    help="Per-gripper calibration yaml.",
)
parser.add_argument(
    "--gripper_torque_scale",
    type=float,
    default=1.0,
    help="Global multiplier applied after per-gripper torque_scale.",
)
parser.add_argument(
    "--gripper_deadband",
    type=float,
    default=0.05,
    help="Fallback gripper deadband if the calibration yaml omits one.",
)
parser.add_argument(
    "--gripper_binary_latch",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Quantize gripper actions to +/-1 and rate-limit direction changes.",
)
parser.add_argument("--gripper_latch_deadband", type=float, default=None, help="Override both latch thresholds unless set separately.")
parser.add_argument(
    "--gripper_latch_open_deadband",
    type=float,
    default=None,
    help="Positive-action threshold for switching a gripper latch to open.",
)
parser.add_argument(
    "--gripper_latch_close_deadband",
    type=float,
    default=None,
    help="Negative-action threshold for switching a gripper latch to close.",
)
parser.add_argument("--gripper_latch_min_hold_s", type=float, default=None, help="Defaults to the checkpoint's latch hold time.")
parser.add_argument(
    "--gripper_latch_initial_action",
    type=float,
    default=None,
    help="Initial latched gripper sign; +1 opens in sim convention, -1 closes.",
)
parser.add_argument("--default_forward_speed", type=float, default=0.05, help="Fallback body x velocity.")
parser.add_argument("--gait_phase_origin", type=float, default=0.0, help="Initial RF gait phase in [0, 1).")
parser.add_argument(
    "--action_smoothing",
    type=float,
    default=1.0,
    help="0..1 smoothing factor for transmitted actions; 1 disables smoothing.",
)
parser.add_argument(
    "--motor_config",
    type=str,
    default=str(THIS_DIR / "damiao.yaml"),
    help="DaMiao motor gain yaml.",
)
parser.add_argument(
    "--enable_camera",
    action="store_true",
    help="Read RealSense depth and use the depth-bar estimator for bar_estimate.",
)
parser.add_argument(
    "--camera_device_id",
    type=str,
    default=None,
    help="RealSense serial number. If omitted, the first available device is used.",
)
parser.add_argument(
    "--camera_flip",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Rotate RealSense color/depth frames by 180 degrees before inference.",
)
parser.add_argument("--camera_img_size", type=int, default=None, help="Defaults to the checkpoint's depth crop size.")
parser.add_argument("--camera_timeout_ms", type=int, default=200, help="Bound the RealSense frame wait.")
parser.add_argument("--camera_stale_timeout_s", type=float, default=0.25, help="Maximum age of a cached camera frame.")
parser.add_argument("--camera_depth_near_m", type=float, default=0.10, help="Minimum valid depth in meters.")
parser.add_argument("--camera_depth_max_m", type=float, default=None, help="Defaults to training depth normalization.")
parser.add_argument(
    "--depth_bar_estimator_checkpoint",
    type=str,
    default=None,
    help="TorchScript depth-bar estimator checkpoint.",
)
parser.add_argument(
    "--bar_dx_scale",
    type=float,
    default=None,
    help="bar_estimate dx normalization scale. Defaults to the selected task cfg.",
)
parser.add_argument(
    "--bar_dy_scale",
    type=float,
    default=None,
    help="bar_estimate dy normalization scale. Defaults to the selected task cfg.",
)
parser.add_argument(
    "--bar_dz_scale",
    type=float,
    default=None,
    help="bar_estimate dz normalization scale. Defaults to the selected task cfg.",
)
parser.add_argument(
    "--bar_visible_x_range",
    type=str,
    default=None,
    help="Estimated metric dx range treated as raw camera-visible.",
)
parser.add_argument(
    "--bar_visible_y_range",
    type=str,
    default=None,
    help="Estimated metric dy range treated as raw camera-visible.",
)
parser.add_argument(
    "--bar_visible_z_range",
    type=str,
    default=None,
    help="Estimated metric dz range treated as raw camera-visible.",
)
parser.add_argument(
    "--bar_visible_min_valid_rate",
    type=float,
    default=0.001,
    help="Minimum valid depth pixel fraction required before accepting a raw bar estimate.",
)
parser.add_argument(
    "--bar_memory_duration_s",
    type=float,
    default=None,
    help="How long to keep the last visible bar estimate after it leaves the camera FOV.",
)
parser.add_argument(
    "--bar_memory_nominal_forward_velocity",
    type=float,
    default=None,
    help="Nominal body forward velocity used to propagate bar memory when no reliable body speed is available.",
)
parser.add_argument(
    "--bar_memory_fallback_delta_b",
    type=str,
    default=None,
    help="Metric fallback bar delta when memory expires, as 'dx,dy,dz'.",
)
parser.add_argument(
    "--bar_estimate_lowpass_alpha",
    type=float,
    default=None,
    help="Low-pass alpha for raw metric depth-bar estimates; 1 disables smoothing.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, checkpoint=DEFAULT_CHECKPOINT)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import importlib.metadata as metadata
import numpy as np
import torch
import yaml
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from motor_session_gripper import create_motor_session
from radar_receiver import RadarReceiver
from serial_threads import start_motor_obs_thread, start_serial_tx_thread
from threads_shared import SharedData
from deployment_config import resolve_loop_hz, restore_training_contract
from runtime_timing import LatestCamera, PeriodicDeadline, TimingStats


INSTALLED_RSL_RL_VERSION = metadata.version("rsl-rl-lib")
POLICY_LEGS = ("RF", "LF", "RB", "LB")
HARDWARE_LEGS = ("LB", "LF", "RB", "RF")
GRAVITY_W = np.array([0.0, 0.0, -1.0], dtype=np.float32)

# Hardware buffer order is interleaved by joint index:
# [LB1, LF1, RB1, RF1, LB2, LF2, ... LB6, LF6, RB6, RF6].
REAL_TO_SIM_SIGN = np.array(
    [
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        -1.0, -1.0, -1.0, -1.0,
        1.0, 1.0, 1.0, 1.0,
    ],
    dtype=np.float32,
)
SIM_TO_REAL_SIGN = REAL_TO_SIM_SIGN.copy()


@dataclass(frozen=True)
class GripperCalibration:
    close_pos: np.ndarray
    torque_sign: np.ndarray
    torque_scale: np.ndarray
    open_torque_scale: np.ndarray
    close_torque_scale: np.ndarray
    deadband: np.ndarray
    max_torque: np.ndarray
    sim_open_angle_rad: float
    sim_closed_angle_rad: float


@dataclass
class DepthBarRuntime:
    estimator: torch.nn.Module
    memory_estimate_b: np.ndarray
    memory_age_s: float
    last_update_time_s: float
    fallback_delta_b: np.ndarray
    filtered_estimate_b: np.ndarray
    filter_initialized: bool
    last_camera_sequence: int | None = None


def _hw_index(leg: str, joint_index: int) -> int:
    return HARDWARE_LEGS.index(leg) + 4 * joint_index


def _policy_index(leg: str, joint_index: int) -> int:
    return POLICY_LEGS.index(leg) * 5 + joint_index


def _quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1.0e-6:
        raise ValueError("Invalid orientation quaternion")
    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _hardware_obs_to_policy_joints(
    obs_np: np.ndarray,
    gripper_calibration: GripperCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if obs_np.shape != (48,) or not np.isfinite(obs_np).all():
        raise ValueError("Motor observation must contain exactly 48 finite values")
    signed = obs_np.astype(np.float32).copy()
    signed[:24] *= REAL_TO_SIM_SIGN
    signed[24:] *= REAL_TO_SIM_SIGN
    policy_pos = np.zeros(20, dtype=np.float32)
    policy_vel = np.zeros(20, dtype=np.float32)
    for leg in POLICY_LEGS:
        for joint_index in range(5):
            pi = _policy_index(leg, joint_index)
            hi = _hw_index(leg, joint_index)
            policy_pos[pi] = signed[hi]
            policy_vel[pi] = signed[24 + hi]
    gripper_angle = np.full(4, gripper_calibration.sim_closed_angle_rad, dtype=np.float32)
    angle_min = min(gripper_calibration.sim_closed_angle_rad, gripper_calibration.sim_open_angle_rad)
    angle_max = max(gripper_calibration.sim_closed_angle_rad, gripper_calibration.sim_open_angle_rad)
    for leg in POLICY_LEGS:
        policy_leg = POLICY_LEGS.index(leg)
        hardware_leg = HARDWARE_LEGS.index(leg)
        raw_pos = signed[20 + hardware_leg]
        travel = gripper_calibration.close_pos[hardware_leg]
        if abs(float(travel)) > 1.0e-6:
            opening_alpha = np.clip(raw_pos / travel, 0.0, 1.0)
            angle = gripper_calibration.sim_closed_angle_rad + opening_alpha * (
                gripper_calibration.sim_open_angle_rad - gripper_calibration.sim_closed_angle_rad
            )
            gripper_angle[policy_leg] = np.clip(angle, angle_min, angle_max)
    return policy_pos, policy_vel, gripper_angle


def _validated_policy_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (24,) or not np.isfinite(action).all():
        raise ValueError("Policy output must contain exactly 24 finite values")
    return np.clip(action, -1.0, 1.0)


def _policy_action_to_hardware_command(
    action: np.ndarray,
    action_offset: np.ndarray,
    action_scale: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    gripper_calibration: GripperCalibration,
) -> np.ndarray:
    action = _validated_policy_action(action)

    target20 = action_offset + action_scale * action[:20]
    target20 = np.clip(target20, joint_lower, joint_upper)
    command_sim = np.zeros(24, dtype=np.float32)
    for leg in POLICY_LEGS:
        for joint_index in range(5):
            command_sim[_hw_index(leg, joint_index)] = target20[_policy_index(leg, joint_index)]

    for leg in POLICY_LEGS:
        policy_leg = POLICY_LEGS.index(leg)
        hardware_leg = HARDWARE_LEGS.index(leg)
        hw_gripper_index = 20 + hardware_leg
        value = float(action[20 + policy_leg])
        deadband = float(gripper_calibration.deadband[hardware_leg])
        max_torque = abs(float(gripper_calibration.max_torque[hardware_leg]))
        if abs(value) < deadband:
            torque = 0.0
        else:
            direction_scale = (
                float(gripper_calibration.open_torque_scale[hardware_leg])
                if value > 0.0
                else float(gripper_calibration.close_torque_scale[hardware_leg])
            )
            torque = (
                value
                * float(gripper_calibration.torque_sign[hardware_leg])
                * float(gripper_calibration.torque_scale[hardware_leg])
                * direction_scale
            )
        command_sim[hw_gripper_index] = float(np.clip(torque, -max_torque, max_torque))

    command = command_sim * SIM_TO_REAL_SIGN
    if not np.isfinite(command).all():
        raise ValueError("Hardware command contains nonfinite values")
    return command


def _read_gripper_calibration(path: str, global_torque_scale: float, fallback_deadband: float) -> GripperCalibration:
    defaults = {
        "LB": {
            "close_pos": -6.96,
            "torque_sign": -1.0,
            "torque_scale": 1.0,
            "open_torque_scale": 1.0,
            "close_torque_scale": 1.0,
            "deadband": fallback_deadband,
        },
        "LF": {
            "close_pos": -7.38,
            "torque_sign": -1.0,
            "torque_scale": 1.0,
            "open_torque_scale": 1.0,
            "close_torque_scale": 1.0,
            "deadband": fallback_deadband,
        },
        "RB": {
            "close_pos": -10.15,
            "torque_sign": -1.0,
            "torque_scale": 1.0,
            "open_torque_scale": 1.0,
            "close_torque_scale": 1.0,
            "deadband": fallback_deadband,
        },
        "RF": {
            "close_pos": -7.0,
            "torque_sign": -1.0,
            "torque_scale": 1.0,
            "open_torque_scale": 1.0,
            "close_torque_scale": 1.0,
            "deadband": fallback_deadband,
        },
    }
    cfg = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    legs_cfg = cfg.get("legs", {})
    sim_angle_cfg = cfg.get("sim_angle", {})
    close_pos = []
    torque_sign = []
    torque_scale = []
    open_torque_scale = []
    close_torque_scale = []
    deadband = []
    max_torque = []
    for leg in HARDWARE_LEGS:
        leg_cfg = {**defaults[leg], **(legs_cfg.get(leg, {}) or {})}
        scale = float(leg_cfg.get("torque_scale", 1.0)) * float(global_torque_scale)
        close_pos.append(float(leg_cfg["close_pos"]))
        torque_sign.append(float(leg_cfg.get("torque_sign", -1.0)))
        torque_scale.append(scale)
        open_torque_scale.append(float(leg_cfg.get("open_torque_scale", 1.0)))
        close_torque_scale.append(float(leg_cfg.get("close_torque_scale", 1.0)))
        deadband.append(float(leg_cfg.get("deadband", fallback_deadband)))
        max_torque.append(float(leg_cfg.get("max_torque", abs(scale))))
    result = GripperCalibration(
        close_pos=np.array(close_pos, dtype=np.float32),
        torque_sign=np.array(torque_sign, dtype=np.float32),
        torque_scale=np.array(torque_scale, dtype=np.float32),
        open_torque_scale=np.array(open_torque_scale, dtype=np.float32),
        close_torque_scale=np.array(close_torque_scale, dtype=np.float32),
        deadband=np.array(deadband, dtype=np.float32),
        max_torque=np.array(max_torque, dtype=np.float32),
        sim_open_angle_rad=math.radians(float(sim_angle_cfg.get("open_deg", -14.0))),
        sim_closed_angle_rad=math.radians(float(sim_angle_cfg.get("closed_deg", -30.0))),
    )
    fields = (result.close_pos, result.torque_sign, result.torque_scale, result.open_torque_scale,
              result.close_torque_scale, result.deadband, result.max_torque)
    if (not all(np.isfinite(v).all() for v in fields)
            or np.any(np.abs(result.close_pos) < 1e-6)
            or not np.isin(result.torque_sign, [-1.0, 1.0]).all()
            or np.any(result.torque_scale < 0) or np.any(result.open_torque_scale < 0)
            or np.any(result.close_torque_scale < 0) or np.any(result.max_torque <= 0)
            or np.any((result.deadband < 0) | (result.deadband > 1))
            or not math.isfinite(result.sim_open_angle_rad) or not math.isfinite(result.sim_closed_angle_rad)
            or result.sim_open_angle_rad <= result.sim_closed_angle_rad):
        raise ValueError("Invalid gripper calibration")
    return result


def _update_gripper_latch(
    raw_gripper_action: np.ndarray,
    latched_action: np.ndarray,
    last_update_time: np.ndarray,
    now: float,
    open_deadband: float,
    close_deadband: float,
    min_hold_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = latched_action.copy()
    candidate[raw_gripper_action > open_deadband] = 1.0
    candidate[raw_gripper_action < -close_deadband] = -1.0
    can_update = (now - last_update_time) >= max(float(min_hold_s), 0.0)
    should_update = can_update & (candidate != latched_action)
    latched_action = latched_action.copy()
    last_update_time = last_update_time.copy()
    latched_action[should_update] = candidate[should_update]
    last_update_time[should_update] = now
    return latched_action, last_update_time


def _parse_float_tuple(value: str, count: int, name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != count:
        raise ValueError(f"Expected {name} to contain {count} comma-separated values, got {value!r}")
    result = tuple(float(part) for part in parts)
    if not all(math.isfinite(v) for v in result) or (count == 2 and result[0] >= result[1]):
        raise ValueError(f"Invalid {name}: {value!r}")
    return result


def _depth_crop_observation_from_camera(
    camera,
    device: torch.device,
    img_size: int,
    depth_near_m: float,
    depth_max_m: float,
) -> tuple[torch.Tensor, float]:
    _, depth = camera.read(img_size=(img_size, img_size))
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]

    if depth.shape != (img_size, img_size):
        raise ValueError(f"Unexpected depth image shape: {depth.shape}")
    finite_depth = depth[np.isfinite(depth)]
    if getattr(camera, "depth_scale_m", None) is not None:
        depth = depth * float(camera.depth_scale_m)
    elif finite_depth.size > 0 and float(finite_depth.max()) > depth_max_m * 2.0:
        depth = depth * 0.001  # RealSense z16 depth is normally millimeters.

    valid = np.isfinite(depth) & (depth > depth_near_m + 1.0e-4) & (depth < depth_max_m - 1.0e-4)
    depth = np.nan_to_num(depth, nan=depth_max_m, posinf=depth_max_m, neginf=0.0)
    depth = np.clip(depth, 0.0, depth_max_m)
    depth_norm = 1.0 - depth / max(depth_max_m, 1.0e-6)
    depth_norm = np.where(valid, np.clip(depth_norm, 0.0, 1.0), 0.0).astype(np.float32)
    valid_map = valid.astype(np.float32)
    obs_np = np.stack((depth_norm, valid_map), axis=0)[None, ...]
    return torch.from_numpy(obs_np).to(device=device, dtype=torch.float32), float(valid_map.mean())


def _load_depth_bar_runtime(
    checkpoint: str,
    device: torch.device,
    memory_duration_s: float,
    fallback_delta_b: np.ndarray,
) -> DepthBarRuntime:
    estimator_path = retrieve_file_path(checkpoint)
    estimator = torch.jit.load(estimator_path, map_location=device)
    estimator.eval()
    return DepthBarRuntime(
        estimator=estimator,
        memory_estimate_b=fallback_delta_b.astype(np.float32).copy(),
        memory_age_s=float(memory_duration_s) + 1.0,
        last_update_time_s=-1.0,
        fallback_delta_b=fallback_delta_b.astype(np.float32).copy(),
        filtered_estimate_b=fallback_delta_b.astype(np.float32).copy(),
        filter_initialized=False,
    )


def _bar_estimate_raw_visible(
    estimate_b: np.ndarray,
    valid_rate: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    min_valid_rate: float,
) -> bool:
    return (
        np.isfinite(estimate_b).all()
        and math.isfinite(valid_rate)
        and valid_rate >= min_valid_rate
        and x_range[0] <= float(estimate_b[0]) <= x_range[1]
        and y_range[0] <= float(estimate_b[1]) <= y_range[1]
        and z_range[0] <= float(estimate_b[2]) <= z_range[1]
    )


def _depth_bar_observation_from_camera(
    camera,
    runtime: DepthBarRuntime,
    device: torch.device,
    now: float,
    img_size: int,
    depth_near_m: float,
    depth_max_m: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    min_valid_rate: float,
    lowpass_alpha: float,
    memory_duration_s: float,
    memory_nominal_forward_velocity: float,
    dx_scale: float,
    dy_scale: float,
    dz_scale: float,
) -> tuple[np.ndarray, np.ndarray, bool, bool, float]:
    depth_obs, valid_rate = _depth_crop_observation_from_camera(
        camera,
        device=device,
        img_size=img_size,
        depth_near_m=depth_near_m,
        depth_max_m=depth_max_m,
    )
    with torch.inference_mode():
        estimate_b = runtime.estimator(depth_obs)[0, :3].detach().cpu().numpy().astype(np.float32)
    # Validate the original output; replacing NaN with zero would create a false detection.
    raw_visible = _bar_estimate_raw_visible(
        estimate_b,
        valid_rate,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        min_valid_rate=min_valid_rate,
    )
    sequence = getattr(camera, "last_read_sequence", None)
    same_frame = sequence is not None and sequence == runtime.last_camera_sequence
    if raw_visible:
        alpha = float(np.clip(lowpass_alpha, 0.0, 1.0))
        if not runtime.filter_initialized or runtime.memory_age_s > memory_duration_s:
            runtime.filtered_estimate_b = estimate_b.copy()
            runtime.filter_initialized = True
        elif not same_frame:
            runtime.filtered_estimate_b = alpha * estimate_b + (1.0 - alpha) * runtime.filtered_estimate_b
        estimate_for_memory = runtime.filtered_estimate_b
    else:
        estimate_for_memory = estimate_b

    dt = 0.0 if runtime.last_update_time_s < 0.0 else max(now - runtime.last_update_time_s, 0.0)
    runtime.last_update_time_s = now
    if raw_visible:
        runtime.memory_estimate_b = estimate_for_memory.copy()
        captured_at = getattr(camera, "last_read_capture_time", None)
        runtime.memory_age_s = max(now - captured_at, 0.) if captured_at is not None else 0.
    else:
        runtime.memory_estimate_b[0] -= float(memory_nominal_forward_velocity) * dt
        runtime.memory_age_s += dt
    runtime.last_camera_sequence = sequence

    memory_valid = runtime.memory_age_s <= float(memory_duration_s)
    memory_used = bool(memory_valid and not raw_visible)
    estimate_out = runtime.memory_estimate_b if memory_valid else runtime.fallback_delta_b
    visible = 1.0 if memory_valid else 0.0
    bar_estimate = np.array(
        [
            np.clip(float(estimate_out[0]) / max(float(dx_scale), 1.0e-6), -2.0, 2.0),
            np.clip(float(estimate_out[1]) / max(float(dy_scale), 1.0e-6), -2.0, 2.0),
            np.clip(float(estimate_out[2]) / max(float(dz_scale), 1.0e-6), -2.0, 2.0),
            visible,
        ],
        dtype=np.float32,
    )
    return bar_estimate, estimate_b, raw_visible, memory_used, valid_rate


def _read_motor_cfg(path: str) -> tuple[list[float], list[float]]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    kp_list = cfg.get("kp_truss")
    kd_list = cfg.get("kd")
    if not kp_list or not kd_list:
        raise ValueError(f"Expected kp_truss and kd in {path}")
    return list(kp_list), list(kd_list)


def _build_policy_observation(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    projected_gravity: np.ndarray,
    angular_velocity_b: np.ndarray,
    linear_velocity_b: np.ndarray,
    gait_phase: float,
    previous_action: np.ndarray,
    gripper_angle: np.ndarray,
    action_offset: np.ndarray,
    action_scale: np.ndarray,
    bar_estimate: np.ndarray | None,
    device: torch.device,
    obs_mode: str,
    deployable_observation_dim: int,
) -> dict[str, torch.Tensor]:
    terms = ((joint_pos, 20), (joint_vel, 20), (projected_gravity, 3), (angular_velocity_b, 3),
             (linear_velocity_b, 3), (previous_action, 24), (gripper_angle, 4))
    if (not math.isfinite(gait_phase)
            or any(np.asarray(value).shape != (size,) or not np.isfinite(value).all() for value, size in terms)):
        raise ValueError("Invalid shape or nonfinite policy observation")
    previous_action = _validated_policy_action(previous_action)
    if bar_estimate is not None and (np.asarray(bar_estimate).shape != (4,) or not np.isfinite(bar_estimate).all()):
        raise ValueError("Invalid bar observation")
    joint_pos_norm = np.clip((joint_pos - action_offset) / np.maximum(action_scale, 1.0e-6), -2.0, 2.0)
    joint_vel_norm = np.clip(joint_vel / 10.0, -2.0, 2.0)
    angular_velocity_norm = np.clip(angular_velocity_b / 5.0, -2.0, 2.0)
    gait_clock = np.array(
        [math.sin(2.0 * math.pi * gait_phase), math.cos(2.0 * math.pi * gait_phase)],
        dtype=np.float32,
    )
    deployable = np.concatenate(
        (
            joint_pos_norm,
            joint_vel_norm,
            projected_gravity.astype(np.float32),
            angular_velocity_norm.astype(np.float32),
            gait_clock,
            previous_action.astype(np.float32),
        ),
        axis=0,
    ).astype(np.float32)
    if deployable_observation_dim == 76:
        deployable = np.concatenate((deployable, gripper_angle.astype(np.float32)), axis=0)
    if deployable.shape[0] != deployable_observation_dim:
        raise RuntimeError(
            f"Expected {deployable_observation_dim}-D deployable observation, got {deployable.shape[0]}"
        )
    if obs_mode == "deployable_bar":
        if bar_estimate is None:
            # Walk/no-camera fallback: keep the 4D low-dimensional bar channel in
            # distribution while placing the crossbar far ahead.
            bar_estimate = np.array([2.0, 0.0, -1.0, 1.0], dtype=np.float32)
        return {
            "deployable": torch.tensor(deployable, dtype=torch.float32, device=device).unsqueeze(0),
            "bar_estimate": torch.tensor(bar_estimate, dtype=torch.float32, device=device).unsqueeze(0),
        }
    if obs_mode != "policy":
        raise RuntimeError(f"Unsupported real-play observation mode: {obs_mode}")

    # Walk-only teacher privileged constants. These match the no-crossbar setup:
    # crossbar very far ahead, foot signed distances far behind it, stage=approach.
    root_lin_vel_priv = np.clip(linear_velocity_b / 0.2, -3.0, 3.0).astype(np.float32)
    bar_delta_priv = np.array([2.0, 0.0, -0.42], dtype=np.float32)
    bar_dim_priv = np.array([0.4, 0.4], dtype=np.float32)
    foot_distance_priv = np.full(4, -2.0, dtype=np.float32)
    stage_one_hot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    privileged = np.concatenate(
        (root_lin_vel_priv, bar_delta_priv, bar_dim_priv, foot_distance_priv, stage_one_hot),
        axis=0,
    )
    policy_obs = np.concatenate((deployable, privileged), axis=0).astype(np.float32)
    expected_policy_dim = deployable_observation_dim + 16
    if policy_obs.shape[0] != expected_policy_dim:
        raise RuntimeError(
            f"Expected {expected_policy_dim}-D V3 walk policy observation, got {policy_obs.shape[0]}"
        )
    return {"policy": torch.tensor(policy_obs, dtype=torch.float32, device=device).unsqueeze(0)}


def _radar_observation(receiver: RadarReceiver | None, default_forward_speed: float, velocity_world: bool,
                       max_age_s: float = 0.5):
    if receiver is None:
        return (
            GRAVITY_W.copy(),
            np.array([default_forward_speed, 0.0, 0.0], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
    data = receiver.get_data(max_age_s=max_age_s)
    quat_xyzw = np.asarray(data.get("quat", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)
    rot_b_to_w = _quat_xyzw_to_rotmat(quat_xyzw)
    projected_gravity_b = rot_b_to_w.T @ GRAVITY_W

    lin_vel = np.asarray(data.get("lin_vel", [default_forward_speed, 0.0, 0.0]), dtype=np.float32)
    ang_vel = np.asarray(data.get("ang_vel", [0.0, 0.0, 0.0]), dtype=np.float32)
    if velocity_world:
        lin_vel = rot_b_to_w.T @ lin_vel
        ang_vel = rot_b_to_w.T @ ang_vel
    return (
        np.nan_to_num(projected_gravity_b, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        np.nan_to_num(lin_vel, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        np.nan_to_num(ang_vel, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
    )


def _resolve_runtime_defaults(args, cfg):
    defaults = {
        "gripper_binary_latch": "gripper_action_latch_enabled",
        "gripper_latch_initial_action": "gripper_action_latch_initial_action",
        "gripper_latch_min_hold_s": "gripper_action_latch_min_hold_s",
        "camera_img_size": "depth_bar_heightmap_size",
        "camera_depth_max_m": "depth_bar_depth_crop_max_depth",
        "bar_estimate_lowpass_alpha": "bar_estimate_lowpass_alpha",
        "bar_memory_duration_s": "depth_bar_memory_duration_s",
        "bar_memory_nominal_forward_velocity": "depth_bar_memory_nominal_forward_velocity",
    }
    for arg, field in defaults.items():
        if getattr(args, arg) is None:
            setattr(args, arg, getattr(cfg, field))
    for direction in ("open", "close"):
        key = f"gripper_latch_{direction}_deadband"
        if getattr(args, key) is None:
            setattr(args, key, args.gripper_latch_deadband if args.gripper_latch_deadband is not None
                    else getattr(cfg, f"gripper_action_latch_{direction}_deadband"))
    for axis in ("x", "y", "z"):
        key = f"bar_visible_{axis}_range"
        if getattr(args, key) is None:
            setattr(args, key, ",".join(str(v) for v in getattr(cfg, f"depth_bar_raw_visible_{axis}_range")))
    if args.bar_memory_fallback_delta_b is None:
        args.bar_memory_fallback_delta_b = ",".join(str(v) for v in cfg.depth_bar_memory_fallback_delta_b)
    for key in ("action_smoothing", "bar_estimate_lowpass_alpha", "bar_visible_min_valid_rate",
                "gripper_latch_open_deadband", "gripper_latch_close_deadband"):
        value = float(getattr(args, key))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"--{key} must be finite and in [0,1]")
    for key in ("feedback_timeout_s", "radar_timeout_s", "command_timeout_s", "sensor_ready_timeout_s",
                "feedback_poll_hz", "camera_stale_timeout_s", "timing_interval_s"):
        value = float(getattr(args, key))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{key} must be positive and finite")
    for key in ("startup_hold_s", "gripper_latch_min_hold_s", "bar_memory_duration_s"):
        value = float(getattr(args, key))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{key} must be nonnegative and finite")
    if (args.camera_img_size <= 0 or args.camera_timeout_ms <= 0
            or not 0 < args.camera_depth_near_m < args.camera_depth_max_m):
        raise ValueError("Invalid camera size, timeout or depth range")
    if args.torch_threads < 1:
        raise ValueError("--torch_threads must be positive")


def _report_timing(stats, args):
    report = stats.snapshot()
    report.update(target_policy_hz=args.loop_hz, tx_mode=args.tx_mode, torch_threads=args.torch_threads)
    rates, samples = report["rates_hz"], report["samples"]
    def value(name, field="p95", unit="ms"):
        result = samples.get(name, {}).get(field)
        return "n/a" if result is None else f"{result:.2f}{unit}"
    print(
        f"[TIMING] policy={rates.get('policy', 0.):.2f}/{args.loop_hz:g}Hz "
        f"tx={rates.get('tx', 0.):.2f}Hz new={rates.get('tx_new', 0.):.2f}Hz "
        f"repeat={report['counts'].get('tx_repeat', 0)} "
        f"rx_complete={rates.get('rx_complete', 0.):.2f}Hz rx_poll={rates.get('rx_poll', 0.):.1f}Hz "
        f"camera={rates.get('camera', 0.):.1f}Hz radar={rates.get('radar', 0.):.1f}Hz "
        f"actor_p95={value('actor_ms')} depth_p95={value('depth_ms')} "
        f"work_p95={value('cycle_ms')} obs_age_max={value('feedback_age_ms', 'max')} "
        f"cmd_age_p95={value('command_age_ms')} obs_to_tx_p95={value('observation_to_tx_ms')} "
        f"tx_p95={value('tx_ms')} camera_age_max={value('camera_age_ms', 'max')} "
        f"radar_age_max={value('radar_age_ms', 'max')} "
        f"rx_queue_max={value('rx_backlog_bytes', 'max', 'B')} "
        f"tx_queue_max={value('tx_pending_bytes', 'max', 'B')} "
        f"rx_invalid={report['counts'].get('rx_invalid_frames', 0)} "
        f"missed={report['counts'].get('policy_missed', 0)} window={report['window_s']:.2f}s"
    )
    if args.timing_jsonl:
        destination = Path(args.timing_jsonl)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report, allow_nan=False) + "\n")
    return report


def _wait_until_ready(check, timeout_s, cancelled):
    deadline = time.monotonic() + timeout_s
    while not cancelled():
        try:
            return check()
        except RuntimeError as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Sensor startup timed out: {exc}") from exc
            time.sleep(0.02)
    raise RuntimeError("Startup interrupted")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    if args_cli.num_envs != 1:
        raise ValueError("sim2real_v3/play_real.py only supports --num_envs 1")
    if args_cli.enable_motors and not args_cli.enable_radar:
        raise ValueError("--enable_motors requires --enable_radar for body orientation/angular velocity")

    stop_requested = False

    def _signal_handler(sig, frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"\n[INFO] Received signal {sig}; stopping after current cycle.")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, INSTALLED_RSL_RL_VERSION)
    log_root_path = os.path.join(DEFAULT_LOG_ROOT, agent_cfg.experiment_name)
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    contract = restore_training_contract(env_cfg, agent_cfg, resume_path, args_cli.training_config_dir)
    args_cli.loop_hz = resolve_loop_hz(args_cli.loop_hz, contract["policy_period_s"])
    _resolve_runtime_defaults(args_cli, env_cfg)
    torch.set_num_threads(args_cli.torch_threads)
    if args_cli.depth_bar_estimator_checkpoint is None:
        args_cli.depth_bar_estimator_checkpoint = (
            contract["env"].get("depth_bar_estimator_checkpoint") or DEFAULT_DEPTH_BAR_ESTIMATOR
        )
    print(f"[INFO] Training interface restored from: {contract['params_dir']}")
    print(f"[INFO] Training gait/control: {env_cfg.gait_frequency:g} Hz / {args_cli.loop_hz:g} Hz")
    if args_cli.enable_motors and not env_cfg.walk_only and not args_cli.enable_camera:
        raise ValueError("Crossing playback requires --enable_camera with --enable_motors")
    if (args_cli.enable_camera and contract["env"].get("enable_depth_bar_camera")
            and contract["env"].get("depth_bar_input_mode") != "front_depth_crop"):
        raise ValueError("RealSense playback only supports front_depth_crop estimator checkpoints")
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    # The real loop supplies its own sensor data; disable training-time randomization
    # in the helper env used for policy construction and action-limit lookup.
    if hasattr(env_cfg, "events"):
        env_cfg.events = None
    if hasattr(env_cfg, "domain_randomization_enabled"):
        env_cfg.domain_randomization_enabled = False
    if hasattr(env_cfg, "max_action_delay_steps"):
        env_cfg.max_action_delay_steps = 0
    if hasattr(env_cfg, "enable_depth_bar_camera"):
        env_cfg.enable_depth_bar_camera = False
    if hasattr(env_cfg, "depth_bar_estimator_checkpoint"):
        env_cfg.depth_bar_estimator_checkpoint = ""

    env_cfg.log_dir = os.path.dirname(resume_path)

    gym_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(gym_env.unwrapped, DirectMARLEnv):
        gym_env = multi_agent_to_single_agent(gym_env)
    raw_env = gym_env.unwrapped
    vec_env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Checkpoint: {resume_path}")
    print(f"[INFO] Motors enabled: {args_cli.enable_motors}")
    print(f"[INFO] Radar enabled: {args_cli.enable_radar}")
    obs_groups_cfg = getattr(agent_cfg, "obs_groups", {})
    if "actor" in obs_groups_cfg:
        actor_obs_groups = list(obs_groups_cfg["actor"])
    elif "student" in obs_groups_cfg:
        actor_obs_groups = list(obs_groups_cfg["student"])
    else:
        actor_obs_groups = ["policy"]
    if actor_obs_groups == ["deployable", "bar_estimate"]:
        obs_mode = "deployable_bar"
    elif actor_obs_groups == ["policy"]:
        if not raw_env.cfg.walk_only:
            raise ValueError("Privileged crossing teachers cannot be deployed; select a deployable student checkpoint")
        obs_mode = "policy"
    else:
        raise RuntimeError(f"Unsupported actor obs_groups for real play: {actor_obs_groups}")
    print(f"[INFO] Actor obs groups: {actor_obs_groups} -> real obs mode: {obs_mode}")

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise RuntimeError(f"Unsupported runner class for real play: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

    device = torch.device(vec_env.unwrapped.device)
    action_offset = raw_env._action_offset.detach().cpu().numpy().astype(np.float32)
    action_scale = raw_env._action_scale.detach().cpu().numpy().astype(np.float32)
    joint_lower = raw_env._joint_lower.detach().cpu().numpy().astype(np.float32)
    joint_upper = raw_env._joint_upper.detach().cpu().numpy().astype(np.float32)
    default_joint_pos = raw_env.robot.data.default_joint_pos[0, raw_env._main_joint_ids].detach().cpu().numpy().astype(np.float32)
    deployable_observation_dim = int(raw_env.cfg.deployable_observation_dim)
    expected_actor_obs_dim = (
        deployable_observation_dim + 4 if obs_mode == "deployable_bar" else deployable_observation_dim + 16
    )
    nominal_action = raw_env.nominal_action.detach().cpu().numpy().astype(np.float32)
    previous_action = nominal_action.copy()
    smoothed_action = nominal_action.copy()
    latched_gripper_action = np.full(
        4,
        1.0 if args_cli.gripper_latch_initial_action > 0.0 else -1.0,
        dtype=np.float32,
    )
    gripper_latch_update_time = np.full(4, -1.0e9, dtype=np.float64)
    gripper_calibration = _read_gripper_calibration(
        args_cli.gripper_calibration,
        args_cli.gripper_torque_scale,
        args_cli.gripper_deadband,
    )
    if not (math.isclose(gripper_calibration.sim_open_angle_rad, raw_env.cfg.gripper_open_angle, abs_tol=1e-6)
            and math.isclose(gripper_calibration.sim_closed_angle_rad, raw_env.cfg.gripper_closed_angle, abs_tol=1e-6)):
        raise ValueError("Gripper calibration angle range differs from the checkpoint's training interface")
    bar_visible_x_range = _parse_float_tuple(args_cli.bar_visible_x_range, 2, "--bar_visible_x_range")
    bar_visible_y_range = _parse_float_tuple(args_cli.bar_visible_y_range, 2, "--bar_visible_y_range")
    bar_visible_z_range = _parse_float_tuple(args_cli.bar_visible_z_range, 2, "--bar_visible_z_range")
    bar_dx_scale = (
        float(args_cli.bar_dx_scale)
        if args_cli.bar_dx_scale is not None
        else float(raw_env.cfg.bar_estimate_dx_scale)
    )
    bar_dy_scale = (
        float(args_cli.bar_dy_scale)
        if args_cli.bar_dy_scale is not None
        else float(raw_env.cfg.bar_estimate_dy_scale)
    )
    bar_dz_scale = (
        float(args_cli.bar_dz_scale)
        if args_cli.bar_dz_scale is not None
        else float(raw_env.cfg.bar_estimate_dz_scale)
    )
    if any(not math.isfinite(v) or v <= 0 for v in (bar_dx_scale, bar_dy_scale, bar_dz_scale)):
        raise ValueError("Bar normalization scales must be positive and finite")
    bar_memory_fallback_delta_b = np.array(
        _parse_float_tuple(args_cli.bar_memory_fallback_delta_b, 3, "--bar_memory_fallback_delta_b"),
        dtype=np.float32,
    )
    if args_cli.enable_camera and obs_mode != "deployable_bar":
        raise RuntimeError("--enable_camera requires actor obs groups ['deployable', 'bar_estimate']")
    if "DepthBar" in args_cli.task and not args_cli.enable_camera:
        print("[WARN] DepthBar task selected without --enable_camera; using dummy bar_estimate fallback.")

    print("[INFO] Policy leg order:", POLICY_LEGS)
    print("[INFO] Hardware leg order:", HARDWARE_LEGS)
    print("[INFO] action_offset[:5]:", action_offset[:5])
    print("[INFO] action_scale[:5]:", action_scale[:5])
    print(f"[INFO] deployable_observation_dim: {deployable_observation_dim}")
    print(f"[INFO] expected actor obs dim: {expected_actor_obs_dim}")
    print("[INFO] nominal_action:", nominal_action)
    print(f"[INFO] Gripper calibration: {args_cli.gripper_calibration}")
    print("[INFO] gripper close_pos:", gripper_calibration.close_pos)
    print("[INFO] gripper torque_sign:", gripper_calibration.torque_sign)
    print("[INFO] gripper torque_scale:", gripper_calibration.torque_scale)
    print("[INFO] gripper max_torque:", gripper_calibration.max_torque)
    print(f"[INFO] Camera enabled: {args_cli.enable_camera}")
    if args_cli.enable_camera:
        print(f"[INFO] Depth-bar estimator: {args_cli.depth_bar_estimator_checkpoint}")
        print(f"[INFO] Camera img/depth: {args_cli.camera_img_size}x{args_cli.camera_img_size}, max={args_cli.camera_depth_max_m}m")
        print(
            "[INFO] Bar visible ranges: "
            f"x={bar_visible_x_range}, y={bar_visible_y_range}, z={bar_visible_z_range}"
        )
        print(f"[INFO] Bar normalization scales: dx={bar_dx_scale}, dy={bar_dy_scale}, dz={bar_dz_scale}")

    shared = SharedData()
    timing = TimingStats()
    io_stop_event = threading.Event()
    feedback_ready_event = threading.Event()
    motor_session = None
    obs_thread = None
    tx_thread = None
    receiver = None
    camera = None
    depth_bar_runtime = None
    step = 0

    try:
        if args_cli.enable_radar:
            receiver = RadarReceiver(port=args_cli.radar_port, source_ip=args_cli.radar_source_ip, stats=timing)
            receiver.start()
            _wait_until_ready(lambda: receiver.get_data(max_age_s=args_cli.radar_timeout_s),
                              args_cli.sensor_ready_timeout_s, lambda: stop_requested)

        if args_cli.enable_camera:
            from realsense_camera import RealSenseCamera

            camera = LatestCamera(
                RealSenseCamera(device_id=args_cli.camera_device_id, flip=args_cli.camera_flip,
                                timeout_ms=args_cli.camera_timeout_ms),
                args_cli.camera_img_size, max_age_s=args_cli.camera_stale_timeout_s, stats=timing,
            ).start()
            depth_bar_runtime = _load_depth_bar_runtime(
                args_cli.depth_bar_estimator_checkpoint,
                device=device,
                memory_duration_s=args_cli.bar_memory_duration_s,
                fallback_delta_b=bar_memory_fallback_delta_b,
            )
            warm_depth, _ = _wait_until_ready(
                lambda: _depth_crop_observation_from_camera(
                    camera, device, args_cli.camera_img_size, args_cli.camera_depth_near_m, args_cli.camera_depth_max_m),
                args_cli.sensor_ready_timeout_s, lambda: stop_requested,
            )
            with torch.inference_mode():
                estimate = depth_bar_runtime.estimator(warm_depth)
            if estimate.shape != (1, 3) or not torch.isfinite(estimate).all():
                raise ValueError("Depth estimator must produce three finite metric coordinates")
            print(f"[INFO] RealSense camera initialized: {camera}")

        if args_cli.enable_motors:
            kp_list, kd_list = _read_motor_cfg(args_cli.motor_config)
            motor_enabled, motor_session = create_motor_session(kp_list, kd_list, args_cli.motor_ports)
            if not motor_enabled or motor_session is None:
                raise RuntimeError("Motor controllers failed to initialize")
            hold_command = _policy_action_to_hardware_command(
                nominal_action,
                action_offset,
                action_scale,
                joint_lower,
                joint_upper,
                gripper_calibration,
            )
            motor_session.enable_all(cancelled=lambda: stop_requested)
            # Enabling is the only serial reader until all acknowledgements arrive.
            shared.set_action(hold_command)
            tx_thread = start_serial_tx_thread(
                shared,
                True,
                motor_session,
                loop_hz=args_cli.loop_hz,
                stop_event=io_stop_event,
                command_timeout_s=args_cli.command_timeout_s,
                feedback_ready_event=feedback_ready_event,
                feedback_timeout_s=args_cli.feedback_timeout_s,
                stats=timing,
                mode=args_cli.tx_mode,
            )
            obs_thread = start_motor_obs_thread(
                shared, motor_session, device, 1, loop_hz=args_cli.feedback_poll_hz,
                stop_event=io_stop_event, stats=timing,
            )
            print(f"[INFO] Holding nominal command for {args_cli.startup_hold_s:.1f}s")
            hold_until = time.monotonic() + args_cli.startup_hold_s
            ready_until = time.monotonic() + max(args_cli.startup_hold_s, args_cli.sensor_ready_timeout_s)
            while not stop_requested:
                shared.raise_if_failed()
                receiver.get_data(max_age_s=args_cli.radar_timeout_s)
                shared.set_action(hold_command)
                try:
                    shared.get_obs(max_age_s=args_cli.feedback_timeout_s)
                    feedback_ready = True
                    feedback_ready_event.set()
                except RuntimeError as exc:
                    feedback_ready = False
                    if feedback_ready_event.is_set() or time.monotonic() >= ready_until:
                        raise RuntimeError(f"Motor feedback startup timed out: {exc}") from exc
                if feedback_ready and time.monotonic() >= hold_until:
                    break
                io_stop_event.wait(1.0 / args_cli.loop_hz)

        timing.snapshot()  # Exclude startup/nominal hold from steady-state rates.
        schedule = PeriodicDeadline(args_cli.loop_hz)
        previous_skipped = 0
        phase_start = time.monotonic()
        last_timing_report = phase_start
        print(f"[INFO] Starting real-policy loop: policy={args_cli.loop_hz:g}Hz, "
              f"feedback polling={args_cli.feedback_poll_hz:g}Hz, TX={args_cli.tx_mode}, "
              f"torch threads={args_cli.torch_threads}")
        while simulation_app.is_running() and not stop_requested:
            if not schedule.wait(io_stop_event):
                break
            start = time.monotonic()
            shared.raise_if_failed()
            timing.observe("deadline_lateness_ms", schedule.last_lateness_s * 1000.)
            timing.tick("policy_missed", schedule.skipped_ticks - previous_skipped)
            previous_skipped = schedule.skipped_ticks
            bar_estimate_obs = None
            bar_metric = None
            bar_raw_visible = False
            bar_memory_used = False
            bar_valid_rate = 0.0
            if camera is not None and depth_bar_runtime is not None:
                depth_started = time.monotonic()
                (
                    bar_estimate_obs,
                    bar_metric,
                    bar_raw_visible,
                    bar_memory_used,
                    bar_valid_rate,
                ) = _depth_bar_observation_from_camera(
                    camera,
                    depth_bar_runtime,
                    device=device,
                    now=start,
                    img_size=args_cli.camera_img_size,
                    depth_near_m=args_cli.camera_depth_near_m,
                    depth_max_m=args_cli.camera_depth_max_m,
                    x_range=bar_visible_x_range,
                    y_range=bar_visible_y_range,
                    z_range=bar_visible_z_range,
                    min_valid_rate=args_cli.bar_visible_min_valid_rate,
                    lowpass_alpha=args_cli.bar_estimate_lowpass_alpha,
                    memory_duration_s=args_cli.bar_memory_duration_s,
                    memory_nominal_forward_velocity=args_cli.bar_memory_nominal_forward_velocity,
                    dx_scale=bar_dx_scale,
                    dy_scale=bar_dy_scale,
                    dz_scale=bar_dz_scale,
                )
                timing.elapsed_ms("depth_ms", depth_started)

            # Sample proprioception after depth inference to avoid aging it during the CNN.
            observed_at = time.monotonic()
            obs_np = shared.get_obs(max_age_s=args_cli.feedback_timeout_s) if args_cli.enable_motors else None
            if obs_np is not None:
                timing.observe("feedback_age_ms", shared.feedback_age() * 1000.)
                joint_pos, joint_vel, gripper_angle = _hardware_obs_to_policy_joints(
                    obs_np[:48], gripper_calibration
                )
            else:
                joint_pos = default_joint_pos.copy()
                joint_vel = np.zeros(20, dtype=np.float32)
                gripper_angle = np.full(4, gripper_calibration.sim_closed_angle_rad, dtype=np.float32)
            projected_gravity, lin_vel_b, ang_vel_b = _radar_observation(
                receiver,
                args_cli.default_forward_speed,
                args_cli.radar_velocity_world,
                max_age_s=args_cli.radar_timeout_s,
            )
            gait_phase = (args_cli.gait_phase_origin + (observed_at - phase_start)
                          * float(raw_env.cfg.gait_frequency)) % 1.0
            obs = _build_policy_observation(
                joint_pos,
                joint_vel,
                projected_gravity,
                ang_vel_b,
                lin_vel_b,
                gait_phase,
                previous_action,
                gripper_angle,
                action_offset,
                action_scale,
                bar_estimate_obs,
                device,
                obs_mode,
                deployable_observation_dim,
            )

            actor_started = time.monotonic()
            with torch.inference_mode():
                action = policy(obs)[0].detach().cpu().numpy().astype(np.float32)
            # CPU conversion also waits for CUDA work, so this includes device completion.
            timing.elapsed_ms("actor_ms", actor_started)
            action = _validated_policy_action(action)

            alpha = float(np.clip(args_cli.action_smoothing, 0.0, 1.0))
            smoothed_action = alpha * action + (1.0 - alpha) * smoothed_action
            command_action = smoothed_action.copy()
            if args_cli.gripper_binary_latch:
                latched_gripper_action, gripper_latch_update_time = _update_gripper_latch(
                    action[20:24],
                    latched_gripper_action,
                    gripper_latch_update_time,
                    time.monotonic(),
                    args_cli.gripper_latch_open_deadband,
                    args_cli.gripper_latch_close_deadband,
                    args_cli.gripper_latch_min_hold_s,
                )
                command_action[20:24] = latched_gripper_action
            command = _policy_action_to_hardware_command(
                command_action,
                action_offset,
                action_scale,
                joint_lower,
                joint_upper,
                gripper_calibration,
            )
            if args_cli.enable_motors:
                shared.raise_if_failed()
                shared.get_obs(max_age_s=args_cli.feedback_timeout_s)
                receiver.get_data(max_age_s=args_cli.radar_timeout_s)
                shared.set_action(command, observed_at=observed_at)

            timing.tick("policy")
            timing.elapsed_ms("cycle_ms", start)

            if args_cli.print_interval > 0 and step % args_cli.print_interval == 0:
                print(
                    "[REAL] "
                    f"step={step} phase={gait_phase:.3f} "
                    f"act_abs_max={np.max(np.abs(action)):.3f} "
                    f"grip_angle={np.round(gripper_angle, 4)} "
                    f"grip_raw={action[20:24]} grip_latch={latched_gripper_action} "
                    f"grip_cmd={np.round(command[20:24], 3)} "
                    f"bar_obs={None if bar_estimate_obs is None else np.round(bar_estimate_obs, 3)} "
                    f"bar_m={None if bar_metric is None else np.round(bar_metric, 3)} "
                    f"bar_vis={bar_raw_visible}/{bar_memory_used} valid={bar_valid_rate:.3f} "
                    f"cmd_leg_deg={np.rad2deg(command[:20]).min():.1f}/{np.rad2deg(command[:20]).max():.1f} "
                    f"gravity={projected_gravity}"
                )

            previous_action = action.copy()
            step += 1
            if time.monotonic() - last_timing_report >= args_cli.timing_interval_s:
                _report_timing(timing, args_cli)
                last_timing_report = time.monotonic()
            if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                break
    finally:
        io_stop_event.set()
        # Close every resource even if one cleanup fails. No additional actuator commands.
        for thread in (tx_thread, obs_thread):
            if thread is not None:
                try:
                    thread.join(timeout=1.0)
                except Exception as exc:
                    print(f"[WARN] I/O thread cleanup: {exc}")
        if step > 0:
            try:
                _report_timing(timing, args_cli)
            except Exception as exc:
                print(f"[WARN] Final timing report: {exc}")
        for resource, method in ((receiver, "stop"), (camera, "stop"),
                                 (motor_session, "close"), (vec_env, "close")):
            if resource is not None:
                try:
                    getattr(resource, method)()
                except Exception as exc:
                    print(f"[WARN] Resource cleanup: {exc}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
