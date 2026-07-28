# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the V3 learned-gripper walk policy on the real robot.

This script reuses the V2 serial/radar hardware interfaces, but loads a V3
RSL-RL checkpoint and builds the V3 walk observation directly from real sensors.
It defaults to dry-run mode. Pass ``--enable_motors`` to send commands.
"""

from __future__ import annotations

import argparse
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
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from isaaclab.app import AppLauncher

import cli_args_v3 as cli_args  # isort: skip


DEFAULT_TASK = "Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Walk-Sim2Real-Direct-v0"
DEFAULT_CHECKPOINT = (
    "logs/rsl_rl/xy4leg_gripper_truss_v3_walk_sim2real/"
    "2026-07-24_18-58-21_deployable_interface/model_28800.pt"
)
DEFAULT_LOG_ROOT = str(Path(__file__).resolve().parents[6] / "logs" / "rsl_rl")


parser = argparse.ArgumentParser(description="Run a V3 RSL-RL walk policy on hardware.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="V3 task name.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="RL agent config entry point.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Must remain 1 for hardware play.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--loop_hz", type=float, default=25.0, help="Real control loop frequency.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after N steps; 0 means run until interrupted.")
parser.add_argument("--print_interval", type=int, default=25, help="Print diagnostics every N control steps.")
parser.add_argument("--enable_motors", action="store_true", help="Actually enable motors and transmit commands.")
parser.add_argument("--enable_radar", action="store_true", help="Read UDP radar pose/velocity.")
parser.add_argument("--radar_port", type=int, default=8888, help="Radar UDP port.")
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
    default=True,
    help="Quantize gripper actions to +/-1 and rate-limit direction changes.",
)
parser.add_argument("--gripper_latch_deadband", type=float, default=0.10, help="Deadband for binary gripper latch.")
parser.add_argument("--gripper_latch_min_hold_s", type=float, default=0.8, help="Minimum hold time per gripper sign.")
parser.add_argument(
    "--gripper_latch_initial_action",
    type=float,
    default=-1.0,
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
from rsl_rl.runners import OnPolicyRunner

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
    deadband: np.ndarray
    max_torque: np.ndarray
    sim_open_angle_rad: float
    sim_closed_angle_rad: float


def _hw_index(leg: str, joint_index: int) -> int:
    return HARDWARE_LEGS.index(leg) + 4 * joint_index


def _policy_index(leg: str, joint_index: int) -> int:
    return POLICY_LEGS.index(leg) * 5 + joint_index


def _quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1.0e-6:
        return np.eye(3, dtype=np.float32)
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


def _policy_action_to_hardware_command(
    action: np.ndarray,
    action_offset: np.ndarray,
    action_scale: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    gripper_calibration: GripperCalibration,
) -> np.ndarray:
    action = np.nan_to_num(action.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    action = np.clip(action, -1.0, 1.0)

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
            torque = (
                value
                * float(gripper_calibration.torque_sign[hardware_leg])
                * float(gripper_calibration.torque_scale[hardware_leg])
            )
        command_sim[hw_gripper_index] = float(np.clip(torque, -max_torque, max_torque))

    return command_sim * SIM_TO_REAL_SIGN


def _read_gripper_calibration(path: str, global_torque_scale: float, fallback_deadband: float) -> GripperCalibration:
    defaults = {
        "LB": {"close_pos": -6.96, "torque_sign": -1.0, "torque_scale": 1.0, "deadband": fallback_deadband},
        "LF": {"close_pos": -7.38, "torque_sign": -1.0, "torque_scale": 1.0, "deadband": fallback_deadband},
        "RB": {"close_pos": -10.15, "torque_sign": -1.0, "torque_scale": 1.0, "deadband": fallback_deadband},
        "RF": {"close_pos": -7.0, "torque_sign": -1.0, "torque_scale": 1.0, "deadband": fallback_deadband},
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
    deadband = []
    max_torque = []
    for leg in HARDWARE_LEGS:
        leg_cfg = {**defaults[leg], **(legs_cfg.get(leg, {}) or {})}
        scale = float(leg_cfg.get("torque_scale", 1.0)) * float(global_torque_scale)
        close_pos.append(float(leg_cfg["close_pos"]))
        torque_sign.append(float(leg_cfg.get("torque_sign", -1.0)))
        torque_scale.append(scale)
        deadband.append(float(leg_cfg.get("deadband", fallback_deadband)))
        max_torque.append(float(leg_cfg.get("max_torque", abs(scale))))
    return GripperCalibration(
        close_pos=np.array(close_pos, dtype=np.float32),
        torque_sign=np.array(torque_sign, dtype=np.float32),
        torque_scale=np.array(torque_scale, dtype=np.float32),
        deadband=np.array(deadband, dtype=np.float32),
        max_torque=np.array(max_torque, dtype=np.float32),
        sim_open_angle_rad=math.radians(float(sim_angle_cfg.get("open_deg", -14.0))),
        sim_closed_angle_rad=math.radians(float(sim_angle_cfg.get("closed_deg", -30.0))),
    )


def _update_gripper_latch(
    raw_gripper_action: np.ndarray,
    latched_action: np.ndarray,
    last_update_time: np.ndarray,
    now: float,
    deadband: float,
    min_hold_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = latched_action.copy()
    candidate[raw_gripper_action > deadband] = 1.0
    candidate[raw_gripper_action < -deadband] = -1.0
    can_update = (now - last_update_time) >= max(float(min_hold_s), 0.0)
    should_update = can_update & (candidate != latched_action)
    latched_action = latched_action.copy()
    last_update_time = last_update_time.copy()
    latched_action[should_update] = candidate[should_update]
    last_update_time[should_update] = now
    return latched_action, last_update_time


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
    device: torch.device,
    obs_mode: str,
    deployable_observation_dim: int,
) -> dict[str, torch.Tensor]:
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
        # Final deployable walk interface: no real crossbar in walk mode, but
        # keep the same 4D low-dimensional bar channel as camera/depth crossing.
        dummy_bar_estimate = np.array([2.0, 0.0, -1.0, 1.0], dtype=np.float32)
        return {
            "deployable": torch.tensor(deployable, dtype=torch.float32, device=device).unsqueeze(0),
            "bar_estimate": torch.tensor(dummy_bar_estimate, dtype=torch.float32, device=device).unsqueeze(0),
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


def _radar_observation(receiver: RadarReceiver | None, default_forward_speed: float, velocity_world: bool):
    if receiver is None:
        return (
            GRAVITY_W.copy(),
            np.array([default_forward_speed, 0.0, 0.0], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
    data = receiver.get_data()
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


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    if args_cli.num_envs != 1:
        raise ValueError("sim2real_v3/play_real.py only supports --num_envs 1")
    if args_cli.loop_hz <= 0.0:
        raise ValueError("--loop_hz must be positive")

    stop_requested = False

    def _signal_handler(sig, frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"\n[INFO] Received signal {sig}; stopping after current cycle.")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, INSTALLED_RSL_RL_VERSION)
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

    log_root_path = os.path.join(DEFAULT_LOG_ROOT, agent_cfg.experiment_name)
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
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
    actor_obs_groups = list(getattr(agent_cfg, "obs_groups", {}).get("actor", ["policy"]))
    if actor_obs_groups == ["deployable", "bar_estimate"]:
        obs_mode = "deployable_bar"
    elif actor_obs_groups == ["policy"]:
        obs_mode = "policy"
    else:
        raise RuntimeError(f"Unsupported actor obs_groups for real play: {actor_obs_groups}")
    print(f"[INFO] Actor obs groups: {actor_obs_groups} -> real obs mode: {obs_mode}")

    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
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

    shared = SharedData()
    io_stop_event = threading.Event()
    motor_session = None
    obs_thread = None
    tx_thread = None
    receiver = None

    try:
        if args_cli.enable_motors:
            kp_list, kd_list = _read_motor_cfg(args_cli.motor_config)
            motor_enabled, motor_session = create_motor_session(kp_list, kd_list)
            if not motor_enabled or motor_session is None:
                raise RuntimeError("Motor controllers failed to initialize")
            obs_thread = start_motor_obs_thread(
                shared, motor_session, device, 1, loop_hz=args_cli.loop_hz, stop_event=io_stop_event
            )
            hold_command = _policy_action_to_hardware_command(
                nominal_action,
                action_offset,
                action_scale,
                joint_lower,
                joint_upper,
                gripper_calibration,
            )
            shared.set_action(hold_command)
            motor_session.enable_all()
            tx_thread = start_serial_tx_thread(
                shared,
                True,
                motor_session,
                loop_hz=args_cli.loop_hz,
                stop_event=io_stop_event,
            )
            print(f"[INFO] Holding nominal command for {args_cli.startup_hold_s:.1f}s")
            time.sleep(max(args_cli.startup_hold_s, 0.0))

        if args_cli.enable_radar:
            receiver = RadarReceiver(port=args_cli.radar_port)
            receiver.start()

        period = 1.0 / args_cli.loop_hz
        step = 0
        print("[INFO] Starting real-policy loop")
        while simulation_app.is_running() and not stop_requested:
            start = time.time()
            obs_np = shared.get_obs() if args_cli.enable_motors else None
            if obs_np is not None and obs_np.shape[0] >= 48:
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
            )
            gait_phase = (args_cli.gait_phase_origin + step * period * float(raw_env.cfg.gait_frequency)) % 1.0
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
                device,
                obs_mode,
                deployable_observation_dim,
            )

            with torch.inference_mode():
                action = policy(obs)[0].detach().cpu().numpy().astype(np.float32)

            alpha = float(np.clip(args_cli.action_smoothing, 0.0, 1.0))
            smoothed_action = alpha * action + (1.0 - alpha) * smoothed_action
            command_action = smoothed_action.copy()
            if args_cli.gripper_binary_latch:
                latched_gripper_action, gripper_latch_update_time = _update_gripper_latch(
                    action[20:24],
                    latched_gripper_action,
                    gripper_latch_update_time,
                    time.time(),
                    args_cli.gripper_latch_deadband,
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
                shared.set_action(command)

            if args_cli.print_interval > 0 and step % args_cli.print_interval == 0:
                print(
                    "[REAL] "
                    f"step={step} phase={gait_phase:.3f} "
                    f"act_abs_max={np.max(np.abs(action)):.3f} "
                    f"grip_angle={np.round(gripper_angle, 4)} "
                    f"grip_raw={action[20:24]} grip_latch={latched_gripper_action} "
                    f"grip_cmd={np.round(command[20:24], 3)} "
                    f"cmd_leg_deg={np.rad2deg(command[:20]).min():.1f}/{np.rad2deg(command[:20]).max():.1f} "
                    f"gravity={projected_gravity}"
                )

            previous_action = action
            step += 1
            if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                break

            remaining = period - (time.time() - start)
            if remaining > 0.0:
                time.sleep(remaining)
            else:
                print(f"[WARN] control loop overran by {-remaining:.4f}s")

    finally:
        io_stop_event.set()
        if tx_thread is not None:
            tx_thread.join(timeout=1.0)
        if obs_thread is not None:
            obs_thread.join(timeout=1.0)
        if receiver is not None:
            receiver.stop()
        if motor_session is not None:
            motor_session.close()
        vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
