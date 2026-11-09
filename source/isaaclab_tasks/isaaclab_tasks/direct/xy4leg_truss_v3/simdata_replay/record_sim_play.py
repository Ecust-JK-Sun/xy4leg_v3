from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


THIS_DIR = Path(__file__).resolve().parent
TASK_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[5]
SIM2REAL_DIR = TASK_DIR / "sim2real_v3"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import cli_args_v3 as cli_args  # isort: skip
from common import (  # isort: skip
    POLICY_LEGS,
    HARDWARE_LEGS,
    SIM_TO_REAL_SIGN,
    policy_action_to_hardware_command,
    read_gripper_calibration,
)


DEFAULT_TASK = "Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Student-DepthBar-Sim2Real-Direct-v0"
DEFAULT_CHECKPOINT = (
    "logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar_sim2real/"
    "model_126000.pt"
)
DEFAULT_DEPTH_BAR_ESTIMATOR = (
    "logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar_sim2real/"
    "depth_crop_estimator_realcam_xmax060.pt"
)
DEFAULT_LOG_ROOT = str(REPO_ROOT / "logs" / "rsl_rl")
DEFAULT_GRIPPER_CALIBRATION = str(SIM2REAL_DIR / "gripper_calibration.yaml")


parser = argparse.ArgumentParser(description="Record a Truss V3 sim-play trajectory for open-loop real replay.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--output", type=str, default="", help="Output .npz. Defaults to simdata_replay/trajectories.")
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--fixed_level", type=int, default=4)
parser.add_argument("--fixed_bar_distance_m", type=float, default=0.5)
parser.add_argument("--record_env_index", type=int, default=0)
parser.add_argument("--stop_on_done", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--stop_on_success", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--real_time", action="store_true", default=False)
parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--command_source",
    type=str,
    choices=("applied", "raw"),
    default="applied",
    help="Use env-applied action or raw policy action to build hardware_command.",
)
parser.add_argument(
    "--use_sim_gripper_latch",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Replace gripper action with env._gripper_latched_actions when available.",
)
parser.add_argument("--gripper_calibration", type=str, default=DEFAULT_GRIPPER_CALIBRATION)
parser.add_argument("--gripper_torque_scale", type=float, default=1.0)
parser.add_argument("--gripper_deadband", type=float, default=0.05)
parser.add_argument(
    "--depth_bar_estimator_checkpoint",
    type=str,
    default=DEFAULT_DEPTH_BAR_ESTIMATOR,
    help="TorchScript depth-bar estimator checkpoint used by DepthBar tasks.",
)
parser.add_argument("--record_deployable", action="store_true", default=False)
parser.add_argument("--record_heightmap", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=False, checkpoint=DEFAULT_CHECKPOINT)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.num_envs != 1:
    raise SystemExit("record_sim_play.py currently supports --num_envs 1 only.")
if args_cli.record_env_index < 0 or args_cli.record_env_index >= args_cli.num_envs:
    raise SystemExit("--record_env_index must be within [0, num_envs).")
if args_cli.max_steps <= 0:
    raise SystemExit("--max_steps must be positive.")

if "DepthBar" in args_cli.task or args_cli.record_heightmap:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import importlib.metadata as metadata
import numpy as np
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


INSTALLED_RSL_RL_VERSION = metadata.version("rsl-rl-lib")


def _to_numpy(value, env_index: int | None = None) -> np.ndarray:
    if value is None:
        raise ValueError("Cannot convert None to numpy")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if env_index is not None:
        value = value[env_index]
    return np.asarray(value).copy()


def _optional_tensor_attr(obj, name: str, env_index: int) -> np.ndarray | None:
    value = getattr(obj, name, None)
    if value is None:
        return None
    try:
        return _to_numpy(value, env_index)
    except Exception:
        return None


def _optional_obs(obs: dict, key: str, env_index: int) -> np.ndarray | None:
    if not isinstance(obs, dict) or key not in obs:
        return None
    try:
        return _to_numpy(obs[key], env_index)
    except Exception:
        return None


def _append(records: dict[str, list[np.ndarray]], key: str, value) -> None:
    if value is None:
        return
    records.setdefault(key, []).append(np.asarray(value).copy())


def _stack_records(records: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, values in records.items():
        if not values:
            continue
        arrays[key] = np.stack(values, axis=0)
    return arrays


def _git_note() -> str:
    try:
        rev = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return f"{rev}{'-dirty' if dirty else ''}"
    except Exception as exc:
        return f"git-unavailable:{exc}"


def _apply_clean_overrides(env_cfg) -> None:
    if hasattr(env_cfg, "events"):
        env_cfg.events = None
    overrides = {
        "domain_randomization_enabled": False,
        "max_action_delay_steps": 0,
        "joint_reset_noise": 0.0,
        "joint_position_bias_range": 0.0,
        "joint_position_noise_std": 0.0,
        "joint_velocity_noise_std": 0.0,
        "bar_estimate_noise_enabled": False,
        "bar_estimate_gaussian_noise_std": (0.0, 0.0, 0.0),
        "bar_estimate_episode_bias_range": (0.0, 0.0, 0.0),
        "bar_estimate_max_delay_steps": 0,
        "bar_estimate_lowpass_alpha": 1.0,
        "bar_estimate_dropout_prob": 0.0,
        "bar_estimate_fov_x_range": (-10.0, 10.0),
        "bar_estimate_fov_y_range": (-10.0, 10.0),
        "bar_estimate_fov_z_range": (-10.0, 10.0),
    }
    for name, value in overrides.items():
        if hasattr(env_cfg, name):
            setattr(env_cfg, name, value)


def _resolve_output_path() -> Path:
    if args_cli.output:
        output = Path(args_cli.output).expanduser()
    else:
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output = THIS_DIR / "trajectories" / f"sim_play_{timestamp}.npz"
    if not output.is_absolute():
        output = Path.cwd() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _command_action_for_step(raw_env, raw_action_np: np.ndarray) -> np.ndarray:
    env_index = args_cli.record_env_index
    if args_cli.command_source == "applied" and hasattr(raw_env, "_applied_actions"):
        command_action = _to_numpy(raw_env._applied_actions, env_index).astype(np.float32)
    else:
        command_action = raw_action_np.astype(np.float32)
    if command_action.shape[0] == 20:
        command_action = np.concatenate((command_action, np.zeros(4, dtype=np.float32)), axis=0)
    if args_cli.use_sim_gripper_latch and hasattr(raw_env, "_gripper_latched_actions"):
        latched = _to_numpy(raw_env._gripper_latched_actions, env_index).astype(np.float32)
        command_action = command_action.copy()
        command_action[20:24] = latched
    return command_action


def _gripper_target_policy_order(raw_env, env_index: int) -> np.ndarray | None:
    target = getattr(raw_env, "_gripper_target", None)
    if target is None:
        return None
    target_np = _to_numpy(target, env_index).reshape(4, 2)
    return target_np.mean(axis=1).astype(np.float32)


def _build_metadata(raw_env, resume_path: str) -> dict:
    cfg = raw_env.cfg
    return {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "recorded_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit_or_dirty_note": _git_note(),
        "policy_dt": float(raw_env.step_dt),
        "sim_dt": float(getattr(cfg.sim, "dt", np.nan)),
        "decimation": int(getattr(cfg, "decimation", 0)),
        "action_offset": _to_numpy(raw_env._action_offset).astype(float).tolist(),
        "action_scale": _to_numpy(raw_env._action_scale).astype(float).tolist(),
        "joint_lower": _to_numpy(raw_env._joint_lower).astype(float).tolist(),
        "joint_upper": _to_numpy(raw_env._joint_upper).astype(float).tolist(),
        "nominal_action": _to_numpy(raw_env.nominal_action).astype(float).tolist(),
        "policy_legs": list(POLICY_LEGS),
        "hardware_legs": list(HARDWARE_LEGS),
        "sim_to_real_sign": SIM_TO_REAL_SIGN.astype(float).tolist(),
        "gripper_calibration_path": str(Path(args_cli.gripper_calibration).expanduser()),
        "depth_bar_estimator_checkpoint": args_cli.depth_bar_estimator_checkpoint,
        "gripper_torque_scale": float(args_cli.gripper_torque_scale),
        "fixed_level": int(args_cli.fixed_level),
        "fixed_bar_distance_m": float(args_cli.fixed_bar_distance_m),
        "seed": None if args_cli.seed is None else int(args_cli.seed),
        "clean": bool(args_cli.clean),
        "command_source": args_cli.command_source,
        "use_sim_gripper_latch": bool(args_cli.use_sim_gripper_latch),
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, INSTALLED_RSL_RL_VERSION)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    if args_cli.clean:
        _apply_clean_overrides(env_cfg)
        print("[INFO] Clean recording overrides enabled.")

    if hasattr(env_cfg, "depth_bar_estimator_checkpoint") and args_cli.depth_bar_estimator_checkpoint:
        env_cfg.depth_bar_estimator_checkpoint = args_cli.depth_bar_estimator_checkpoint
        print(f"[INFO] Depth-bar estimator: {args_cli.depth_bar_estimator_checkpoint}")

    if args_cli.fixed_level >= 0:
        max_level = int(getattr(env_cfg, "curriculum_max_level", 0))
        if args_cli.fixed_level > max_level:
            raise ValueError(f"--fixed_level {args_cli.fixed_level} exceeds curriculum_max_level={max_level}")
        env_cfg.initial_curriculum_level = int(args_cli.fixed_level)
        env_cfg.curriculum_enabled = False
        print(f"[INFO] Fixed curriculum level: {args_cli.fixed_level}")

    if args_cli.fixed_bar_distance_m >= 0.0 and hasattr(env_cfg, "level_bar_distance_ranges"):
        level_count = int(getattr(env_cfg, "curriculum_max_level", len(env_cfg.level_bar_distance_ranges) - 1)) + 1
        fixed_range = (float(args_cli.fixed_bar_distance_m), float(args_cli.fixed_bar_distance_m))
        env_cfg.level_bar_distance_ranges = tuple(fixed_range for _ in range(level_count))
        print(f"[INFO] Fixed crossbar distance: root_x + {args_cli.fixed_bar_distance_m:.2f} m")

    log_root_path = os.path.join(DEFAULT_LOG_ROOT, agent_cfg.experiment_name)
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    gym_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(gym_env.unwrapped, DirectMARLEnv):
        gym_env = multi_agent_to_single_agent(gym_env)
    vec_env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    raw_env = vec_env.unwrapped

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise RuntimeError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=raw_env.device)

    if version.parse(INSTALLED_RSL_RL_VERSION) >= version.parse("4.0.0"):
        reset_policy = policy.reset
    else:
        policy_module = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
        reset_policy = policy_module.reset if hasattr(policy_module, "reset") else None

    gripper_calibration = read_gripper_calibration(
        args_cli.gripper_calibration,
        global_torque_scale=args_cli.gripper_torque_scale,
        fallback_deadband=args_cli.gripper_deadband,
    )
    step_dt = float(raw_env.step_dt)
    output = _resolve_output_path()
    metadata_out = _build_metadata(raw_env, resume_path)
    records: dict[str, list[np.ndarray]] = {}

    obs = vec_env.get_observations()
    step = 0
    print(f"[INFO] Recording to {output}")
    print(f"[INFO] Checkpoint: {resume_path}")
    print(f"[INFO] step_dt={raw_env.step_dt:.4f}s, command_source={args_cli.command_source}")

    try:
        while simulation_app.is_running() and step < args_cli.max_steps:
            start_time = _dt.datetime.now().timestamp()
            obs_used = obs
            with torch.inference_mode():
                actions = policy(obs)
                raw_action_np = _to_numpy(actions, args_cli.record_env_index).astype(np.float32)
                obs, _, dones, _ = vec_env.step(actions)
                if reset_policy is not None:
                    reset_policy(dones)

            command_action = _command_action_for_step(raw_env, raw_action_np)
            joint_target = (
                _to_numpy(raw_env._action_offset).astype(np.float32)
                + _to_numpy(raw_env._action_scale).astype(np.float32) * command_action[:20]
            )
            joint_target = np.clip(
                joint_target,
                _to_numpy(raw_env._joint_lower).astype(np.float32),
                _to_numpy(raw_env._joint_upper).astype(np.float32),
            )
            hardware_command = policy_action_to_hardware_command(
                command_action,
                _to_numpy(raw_env._action_offset),
                _to_numpy(raw_env._action_scale),
                _to_numpy(raw_env._joint_lower),
                _to_numpy(raw_env._joint_upper),
                gripper_calibration,
            )

            env_index = args_cli.record_env_index
            done_value = bool(_to_numpy(dones, env_index))
            success_value = bool(_optional_tensor_attr(raw_env, "_success", env_index))

            _append(records, "time_s", np.array(step * step_dt, dtype=np.float32))
            _append(records, "wall_time_s", np.array(start_time, dtype=np.float64))
            _append(records, "raw_action", raw_action_np.astype(np.float32))
            _append(records, "command_action", command_action.astype(np.float32))
            if hasattr(raw_env, "_applied_actions"):
                _append(records, "applied_action", _to_numpy(raw_env._applied_actions, env_index).astype(np.float32))
            _append(records, "joint_target_policy_order", joint_target.astype(np.float32))
            _append(records, "hardware_command", hardware_command.astype(np.float32))
            _append(records, "joint_pos_policy_order", _to_numpy(raw_env.robot.data.joint_pos[:, raw_env._main_joint_ids], env_index).astype(np.float32))
            _append(records, "joint_vel_policy_order", _to_numpy(raw_env.robot.data.joint_vel[:, raw_env._main_joint_ids], env_index).astype(np.float32))
            _append(records, "gripper_action_policy_order", command_action[20:24].astype(np.float32))
            _append(records, "gripper_target_angle_policy_order", _gripper_target_policy_order(raw_env, env_index))
            _append(records, "bar_estimate", _optional_obs(obs_used, "bar_estimate", env_index))
            _append(records, "bar_delta_b", _optional_obs(obs_used, "bar_estimate_label", env_index))
            if args_cli.record_deployable:
                _append(records, "deployable", _optional_obs(obs_used, "deployable", env_index))
            if args_cli.record_heightmap:
                _append(records, "heightmap", _optional_obs(obs_used, "heightmap", env_index))
            _append(records, "root_pos_w", _to_numpy(raw_env.robot.data.root_pos_w, env_index).astype(np.float32))
            _append(records, "root_quat_w", _to_numpy(raw_env.robot.data.root_quat_w, env_index).astype(np.float32))
            _append(records, "root_lin_vel_b", _to_numpy(raw_env.robot.data.root_lin_vel_b, env_index).astype(np.float32))
            _append(records, "root_ang_vel_b", _to_numpy(raw_env.robot.data.root_ang_vel_b, env_index).astype(np.float32))
            _append(records, "projected_gravity_b", _to_numpy(raw_env.robot.data.projected_gravity_b, env_index).astype(np.float32))
            if hasattr(raw_env, "_foot_body_ids"):
                _append(
                    records,
                    "foot_pos_w",
                    _to_numpy(raw_env.robot.data.body_pos_w[:, raw_env._foot_body_ids], env_index).astype(np.float32),
                )
            _append(records, "foot_pos_b", _optional_tensor_attr(raw_env, "_foot_pos_b", env_index))
            _append(records, "foot_clearance", _optional_tensor_attr(raw_env, "_foot_clearance", env_index))
            _append(records, "foot_contacts", _optional_tensor_attr(raw_env, "_foot_contacts", env_index))
            _append(records, "forbidden_contact", _optional_tensor_attr(raw_env, "_forbidden_contact", env_index))
            _append(records, "forbidden_contact_force", _optional_tensor_attr(raw_env, "_forbidden_contact_force", env_index))
            _append(records, "stage", _optional_tensor_attr(raw_env, "_stage", env_index))
            _append(records, "done", np.array(done_value, dtype=np.bool_))
            _append(records, "success", np.array(success_value, dtype=np.bool_))

            step += 1
            if step % 25 == 0 or success_value or done_value:
                print(
                    f"[REC] step={step} success={success_value} done={done_value} "
                    f"cmd_abs_max={np.max(np.abs(hardware_command[:20])):.3f} "
                    f"grip={np.round(hardware_command[20:24], 3)}"
                )

            if success_value and args_cli.stop_on_success:
                print(f"[INFO] Stopping after success at step {step}.")
                break
            if done_value and args_cli.stop_on_done:
                print(f"[INFO] Stopping after done at step {step}.")
                break

            if args_cli.real_time:
                elapsed = _dt.datetime.now().timestamp() - start_time
                sleep_time = step_dt - elapsed
                if sleep_time > 0.0:
                    import time

                    time.sleep(sleep_time)
    finally:
        vec_env.close()

    arrays = _stack_records(records)
    if "hardware_command" not in arrays:
        raise RuntimeError("No trajectory data was recorded.")
    metadata_out["recorded_steps"] = int(arrays["hardware_command"].shape[0])
    metadata_out["duration_s"] = float(arrays["hardware_command"].shape[0] * step_dt)
    metadata_json = np.array(json.dumps(metadata_out, sort_keys=True), dtype=np.str_)
    np.savez_compressed(
        output,
        metadata_json=metadata_json,
        dt=np.array(step_dt, dtype=np.float32),
        loop_hz=np.array(1.0 / step_dt, dtype=np.float32),
        **arrays,
    )
    print(f"[INFO] Saved {arrays['hardware_command'].shape[0]} steps to {output}")


if __name__ == "__main__":
    main()
    simulation_app.close()
