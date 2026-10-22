"""Restore policy interface data from a checkpoint's training snapshots."""

from __future__ import annotations

import math
from pathlib import Path

import yaml


class TrainingSnapshotLoader(yaml.SafeLoader):
    """Allow Isaac Lab's tuple/slice data tags without evaluating Python objects."""


def _load_slice(loader, node):
    values = loader.construct_sequence(node, deep=True)
    if len(values) != 3 or any(value is not None and type(value) is not int for value in values):
        raise ValueError("Invalid slice in training snapshot")
    return slice(*values)


TrainingSnapshotLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node, deep=True))
)
TrainingSnapshotLoader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice", _load_slice)

ENV_INTERFACE_FIELDS = (
    "action_space", "observation_space", "state_space", "deployable_observation_dim",
    "bar_estimate_observation_dim", "include_gripper_position_obs", "learn_gripper_actions",
    "gait_frequency", "gait_duty_factor", "gait_phase_offsets", "walk_only",
    "gripper_open_angle", "gripper_closed_angle", "gripper_action_step",
    "gripper_action_latch_enabled", "gripper_action_latch_deadband",
    "gripper_action_latch_open_deadband", "gripper_action_latch_close_deadband",
    "gripper_action_latch_min_hold_s", "gripper_action_latch_initial_action",
    "bar_estimate_dx_scale", "bar_estimate_dy_scale", "bar_estimate_dz_scale",
    "bar_estimate_lowpass_alpha", "depth_bar_input_mode", "depth_bar_heightmap_size",
    "depth_bar_max_depth", "depth_bar_depth_crop_max_depth", "depth_bar_memory_enabled",
    "depth_bar_memory_duration_s", "depth_bar_memory_nominal_forward_velocity",
    "depth_bar_memory_fallback_delta_b", "depth_bar_raw_visible_x_range",
    "depth_bar_raw_visible_y_range", "depth_bar_raw_visible_z_range",
)


def load_training_snapshot(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing training snapshot: {path}. Copy params/ alongside the checkpoint.")
    with path.open(encoding="utf-8") as stream:
        result = yaml.load(stream, Loader=TrainingSnapshotLoader)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return result


def _update_object(config, values: dict) -> None:
    for key, value in values.items():
        if not hasattr(config, key):
            raise ValueError(f"Training configuration field is unsupported by this code: {key}")
        current = getattr(config, key)
        if isinstance(value, dict) and current is not None and not isinstance(current, dict):
            _update_object(current, value)
        else:
            setattr(config, key, value)


def restore_training_contract(env_cfg, agent_cfg, checkpoint: str, params_dir: str | None = None) -> dict:
    """Restore interface/model fields before constructing the helper environment or runner."""
    directory = Path(params_dir) if params_dir else Path(checkpoint).resolve().parent / "params"
    saved_env = load_training_snapshot(directory / "env.yaml")
    saved_agent = load_training_snapshot(directory / "agent.yaml")
    required = ("gait_frequency", "deployable_observation_dim", "action_space",
                "bar_estimate_dx_scale", "bar_estimate_dy_scale", "bar_estimate_dz_scale")
    missing = [key for key in required if key not in saved_env]
    if missing:
        raise ValueError(f"Training snapshot is missing policy interface fields: {missing}")
    for key in ("gait_frequency", "bar_estimate_dx_scale", "bar_estimate_dy_scale", "bar_estimate_dz_scale"):
        value = float(saved_env[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {key} in training snapshot: {value}")
    if saved_env["action_space"] != 24:
        raise ValueError("Real learned-gripper playback requires a 24-action checkpoint")
    if saved_agent.get("class_name") != agent_cfg.class_name:
        raise ValueError(
            f"Checkpoint runner {saved_agent.get('class_name')} does not match selected task runner "
            f"{agent_cfg.class_name}; select the checkpoint's original task."
        )
    _update_object(env_cfg, {key: saved_env[key] for key in ENV_INTERFACE_FIELDS if key in saved_env})
    # Old snapshots used one threshold; never inherit newer asymmetric defaults.
    deadband = float(saved_env.get("gripper_action_latch_deadband", 0.0))
    for direction in ("open", "close"):
        key = f"gripper_action_latch_{direction}_deadband"
        setattr(env_cfg, key, float(saved_env.get(key, deadband)))
    sim = saved_env["sim"]
    dt = float(sim["dt"])
    decimation = int(saved_env["decimation"])
    if not math.isfinite(dt) or dt <= 0 or decimation <= 0:
        raise ValueError("Invalid training policy period")
    env_cfg.sim.dt = dt
    env_cfg.decimation = decimation
    robot = saved_env.get("robot", {})
    if "soft_joint_pos_limit_factor" in robot:
        env_cfg.robot.soft_joint_pos_limit_factor = robot["soft_joint_pos_limit_factor"]
    if "joint_pos" in robot.get("init_state", {}):
        env_cfg.robot.init_state.joint_pos = robot["init_state"]["joint_pos"]
    saved_usd = robot.get("spawn", {}).get("usd_path")
    if saved_usd:
        if Path(saved_usd).is_file():
            env_cfg.robot.spawn.usd_path = saved_usd
        elif Path(env_cfg.robot.spawn.usd_path).name != Path(saved_usd).name:
            raise FileNotFoundError(f"Checkpoint robot asset is unavailable: {saved_usd}")
    model_fields = ("class_name", "obs_groups", "clip_actions", "actor", "critic", "student", "teacher")
    _update_object(agent_cfg, {key: saved_agent[key] for key in model_fields if key in saved_agent})
    return {"params_dir": str(directory), "policy_period_s": dt * decimation,
            "env": saved_env, "agent": saved_agent}


def resolve_loop_hz(requested: float | None, policy_period_s: float) -> float:
    expected = 1.0 / policy_period_s
    if requested is not None and (not math.isfinite(requested) or not math.isclose(requested, expected, rel_tol=1e-6)):
        raise ValueError(f"--loop_hz must match training: {expected:g} Hz (got {requested})")
    return expected
