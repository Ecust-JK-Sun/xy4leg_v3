from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


THIS_DIR = Path(__file__).resolve().parent
TASK_DIR = THIS_DIR.parent
SIM2REAL_DIR = TASK_DIR / "sim2real_v3"

POLICY_LEGS = ("RF", "LF", "RB", "LB")
HARDWARE_LEGS = ("LB", "LF", "RB", "RF")

# Hardware buffer order is joint-major:
# [LB1, LF1, RB1, RF1, LB2, LF2, ..., LB6, LF6, RB6, RF6].
REAL_TO_SIM_SIGN = np.array(
    [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
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


def hw_index(leg: str, joint_index: int) -> int:
    return HARDWARE_LEGS.index(leg) + 4 * joint_index


def policy_index(leg: str, joint_index: int) -> int:
    return POLICY_LEGS.index(leg) * 5 + joint_index


def policy20_to_hardware20(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(20)
    hardware = np.zeros(20, dtype=np.float32)
    for leg in POLICY_LEGS:
        for joint_index in range(5):
            hardware[hw_index(leg, joint_index)] = values[policy_index(leg, joint_index)]
    return hardware


# Gather tables: index by hardware position, value is the policy-order position to read.
# Used as `hardware = policy_order_values[..., POLICY20_TO_HARDWARE20_INDEX]`.
POLICY20_TO_HARDWARE20_INDEX = np.zeros(20, dtype=np.int64)
for _leg in POLICY_LEGS:
    for _joint_index in range(5):
        POLICY20_TO_HARDWARE20_INDEX[hw_index(_leg, _joint_index)] = policy_index(_leg, _joint_index)

# Scatter table: index by policy gripper slot (RF, LF, RB, LB), value is the hardware slot
# (LB, LF, RB, RF). Used as `hardware_command[..., 20 + idx] = policy_gripper_torque`.
POLICY_GRIPPER_TO_HARDWARE_GRIPPER_INDEX = np.array(
    [HARDWARE_LEGS.index(leg) for leg in POLICY_LEGS],
    dtype=np.int64,
)


def policy20_array_to_hardware20_array(values: np.ndarray) -> np.ndarray:
    """Convert joint values shaped [..., 20] from policy leg order to hardware joint-major order."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != 20:
        raise ValueError(f"Expected last axis to be 20 policy-order leg joints, got shape {values.shape}")
    return values[..., POLICY20_TO_HARDWARE20_INDEX]


def parse_float_tuple(value: str, count: int, name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != count:
        raise ValueError(f"Expected {name} to contain {count} comma-separated values, got {value!r}")
    return tuple(float(part) for part in parts)


def read_gripper_calibration(
    path: str | Path,
    global_torque_scale: float = 1.0,
    fallback_deadband: float = 0.05,
) -> GripperCalibration:
    defaults = {
        "LB": {"close_pos": -6.96, "torque_sign": -1.0, "torque_scale": 1.0, "deadband": fallback_deadband},
        "LF": {"close_pos": -7.38, "torque_sign": -1.0, "torque_scale": 1.0, "deadband": fallback_deadband},
        "RB": {"close_pos": -9.15, "torque_sign": -1.0, "torque_scale": 1.0, "deadband": fallback_deadband},
        "RF": {"close_pos": -7.0, "torque_sign": -1.0, "torque_scale": 1.0, "deadband": fallback_deadband},
    }
    cfg: dict[str, Any] = {}
    if path:
        with open(Path(path).expanduser(), "r", encoding="utf-8") as f:
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


def policy_action_to_hardware_command(
    action: np.ndarray,
    action_offset: np.ndarray,
    action_scale: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    gripper_calibration: GripperCalibration,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] == 20:
        action = np.concatenate((action, np.zeros(4, dtype=np.float32)), axis=0)
    if action.shape[0] != 24:
        raise ValueError(f"Expected 20D or 24D policy action, got shape {action.shape}")
    action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
    action = np.clip(action, -1.0, 1.0)

    action_offset = np.asarray(action_offset, dtype=np.float32).reshape(20)
    action_scale = np.asarray(action_scale, dtype=np.float32).reshape(20)
    joint_lower = np.asarray(joint_lower, dtype=np.float32).reshape(20)
    joint_upper = np.asarray(joint_upper, dtype=np.float32).reshape(20)

    target20 = action_offset + action_scale * action[:20]
    target20 = np.clip(target20, joint_lower, joint_upper)
    command_sim = np.zeros(24, dtype=np.float32)
    command_sim[:20] = policy20_to_hardware20(target20)

    for leg in POLICY_LEGS:
        policy_leg = POLICY_LEGS.index(leg)
        hardware_leg = HARDWARE_LEGS.index(leg)
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
        command_sim[20 + hardware_leg] = float(np.clip(torque, -max_torque, max_torque))

    return command_sim * SIM_TO_REAL_SIGN


def policy_joint_targets_to_hardware_command(
    joint_targets_policy_order: np.ndarray,
    gripper_action_policy_order: np.ndarray,
    gripper_calibration: GripperCalibration,
) -> np.ndarray:
    joint_targets_policy_order = np.asarray(joint_targets_policy_order, dtype=np.float32).reshape(20)
    gripper_action_policy_order = np.asarray(gripper_action_policy_order, dtype=np.float32).reshape(4)
    gripper_action_policy_order = np.nan_to_num(
        gripper_action_policy_order,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )
    gripper_action_policy_order = np.clip(gripper_action_policy_order, -1.0, 1.0)

    command_sim = np.zeros(24, dtype=np.float32)
    command_sim[:20] = policy20_to_hardware20(joint_targets_policy_order)
    for leg in POLICY_LEGS:
        policy_leg = POLICY_LEGS.index(leg)
        hardware_leg = HARDWARE_LEGS.index(leg)
        value = float(gripper_action_policy_order[policy_leg])
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
        command_sim[20 + hardware_leg] = float(np.clip(torque, -max_torque, max_torque))
    return command_sim * SIM_TO_REAL_SIGN


def policy_joint_targets_array_to_hardware_commands(
    joint_targets_policy_order: np.ndarray,
    gripper_desired_policy_order: np.ndarray,
    gripper_calibration: GripperCalibration,
) -> np.ndarray:
    """Convert a whole [T,20] policy-order joint-target array to [T,24] hardware commands.

    `joint_targets_policy_order` uses policy leg order (RF, LF, RB, LB) and the sim/URDF joint
    sign convention. `gripper_desired_policy_order` is [T,4] in policy leg order; booleans are
    treated as True = open, False = close, other dtypes as continuous action clipped to [-1, 1].
    """
    targets = np.asarray(joint_targets_policy_order, dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] != 20:
        raise ValueError(f"Expected joint_target shape [T,20], got {targets.shape}")
    gripper = np.asarray(gripper_desired_policy_order)
    if gripper.ndim != 2 or gripper.shape[1] != 4 or gripper.shape[0] != targets.shape[0]:
        raise ValueError(f"Expected gripper desired shape [T,4] matching joint_target, got {gripper.shape}")
    if gripper.dtype == bool:
        desired = np.where(gripper, 1.0, -1.0).astype(np.float32)
    else:
        desired = np.nan_to_num(gripper.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        desired = np.clip(desired, -1.0, 1.0)

    gripper_index = POLICY_GRIPPER_TO_HARDWARE_GRIPPER_INDEX
    torque_sign = np.asarray(gripper_calibration.torque_sign, dtype=np.float32)[gripper_index]
    torque_scale = np.asarray(gripper_calibration.torque_scale, dtype=np.float32)[gripper_index]
    deadband = np.asarray(gripper_calibration.deadband, dtype=np.float32)[gripper_index]
    max_torque = np.abs(np.asarray(gripper_calibration.max_torque, dtype=np.float32))[gripper_index]

    torque = desired * (torque_sign * torque_scale)[None, :]
    torque = np.where(np.abs(desired) < deadband[None, :], 0.0, torque)
    torque = np.clip(torque, -max_torque[None, :], max_torque[None, :])

    commands = np.zeros((targets.shape[0], 24), dtype=np.float32)
    commands[:, :20] = policy20_array_to_hardware20_array(targets)
    commands[:, 20 + gripper_index] = torque
    return commands * SIM_TO_REAL_SIGN[None, :]


def resolve_hardware_commands(
    arrays: dict[str, np.ndarray],
    gripper_calibration: GripperCalibration,
) -> tuple[np.ndarray, str]:
    """Return [T,24] hardware commands plus a label describing where they came from.

    Supports both the recorded replay format (`hardware_command`) and crossing-demo files that
    only store `joint_target[20]` (policy order, sim sign) plus `gripper_open[4]` booleans.
    """
    if "hardware_command" in arrays:
        return np.asarray(arrays["hardware_command"], dtype=np.float32), "hardware_command"
    if "joint_target" in arrays:
        if "gripper_open" not in arrays:
            raise ValueError(
                "Trajectory has joint_target but no gripper_open; cannot build gripper torque commands"
            )
        commands = policy_joint_targets_array_to_hardware_commands(
            arrays["joint_target"],
            arrays["gripper_open"],
            gripper_calibration,
        )
        return commands, "joint_target+gripper_open"
    raise ValueError("Trajectory has neither hardware_command nor joint_target")


def recorded_rate_hz(arrays: dict[str, np.ndarray], metadata: dict[str, Any] | None = None) -> float:
    """Best-effort sampling rate of a trajectory file in Hz."""
    metadata = metadata or {}
    for key in ("loop_hz",):
        if key in arrays:
            value = float(np.asarray(arrays[key]).reshape(()))
            if value > 0.0:
                return value
    for key in ("dt", "sample_dt"):
        if key in arrays:
            dt = float(np.asarray(arrays[key]).reshape(()))
            if dt > 0.0:
                return 1.0 / dt
    policy_dt = float(metadata.get("policy_dt", 0.04))
    return 1.0 / max(policy_dt, 1.0e-6)


def metadata_from_npz(data: Any) -> dict[str, Any]:
    if "metadata_json" not in data:
        return {}
    raw = data["metadata_json"]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not raw:
        return {}
    return json.loads(str(raw))


def load_trajectory(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz = np.load(Path(path).expanduser(), allow_pickle=False)
    arrays = {key: npz[key] for key in npz.files}
    metadata = metadata_from_npz(npz)
    return arrays, metadata


def hardware_joint_limits_from_policy_metadata(metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    if "joint_lower" not in metadata or "joint_upper" not in metadata:
        return None
    lower_policy = np.asarray(metadata["joint_lower"], dtype=np.float32).reshape(20)
    upper_policy = np.asarray(metadata["joint_upper"], dtype=np.float32).reshape(20)
    lower_hw_sim = policy20_to_hardware20(lower_policy)
    upper_hw_sim = policy20_to_hardware20(upper_policy)
    lower_hw_real = lower_hw_sim * SIM_TO_REAL_SIGN[:20]
    upper_hw_real = upper_hw_sim * SIM_TO_REAL_SIGN[:20]
    return np.minimum(lower_hw_real, upper_hw_real), np.maximum(lower_hw_real, upper_hw_real)


def validate_hardware_commands(
    commands: np.ndarray,
    metadata: dict[str, Any] | None = None,
    max_joint_step_rad: float = 0.04,
    max_joint_abs_rad: float = 2.2,
    joint_limit_margin_rad: float = 0.05,
) -> list[str]:
    metadata = metadata or {}
    commands = np.asarray(commands, dtype=np.float32)
    warnings: list[str] = []
    if commands.ndim != 2 or commands.shape[1] != 24:
        raise ValueError(f"Expected hardware_command shape [T,24], got {commands.shape}")
    if commands.shape[0] == 0:
        raise ValueError("Trajectory contains zero commands")
    if not np.isfinite(commands).all():
        raise ValueError("Trajectory contains NaN or Inf commands")

    leg_commands = commands[:, :20]
    if max_joint_abs_rad > 0.0:
        abs_max = float(np.max(np.abs(leg_commands)))
        if abs_max > max_joint_abs_rad:
            warnings.append(
                f"leg command abs max {abs_max:.4f} rad exceeds --max_joint_abs_rad {max_joint_abs_rad:.4f}"
            )
    if commands.shape[0] > 1 and max_joint_step_rad > 0.0:
        step_abs_max = float(np.max(np.abs(np.diff(leg_commands, axis=0))))
        if step_abs_max > max_joint_step_rad:
            warnings.append(
                f"leg command max frame step {step_abs_max:.4f} rad exceeds --max_joint_step_rad "
                f"{max_joint_step_rad:.4f}"
            )

    limits = hardware_joint_limits_from_policy_metadata(metadata)
    if limits is not None:
        lower_hw, upper_hw = limits
        below = leg_commands < (lower_hw[None, :] - float(joint_limit_margin_rad))
        above = leg_commands > (upper_hw[None, :] + float(joint_limit_margin_rad))
        if np.any(below) or np.any(above):
            warnings.append(
                "leg command exceeds recorded joint limits "
                f"with margin {joint_limit_margin_rad:.4f} rad"
            )
    return warnings


def command_summary(commands: np.ndarray) -> dict[str, float]:
    commands = np.asarray(commands, dtype=np.float32)
    leg = commands[:, :20]
    grip = commands[:, 20:24]
    if commands.shape[0] > 1:
        max_step = float(np.max(np.abs(np.diff(leg, axis=0))))
    else:
        max_step = 0.0
    return {
        "steps": float(commands.shape[0]),
        "leg_min_rad": float(np.min(leg)),
        "leg_max_rad": float(np.max(leg)),
        "leg_abs_max_rad": float(np.max(np.abs(leg))),
        "leg_max_step_rad": max_step,
        "gripper_min": float(np.min(grip)),
        "gripper_max": float(np.max(grip)),
    }


def slice_commands(commands: np.ndarray, start_step: int = 0, end_step: int = 0) -> np.ndarray:
    commands = np.asarray(commands, dtype=np.float32)
    start = max(int(start_step), 0)
    end = int(end_step) if int(end_step) > 0 else commands.shape[0]
    end = min(end, commands.shape[0])
    if start >= end:
        raise ValueError(f"Invalid slice start={start}, end={end}, total={commands.shape[0]}")
    return commands[start:end].copy()
