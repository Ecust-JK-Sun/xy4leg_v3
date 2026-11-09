from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    HARDWARE_LEGS,
    POLICY_LEGS,
    SIM_TO_REAL_SIGN,
    command_summary,
    parse_float_tuple,
    policy_joint_targets_to_hardware_command,
    read_gripper_calibration,
    validate_hardware_commands,
)


THIS_DIR = Path(__file__).resolve().parent
TASK_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[5]
DEFAULT_URDF = REPO_ROOT / "Collect_data_pybullet" / "xy4legv5_gripper" / "urdf" / "xy4legv5_gripper.urdf"
DEFAULT_OUTPUT = THIS_DIR / "trajectories" / "pybullet_footpath_ik.npz"
DEFAULT_GRIPPER_CALIBRATION = TASK_DIR / "sim2real_v3" / "gripper_calibration.yaml"

INIT_JOINT_POS_DEG = {
    "RF": (65.0, 45.0, -20.0, 24.0, 24.0),
    "LF": (-65.0, -85.0, -20.0, 24.0, 24.0),
    "RB": (-65.0, -85.0, -20.0, 24.0, 24.0),
    "LB": (65.0, 45.0, -20.0, 24.0, 24.0),
}

SAVE_DOF_NAMES = tuple(
    name
    for leg in POLICY_LEGS
    for name in (
        f"{leg}_Joint1",
        f"{leg}_Joint2",
        f"{leg}_Joint3",
        f"{leg}_Joint4",
        f"{leg}_Joint5",
        f"{leg}_Joint6",
    )
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Truss V3 open-loop replay trajectory from analytic foot-end paths "
            "solved with PyBullet IK."
        )
    )
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_URDF))
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("--base_position", type=str, default="0.0,0.0,1.02")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--duration_s", type=float, default=24.0)
    parser.add_argument(
        "--target_progress_m",
        type=float,
        default=0.0,
        help="If >0, override duration_s using target_progress_m / (step_length_m * gait_freq).",
    )
    parser.add_argument("--bar_distance_m", type=float, default=0.0, help="Metadata note for the intended crossbar distance.")
    parser.add_argument("--prepend_hold_s", type=float, default=0.2)
    parser.add_argument("--append_hold_s", type=float, default=0.0)
    parser.add_argument("--gait_freq", type=float, default=0.16)
    parser.add_argument("--duty", type=float, default=0.75)
    parser.add_argument("--step_length_m", type=float, default=0.040)
    parser.add_argument("--swing_height_m", type=float, default=0.045)
    parser.add_argument(
        "--warmup_cycles",
        type=float,
        default=0.0,
        help="Use a scaled swing height for the first N global gait cycles while keeping step length unchanged.",
    )
    parser.add_argument(
        "--warmup_swing_height_scale",
        type=float,
        default=1.0,
        help="Swing-height multiplier during warmup_cycles. For example 0.5 halves the lift height.",
    )
    parser.add_argument("--bar_x_m", type=float, default=0.0)
    parser.add_argument("--bar_window_m", type=float, default=0.050)
    parser.add_argument("--bar_extra_height_m", type=float, default=0.010)
    parser.add_argument(
        "--phase_offsets",
        type=str,
        default="RF:0.0,LB:0.25,LF:0.5,RB:0.75",
        help="Comma-separated leg:phase values in cycles. Default gives one-leg-at-a-time crawl timing.",
    )
    parser.add_argument(
        "--direction",
        choices=("forward", "backward"),
        default="forward",
        help="Forward means stance moves foot target from +x to -x while swing returns -x to +x.",
    )
    parser.add_argument(
        "--gripper_mode",
        choices=("phase", "closed", "open", "zero"),
        default="phase",
        help="phase: close during stance, open during swing, using sim action convention -1 close / +1 open.",
    )
    parser.add_argument("--gripper_calibration", type=str, default=str(DEFAULT_GRIPPER_CALIBRATION))
    parser.add_argument("--gripper_torque_scale", type=float, default=1.0)
    parser.add_argument("--gripper_deadband", type=float, default=0.05)
    parser.add_argument("--joint_abs_limit_rad", type=float, default=2.2)
    parser.add_argument("--joint_limit_soft_factor", type=float, default=0.95)
    parser.add_argument("--max_joint_step_rad", type=float, default=0.04)
    parser.add_argument("--joint_limit_margin_rad", type=float, default=0.05)
    parser.add_argument("--ik_max_iterations", type=int, default=400)
    parser.add_argument("--ik_residual_threshold", type=float, default=1.0e-6)
    parser.add_argument(
        "--use_ee_orientation",
        action="store_true",
        default=False,
        help="Also constrain Link5 orientation during IK. Default solves foot-end position only for smoother joints.",
    )
    parser.add_argument("--load_plane", action="store_true", default=False)
    parser.add_argument("--debug_draw", action="store_true", default=False)
    parser.add_argument("--print_interval", type=int, default=50)
    return parser


def import_pybullet():
    try:
        import pybullet as p
    except ImportError as exc:
        raise SystemExit(
            "PyBullet is not installed in the current Python environment.\n"
            "Install it in the IsaacLab env first, for example:\n"
            "  pip install pybullet\n"
            "or:\n"
            "  conda install -c conda-forge pybullet"
        ) from exc
    try:
        import pybullet_data
    except ImportError:
        pybullet_data = None
    return p, pybullet_data


def min_jerk(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return 6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3


def swing_bump(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return 16.0 * s**2 * (1.0 - s) ** 2


def raised_cosine_bump(x: float, center: float, half_width: float) -> float:
    half_width = float(half_width)
    if half_width <= 0.0:
        return 0.0
    d = abs(float(x) - float(center)) / half_width
    if d >= 1.0:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * d))


def parse_phase_offsets(value: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for part in value.split(","):
        if not part.strip():
            continue
        if ":" not in part:
            raise ValueError(f"Invalid phase offset item {part!r}; expected LEG:VALUE")
        leg, raw_phase = part.split(":", 1)
        leg = leg.strip().upper()
        if leg not in POLICY_LEGS:
            raise ValueError(f"Unknown leg {leg!r}; expected one of {POLICY_LEGS}")
        result[leg] = float(raw_phase) % 1.0
    missing = [leg for leg in POLICY_LEGS if leg not in result]
    if missing:
        raise ValueError(f"Missing phase offsets for legs: {missing}")
    return result


def init_joint_policy_order() -> np.ndarray:
    values = []
    for leg in POLICY_LEGS:
        values.extend(math.radians(deg) for deg in INIT_JOINT_POS_DEG[leg])
    return np.array(values, dtype=np.float32)


def build_name_index(p, robot_id: int) -> dict[str, int]:
    name_to_idx = {}
    for joint_index in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, joint_index)
        name_to_idx[info[1].decode("utf-8")] = joint_index
    return name_to_idx


def build_link_index(p, robot_id: int) -> dict[str, int]:
    link_to_idx = {}
    for joint_index in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, joint_index)
        link_to_idx[info[12].decode("utf-8")] = joint_index
    return link_to_idx


def policy_joint_names() -> tuple[str, ...]:
    return tuple(f"{leg}_Joint{joint_index}" for leg in POLICY_LEGS for joint_index in range(1, 6))


def joint_policy_index(leg: str, joint_number: int) -> int:
    return POLICY_LEGS.index(leg) * 5 + (joint_number - 1)


def reset_main_joints(p, robot_id: int, name_to_idx: dict[str, int], policy_joints: np.ndarray) -> None:
    policy_joints = np.asarray(policy_joints, dtype=np.float32).reshape(20)
    for leg in POLICY_LEGS:
        for joint_number in range(1, 6):
            name = f"{leg}_Joint{joint_number}"
            p.resetJointState(
                robot_id,
                name_to_idx[name],
                targetValue=float(policy_joints[joint_policy_index(leg, joint_number)]),
                targetVelocity=0.0,
            )


def reset_gripper_joints(p, robot_id: int, name_to_idx: dict[str, int], gripper_angle: float) -> None:
    for leg in POLICY_LEGS:
        for suffix in ("1out", "2out"):
            name = f"{leg}_{suffix}_Joint"
            if name in name_to_idx:
                p.resetJointState(robot_id, name_to_idx[name], targetValue=float(gripper_angle), targetVelocity=0.0)


def get_policy_joint_limits(
    p,
    robot_id: int,
    name_to_idx: dict[str, int],
    joint_abs_limit_rad: float,
    soft_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.zeros(20, dtype=np.float32)
    upper = np.zeros(20, dtype=np.float32)
    abs_limit = abs(float(joint_abs_limit_rad))
    soft_factor = float(np.clip(soft_factor, 0.0, 1.0))
    for leg in POLICY_LEGS:
        for joint_number in range(1, 6):
            name = f"{leg}_Joint{joint_number}"
            info = p.getJointInfo(robot_id, name_to_idx[name])
            lo = float(info[8])
            hi = float(info[9])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                lo, hi = -math.pi, math.pi
            lo *= soft_factor
            hi *= soft_factor
            if abs_limit > 0.0:
                lo = max(lo, -abs_limit)
                hi = min(hi, abs_limit)
            idx = joint_policy_index(leg, joint_number)
            lower[idx] = lo
            upper[idx] = hi
    return lower, upper


def get_movable_joint_indices(p, robot_id: int) -> list[int]:
    return [
        joint_index
        for joint_index in range(p.getNumJoints(robot_id))
        if p.getJointInfo(robot_id, joint_index)[2] != p.JOINT_FIXED
    ]


def get_joint_limits_from_info(info: tuple[Any, ...]) -> tuple[float, float]:
    lo = float(info[8])
    hi = float(info[9])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return -math.pi, math.pi
    return lo, hi


def build_ik_limits_and_rest(
    p,
    robot_id: int,
    name_to_idx: dict[str, int],
    movable_indices: list[int],
    policy_rest: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    gripper_rest_angle: float,
) -> tuple[list[float], list[float], list[float], list[float]]:
    name_by_idx = {idx: name for name, idx in name_to_idx.items()}
    lower_limits: list[float] = []
    upper_limits: list[float] = []
    joint_ranges: list[float] = []
    rest_poses: list[float] = []
    for joint_index in movable_indices:
        name = name_by_idx[joint_index]
        if name in policy_joint_names():
            leg = name[:2]
            joint_number = int(name[-1])
            policy_idx = joint_policy_index(leg, joint_number)
            lo = float(joint_lower[policy_idx])
            hi = float(joint_upper[policy_idx])
            rest = float(policy_rest[policy_idx])
        else:
            info = p.getJointInfo(robot_id, joint_index)
            lo, hi = get_joint_limits_from_info(info)
            rest = float(gripper_rest_angle) if name.endswith("out_Joint") else 0.0
        lower_limits.append(lo)
        upper_limits.append(hi)
        joint_ranges.append(max(hi - lo, 1.0e-3))
        rest_poses.append(float(np.clip(rest, lo, hi)))
    return lower_limits, upper_limits, joint_ranges, rest_poses


def ik_value_for_joint(ik_result: tuple[float, ...], joint_index: int, movable_index_map: dict[int, int]) -> float:
    if joint_index < len(ik_result):
        return float(ik_result[joint_index])
    if joint_index not in movable_index_map:
        raise KeyError(f"Joint index {joint_index} is not movable and not present in IK result")
    return float(ik_result[movable_index_map[joint_index]])


def nearest_angle_in_limits(raw: float, previous: float, lo: float, hi: float) -> float:
    candidates = np.array([raw + 2.0 * math.pi * k for k in range(-2, 3)], dtype=np.float64)
    valid = candidates[(candidates >= lo) & (candidates <= hi)]
    if valid.size:
        candidates = valid
    best = float(candidates[np.argmin(np.abs(candidates - previous))])
    return float(np.clip(best, lo, hi))


def foot_relative_offset(
    traj_time: float,
    phase_offset: float,
    gait_freq: float,
    duty: float,
    step_length: float,
    swing_height: float,
    bar_x: float,
    bar_window: float,
    bar_extra_height: float,
    direction_sign: float,
) -> tuple[np.ndarray, float, bool, float]:
    phase = (float(traj_time) * float(gait_freq) + float(phase_offset)) % 1.0
    duty = float(np.clip(duty, 1.0e-3, 1.0 - 1.0e-3))
    if phase <= duty:
        s = phase / duty
        x = step_length * (0.5 - min_jerk(s))
        z = 0.0
        is_swing = False
        gripper_action = -1.0
    else:
        s = (phase - duty) / (1.0 - duty)
        x = step_length * (-0.5 + min_jerk(s))
        z_shape = swing_bump(s)
        z = swing_height * z_shape
        z += bar_extra_height * raised_cosine_bump(x, bar_x, bar_window) * z_shape
        is_swing = True
        gripper_action = 1.0
    rel = np.array([direction_sign * x, 0.0, z], dtype=np.float32)
    return rel, gripper_action, is_swing, phase


def warmup_height_scale(traj_time: float, gait_freq: float, warmup_cycles: float, warmup_scale: float) -> float:
    if float(warmup_cycles) <= 0.0:
        return 1.0
    cycle_position = float(traj_time) * float(gait_freq)
    if cycle_position < float(warmup_cycles):
        return float(warmup_scale)
    return 1.0


def get_link_pose(p, robot_id: int, link_index: int) -> tuple[np.ndarray, np.ndarray]:
    link_state = p.getLinkState(robot_id, link_index, computeForwardKinematics=1)
    return np.array(link_state[4], dtype=np.float32), np.array(link_state[5], dtype=np.float32)


def get_foot_positions(p, robot_id: int, ee_indices: dict[str, int]) -> np.ndarray:
    positions = []
    for leg in POLICY_LEGS:
        pos, _ = get_link_pose(p, robot_id, ee_indices[leg])
        positions.append(pos)
    return np.stack(positions, axis=0).astype(np.float32)


def solve_frame_ik(
    p,
    robot_id: int,
    name_to_idx: dict[str, int],
    movable_indices: list[int],
    movable_index_map: dict[int, int],
    ee_indices: dict[str, int],
    target_positions: dict[str, np.ndarray],
    target_orientations: dict[str, np.ndarray] | None,
    previous_policy_joints: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    gripper_rest_angle: float,
    max_iterations: int,
    residual_threshold: float,
) -> np.ndarray:
    solution = np.asarray(previous_policy_joints, dtype=np.float32).reshape(20).copy()
    reset_main_joints(p, robot_id, name_to_idx, solution)
    reset_gripper_joints(p, robot_id, name_to_idx, gripper_rest_angle)

    for leg in POLICY_LEGS:
        lower_limits, upper_limits, joint_ranges, rest_poses = build_ik_limits_and_rest(
            p,
            robot_id,
            name_to_idx,
            movable_indices,
            solution,
            joint_lower,
            joint_upper,
            gripper_rest_angle,
        )
        ik_kwargs = {
            "bodyUniqueId": robot_id,
            "endEffectorLinkIndex": ee_indices[leg],
            "targetPosition": target_positions[leg].tolist(),
            "lowerLimits": lower_limits,
            "upperLimits": upper_limits,
            "jointRanges": joint_ranges,
            "restPoses": rest_poses,
            "maxNumIterations": int(max_iterations),
            "residualThreshold": float(residual_threshold),
        }
        if target_orientations is not None:
            ik_kwargs["targetOrientation"] = target_orientations[leg].tolist()
        ik_result = p.calculateInverseKinematics(**ik_kwargs)
        for joint_number in range(1, 6):
            joint_name = f"{leg}_Joint{joint_number}"
            joint_index = name_to_idx[joint_name]
            policy_idx = joint_policy_index(leg, joint_number)
            raw = ik_value_for_joint(ik_result, joint_index, movable_index_map)
            solution[policy_idx] = nearest_angle_in_limits(
                raw,
                previous=float(solution[policy_idx]),
                lo=float(joint_lower[policy_idx]),
                hi=float(joint_upper[policy_idx]),
            )
        reset_main_joints(p, robot_id, name_to_idx, solution)
    return solution.astype(np.float32)


def gripper_actions_for_mode(mode: str, phase_actions: np.ndarray) -> np.ndarray:
    if mode == "phase":
        return phase_actions.astype(np.float32)
    if mode == "closed":
        return -np.ones(4, dtype=np.float32)
    if mode == "open":
        return np.ones(4, dtype=np.float32)
    if mode == "zero":
        return np.zeros(4, dtype=np.float32)
    raise ValueError(f"Unknown gripper mode: {mode}")


def make_dof_positions(joint_policy: np.ndarray, gripper_policy: np.ndarray) -> np.ndarray:
    joint_policy = np.asarray(joint_policy, dtype=np.float32)
    gripper_policy = np.asarray(gripper_policy, dtype=np.float32)
    output = np.zeros((joint_policy.shape[0], 24), dtype=np.float32)
    for leg_index in range(4):
        output[:, leg_index * 6 : leg_index * 6 + 5] = joint_policy[:, leg_index * 5 : leg_index * 5 + 5]
        output[:, leg_index * 6 + 5] = gripper_policy[:, leg_index]
    return output


def append_repeated_frames(arrays: dict[str, np.ndarray], prepend: int, append: int, dt: float) -> dict[str, np.ndarray]:
    if prepend <= 0 and append <= 0:
        return arrays
    out: dict[str, np.ndarray] = {}
    frame_count = arrays["hardware_command"].shape[0]
    for key, value in arrays.items():
        if not isinstance(value, np.ndarray) or value.shape[:1] != (frame_count,):
            out[key] = value
            continue
        pieces = []
        if prepend > 0:
            pieces.append(np.repeat(value[:1], prepend, axis=0))
        pieces.append(value)
        if append > 0:
            pieces.append(np.repeat(value[-1:], append, axis=0))
        out[key] = np.concatenate(pieces, axis=0)
    out["time_s"] = (np.arange(out["hardware_command"].shape[0], dtype=np.float32) * float(dt)).astype(np.float32)
    return out


def draw_target_points(p, points: np.ndarray, color: tuple[float, float, float]) -> None:
    for point in points:
        p.addUserDebugLine(
            [float(point[0]), float(point[1]), float(point[2]) - 0.005],
            [float(point[0]), float(point[1]), float(point[2]) + 0.005],
            color,
            lineWidth=1.0,
            lifeTime=0.0,
        )


def generate(args: argparse.Namespace) -> None:
    p, pybullet_data = import_pybullet()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    urdf_path = Path(args.urdf).expanduser().resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    base_position = parse_float_tuple(args.base_position, 3, "--base_position")
    phase_offsets = parse_phase_offsets(args.phase_offsets)
    fps = float(args.fps)
    if fps <= 0.0:
        raise ValueError("--fps must be positive")
    dt = 1.0 / fps
    nominal_speed_mps = abs(float(args.step_length_m) * float(args.gait_freq))
    if float(args.target_progress_m) > 0.0:
        if nominal_speed_mps <= 1.0e-6:
            raise ValueError("--target_progress_m requires positive --step_length_m and --gait_freq")
        duration_s = float(args.target_progress_m) / nominal_speed_mps
    else:
        duration_s = max(float(args.duration_s), 0.0)
    frame_count = max(int(round(duration_s * fps)), 1)
    prepend_frames = max(int(round(float(args.prepend_hold_s) * fps)), 0)
    append_frames = max(int(round(float(args.append_hold_s) * fps)), 0)
    direction_sign = 1.0 if args.direction == "forward" else -1.0

    client_id = p.connect(p.GUI if args.gui else p.DIRECT, options="--opengl2")
    try:
        if pybullet_data is not None:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if len(urdf_path.parents) >= 3:
            p.setAdditionalSearchPath(str(urdf_path.parents[2]))
        p.setAdditionalSearchPath(str(urdf_path.parent))
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(dt)

        if args.load_plane:
            p.loadURDF("plane.urdf")
        robot_id = p.loadURDF(str(urdf_path), basePosition=list(base_position), useFixedBase=True)

        name_to_idx = build_name_index(p, robot_id)
        link_to_idx = build_link_index(p, robot_id)
        missing_joints = [name for name in policy_joint_names() if name not in name_to_idx]
        if missing_joints:
            raise ValueError(f"URDF missing main joints: {missing_joints}")
        ee_indices = {}
        for leg in POLICY_LEGS:
            link_name = f"{leg}_Link5"
            if link_name not in link_to_idx:
                raise ValueError(f"URDF missing end-effector link: {link_name}")
            ee_indices[leg] = link_to_idx[link_name]

        movable_indices = get_movable_joint_indices(p, robot_id)
        movable_index_map = {joint_index: i for i, joint_index in enumerate(movable_indices)}
        joint_lower, joint_upper = get_policy_joint_limits(
            p,
            robot_id,
            name_to_idx,
            joint_abs_limit_rad=float(args.joint_abs_limit_rad),
            soft_factor=float(args.joint_limit_soft_factor),
        )
        init_policy = init_joint_policy_order()
        init_policy = np.clip(init_policy, joint_lower, joint_upper)
        gripper_calibration = read_gripper_calibration(
            args.gripper_calibration,
            global_torque_scale=float(args.gripper_torque_scale),
            fallback_deadband=float(args.gripper_deadband),
        )

        reset_main_joints(p, robot_id, name_to_idx, init_policy)
        reset_gripper_joints(p, robot_id, name_to_idx, gripper_calibration.sim_open_angle_rad)
        p.stepSimulation()

        init_foot_pos = {}
        init_foot_orn = {}
        anchor_pos = {}
        height_scale0 = warmup_height_scale(
            traj_time=0.0,
            gait_freq=float(args.gait_freq),
            warmup_cycles=float(args.warmup_cycles),
            warmup_scale=float(args.warmup_swing_height_scale),
        )
        for leg in POLICY_LEGS:
            pos, orn = get_link_pose(p, robot_id, ee_indices[leg])
            init_foot_pos[leg] = pos
            init_foot_orn[leg] = orn
            rel0, _, _, _ = foot_relative_offset(
                traj_time=0.0,
                phase_offset=phase_offsets[leg],
                gait_freq=float(args.gait_freq),
                duty=float(args.duty),
                step_length=float(args.step_length_m),
                swing_height=float(args.swing_height_m) * height_scale0,
                bar_x=float(args.bar_x_m),
                bar_window=float(args.bar_window_m),
                bar_extra_height=float(args.bar_extra_height_m) * height_scale0,
                direction_sign=direction_sign,
            )
            anchor_pos[leg] = pos - rel0

        joint_policy_frames = np.zeros((frame_count, 20), dtype=np.float32)
        gripper_policy_frames = np.zeros((frame_count, 4), dtype=np.float32)
        hardware_command_frames = np.zeros((frame_count, 24), dtype=np.float32)
        foot_target_frames = np.zeros((frame_count, 4, 3), dtype=np.float32)
        foot_actual_frames = np.zeros((frame_count, 4, 3), dtype=np.float32)
        ik_error_frames = np.zeros((frame_count, 4), dtype=np.float32)
        phase_frames = np.zeros((frame_count, 4), dtype=np.float32)
        is_swing_frames = np.zeros((frame_count, 4), dtype=np.bool_)
        swing_height_scale_frames = np.zeros((frame_count,), dtype=np.float32)

        previous_policy = init_policy.copy()
        start_wall = time.time()
        for frame in range(frame_count):
            traj_time = frame * dt
            height_scale = warmup_height_scale(
                traj_time=traj_time,
                gait_freq=float(args.gait_freq),
                warmup_cycles=float(args.warmup_cycles),
                warmup_scale=float(args.warmup_swing_height_scale),
            )
            swing_height_scale_frames[frame] = height_scale
            targets = {}
            phase_actions = np.zeros(4, dtype=np.float32)
            for leg_index, leg in enumerate(POLICY_LEGS):
                rel, grip_action, is_swing, phase = foot_relative_offset(
                    traj_time=traj_time,
                    phase_offset=phase_offsets[leg],
                    gait_freq=float(args.gait_freq),
                    duty=float(args.duty),
                    step_length=float(args.step_length_m),
                    swing_height=float(args.swing_height_m) * height_scale,
                    bar_x=float(args.bar_x_m),
                    bar_window=float(args.bar_window_m),
                    bar_extra_height=float(args.bar_extra_height_m) * height_scale,
                    direction_sign=direction_sign,
                )
                target = anchor_pos[leg] + rel
                targets[leg] = target.astype(np.float32)
                phase_actions[leg_index] = grip_action
                phase_frames[frame, leg_index] = phase
                is_swing_frames[frame, leg_index] = is_swing
                foot_target_frames[frame, leg_index] = target

            solution = solve_frame_ik(
                p,
                robot_id,
                name_to_idx,
                movable_indices,
                movable_index_map,
                ee_indices,
                targets,
                init_foot_orn if args.use_ee_orientation else None,
                previous_policy,
                joint_lower,
                joint_upper,
                gripper_calibration.sim_open_angle_rad,
                max_iterations=int(args.ik_max_iterations),
                residual_threshold=float(args.ik_residual_threshold),
            )
            reset_main_joints(p, robot_id, name_to_idx, solution)
            reset_gripper_joints(p, robot_id, name_to_idx, gripper_calibration.sim_open_angle_rad)
            p.stepSimulation()

            actual = get_foot_positions(p, robot_id, ee_indices)
            gripper_actions = gripper_actions_for_mode(args.gripper_mode, phase_actions)
            command = policy_joint_targets_to_hardware_command(solution, gripper_actions, gripper_calibration)

            joint_policy_frames[frame] = solution
            gripper_policy_frames[frame] = gripper_actions
            hardware_command_frames[frame] = command
            foot_actual_frames[frame] = actual
            ik_error_frames[frame] = np.linalg.norm(actual - foot_target_frames[frame], axis=1)
            previous_policy = solution

            if args.debug_draw and args.gui and frame % max(int(fps // 4), 1) == 0:
                draw_target_points(p, foot_target_frames[frame], (0.2, 0.8, 1.0))
            if args.print_interval > 0 and frame % int(args.print_interval) == 0:
                print(
                    f"[IK] frame={frame}/{frame_count} "
                    f"max_err={float(np.max(ik_error_frames[frame])):.4f} m "
                    f"joint_deg_minmax={np.rad2deg(solution).min():.1f}/{np.rad2deg(solution).max():.1f} "
                    f"grip={np.round(gripper_actions, 2)}"
                )
            if args.gui:
                target_wall = start_wall + (frame + 1) * dt
                sleep_s = target_wall - time.time()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

        arrays = {
            "time_s": (np.arange(frame_count, dtype=np.float32) * dt).astype(np.float32),
            "joint_target_policy_order": joint_policy_frames,
            "gripper_action_policy_order": gripper_policy_frames,
            "hardware_command": hardware_command_frames,
            "foot_target_w": foot_target_frames,
            "foot_actual_w": foot_actual_frames,
            "ik_error_m": ik_error_frames,
            "phase": phase_frames,
            "is_swing": is_swing_frames,
            "swing_height_scale": swing_height_scale_frames,
        }
        arrays = append_repeated_frames(arrays, prepend_frames, append_frames, dt)
        arrays["dof_positions"] = make_dof_positions(
            arrays["joint_target_policy_order"], arrays["gripper_action_policy_order"]
        )

        metadata = {
            "task": "PyBullet-FootPath-IK-Truss-V3",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generator": Path(__file__).name,
            "urdf": str(urdf_path),
            "policy_legs": list(POLICY_LEGS),
            "hardware_legs": list(HARDWARE_LEGS),
            "hardware_order": "joint-major [LB,LF,RB,RF] for Joint1-5, then gripper torque [LB,LF,RB,RF]",
            "sim_to_real_sign": SIM_TO_REAL_SIGN.astype(float).tolist(),
            "loop_hz": fps,
            "dt": dt,
            "duration_s": duration_s,
            "target_progress_m": float(args.target_progress_m),
            "nominal_progress_m": float(nominal_speed_mps * duration_s),
            "nominal_speed_mps": float(nominal_speed_mps),
            "bar_distance_m": float(args.bar_distance_m),
            "prepend_hold_s": float(args.prepend_hold_s),
            "append_hold_s": float(args.append_hold_s),
            "base_position": list(base_position),
            "phase_offsets": phase_offsets,
            "gait_freq": float(args.gait_freq),
            "duty": float(args.duty),
            "step_length_m": float(args.step_length_m),
            "swing_height_m": float(args.swing_height_m),
            "warmup_cycles": float(args.warmup_cycles),
            "warmup_swing_height_scale": float(args.warmup_swing_height_scale),
            "bar_x_m": float(args.bar_x_m),
            "bar_window_m": float(args.bar_window_m),
            "bar_extra_height_m": float(args.bar_extra_height_m),
            "direction": args.direction,
            "gripper_mode": args.gripper_mode,
            "gripper_calibration_path": str(Path(args.gripper_calibration).expanduser()),
            "gripper_torque_scale": float(args.gripper_torque_scale),
            "joint_abs_limit_rad": float(args.joint_abs_limit_rad),
            "joint_limit_soft_factor": float(args.joint_limit_soft_factor),
            "joint_lower": joint_lower.astype(float).tolist(),
            "joint_upper": joint_upper.astype(float).tolist(),
            "init_joint_policy_order": init_policy.astype(float).tolist(),
            "init_joint_policy_order_deg": np.rad2deg(init_policy).astype(float).tolist(),
            "init_foot_pos_w": {leg: init_foot_pos[leg].astype(float).tolist() for leg in POLICY_LEGS},
            "anchor_foot_pos_w": {leg: anchor_pos[leg].astype(float).tolist() for leg in POLICY_LEGS},
            "ik_max_iterations": int(args.ik_max_iterations),
            "ik_residual_threshold": float(args.ik_residual_threshold),
            "use_ee_orientation": bool(args.use_ee_orientation),
            "dof_names": list(SAVE_DOF_NAMES),
        }
        warnings = validate_hardware_commands(
            arrays["hardware_command"],
            metadata,
            max_joint_step_rad=float(args.max_joint_step_rad),
            max_joint_abs_rad=float(args.joint_abs_limit_rad),
            joint_limit_margin_rad=float(args.joint_limit_margin_rad),
        )
        metadata["recorded_steps"] = int(arrays["hardware_command"].shape[0])
        metadata["recorded_duration_s"] = float(arrays["hardware_command"].shape[0] * dt)
        metadata["ik_error_max_m"] = float(np.max(arrays["ik_error_m"]))
        metadata["ik_error_mean_m"] = float(np.mean(arrays["ik_error_m"]))
        metadata["warnings"] = warnings

        np.savez(
            output,
            metadata_json=np.array(json.dumps(metadata, sort_keys=True), dtype=np.str_),
            dt=np.array(dt, dtype=np.float32),
            loop_hz=np.array(fps, dtype=np.float32),
            dof_names=np.array(SAVE_DOF_NAMES, dtype="<U32"),
            **arrays,
        )

        summary = command_summary(arrays["hardware_command"])
        print(f"[INFO] Saved {arrays['hardware_command'].shape[0]} steps to {output}")
        print(f"[INFO] duration={arrays['hardware_command'].shape[0] * dt:.2f}s, fps={fps:.1f}")
        print(f"[INFO] ik_error max={metadata['ik_error_max_m']:.4f}m mean={metadata['ik_error_mean_m']:.4f}m")
        print(
            "[INFO] leg command "
            f"min={summary['leg_min_rad']:.4f} max={summary['leg_max_rad']:.4f} "
            f"abs_max={summary['leg_abs_max_rad']:.4f} max_step={summary['leg_max_step_rad']:.4f}"
        )
        print(f"[INFO] gripper command min={summary['gripper_min']:.3f} max={summary['gripper_max']:.3f}")
        if warnings:
            print("[WARN] trajectory warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("[INFO] trajectory warnings: none")
    finally:
        p.disconnect(client_id)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
