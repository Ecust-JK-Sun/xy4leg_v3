from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from common import (
    HARDWARE_LEGS,
    POLICY_LEGS,
    SIM_TO_REAL_SIGN,
    command_summary,
    load_trajectory,
    policy_joint_targets_to_hardware_command,
    read_gripper_calibration,
    validate_hardware_commands,
)
from pybullet_footpath_ik_collect import (
    DEFAULT_GRIPPER_CALIBRATION,
    DEFAULT_URDF,
    SAVE_DOF_NAMES,
    build_link_index,
    build_name_index,
    get_link_pose,
    get_movable_joint_indices,
    get_policy_joint_limits,
    make_dof_positions,
    policy_joint_names,
    reset_gripper_joints,
    reset_main_joints,
    solve_frame_ik,
)


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = THIS_DIR / "trajectories" / "sim_footpath_optimized.npz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a sim-recorded trajectory, edit foot-end paths, solve PyBullet IK, and save real replay npz."
    )
    parser.add_argument("--trajectory", type=str, required=True)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_URDF))
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--end_step", type=int, default=0, help="0 means use trajectory end.")
    parser.add_argument("--fps", type=float, default=0.0, help="0 uses recorded loop_hz/dt.")
    parser.add_argument("--base_position", type=str, default="", help="Defaults to source metadata base_position, then 0,0,1.02.")
    parser.add_argument("--gripper_calibration", type=str, default=str(DEFAULT_GRIPPER_CALIBRATION))
    parser.add_argument("--gripper_torque_scale", type=float, default=1.0)
    parser.add_argument("--gripper_deadband", type=float, default=0.05)
    parser.add_argument("--joint_abs_limit_rad", type=float, default=2.60)
    parser.add_argument("--joint_limit_soft_factor", type=float, default=0.95)
    parser.add_argument("--max_joint_step_rad", type=float, default=0.04)
    parser.add_argument("--joint_limit_margin_rad", type=float, default=0.05)
    parser.add_argument("--ik_max_iterations", type=int, default=500)
    parser.add_argument("--ik_residual_threshold", type=float, default=1.0e-6)
    parser.add_argument("--use_ee_orientation", action="store_true", default=False)
    parser.add_argument("--swing_height_scale", type=float, default=1.0)
    parser.add_argument("--swing_height_offset_m", type=float, default=0.0)
    parser.add_argument("--min_clearance_for_swing_m", type=float, default=0.005)
    parser.add_argument("--only_swing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--x_stride_scale", type=float, default=1.0)
    parser.add_argument("--swing_x_offset_m", type=float, default=0.0)
    parser.add_argument("--bar_x_m", type=float, default=0.5)
    parser.add_argument("--bar_window_m", type=float, default=0.14)
    parser.add_argument("--bar_extra_height_m", type=float, default=0.0)
    parser.add_argument("--bar_top_z_m", type=float, default=0.5)
    parser.add_argument("--bar_min_clearance_m", type=float, default=0.0)
    parser.add_argument("--smooth_targets_window", type=int, default=1, help="Odd moving-average window for edited foot targets.")
    parser.add_argument("--interpolate_output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interpolate_max_step_rad", type=float, default=0.04)
    parser.add_argument("--keep_source_gripper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gripper_from_swing",
        action="store_true",
        default=False,
        help="Ignore source gripper actions and set swing/stance gripper actions from the foot swing mask.",
    )
    parser.add_argument("--swing_gripper_action", type=float, default=1.0, help="Policy-order gripper action during swing.")
    parser.add_argument("--stance_gripper_action", type=float, default=-1.0, help="Policy-order gripper action during stance.")
    parser.add_argument("--invert_gripper_action", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print_interval", type=int, default=50)
    return parser


def import_pybullet():
    try:
        import pybullet as p
    except ImportError as exc:
        raise SystemExit(
            "PyBullet is not installed in the current Python environment.\n"
            "Use the isaaclab env or install it with: pip install pybullet"
        ) from exc
    try:
        import pybullet_data
    except ImportError:
        pybullet_data = None
    return p, pybullet_data


def slice_range(total: int, start_step: int, end_step: int) -> tuple[int, int]:
    start = max(int(start_step), 0)
    end = int(end_step) if int(end_step) > 0 else total
    end = min(end, total)
    if start >= end:
        raise ValueError(f"Invalid slice start={start}, end={end}, total={total}")
    return start, end


def recorded_hz(arrays: dict[str, np.ndarray], metadata: dict) -> float:
    if "loop_hz" in metadata and float(metadata["loop_hz"]) > 0.0:
        return float(metadata["loop_hz"])
    if "loop_hz" in arrays:
        return float(np.asarray(arrays["loop_hz"]).reshape(()))
    if "dt" in arrays:
        return 1.0 / float(np.asarray(arrays["dt"]).reshape(()))
    return 50.0


def base_position_from_metadata(metadata: dict) -> tuple[float, float, float]:
    raw = metadata.get("base_position", (0.0, 0.0, 1.02))
    return tuple(float(x) for x in raw)


def joint_targets_from_source(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if "joint_target_policy_order" in arrays:
        return np.asarray(arrays["joint_target_policy_order"], dtype=np.float32).reshape(-1, 20)
    if "joint_pos_policy_order" in arrays:
        return np.asarray(arrays["joint_pos_policy_order"], dtype=np.float32).reshape(-1, 20)
    raise SystemExit("Source trajectory needs joint_target_policy_order or joint_pos_policy_order.")


def gripper_actions_from_source(arrays: dict[str, np.ndarray], frame_count: int) -> np.ndarray:
    if "gripper_action_policy_order" in arrays:
        return np.asarray(arrays["gripper_action_policy_order"], dtype=np.float32).reshape(-1, 4)
    if "command_action" in arrays and arrays["command_action"].shape[1] >= 24:
        return np.asarray(arrays["command_action"][:, 20:24], dtype=np.float32)
    return -np.ones((frame_count, 4), dtype=np.float32)


def compute_fk_foot_positions(
    p,
    robot_id: int,
    name_to_idx: dict[str, int],
    ee_indices: dict[str, int],
    joints: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    foot_pos = np.zeros((joints.shape[0], 4, 3), dtype=np.float32)
    foot_orn: dict[str, np.ndarray] = {}
    for frame, joint_target in enumerate(joints):
        reset_main_joints(p, robot_id, name_to_idx, joint_target)
        p.stepSimulation()
        for leg_index, leg in enumerate(POLICY_LEGS):
            pos, orn = get_link_pose(p, robot_id, ee_indices[leg])
            foot_pos[frame, leg_index] = pos
            if frame == 0:
                foot_orn[leg] = orn
    return foot_pos, foot_orn


def swing_mask_from_source(arrays: dict[str, np.ndarray], foot_pos: np.ndarray, min_clearance: float) -> np.ndarray:
    if "foot_contacts" in arrays:
        contacts = np.asarray(arrays["foot_contacts"], dtype=np.bool_)
        if contacts.shape[:2] == foot_pos.shape[:2]:
            return ~contacts
    if "foot_clearance" in arrays:
        clearance = np.asarray(arrays["foot_clearance"], dtype=np.float32)
        if clearance.shape[:2] == foot_pos.shape[:2]:
            return clearance > float(min_clearance)
    floor_z = np.min(foot_pos[:, :, 2], axis=0, keepdims=True)
    return foot_pos[:, :, 2] > floor_z + float(min_clearance)


def smooth_targets(targets: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1:
        return targets
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(targets, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    out = np.zeros_like(targets)
    for i in range(targets.shape[0]):
        out[i] = np.mean(padded[i : i + window], axis=0)
    return out.astype(np.float32)


def interpolation_segments(joints: np.ndarray, max_step_rad: float) -> list[int]:
    if joints.shape[0] <= 1 or float(max_step_rad) <= 0.0:
        return [1] * max(joints.shape[0] - 1, 0)
    segments: list[int] = []
    for prev, current in zip(joints[:-1], joints[1:]):
        max_delta = float(np.max(np.abs(current - prev)))
        segments.append(max(int(np.ceil(max_delta / max(float(max_step_rad), 1.0e-6))), 1))
    return segments


def densify_array(values: np.ndarray, segments: list[int]) -> np.ndarray:
    if not segments:
        return values.copy()
    output = [values[0].copy()]
    is_bool = values.dtype == np.bool_
    for prev, current, count in zip(values[:-1], values[1:], segments):
        for index in range(1, count + 1):
            alpha = float(index) / float(count)
            if is_bool:
                output.append((current if alpha >= 0.5 else prev).copy())
            else:
                output.append(((1.0 - alpha) * prev + alpha * current).astype(values.dtype))
    return np.stack(output, axis=0)


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, 1.0e-8)


def edit_foot_targets(
    foot_pos: np.ndarray,
    swing_mask: np.ndarray,
    swing_height_scale: float,
    swing_height_offset: float,
    only_swing: bool,
    x_stride_scale: float,
    swing_x_offset: float,
    root_pos_w: np.ndarray | None,
    bar_x: float,
    bar_window: float,
    bar_extra_height: float,
    bar_top_z: float,
    bar_min_clearance: float,
) -> np.ndarray:
    targets = foot_pos.copy()
    if abs(float(x_stride_scale) - 1.0) > 1.0e-6:
        center_x = np.median(foot_pos[:, :, 0], axis=0, keepdims=True)
        targets[:, :, 0] = center_x + (targets[:, :, 0] - center_x) * float(x_stride_scale)
    if abs(float(swing_x_offset)) > 1.0e-6:
        targets[:, :, 0] = np.where(swing_mask, targets[:, :, 0] + float(swing_x_offset), targets[:, :, 0])

    base_z = np.min(foot_pos[:, :, 2], axis=0, keepdims=True)
    height = foot_pos[:, :, 2] - base_z
    edited_z = base_z + height * float(swing_height_scale) + float(swing_height_offset)
    if only_swing:
        targets[:, :, 2] = np.where(swing_mask, edited_z, foot_pos[:, :, 2])
    else:
        targets[:, :, 2] = edited_z
    if float(bar_window) > 0.0 and (abs(float(bar_extra_height)) > 1.0e-6 or float(bar_min_clearance) > 0.0):
        if root_pos_w is not None and root_pos_w.shape[0] == targets.shape[0]:
            root_x_offset = root_pos_w[:, 0:1] - root_pos_w[0:1, 0:1]
        else:
            root_x_offset = np.zeros((targets.shape[0], 1), dtype=np.float32)
        foot_x_w = targets[:, :, 0] + root_x_offset
        normalized = np.abs(foot_x_w - float(bar_x)) / float(bar_window)
        bump = np.where(normalized < 1.0, 0.5 * (1.0 + np.cos(np.pi * normalized)), 0.0).astype(np.float32)
        if only_swing:
            bump *= swing_mask.astype(np.float32)
        targets[:, :, 2] += float(bar_extra_height) * bump
        if float(bar_min_clearance) > 0.0:
            minimum_z = float(bar_top_z) + float(bar_min_clearance)
            targets[:, :, 2] = np.where(bump > 0.0, np.maximum(targets[:, :, 2], minimum_z), targets[:, :, 2])
    return targets.astype(np.float32)


def solve_edited_trajectory(
    p,
    robot_id: int,
    name_to_idx: dict[str, int],
    ee_indices: dict[str, int],
    target_foot_pos: np.ndarray,
    source_joints: np.ndarray,
    source_orn: dict[str, np.ndarray] | None,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    gripper_rest_angle: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    movable_indices = get_movable_joint_indices(p, robot_id)
    movable_index_map = {joint_index: i for i, joint_index in enumerate(movable_indices)}
    solved = np.zeros_like(source_joints, dtype=np.float32)
    actual = np.zeros_like(target_foot_pos, dtype=np.float32)
    error = np.zeros((source_joints.shape[0], 4), dtype=np.float32)
    previous = source_joints[0].copy()
    for frame in range(source_joints.shape[0]):
        targets = {leg: target_foot_pos[frame, leg_index] for leg_index, leg in enumerate(POLICY_LEGS)}
        solution = solve_frame_ik(
            p,
            robot_id,
            name_to_idx,
            movable_indices,
            movable_index_map,
            ee_indices,
            targets,
            source_orn if args.use_ee_orientation else None,
            previous,
            joint_lower,
            joint_upper,
            gripper_rest_angle,
            max_iterations=int(args.ik_max_iterations),
            residual_threshold=float(args.ik_residual_threshold),
        )
        reset_main_joints(p, robot_id, name_to_idx, solution)
        reset_gripper_joints(p, robot_id, name_to_idx, gripper_rest_angle)
        p.stepSimulation()
        for leg_index, leg in enumerate(POLICY_LEGS):
            pos, _ = get_link_pose(p, robot_id, ee_indices[leg])
            actual[frame, leg_index] = pos
        error[frame] = np.linalg.norm(actual[frame] - target_foot_pos[frame], axis=1)
        solved[frame] = solution
        previous = solution
        if args.print_interval > 0 and frame % int(args.print_interval) == 0:
            print(
                f"[OPT-IK] frame={frame}/{source_joints.shape[0]} "
                f"max_err={float(np.max(error[frame])):.4f}m "
                f"joint_deg_minmax={np.rad2deg(solution).min():.1f}/{np.rad2deg(solution).max():.1f}"
            )
    return solved, actual, error


def main() -> None:
    args = build_parser().parse_args()
    p, pybullet_data = import_pybullet()
    trajectory = Path(args.trajectory).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays, source_metadata = load_trajectory(trajectory)

    source_joints_all = joint_targets_from_source(arrays)
    start, end = slice_range(source_joints_all.shape[0], args.start_step, args.end_step)
    source_joints = source_joints_all[start:end].astype(np.float32)
    gripper_actions = gripper_actions_from_source(arrays, source_joints_all.shape[0])[start:end].astype(np.float32)
    root_pos_w = None
    root_quat_w = None
    if "root_pos_w" in arrays and arrays["root_pos_w"].shape[:1] == (source_joints_all.shape[0],):
        root_pos_w = np.asarray(arrays["root_pos_w"][start:end], dtype=np.float32)
    if "root_quat_w" in arrays and arrays["root_quat_w"].shape[:1] == (source_joints_all.shape[0],):
        root_quat_w = normalize_quat_wxyz(np.asarray(arrays["root_quat_w"][start:end], dtype=np.float32))

    loop_hz = float(args.fps) if float(args.fps) > 0.0 else recorded_hz(arrays, source_metadata)
    dt = 1.0 / loop_hz
    if args.base_position:
        base_position = tuple(float(x.strip()) for x in args.base_position.split(","))
    else:
        base_position = base_position_from_metadata(source_metadata)

    urdf_path = Path(args.urdf).expanduser().resolve()
    gripper_calibration = read_gripper_calibration(
        args.gripper_calibration,
        global_torque_scale=float(args.gripper_torque_scale),
        fallback_deadband=float(args.gripper_deadband),
    )

    client_id = p.connect(p.DIRECT, options="--opengl2")
    try:
        if pybullet_data is not None:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if len(urdf_path.parents) >= 3:
            p.setAdditionalSearchPath(str(urdf_path.parents[2]))
        p.setAdditionalSearchPath(str(urdf_path.parent))
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(dt)
        robot_id = p.loadURDF(str(urdf_path), basePosition=list(base_position), useFixedBase=True)
        name_to_idx = build_name_index(p, robot_id)
        link_to_idx = build_link_index(p, robot_id)
        missing = [name for name in policy_joint_names() if name not in name_to_idx]
        if missing:
            raise ValueError(f"URDF missing main joints: {missing}")
        ee_indices = {leg: link_to_idx[f"{leg}_Link5"] for leg in POLICY_LEGS}
        joint_lower, joint_upper = get_policy_joint_limits(
            p,
            robot_id,
            name_to_idx,
            joint_abs_limit_rad=float(args.joint_abs_limit_rad),
            soft_factor=float(args.joint_limit_soft_factor),
        )

        source_foot_pos, source_foot_orn = compute_fk_foot_positions(p, robot_id, name_to_idx, ee_indices, source_joints)
        swing_mask = swing_mask_from_source(
            {key: value[start:end] if value.shape[:1] == (source_joints_all.shape[0],) else value for key, value in arrays.items()},
            source_foot_pos,
            args.min_clearance_for_swing_m,
        )
        if args.gripper_from_swing:
            gripper_actions = np.where(
                swing_mask,
                float(args.swing_gripper_action),
                float(args.stance_gripper_action),
            ).astype(np.float32)
        elif not args.keep_source_gripper:
            gripper_actions[:] = -1.0
        if args.invert_gripper_action:
            gripper_actions *= -1.0
        edited_foot_pos = edit_foot_targets(
            source_foot_pos,
            swing_mask,
            swing_height_scale=float(args.swing_height_scale),
            swing_height_offset=float(args.swing_height_offset_m),
            only_swing=bool(args.only_swing),
            x_stride_scale=float(args.x_stride_scale),
            swing_x_offset=float(args.swing_x_offset_m),
            root_pos_w=root_pos_w,
            bar_x=float(args.bar_x_m),
            bar_window=float(args.bar_window_m),
            bar_extra_height=float(args.bar_extra_height_m),
            bar_top_z=float(args.bar_top_z_m),
            bar_min_clearance=float(args.bar_min_clearance_m),
        )
        edited_foot_pos = smooth_targets(edited_foot_pos, args.smooth_targets_window)

        solved_joints, actual_foot_pos, ik_error = solve_edited_trajectory(
            p,
            robot_id,
            name_to_idx,
            ee_indices,
            edited_foot_pos,
            source_joints,
            source_foot_orn,
            joint_lower,
            joint_upper,
            gripper_calibration.sim_open_angle_rad,
            args,
        )
    finally:
        p.disconnect(client_id)

    interpolation_info = {
        "enabled": bool(args.interpolate_output),
        "input_steps": int(solved_joints.shape[0]),
        "output_steps": int(solved_joints.shape[0]),
        "max_inserted_frames": 0,
    }
    if args.interpolate_output:
        segments = interpolation_segments(solved_joints, args.interpolate_max_step_rad)
        interpolation_info["max_inserted_frames"] = int(max((segment - 1 for segment in segments), default=0))
        if interpolation_info["max_inserted_frames"] > 0:
            solved_joints = densify_array(solved_joints, segments).astype(np.float32)
            gripper_actions = densify_array(gripper_actions, segments).astype(np.float32)
            source_foot_pos = densify_array(source_foot_pos, segments).astype(np.float32)
            edited_foot_pos = densify_array(edited_foot_pos, segments).astype(np.float32)
            actual_foot_pos = densify_array(actual_foot_pos, segments).astype(np.float32)
            ik_error = densify_array(ik_error, segments).astype(np.float32)
            swing_mask = densify_array(swing_mask.astype(np.bool_), segments).astype(np.bool_)
            if root_pos_w is not None:
                root_pos_w = densify_array(root_pos_w, segments).astype(np.float32)
            if root_quat_w is not None:
                root_quat_w = normalize_quat_wxyz(densify_array(root_quat_w, segments).astype(np.float32))
            interpolation_info["output_steps"] = int(solved_joints.shape[0])

    hardware_commands = np.zeros((solved_joints.shape[0], 24), dtype=np.float32)
    for frame in range(solved_joints.shape[0]):
        hardware_commands[frame] = policy_joint_targets_to_hardware_command(
            solved_joints[frame], gripper_actions[frame], gripper_calibration
        )

    dof_positions = make_dof_positions(solved_joints, gripper_actions)
    time_s = (np.arange(solved_joints.shape[0], dtype=np.float32) * dt).astype(np.float32)
    metadata = {
        "task": "PyBullet-Optimized-Sim-FootPath-Truss-V3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": Path(__file__).name,
        "source_trajectory": str(trajectory),
        "source_start_step": int(start),
        "source_end_step": int(end),
        "urdf": str(urdf_path),
        "policy_legs": list(POLICY_LEGS),
        "hardware_legs": list(HARDWARE_LEGS),
        "sim_to_real_sign": SIM_TO_REAL_SIGN.astype(float).tolist(),
        "loop_hz": float(loop_hz),
        "dt": float(dt),
        "base_position": list(base_position),
        "joint_lower": joint_lower.astype(float).tolist(),
        "joint_upper": joint_upper.astype(float).tolist(),
        "joint_abs_limit_rad": float(args.joint_abs_limit_rad),
        "swing_height_scale": float(args.swing_height_scale),
        "swing_height_offset_m": float(args.swing_height_offset_m),
        "only_swing": bool(args.only_swing),
        "x_stride_scale": float(args.x_stride_scale),
        "swing_x_offset_m": float(args.swing_x_offset_m),
        "bar_x_m": float(args.bar_x_m),
        "bar_window_m": float(args.bar_window_m),
        "bar_extra_height_m": float(args.bar_extra_height_m),
        "bar_top_z_m": float(args.bar_top_z_m),
        "bar_min_clearance_m": float(args.bar_min_clearance_m),
        "min_clearance_for_swing_m": float(args.min_clearance_for_swing_m),
        "smooth_targets_window": int(args.smooth_targets_window),
        "interpolate_output": bool(args.interpolate_output),
        "interpolate_max_step_rad": float(args.interpolate_max_step_rad),
        "interpolation_info": interpolation_info,
        "keep_source_gripper": bool(args.keep_source_gripper),
        "gripper_from_swing": bool(args.gripper_from_swing),
        "swing_gripper_action": float(args.swing_gripper_action),
        "stance_gripper_action": float(args.stance_gripper_action),
        "invert_gripper_action": bool(args.invert_gripper_action),
        "use_ee_orientation": bool(args.use_ee_orientation),
        "ik_error_max_m": float(np.max(ik_error)),
        "ik_error_mean_m": float(np.mean(ik_error)),
    }
    warnings = validate_hardware_commands(
        hardware_commands,
        metadata,
        max_joint_step_rad=float(args.max_joint_step_rad),
        max_joint_abs_rad=float(args.joint_abs_limit_rad),
        joint_limit_margin_rad=float(args.joint_limit_margin_rad),
    )
    metadata["recorded_steps"] = int(hardware_commands.shape[0])
    metadata["recorded_duration_s"] = float(hardware_commands.shape[0] * dt)
    metadata["warnings"] = warnings

    save_arrays = {
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        "dt": np.array(dt, dtype=np.float32),
        "loop_hz": np.array(loop_hz, dtype=np.float32),
        "dof_names": np.array(SAVE_DOF_NAMES, dtype="<U32"),
        "time_s": time_s,
        "joint_target_policy_order": solved_joints.astype(np.float32),
        "gripper_action_policy_order": gripper_actions.astype(np.float32),
        "hardware_command": hardware_commands.astype(np.float32),
        "source_foot_pos_w": source_foot_pos.astype(np.float32),
        "foot_target_w": edited_foot_pos.astype(np.float32),
        "foot_actual_w": actual_foot_pos.astype(np.float32),
        "ik_error_m": ik_error.astype(np.float32),
        "is_swing": swing_mask.astype(np.bool_),
        "dof_positions": dof_positions.astype(np.float32),
    }
    if root_pos_w is not None:
        save_arrays["root_pos_w"] = root_pos_w.astype(np.float32)
    if root_quat_w is not None:
        save_arrays["root_quat_w"] = root_quat_w.astype(np.float32)
    np.savez(output, **save_arrays)

    summary = command_summary(hardware_commands)
    print(f"[INFO] Saved {hardware_commands.shape[0]} steps to {output}")
    print(f"[INFO] duration={hardware_commands.shape[0] * dt:.2f}s, loop_hz={loop_hz:.1f}")
    if interpolation_info["output_steps"] != interpolation_info["input_steps"]:
        print(
            "[INFO] interpolation: "
            f"{interpolation_info['input_steps']} -> {interpolation_info['output_steps']} steps, "
            f"max_inserted_frames={interpolation_info['max_inserted_frames']}"
        )
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


if __name__ == "__main__":
    main()
