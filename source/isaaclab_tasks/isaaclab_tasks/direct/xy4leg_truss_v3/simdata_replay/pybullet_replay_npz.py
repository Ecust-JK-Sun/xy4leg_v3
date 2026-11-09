from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import POLICY_LEGS, load_trajectory, parse_float_tuple, read_gripper_calibration
from pybullet_footpath_ik_collect import (
    DEFAULT_GRIPPER_CALIBRATION,
    DEFAULT_URDF,
    build_link_index,
    build_name_index,
    get_link_pose,
    policy_joint_names,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview a simdata replay npz trajectory in PyBullet.")
    parser.add_argument("--trajectory", type=str, required=True)
    parser.add_argument("--urdf", type=str, default="", help="Defaults to metadata urdf, then xy4legv5_gripper URDF.")
    parser.add_argument("--direct", action="store_true", default=False)
    parser.add_argument("--base_position", type=str, default="", help="Defaults to metadata base_position, then 0,0,1.02.")
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--end_step", type=int, default=0, help="0 means use the trajectory end.")
    parser.add_argument("--speed_scale", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true", default=False)
    parser.add_argument("--hold_final_s", type=float, default=3.0)
    parser.add_argument("--mode", choices=("reset", "position"), default="reset")
    parser.add_argument("--motor_force", type=float, default=80.0)
    parser.add_argument(
        "--replay_root_pose",
        action="store_true",
        default=False,
        help="Replay root_pos_w/root_quat_w from sim data so the robot body moves in PyBullet.",
    )
    parser.add_argument(
        "--root_pose_mode",
        choices=("absolute", "relative_xy", "relative_xyz"),
        default="absolute",
        help="absolute uses recorded world root pose; relative modes apply recorded displacement to --base_position.",
    )
    parser.add_argument("--follow_root_camera", action="store_true", default=False)
    parser.add_argument("--draw_root_trail", action="store_true", default=False)
    parser.add_argument("--gripper_calibration", type=str, default=str(DEFAULT_GRIPPER_CALIBRATION))
    parser.add_argument("--gripper_visual_mode", choices=("action", "open", "closed"), default="action")
    parser.add_argument("--show_targets", action="store_true", default=False)
    parser.add_argument("--show_fk_trails", action="store_true", default=False)
    parser.add_argument("--trail_stride", type=int, default=5)
    parser.add_argument("--draw_crossbar", action="store_true", default=False)
    parser.add_argument("--crossbar_x_m", type=float, default=0.5)
    parser.add_argument("--crossbar_y_half_m", type=float, default=0.27)
    parser.add_argument("--crossbar_z_m", type=float, default=0.48)
    parser.add_argument("--crossbar_radius_m", type=float, default=0.012)
    parser.add_argument("--crossbar_line_width", type=float, default=8.0)
    parser.add_argument("--screenshot_output", type=str, default="")
    parser.add_argument("--screenshot_step", type=int, default=0)
    parser.add_argument("--screenshot_width", type=int, default=1280)
    parser.add_argument("--screenshot_height", type=int, default=900)
    parser.add_argument("--load_plane", action="store_true", default=False)
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


def joint_targets_from_arrays(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if "joint_target_policy_order" in arrays:
        return np.asarray(arrays["joint_target_policy_order"], dtype=np.float32).reshape(-1, 20)
    if "dof_positions" in arrays:
        dof = np.asarray(arrays["dof_positions"], dtype=np.float32)
        if dof.ndim != 2 or dof.shape[1] != 24:
            raise ValueError(f"Expected dof_positions shape [T,24], got {dof.shape}")
        joints = np.zeros((dof.shape[0], 20), dtype=np.float32)
        for leg_index in range(4):
            joints[:, leg_index * 5 : leg_index * 5 + 5] = dof[:, leg_index * 6 : leg_index * 6 + 5]
        return joints
    raise SystemExit("Trajectory needs joint_target_policy_order or dof_positions for PyBullet preview.")


def gripper_actions_from_arrays(arrays: dict[str, np.ndarray], frame_count: int) -> np.ndarray:
    if "gripper_action_policy_order" in arrays:
        return np.asarray(arrays["gripper_action_policy_order"], dtype=np.float32).reshape(-1, 4)
    if "dof_positions" in arrays:
        dof = np.asarray(arrays["dof_positions"], dtype=np.float32)
        if dof.ndim == 2 and dof.shape[1] == 24:
            actions = np.zeros((dof.shape[0], 4), dtype=np.float32)
            for leg_index in range(4):
                actions[:, leg_index] = dof[:, leg_index * 6 + 5]
            return actions
    return -np.ones((frame_count, 4), dtype=np.float32)


def quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)


def root_pose_from_arrays(
    arrays: dict[str, np.ndarray],
    base_position: tuple[float, float, float],
    mode: str,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if "root_pos_w" not in arrays or "root_quat_w" not in arrays:
        return None, None
    root_pos = np.asarray(arrays["root_pos_w"], dtype=np.float32).reshape(-1, 3).copy()
    root_quat_wxyz = np.asarray(arrays["root_quat_w"], dtype=np.float32).reshape(-1, 4)
    root_quat = np.stack([quat_wxyz_to_xyzw(quat) for quat in root_quat_wxyz], axis=0).astype(np.float32)
    if mode == "absolute":
        return root_pos, root_quat

    offset = root_pos - root_pos[:1]
    base = np.array(base_position, dtype=np.float32).reshape(1, 3)
    if mode == "relative_xy":
        root_pos = np.repeat(base, offset.shape[0], axis=0)
        root_pos[:, :2] += offset[:, :2]
    elif mode == "relative_xyz":
        root_pos = base + offset
    return root_pos.astype(np.float32), root_quat


def reset_root_pose(p, robot_id: int, root_pos: np.ndarray | None, root_quat: np.ndarray | None, frame: int) -> None:
    if root_pos is None or root_quat is None:
        return
    p.resetBasePositionAndOrientation(robot_id, root_pos[frame].tolist(), root_quat[frame].tolist())
    p.resetBaseVelocity(robot_id, linearVelocity=[0.0, 0.0, 0.0], angularVelocity=[0.0, 0.0, 0.0])


def reset_main_joints(p, robot_id: int, name_to_idx: dict[str, int], joint_targets: np.ndarray) -> None:
    for leg_index, leg in enumerate(POLICY_LEGS):
        for joint_number in range(1, 6):
            name = f"{leg}_Joint{joint_number}"
            value = float(joint_targets[leg_index * 5 + joint_number - 1])
            p.resetJointState(robot_id, name_to_idx[name], targetValue=value, targetVelocity=0.0)


def command_main_joints(
    p,
    robot_id: int,
    name_to_idx: dict[str, int],
    joint_targets: np.ndarray,
    force: float,
) -> None:
    joint_indices = []
    targets = []
    for leg_index, leg in enumerate(POLICY_LEGS):
        for joint_number in range(1, 6):
            name = f"{leg}_Joint{joint_number}"
            joint_indices.append(name_to_idx[name])
            targets.append(float(joint_targets[leg_index * 5 + joint_number - 1]))
    p.setJointMotorControlArray(
        robot_id,
        joint_indices,
        p.POSITION_CONTROL,
        targetPositions=targets,
        forces=[float(force)] * len(targets),
    )


def set_gripper_visual(
    p,
    robot_id: int,
    name_to_idx: dict[str, int],
    gripper_actions: np.ndarray,
    open_angle: float,
    closed_angle: float,
    mode: str,
) -> None:
    for leg_index, leg in enumerate(POLICY_LEGS):
        if mode == "open":
            angle = open_angle
        elif mode == "closed":
            angle = closed_angle
        else:
            angle = open_angle if float(gripper_actions[leg_index]) > 0.0 else closed_angle
        for suffix in ("1out", "2out"):
            name = f"{leg}_{suffix}_Joint"
            if name in name_to_idx:
                p.resetJointState(robot_id, name_to_idx[name], targetValue=float(angle), targetVelocity=0.0)


def draw_target_trails(p, foot_targets: np.ndarray, stride: int) -> None:
    colors = {
        "RF": (1.0, 0.25, 0.25),
        "LF": (0.25, 0.8, 1.0),
        "RB": (1.0, 0.8, 0.2),
        "LB": (0.4, 1.0, 0.4),
    }
    stride = max(int(stride), 1)
    for leg_index, leg in enumerate(POLICY_LEGS):
        color = colors[leg]
        points = foot_targets[::stride, leg_index, :]
        for i in range(max(points.shape[0] - 1, 0)):
            p.addUserDebugLine(points[i].tolist(), points[i + 1].tolist(), color, lineWidth=2.0, lifeTime=0.0)


def compute_fk_foot_trails(
    p,
    robot_id: int,
    name_to_idx: dict[str, int],
    link_to_idx: dict[str, int],
    joints: np.ndarray,
    root_pos: np.ndarray | None,
    root_quat: np.ndarray | None,
    start: int,
    end: int,
    stride: int,
) -> np.ndarray:
    ee_indices = {leg: link_to_idx[f"{leg}_Link5"] for leg in POLICY_LEGS}
    frames = range(start, end, max(int(stride), 1))
    trails = []
    for frame in frames:
        reset_root_pose(p, robot_id, root_pos, root_quat, frame)
        reset_main_joints(p, robot_id, name_to_idx, joints[frame])
        p.stepSimulation()
        foot_positions = []
        for leg in POLICY_LEGS:
            pos, _ = get_link_pose(p, robot_id, ee_indices[leg])
            foot_positions.append(pos)
        trails.append(np.stack(foot_positions, axis=0))
    if not trails:
        return np.zeros((0, 4, 3), dtype=np.float32)
    return np.stack(trails, axis=0).astype(np.float32)


def print_preview_info(
    trajectory: Path,
    joints: np.ndarray,
    grippers: np.ndarray,
    start: int,
    end: int,
    hz: float,
) -> None:
    sliced = joints[start:end]
    max_step = float(np.max(np.abs(np.diff(sliced, axis=0)))) if sliced.shape[0] > 1 else 0.0
    print(f"trajectory: {trajectory}")
    print(f"steps: {end - start} ({start}:{end})")
    print(f"preview_hz: {hz:.3f}")
    print(f"duration: {(end - start) / hz:.2f} s")
    print(
        "joint target: "
        f"min={float(np.min(sliced)):.4f} rad, max={float(np.max(sliced)):.4f} rad, "
        f"max_step={max_step:.4f} rad"
    )
    print(f"gripper action: min={float(np.min(grippers[start:end])):.3f}, max={float(np.max(grippers[start:end])):.3f}")


def draw_root_trail(p, root_pos: np.ndarray, start: int, end: int, stride: int) -> None:
    points = root_pos[start:end:max(int(stride), 1)]
    for i in range(max(points.shape[0] - 1, 0)):
        p.addUserDebugLine(points[i].tolist(), points[i + 1].tolist(), [1.0, 1.0, 1.0], lineWidth=3.0, lifeTime=0.0)


def draw_crossbar(p, args: argparse.Namespace) -> int:
    start = [float(args.crossbar_x_m), -float(args.crossbar_y_half_m), float(args.crossbar_z_m)]
    end = [float(args.crossbar_x_m), float(args.crossbar_y_half_m), float(args.crossbar_z_m)]
    p.addUserDebugLine(start, end, [0.0, 0.0, 0.0], lineWidth=float(args.crossbar_line_width), lifeTime=0.0)
    length = 2.0 * float(args.crossbar_y_half_m)
    visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=float(args.crossbar_radius_m),
        length=length,
        rgbaColor=[0.0, 0.0, 0.0, 1.0],
    )
    collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=float(args.crossbar_radius_m),
        height=length,
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[float(args.crossbar_x_m), 0.0, float(args.crossbar_z_m)],
        baseOrientation=p.getQuaternionFromEuler([1.57079632679, 0.0, 0.0]),
    )


def save_screenshot(p, args: argparse.Namespace, camera_target: np.ndarray | list[float]) -> None:
    if not args.screenshot_output:
        return
    from PIL import Image

    target = np.asarray(camera_target, dtype=np.float32).reshape(3)
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target.tolist(),
        distance=1.45,
        yaw=45.0,
        pitch=-28.0,
        roll=0.0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=55.0,
        aspect=float(args.screenshot_width) / float(args.screenshot_height),
        nearVal=0.01,
        farVal=10.0,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width=int(args.screenshot_width),
        height=int(args.screenshot_height),
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
    )
    image = np.reshape(np.asarray(rgba, dtype=np.uint8), (int(args.screenshot_height), int(args.screenshot_width), 4))
    output = Path(args.screenshot_output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image[:, :, :3]).save(output)
    print(f"screenshot: {output}")


def main() -> None:
    args = build_parser().parse_args()
    p, pybullet_data = import_pybullet()

    trajectory = Path(args.trajectory).expanduser().resolve()
    arrays, metadata = load_trajectory(trajectory)
    joints = joint_targets_from_arrays(arrays)
    grippers = gripper_actions_from_arrays(arrays, joints.shape[0])
    if joints.shape[0] != grippers.shape[0]:
        raise ValueError(f"Joint/gripper frame mismatch: {joints.shape[0]} vs {grippers.shape[0]}")
    start, end = slice_range(joints.shape[0], args.start_step, args.end_step)

    base_hz = float(metadata.get("loop_hz", 0.0) or 0.0)
    if base_hz <= 0.0 and "loop_hz" in arrays:
        base_hz = float(np.asarray(arrays["loop_hz"]).reshape(()))
    if base_hz <= 0.0 and "dt" in arrays:
        base_hz = 1.0 / float(np.asarray(arrays["dt"]).reshape(()))
    if base_hz <= 0.0:
        base_hz = 50.0
    hz = base_hz * float(args.speed_scale)
    if hz <= 0.0:
        raise ValueError("--speed_scale must produce a positive preview rate")

    urdf = Path(args.urdf or metadata.get("urdf", "") or DEFAULT_URDF).expanduser().resolve()
    if not urdf.exists():
        raise FileNotFoundError(f"URDF not found: {urdf}")
    if args.base_position:
        base_position = parse_float_tuple(args.base_position, 3, "--base_position")
    else:
        base_position = tuple(float(x) for x in metadata.get("base_position", (0.0, 0.0, 1.02)))
    root_pos, root_quat = root_pose_from_arrays(arrays, base_position, args.root_pose_mode)
    if args.replay_root_pose and (root_pos is None or root_quat is None):
        raise SystemExit("Trajectory does not contain root_pos_w/root_quat_w; cannot use --replay_root_pose.")
    if not args.replay_root_pose:
        root_pos, root_quat = None, None

    gripper_calibration = read_gripper_calibration(args.gripper_calibration)
    print_preview_info(trajectory, joints, grippers, start, end, hz)
    if root_pos is not None:
        progress = root_pos[end - 1] - root_pos[start]
        print(f"root replay: mode={args.root_pose_mode}, delta={np.round(progress, 4)} m")

    client_id = p.connect(p.DIRECT if args.direct else p.GUI, options="--opengl2")
    try:
        if pybullet_data is not None:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if len(urdf.parents) >= 3:
            p.setAdditionalSearchPath(str(urdf.parents[2]))
        p.setAdditionalSearchPath(str(urdf.parent))
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(1.0 / max(base_hz, 1.0))
        if args.load_plane:
            p.loadURDF("plane.urdf")
        if args.draw_crossbar:
            draw_crossbar(p, args)
        if root_pos is not None and root_quat is not None:
            load_base_position = root_pos[start].tolist()
            load_base_orientation = root_quat[start].tolist()
        else:
            load_base_position = list(base_position)
            load_base_orientation = [0.0, 0.0, 0.0, 1.0]
        robot_id = p.loadURDF(
            str(urdf),
            basePosition=load_base_position,
            baseOrientation=load_base_orientation,
            useFixedBase=root_pos is None,
        )
        name_to_idx = build_name_index(p, robot_id)
        link_to_idx = build_link_index(p, robot_id)
        missing = [name for name in policy_joint_names() if name not in name_to_idx]
        if missing:
            raise ValueError(f"URDF missing main joints: {missing}")
        missing_links = [f"{leg}_Link5" for leg in POLICY_LEGS if f"{leg}_Link5" not in link_to_idx]
        if missing_links:
            raise ValueError(f"URDF missing foot links: {missing_links}")

        camera_target = root_pos[start].tolist() if root_pos is not None else list(base_position)
        if not args.direct:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.4,
                cameraYaw=45.0,
                cameraPitch=-28.0,
                cameraTargetPosition=camera_target,
            )
        if not args.direct or args.screenshot_output:
            if args.draw_root_trail and root_pos is not None:
                draw_root_trail(p, root_pos, start, end, args.trail_stride)
            if args.show_targets and "foot_target_w" in arrays:
                draw_target_trails(p, np.asarray(arrays["foot_target_w"])[start:end], args.trail_stride)
            if args.show_fk_trails:
                fk_trails = compute_fk_foot_trails(
                    p, robot_id, name_to_idx, link_to_idx, joints, root_pos, root_quat, start, end, args.trail_stride
                )
                draw_target_trails(p, fk_trails, 1)
            reset_root_pose(p, robot_id, root_pos, root_quat, start)

        period = 1.0 / hz
        screenshot_frame = min(max(int(args.screenshot_step), start), end - 1) if args.screenshot_output else -1
        screenshot_saved = False
        try:
            while True:
                next_tick = time.time()
                for frame in range(start, end):
                    reset_root_pose(p, robot_id, root_pos, root_quat, frame)
                    if args.mode == "position":
                        command_main_joints(p, robot_id, name_to_idx, joints[frame], args.motor_force)
                    else:
                        reset_main_joints(p, robot_id, name_to_idx, joints[frame])
                    set_gripper_visual(
                        p,
                        robot_id,
                        name_to_idx,
                        grippers[frame],
                        gripper_calibration.sim_open_angle_rad,
                        gripper_calibration.sim_closed_angle_rad,
                        args.gripper_visual_mode,
                    )
                    p.stepSimulation()
                    if args.screenshot_output and frame == screenshot_frame and not screenshot_saved:
                        camera_target = root_pos[frame] if root_pos is not None else np.asarray(base_position, dtype=np.float32)
                        save_screenshot(p, args, camera_target)
                        screenshot_saved = True
                        if args.direct:
                            break
                    if not args.direct:
                        if args.follow_root_camera and root_pos is not None:
                            p.resetDebugVisualizerCamera(
                                cameraDistance=1.4,
                                cameraYaw=45.0,
                                cameraPitch=-28.0,
                                cameraTargetPosition=root_pos[frame].tolist(),
                            )
                        next_tick += period
                        sleep_s = next_tick - time.time()
                        if sleep_s > 0.0:
                            time.sleep(sleep_s)
                if args.direct and screenshot_saved:
                    break
                if not args.loop:
                    break
            if not args.direct and args.hold_final_s > 0.0:
                time.sleep(float(args.hold_final_s))
        except KeyboardInterrupt:
            print("[INFO] Preview stopped.")
    finally:
        p.disconnect(client_id)


if __name__ == "__main__":
    main()
