from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import POLICY_LEGS, load_trajectory, parse_float_tuple
from pybullet_footpath_ik_collect import DEFAULT_URDF, build_link_index, build_name_index, get_link_pose
from pybullet_replay_npz import (
    joint_targets_from_arrays,
    reset_main_joints,
    reset_root_pose,
    root_pose_from_arrays,
    slice_range,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure PyBullet FK foot clearance near the crossbar.")
    parser.add_argument("--trajectory", type=str, required=True)
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_URDF))
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--end_step", type=int, default=0, help="0 means use the trajectory end.")
    parser.add_argument("--base_position", type=str, default="", help="Defaults to metadata base_position, then 0,0,1.02.")
    parser.add_argument("--replay_root_pose", action="store_true", default=False)
    parser.add_argument("--root_pose_mode", choices=("absolute", "relative_xy", "relative_xyz"), default="relative_xy")
    parser.add_argument("--bar_x_m", type=float, default=0.5)
    parser.add_argument("--bar_top_z_m", type=float, default=0.5)
    parser.add_argument("--bar_window_m", type=float, default=0.14)
    parser.add_argument("--worst", type=int, default=12)
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


def main() -> None:
    args = build_parser().parse_args()
    p, pybullet_data = import_pybullet()

    trajectory = Path(args.trajectory).expanduser().resolve()
    arrays, metadata = load_trajectory(trajectory)
    joints = joint_targets_from_arrays(arrays)
    start, end = slice_range(joints.shape[0], args.start_step, args.end_step)
    if args.base_position:
        base_position = parse_float_tuple(args.base_position, 3, "--base_position")
    else:
        base_position = tuple(float(x) for x in metadata.get("base_position", (0.0, 0.0, 1.02)))
    root_pos, root_quat = root_pose_from_arrays(arrays, base_position, args.root_pose_mode)
    if args.replay_root_pose and (root_pos is None or root_quat is None):
        raise SystemExit("Trajectory does not contain root_pos_w/root_quat_w; cannot use --replay_root_pose.")
    if not args.replay_root_pose:
        root_pos, root_quat = None, None

    client_id = p.connect(p.DIRECT, options="--opengl2")
    try:
        if pybullet_data is not None:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        urdf = Path(args.urdf).expanduser().resolve()
        if len(urdf.parents) >= 3:
            p.setAdditionalSearchPath(str(urdf.parents[2]))
        p.setAdditionalSearchPath(str(urdf.parent))

        if root_pos is not None and root_quat is not None:
            base_pos = root_pos[start].tolist()
            base_orn = root_quat[start].tolist()
            fixed_base = False
        else:
            base_pos = list(base_position)
            base_orn = [0.0, 0.0, 0.0, 1.0]
            fixed_base = True
        robot_id = p.loadURDF(str(urdf), basePosition=base_pos, baseOrientation=base_orn, useFixedBase=fixed_base)
        name_to_idx = build_name_index(p, robot_id)
        link_to_idx = build_link_index(p, robot_id)

        samples: list[tuple[float, int, str, float, float, float]] = []
        per_leg_min: dict[str, tuple[float, int, float, float, float] | None] = {leg: None for leg in POLICY_LEGS}
        for frame in range(start, end):
            reset_root_pose(p, robot_id, root_pos, root_quat, frame)
            reset_main_joints(p, robot_id, name_to_idx, joints[frame])
            p.stepSimulation()
            for leg in POLICY_LEGS:
                pos, _ = get_link_pose(p, robot_id, link_to_idx[f"{leg}_Link5"])
                dx = float(pos[0]) - float(args.bar_x_m)
                if abs(dx) > float(args.bar_window_m):
                    continue
                clearance = float(pos[2]) - float(args.bar_top_z_m)
                item = (clearance, frame, leg, float(pos[0]), float(pos[2]), dx)
                samples.append(item)
                previous = per_leg_min[leg]
                if previous is None or clearance < previous[0]:
                    per_leg_min[leg] = (clearance, frame, float(pos[0]), float(pos[2]), dx)
    finally:
        p.disconnect(client_id)

    print(f"trajectory: {trajectory}")
    print(f"frames: {start}:{end}")
    print(
        "bar: "
        f"x={float(args.bar_x_m):.3f}m, top_z={float(args.bar_top_z_m):.3f}m, "
        f"window=+/-{float(args.bar_window_m):.3f}m"
    )
    if root_pos is not None:
        print(f"root replay: mode={args.root_pose_mode}, delta={np.round(root_pos[end - 1] - root_pos[start], 4)} m")
    if not samples:
        print("no foot samples inside bar window")
        return

    print("per-leg worst clearance:")
    for leg in POLICY_LEGS:
        value = per_leg_min[leg]
        if value is None:
            print(f"- {leg}: no samples")
        else:
            clearance, frame, x, z, dx = value
            print(f"- {leg}: frame={frame} clearance={clearance:+.4f}m x={x:.4f} z={z:.4f} dx={dx:+.4f}")

    print("overall worst samples:")
    for clearance, frame, leg, x, z, dx in sorted(samples)[: max(int(args.worst), 0)]:
        print(f"- frame={frame} leg={leg} clearance={clearance:+.4f}m x={x:.4f} z={z:.4f} dx={dx:+.4f}")


if __name__ == "__main__":
    main()
