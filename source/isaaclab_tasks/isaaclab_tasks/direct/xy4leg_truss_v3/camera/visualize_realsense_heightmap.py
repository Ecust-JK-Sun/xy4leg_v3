from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2

DEFAULT_CAPTURE_DIR = Path(__file__).resolve().parent / "realsense_heightmap_captures"


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def _quat_multiply_xyzw(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _quat_from_axis_angle_xyzw(axis: str, angle_deg: float) -> tuple[float, float, float, float]:
    angle = math.radians(angle_deg)
    half_sin = math.sin(0.5 * angle)
    half_cos = math.cos(0.5 * angle)
    if axis == "x":
        return (half_sin, 0.0, 0.0, half_cos)
    if axis == "y":
        return (0.0, half_sin, 0.0, half_cos)
    if axis == "z":
        return (0.0, 0.0, half_sin, half_cos)
    raise ValueError(f"Unsupported axis: {axis}")


def _quat_from_euler_wxyz(seq: str, angles_deg: tuple[float, ...]) -> tuple[float, float, float, float]:
    q_xyzw = (0.0, 0.0, 0.0, 1.0)
    for axis, angle_deg in zip(seq, angles_deg):
        q_xyzw = _quat_multiply_xyzw(q_xyzw, _quat_from_axis_angle_xyzw(axis, angle_deg))
    x, y, z, w = q_xyzw
    return (w, x, y, z)


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1.0e-8)
    w, x, y, z = q
    return np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float32,
    )


def _validate_range(name: str, value: list[float]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    lower = float(value[0])
    upper = float(value[1])
    if not lower < upper:
        raise ValueError(f"{name} must be increasing, got {value}")
    return lower, upper


def _depth_to_heightmap(
    depth_raw: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_scale: float,
    rot_base_cam: np.ndarray,
    pos_base_cam: np.ndarray,
    grid_size: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    min_depth_m: float,
    max_depth_m: float,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float] | None]:
    depth_m = depth_raw.astype(np.float32) * float(depth_scale)
    depth_m = depth_m[::stride, ::stride]
    height, width = depth_m.shape
    u_values = np.arange(0, intrinsics.width, stride, dtype=np.float32)[:width]
    v_values = np.arange(0, intrinsics.height, stride, dtype=np.float32)[:height]
    u, v = np.meshgrid(u_values, v_values)

    valid_depth = np.isfinite(depth_m) & (depth_m > min_depth_m) & (depth_m < max_depth_m)
    z_cam = np.nan_to_num(depth_m, nan=max_depth_m, posinf=max_depth_m, neginf=0.0)
    x_cam = (u - float(intrinsics.cx)) * z_cam / max(float(intrinsics.fx), 1.0e-6)
    y_cam = (v - float(intrinsics.cy)) * z_cam / max(float(intrinsics.fy), 1.0e-6)
    points_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).reshape(-1, 3)
    valid = valid_depth.reshape(-1)

    points_b = points_cam @ rot_base_cam.T + pos_base_cam.reshape(1, 3)
    x_b = points_b[:, 0]
    y_b = points_b[:, 1]
    z_b_raw = points_b[:, 2]
    z_min, z_max = z_range
    z_b = np.clip(z_b_raw, z_min, z_max)

    in_grid = (
        valid
        & (x_b >= x_range[0])
        & (x_b <= x_range[1])
        & (y_b >= y_range[0])
        & (y_b <= y_range[1])
        & (z_b_raw >= z_min)
        & (z_b_raw <= z_max)
    )

    grid_cells = grid_size * grid_size
    height_flat = np.full((grid_cells,), z_min, dtype=np.float32)
    valid_flat = np.zeros((grid_cells,), dtype=bool)

    if np.any(in_grid):
        cols = ((x_b[in_grid] - x_range[0]) / max(x_range[1] - x_range[0], 1.0e-6) * grid_size).astype(np.int64)
        rows = ((y_b[in_grid] - y_range[0]) / max(y_range[1] - y_range[0], 1.0e-6) * grid_size).astype(np.int64)
        cols = np.clip(cols, 0, grid_size - 1)
        rows = np.clip(rows, 0, grid_size - 1)
        flat_index = rows * grid_size + cols
        np.maximum.at(height_flat, flat_index, z_b[in_grid])
        valid_flat[flat_index] = True

    height_m = height_flat.reshape(grid_size, grid_size)
    valid_map = valid_flat.reshape(grid_size, grid_size)
    height_norm = (height_m - z_min) / max(z_max - z_min, 1.0e-6)
    height_norm = np.where(valid_map, np.clip(height_norm, 0.0, 1.0), 0.0).astype(np.float32)
    heightmap = np.stack((height_norm, valid_map.astype(np.float32)), axis=0)

    valid_bounds = None
    valid_indices = np.argwhere(valid_map)
    if valid_indices.size > 0:
        rows = valid_indices[:, 0].astype(np.float32)
        cols = valid_indices[:, 1].astype(np.float32)
        denom = float(max(grid_size - 1, 1))
        x_values = x_range[0] + cols / denom * (x_range[1] - x_range[0])
        y_values = y_range[0] + rows / denom * (y_range[1] - y_range[0])
        valid_bounds = (float(x_values.min()), float(x_values.max()), float(y_values.min()), float(y_values.max()))

    return heightmap, depth_m, valid_bounds


def _depth_to_color(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m < max_depth_m)
    normalized = 1.0 - np.clip(depth_m / max(max_depth_m, 1.0e-6), 0.0, 1.0)
    image_u8 = (normalized * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(image_u8, cv2.COLORMAP_TURBO)
    color[~valid] = (0, 0, 0)
    return color


def _heightmap_to_color(heightmap: np.ndarray, scale: int) -> np.ndarray:
    height = heightmap[0]
    valid = heightmap[1] > 0.5
    image_u8 = (np.clip(height, 0.0, 1.0) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(image_u8, cv2.COLORMAP_TURBO)
    color[~valid] = (0, 0, 0)
    color = cv2.resize(color, (height.shape[1] * scale, height.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    cv2.arrowedLine(color, (12, color.shape[0] - 14), (64, color.shape[0] - 14), (255, 255, 255), 1)
    cv2.putText(color, "+x", (68, color.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.arrowedLine(color, (12, color.shape[0] - 14), (12, color.shape[0] - 52), (255, 255, 255), 1)
    cv2.putText(color, "-y", (16, color.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return color


def _put_status(image: np.ndarray, lines: list[str]) -> np.ndarray:
    output = image.copy()
    y = 22
    for line in lines:
        cv2.putText(output, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return output


def _save_frame(
    save_dir: Path,
    save_index: int,
    frame_index: int,
    color_bgr: np.ndarray,
    depth_raw: np.ndarray,
    depth_vis: np.ndarray,
    heightmap: np.ndarray,
    heightmap_vis: np.ndarray,
    metadata: dict,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    metadata = dict(metadata)
    metadata["saved_at"] = timestamp
    prefix = save_dir / f"{timestamp}_save_{save_index:04d}_frame_{frame_index:06d}"
    cv2.imwrite(str(prefix.with_suffix(".color.png")), color_bgr)
    cv2.imwrite(str(prefix.with_suffix(".depth_vis.png")), depth_vis)
    cv2.imwrite(str(prefix.with_suffix(".heightmap.png")), heightmap_vis)
    np.savez_compressed(
        str(prefix.with_suffix(".npz")),
        depth_raw=depth_raw,
        heightmap=heightmap,
        metadata=json.dumps(metadata),
    )
    return prefix


def _parse_args() -> argparse.Namespace:
    default_front_rot = _quat_from_euler_wxyz("yyz", (180.0, -30.0, -90.0))
    parser = argparse.ArgumentParser(
        description="Visualize RealSense color/depth and Truss V3 base-frame heightmap.",
    )
    parser.add_argument("--list_devices", action="store_true", help="List RealSense devices and exit.")
    parser.add_argument("--serial", type=str, default=None, help="RealSense serial number.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hardware_reset", action="store_true", help="Reset connected RealSense devices before streaming.")
    parser.add_argument("--stride", type=int, default=2, help="Use every Nth depth pixel for heightmap rasterization.")
    parser.add_argument("--median_ksize", type=int, default=0, help="Optional raw depth median blur kernel; use 0 to disable.")
    parser.add_argument("--grid_size", type=int, default=64)
    parser.add_argument(
        "--x_range",
        type=float,
        nargs=2,
        default=(-0.10, 0.75),
        help="Heightmap x range in base_link meters. Use -0.85 1.45 to match the current two-camera sim cfg.",
    )
    parser.add_argument("--y_range", type=float, nargs=2, default=(-0.45, 0.45))
    parser.add_argument("--z_range", type=float, nargs=2, default=(-0.85, 0.25))
    parser.add_argument("--min_depth_m", type=float, default=0.10)
    parser.add_argument("--max_depth_m", type=float, default=3.0)
    parser.add_argument("--depth_vis_max_m", type=float, default=2.0)
    parser.add_argument(
        "--camera_pos_m",
        type=float,
        nargs=3,
        default=(-0.06, 0.0, -0.16),
        help="RealSense optical-frame origin in base_link frame.",
    )
    parser.add_argument(
        "--camera_rot_wxyz",
        type=float,
        nargs=4,
        default=default_front_rot,
        help="Quaternion rotating RealSense optical-frame points into base_link frame.",
    )
    parser.add_argument("--depth_scale", type=float, default=0.0, help="Override RealSense depth scale; 0 uses device value.")
    parser.add_argument("--heightmap_scale", type=int, default=6)
    parser.add_argument("--print_interval", type=int, default=30)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(DEFAULT_CAPTURE_DIR),
        help="Root directory for saved frames. Press 's' to save into a timestamped run directory.",
    )
    return parser.parse_args()


def main() -> None:
    global cv2, np

    args = _parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.grid_size <= 0:
        raise ValueError("--grid_size must be positive")
    if args.median_ksize < 0 or (args.median_ksize > 0 and args.median_ksize % 2 == 0):
        raise ValueError("--median_ksize must be 0 or a positive odd integer")

    x_range = _validate_range("x_range", list(args.x_range))
    y_range = _validate_range("y_range", list(args.y_range))
    z_range = _validate_range("z_range", list(args.z_range))
    if not args.min_depth_m < args.max_depth_m:
        raise ValueError("--min_depth_m must be smaller than --max_depth_m")

    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    if args.list_devices:
        for device in devices:
            print(device.get_info(rs.camera_info.serial_number), device.get_info(rs.camera_info.name))
        return
    if len(devices) == 0:
        raise RuntimeError("No RealSense devices found")
    if args.hardware_reset:
        for device in devices:
            device.hardware_reset()
        time.sleep(2.0)

    try:
        import numpy as np_module
    except ImportError as exc:
        raise RuntimeError("NumPy is required for RealSense heightmap preprocessing.") from exc
    np = np_module

    try:
        import cv2 as cv2_module
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for visualization. Install python package 'opencv-python'.") from exc
    cv2 = cv2_module

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial is not None:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(args.depth_scale) if args.depth_scale > 0.0 else float(depth_sensor.get_depth_scale())
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    depth_intrinsics = depth_profile.get_intrinsics()
    intrinsics = CameraIntrinsics(
        width=int(depth_intrinsics.width),
        height=int(depth_intrinsics.height),
        fx=float(depth_intrinsics.fx),
        fy=float(depth_intrinsics.fy),
        cx=float(depth_intrinsics.ppx),
        cy=float(depth_intrinsics.ppy),
    )
    rot_base_cam = _quat_wxyz_to_rotmat(np.asarray(args.camera_rot_wxyz, dtype=np.float32))
    pos_base_cam = np.asarray(args.camera_pos_m, dtype=np.float32)
    save_root = Path(args.save_dir).expanduser().resolve() if args.save_dir else None
    save_dir = save_root / datetime.now().strftime("%Y%m%d_%H%M%S") if save_root is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Streaming RealSense")
    print(f"[INFO] depth_scale={depth_scale:.8f}")
    print(
        "[INFO] depth intrinsics: "
        f"width={intrinsics.width}, height={intrinsics.height}, "
        f"fx={intrinsics.fx:.3f}, fy={intrinsics.fy:.3f}, cx={intrinsics.cx:.3f}, cy={intrinsics.cy:.3f}"
    )
    print(f"[INFO] camera_pos_m={tuple(float(v) for v in pos_base_cam)}")
    print(f"[INFO] camera_rot_wxyz={tuple(float(v) for v in args.camera_rot_wxyz)}")
    print(f"[INFO] heightmap ranges: x={x_range}, y={y_range}, z={z_range}, grid={args.grid_size}")
    if save_dir is not None:
        print(f"[INFO] Press 'q' or Esc to quit; press 's' to save a frame to {save_dir}")
    else:
        print("[INFO] Press 'q' or Esc to quit; saving is disabled because --save_dir is empty.")

    cv2.namedWindow("color", cv2.WINDOW_NORMAL)
    cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
    cv2.namedWindow("heightmap", cv2.WINDOW_NORMAL)

    frame_index = 0
    saved_index = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            if args.median_ksize > 0:
                depth_raw = cv2.medianBlur(depth_raw, int(args.median_ksize))

            heightmap, depth_m_sampled, valid_bounds = _depth_to_heightmap(
                depth_raw=depth_raw,
                intrinsics=intrinsics,
                depth_scale=depth_scale,
                rot_base_cam=rot_base_cam,
                pos_base_cam=pos_base_cam,
                grid_size=int(args.grid_size),
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                min_depth_m=float(args.min_depth_m),
                max_depth_m=float(args.max_depth_m),
                stride=int(args.stride),
            )

            depth_vis = _depth_to_color(depth_raw.astype(np.float32) * depth_scale, float(args.depth_vis_max_m))
            heightmap_vis = _heightmap_to_color(heightmap, int(args.heightmap_scale))
            valid_rate = float(np.mean(heightmap[1] > 0.5))
            if valid_bounds is None:
                bounds_text = "valid bounds: none"
            else:
                bounds_text = (
                    f"x[{valid_bounds[0]:.2f},{valid_bounds[1]:.2f}] "
                    f"y[{valid_bounds[2]:.2f},{valid_bounds[3]:.2f}]"
                )
            heightmap_vis = _put_status(
                heightmap_vis,
                [
                    f"valid={valid_rate:.3f}",
                    bounds_text,
                    f"x={x_range[0]:.2f}..{x_range[1]:.2f} y={y_range[0]:.2f}..{y_range[1]:.2f}",
                    f"z={z_range[0]:.2f}..{z_range[1]:.2f}",
                ],
            )

            if args.print_interval > 0 and frame_index % int(args.print_interval) == 0:
                depth_valid = depth_m_sampled[
                    np.isfinite(depth_m_sampled)
                    & (depth_m_sampled > float(args.min_depth_m))
                    & (depth_m_sampled < float(args.max_depth_m))
                ]
                if depth_valid.size > 0:
                    depth_stats = (
                        f"depth_m min/mean/max="
                        f"{float(depth_valid.min()):.3f}/{float(depth_valid.mean()):.3f}/{float(depth_valid.max()):.3f}"
                    )
                else:
                    depth_stats = "depth_m no valid pixels"
                print(f"[FRAME {frame_index:06d}] {depth_stats}, heightmap_valid={valid_rate:.4f}, {bounds_text}")

            cv2.imshow("color", color_bgr)
            cv2.imshow("depth", depth_vis)
            cv2.imshow("heightmap", heightmap_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s") and save_dir is not None:
                metadata = {
                    "frame_index": frame_index,
                    "depth_scale": depth_scale,
                    "intrinsics": intrinsics.__dict__,
                    "camera_pos_m": [float(v) for v in pos_base_cam],
                    "camera_rot_wxyz": [float(v) for v in args.camera_rot_wxyz],
                    "x_range": list(x_range),
                    "y_range": list(y_range),
                    "z_range": list(z_range),
                    "grid_size": int(args.grid_size),
                    "stride": int(args.stride),
                    "valid_rate": valid_rate,
                    "valid_bounds": valid_bounds,
                }
                prefix = _save_frame(
                    save_dir,
                    saved_index,
                    frame_index,
                    color_bgr,
                    depth_raw,
                    depth_vis,
                    heightmap,
                    heightmap_vis,
                    metadata,
                )
                print(f"[INFO] Saved frame {saved_index:06d} to {prefix.parent}")
                saved_index += 1

            frame_index += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
