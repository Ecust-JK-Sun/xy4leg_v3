from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import command_summary, load_trajectory, validate_hardware_commands


parser = argparse.ArgumentParser(description="Inspect a recorded Truss V3 simdata replay trajectory.")
parser.add_argument("--trajectory", type=str, required=True)
parser.add_argument("--max_joint_step_rad", type=float, default=0.04)
parser.add_argument("--max_joint_abs_rad", type=float, default=2.2)
parser.add_argument("--joint_limit_margin_rad", type=float, default=0.05)
parser.add_argument("--fail_on_warning", action="store_true", default=False)
parser.add_argument("--plot", action="store_true", default=False)
parser.add_argument("--plot_output", type=str, default="")


def _first_true_index(values: np.ndarray) -> int | None:
    values = np.asarray(values).astype(bool).reshape(-1)
    hits = np.nonzero(values)[0]
    if hits.size == 0:
        return None
    return int(hits[0])


def _print_array_range(name: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    if values.size == 0:
        return
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        print(f"{name}: no finite values")
        return
    print(f"{name}: shape={values.shape}, min={finite.min():.4f}, max={finite.max():.4f}")


def _count_sign_switches(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        return np.zeros(values.shape[1] if values.ndim == 2 else 1, dtype=np.int64)
    signs = np.sign(values)
    signs[np.abs(values) < 1.0e-6] = 0.0
    return np.count_nonzero(np.diff(signs, axis=0) != 0.0, axis=0)


def _save_plot(path: Path, arrays: dict[str, np.ndarray]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for --plot") from exc

    commands = arrays["hardware_command"]
    time_s = arrays.get("time_s")
    if time_s is None:
        dt = float(np.asarray(arrays.get("dt", 0.04)).reshape(()))
        time_s = np.arange(commands.shape[0], dtype=np.float32) * dt

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(time_s, commands[:, :20])
    axes[0].set_ylabel("leg cmd rad")
    axes[0].grid(True)

    axes[1].plot(time_s, commands[:, 20:24])
    axes[1].set_ylabel("grip cmd")
    axes[1].grid(True)

    if "bar_delta_b" in arrays:
        axes[2].plot(time_s, arrays["bar_delta_b"][:, :3])
        axes[2].set_ylabel("bar delta m")
    elif "bar_estimate" in arrays:
        axes[2].plot(time_s, arrays["bar_estimate"])
        axes[2].set_ylabel("bar estimate")
    axes[2].grid(True)

    if "projected_gravity_b" in arrays:
        axes[3].plot(time_s, arrays["projected_gravity_b"])
        axes[3].set_ylabel("gravity b")
    elif "root_ang_vel_b" in arrays:
        axes[3].plot(time_s, arrays["root_ang_vel_b"])
        axes[3].set_ylabel("ang vel b")
    axes[3].set_xlabel("time s")
    axes[3].grid(True)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"plot: {path}")


def main() -> None:
    args = parser.parse_args()
    arrays, metadata = load_trajectory(args.trajectory)
    if "hardware_command" not in arrays:
        raise SystemExit("Trajectory is missing required key: hardware_command")
    commands = arrays["hardware_command"].astype(np.float32)

    warnings = validate_hardware_commands(
        commands,
        metadata,
        max_joint_step_rad=args.max_joint_step_rad,
        max_joint_abs_rad=args.max_joint_abs_rad,
        joint_limit_margin_rad=args.joint_limit_margin_rad,
    )
    summary = command_summary(commands)
    dt = float(np.asarray(arrays.get("dt", metadata.get("policy_dt", 0.04))).reshape(()))
    duration_s = commands.shape[0] * dt

    print(f"trajectory: {Path(args.trajectory).expanduser()}")
    print(f"steps: {commands.shape[0]}")
    print(f"dt: {dt:.4f} s")
    print(f"duration: {duration_s:.2f} s")
    print(f"task: {metadata.get('task', '<unknown>')}")
    print(f"checkpoint: {metadata.get('checkpoint', '<unknown>')}")
    print(
        "leg command: "
        f"min={summary['leg_min_rad']:.4f} rad, max={summary['leg_max_rad']:.4f} rad, "
        f"abs_max={summary['leg_abs_max_rad']:.4f} rad, "
        f"max_step={summary['leg_max_step_rad']:.4f} rad"
    )
    print(f"gripper command: min={summary['gripper_min']:.4f}, max={summary['gripper_max']:.4f}")

    if "success" in arrays:
        first_success = _first_true_index(arrays["success"])
        print(f"success: last={bool(arrays['success'][-1])}, first_true_step={first_success}")
    if "done" in arrays:
        first_done = _first_true_index(arrays["done"])
        print(f"done: last={bool(arrays['done'][-1])}, first_true_step={first_done}")
    if "gripper_action_policy_order" in arrays:
        switches = _count_sign_switches(arrays["gripper_action_policy_order"])
        print(f"gripper policy sign switches RF/LF/RB/LB: {switches.tolist()}")

    for key in (
        "raw_action",
        "command_action",
        "applied_action",
        "joint_target_policy_order",
        "joint_pos_policy_order",
        "joint_vel_policy_order",
        "bar_estimate",
        "bar_delta_b",
        "foot_clearance",
        "forbidden_contact_force",
    ):
        if key in arrays:
            _print_array_range(key, arrays[key])

    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        if args.fail_on_warning:
            raise SystemExit(2)
    else:
        print("warnings: none")

    if args.plot:
        plot_output = Path(args.plot_output).expanduser() if args.plot_output else Path(args.trajectory).with_suffix(".summary.png")
        _save_plot(plot_output, arrays)


if __name__ == "__main__":
    main()
