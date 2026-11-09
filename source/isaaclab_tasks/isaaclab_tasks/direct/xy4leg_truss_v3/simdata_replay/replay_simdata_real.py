from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from common import (
    SIM2REAL_DIR,
    command_summary,
    load_trajectory,
    read_gripper_calibration,
    recorded_rate_hz,
    resolve_hardware_commands,
    slice_commands,
    validate_hardware_commands,
)


DEFAULT_MOTOR_CONFIG = str(SIM2REAL_DIR / "damiao.yaml")
DEFAULT_GRIPPER_CALIBRATION = str(SIM2REAL_DIR / "gripper_calibration.yaml")


parser = argparse.ArgumentParser(description="Replay a recorded Truss V3 sim hardware_command trajectory.")
parser.add_argument("--trajectory", type=str, required=True)
parser.add_argument("--enable_motors", action="store_true", default=False)
parser.add_argument("--allow_warnings", action="store_true", default=False)
parser.add_argument("--loop_hz", type=float, default=0.0, help="0 uses the recorded loop_hz/dt.")
parser.add_argument("--speed_scale", type=float, default=0.5, help="0.5 plays twice as slow as recorded.")
parser.add_argument("--start_step", type=int, default=0)
parser.add_argument("--end_step", type=int, default=0, help="0 means use the trajectory end.")
parser.add_argument(
    "--interpolate",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Insert intermediate frames so large joint-target jumps become smaller steps.",
)
parser.add_argument(
    "--interpolate_max_step_rad",
    type=float,
    default=0.04,
    help="Maximum leg-joint delta per inserted replay frame when --interpolate is enabled.",
)
parser.add_argument("--startup_hold_s", type=float, default=0.5)
parser.add_argument("--soft_start_s", type=float, default=0.0)
parser.add_argument(
    "--use_feedback_soft_start",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use motor feedback as the start of the soft-start ramp. Disabled by default because zero feedback is unsafe.",
)
parser.add_argument(
    "--feedback_min_abs_rad",
    type=float,
    default=0.05,
    help="Minimum max(abs(joint feedback)) required before feedback is trusted for soft start.",
)
parser.add_argument("--hold_final_s", type=float, default=1.0)
parser.add_argument("--zero_gripper_on_exit", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--motor_config", type=str, default=DEFAULT_MOTOR_CONFIG)
parser.add_argument("--gripper_calibration", type=str, default=DEFAULT_GRIPPER_CALIBRATION)
parser.add_argument("--gripper_torque_scale", type=float, default=0.5)
parser.add_argument("--gripper_deadband", type=float, default=0.05)
parser.add_argument(
    "--invert_gripper_command",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Flip the final hardware gripper torque sign if real gripper open/close is reversed.",
)
parser.add_argument(
    "--gripper_override",
    type=str,
    choices=("recorded", "open", "close", "zero"),
    default="recorded",
    help="Override the 4 gripper torques for direction/strength tests.",
)
parser.add_argument("--max_joint_step_rad", type=float, default=0.04)
parser.add_argument("--max_joint_abs_rad", type=float, default=2.2)
parser.add_argument("--joint_limit_margin_rad", type=float, default=0.05)
parser.add_argument("--print_interval", type=int, default=10)
parser.add_argument("--print_preview_steps", type=int, default=5)
parser.add_argument("--feedback_wait_s", type=float, default=0.5)
parser.add_argument("--feedback_timeout_s", type=float, default=0.5)
parser.add_argument("--compensation", choices=("none", "nominal", "fitted", "fitted_j1"), default="fitted_j1",
                    help="Leg feedforward model, shared with init_default_joint_pose and play_real.")
parser.add_argument("--gravity_config", type=str, default=str(SIM2REAL_DIR / "gravity_compensation.yaml"))
parser.add_argument("--gravity_urdf", type=str, default=None)
parser.add_argument("--gravity_node_mass_file", type=str, default=None)
parser.add_argument("--j1_residual_file", type=str, default=None)
parser.add_argument("--gravity_scale", type=float, default=1.)
parser.add_argument("--j1_residual_scale", type=float, default=1.)
parser.add_argument("--j1_residual_limit_nm", type=float, default=1.5)


def _make_replay_feedforward(args, effective_hz):
    sys.path.insert(0, str(SIM2REAL_DIR))
    from gravity_compensation import GravityFeedforward
    from initialization_models import make_compensation_model

    if not np.isfinite(args.feedback_timeout_s) or args.feedback_timeout_s <= 0:
        raise ValueError("--feedback_timeout_s must be finite and positive")
    if (not np.isfinite(args.feedback_wait_s) or args.feedback_wait_s < 0
            or not np.isfinite(effective_hz) or effective_hz <= 0):
        raise ValueError("Feedback wait and replay frequency must be finite and valid")
    if not np.isfinite(args.gravity_scale) or not 0 <= args.gravity_scale <= 1:
        raise ValueError("--gravity_scale must be in [0,1]")
    if (not np.isfinite(args.j1_residual_scale) or not 0 <= args.j1_residual_scale <= 1
            or not np.isfinite(args.j1_residual_limit_nm) or not 0 < args.j1_residual_limit_nm <= 28):
        raise ValueError("Invalid J1 residual scale or limit")
    if args.gravity_node_mass_file and args.compensation not in ("fitted", "fitted_j1"):
        raise ValueError("Fitted masses require fitted or fitted_j1 compensation")
    model = make_compensation_model(
        args.compensation, args.gravity_config, args.gravity_urdf, args.gravity_node_mass_file,
        args.j1_residual_file, args.j1_residual_scale, args.j1_residual_limit_nm)
    return None if model is None else GravityFeedforward(model, args.gravity_scale, 1. / effective_hz)


def _publish_replay_command(shared, command, controller, feedback_timeout_s):
    """Pair the replay target with bounded feedforward from fresh measured joints."""
    shared.raise_if_failed()
    observed_at = time.monotonic()
    observation = shared.get_obs(max_age_s=feedback_timeout_s)
    torque = None if controller is None else controller.update(
        observation[:20], (0., 0., -9.81), now=observed_at)
    shared.set_action(command, observed_at=observed_at, feedforward=torque)


def _read_motor_cfg(path: str | Path) -> tuple[list[float], list[float]]:
    with open(Path(path).expanduser(), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    kp_list = cfg.get("kp_truss")
    kd_list = cfg.get("kd")
    if not kp_list or not kd_list:
        raise ValueError(f"Expected kp_truss and kd in {path}")
    return list(kp_list), list(kd_list)


def _interpolate_commands(commands: np.ndarray, max_step_rad: float) -> tuple[np.ndarray, dict[str, float]]:
    commands = np.asarray(commands, dtype=np.float32)
    if commands.shape[0] <= 1 or max_step_rad <= 0.0:
        return commands.copy(), {
            "input_steps": float(commands.shape[0]),
            "output_steps": float(commands.shape[0]),
            "max_inserted_frames": 0.0,
        }

    output = [commands[0].copy()]
    max_inserted_frames = 0
    for prev, current in zip(commands[:-1], commands[1:]):
        max_delta = float(np.max(np.abs(current[:20] - prev[:20])))
        segments = max(int(np.ceil(max_delta / max(float(max_step_rad), 1.0e-6))), 1)
        max_inserted_frames = max(max_inserted_frames, segments - 1)
        for index in range(1, segments + 1):
            alpha = float(index) / float(segments)
            output.append(((1.0 - alpha) * prev + alpha * current).astype(np.float32))

    interpolated = np.stack(output, axis=0)
    return interpolated, {
        "input_steps": float(commands.shape[0]),
        "output_steps": float(interpolated.shape[0]),
        "max_inserted_frames": float(max_inserted_frames),
    }


def _gripper_override_command(mode: str, torque_scale: float, gripper_calibration) -> np.ndarray | None:
    if mode == "recorded":
        return None
    if mode == "zero":
        return np.zeros(4, dtype=np.float32)
    scale = abs(float(torque_scale))
    max_torque = np.abs(gripper_calibration.max_torque).astype(np.float32)
    magnitude = np.minimum(max_torque, scale).astype(np.float32)
    # Sim convention: +1 opens, -1 closes. Hardware command sign follows
    # gripper_calibration.torque_sign, already in hardware leg order.
    sim_action = 1.0 if mode == "open" else -1.0
    return (sim_action * gripper_calibration.torque_sign * magnitude).astype(np.float32)


def _prepare_commands(args) -> tuple[np.ndarray, dict, float, list[str], dict[str, float]]:
    arrays, metadata = load_trajectory(args.trajectory)

    gripper_calibration = read_gripper_calibration(
        args.gripper_calibration,
        global_torque_scale=1.0,
        fallback_deadband=args.gripper_deadband,
    )
    try:
        raw_commands, source = resolve_hardware_commands(arrays, gripper_calibration)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if source != "hardware_command":
        metadata = {**metadata, "replay_source": source}
    commands = slice_commands(raw_commands, args.start_step, args.end_step)
    commands = commands.astype(np.float32)

    override = _gripper_override_command(args.gripper_override, args.gripper_torque_scale, gripper_calibration)
    if override is None:
        commands[:, 20:24] *= float(args.gripper_torque_scale)
        max_torque = np.abs(gripper_calibration.max_torque).astype(np.float32)
        commands[:, 20:24] = np.clip(commands[:, 20:24], -max_torque[None, :], max_torque[None, :])
    else:
        commands[:, 20:24] = override[None, :]
    if args.invert_gripper_command:
        commands[:, 20:24] *= -1.0

    interpolation_info = {
        "input_steps": float(commands.shape[0]),
        "output_steps": float(commands.shape[0]),
        "max_inserted_frames": 0.0,
    }
    if args.interpolate:
        commands, interpolation_info = _interpolate_commands(commands, args.interpolate_max_step_rad)

    warnings = validate_hardware_commands(
        commands,
        metadata,
        max_joint_step_rad=args.max_joint_step_rad,
        max_joint_abs_rad=args.max_joint_abs_rad,
        joint_limit_margin_rad=args.joint_limit_margin_rad,
    )
    recorded_hz = recorded_rate_hz(arrays, metadata)
    base_hz = float(args.loop_hz) if args.loop_hz > 0.0 else recorded_hz
    effective_hz = base_hz * float(args.speed_scale)
    if effective_hz <= 0.0:
        raise SystemExit("--speed_scale and --loop_hz must produce a positive replay rate")
    return commands, metadata, effective_hz, warnings, interpolation_info


def _print_summary(
    commands: np.ndarray,
    metadata: dict,
    effective_hz: float,
    warnings: list[str],
    interpolation_info: dict[str, float],
    args,
) -> None:
    summary = command_summary(commands)
    print(f"trajectory: {Path(args.trajectory).expanduser()}")
    print(f"task: {metadata.get('task', '<unknown>')}")
    print(f"checkpoint: {metadata.get('checkpoint', '<unknown>')}")
    print(f"command_source: {metadata.get('replay_source', 'hardware_command')}")
    print(f"steps: {commands.shape[0]}")
    print(f"replay_hz: {effective_hz:.3f}")
    print(f"duration: {commands.shape[0] / effective_hz:.2f} s")
    if interpolation_info["output_steps"] != interpolation_info["input_steps"]:
        print(
            "interpolation: "
            f"{int(interpolation_info['input_steps'])} -> {int(interpolation_info['output_steps'])} steps, "
            f"max_inserted_frames={int(interpolation_info['max_inserted_frames'])}"
        )
    if args.gripper_override != "recorded":
        print(f"gripper_override: {args.gripper_override}")
    if args.invert_gripper_command:
        print("gripper_command: inverted")
    print(
        "leg command: "
        f"min={summary['leg_min_rad']:.4f} rad, max={summary['leg_max_rad']:.4f} rad, "
        f"abs_max={summary['leg_abs_max_rad']:.4f} rad, "
        f"max_step={summary['leg_max_step_rad']:.4f} rad"
    )
    print(f"gripper command: min={summary['gripper_min']:.4f}, max={summary['gripper_max']:.4f}")
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("warnings: none")
    preview_count = min(max(int(args.print_preview_steps), 0), commands.shape[0])
    for i in range(preview_count):
        print(
            f"preview[{i}] leg_deg={np.round(np.rad2deg(commands[i, :20]), 2)} "
            f"grip={np.round(commands[i, 20:24], 3)}"
        )


def _sleep_until(target_time: float) -> None:
    remaining = target_time - time.time()
    if remaining > 0.0:
        time.sleep(remaining)


def _ramp(publish, start_command: np.ndarray, end_command: np.ndarray, duration_s: float, hz: float) -> None:
    steps = max(int(round(max(duration_s, 0.0) * hz)), 1)
    period = 1.0 / hz
    for i in range(steps):
        alpha = float(i + 1) / float(steps)
        command = (1.0 - alpha) * start_command + alpha * end_command
        publish(command.astype(np.float32))
        _sleep_until(time.time() + period)


def _feedback_looks_valid(obs_np: np.ndarray | None, min_abs_rad: float) -> bool:
    if obs_np is None or obs_np.shape[0] < 20:
        return False
    leg_pos = np.asarray(obs_np[:20], dtype=np.float32)
    if not np.isfinite(leg_pos).all():
        return False
    return float(np.max(np.abs(leg_pos))) >= float(min_abs_rad)


def _run_hardware_replay(commands: np.ndarray, effective_hz: float, args, feedforward_controller) -> None:
    sys.path.insert(0, str(SIM2REAL_DIR))
    import torch
    from motor_session_gripper import create_motor_session
    from serial_threads import start_motor_obs_thread, start_serial_tx_thread
    from threads_shared import SharedData

    stop_requested = False

    def _signal_handler(sig, frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"\n[INFO] Received signal {sig}; stopping after current cycle.")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    kp_list, kd_list = _read_motor_cfg(args.motor_config)
    motor_enabled, motor_session = create_motor_session(kp_list, kd_list)
    if not motor_enabled or motor_session is None:
        raise RuntimeError("Motor controllers failed to initialize")

    shared = SharedData()
    stop_event = threading.Event()
    obs_thread = None
    tx_thread = None
    last_command = commands[0].copy()
    def publish(command):
        _publish_replay_command(shared, command, feedforward_controller, args.feedback_timeout_s)
    try:
        # Enable replies have one reader; normal RX starts only afterwards.
        motor_session.enable_all()
        obs_thread = start_motor_obs_thread(
            shared,
            motor_session,
            torch.device("cpu"),
            1,
            loop_hz=200.,
            stop_event=stop_event,
        )
        ready_deadline = time.monotonic() + max(args.feedback_wait_s, args.feedback_timeout_s)
        while True:
            shared.raise_if_failed()
            try:
                obs_np = shared.get_obs(max_age_s=args.feedback_timeout_s)
                break
            except RuntimeError:
                if stop_requested or time.monotonic() >= ready_deadline:
                    raise
                time.sleep(.005)
        start_command = commands[0].copy()
        if args.use_feedback_soft_start and _feedback_looks_valid(obs_np, args.feedback_min_abs_rad):
            start_command[:20] = obs_np[:20].astype(np.float32)
            start_command[20:24] = 0.0
            print("[INFO] Soft-starting from latest motor feedback.")
        elif args.use_feedback_soft_start:
            print("[WARN] Feedback soft start requested, but feedback looked invalid/zero. Using first trajectory command.")
        else:
            print("[INFO] Feedback soft start disabled; using first trajectory command as startup pose.")

        publish(start_command)
        feedback_ready = threading.Event()
        feedback_ready.set()
        tx_thread = start_serial_tx_thread(
            shared,
            True,
            motor_session,
            loop_hz=effective_hz,
            stop_event=stop_event,
            feedback_ready_event=feedback_ready,
            feedback_timeout_s=args.feedback_timeout_s,
        )

        if args.soft_start_s > 0.0 and not np.allclose(start_command, commands[0], atol=1.0e-6):
            print(f"[INFO] Soft start for {args.soft_start_s:.2f}s")
            _ramp(publish, start_command, commands[0], args.soft_start_s, effective_hz)
        elif args.soft_start_s > 0.0:
            print("[INFO] Skipping no-op soft start because startup pose already equals first trajectory command.")

        print(f"[INFO] Holding first command for {args.startup_hold_s:.2f}s")
        hold_end = time.time() + max(float(args.startup_hold_s), 0.0)
        while time.time() < hold_end and not stop_requested:
            publish(commands[0])
            _sleep_until(time.time() + 1.0 / effective_hz)

        print("[INFO] Starting trajectory replay")
        period = 1.0 / effective_hz
        next_tick = time.time()
        for step, command in enumerate(commands):
            if stop_requested:
                break
            publish(command)
            last_command = command.copy()
            if args.print_interval > 0 and step % args.print_interval == 0:
                print(
                    f"[REPLAY] step={step}/{commands.shape[0]} "
                    f"leg_deg_minmax={np.rad2deg(command[:20]).min():.1f}/"
                    f"{np.rad2deg(command[:20]).max():.1f} "
                    f"grip={np.round(command[20:24], 3)}"
                )
            next_tick += period
            _sleep_until(next_tick)

        if args.hold_final_s > 0.0 and not stop_requested:
            print(f"[INFO] Holding final command for {args.hold_final_s:.2f}s")
            hold_end = time.time() + float(args.hold_final_s)
            while time.time() < hold_end and not stop_requested:
                publish(last_command)
                _sleep_until(time.time() + period)

        if args.zero_gripper_on_exit:
            safe_command = last_command.copy()
            safe_command[20:24] = 0.0
            publish(safe_command)
            time.sleep(0.2)
            shared.raise_if_failed()
            print("[INFO] Zeroed gripper torque on exit.")
    finally:
        stop_event.set()
        if tx_thread is not None:
            tx_thread.join(timeout=1.0)
        if obs_thread is not None:
            obs_thread.join(timeout=1.0)
        motor_session.close()


def main() -> None:
    args = parser.parse_args()
    commands, metadata, effective_hz, warnings, interpolation_info = _prepare_commands(args)
    _print_summary(commands, metadata, effective_hz, warnings, interpolation_info, args)
    feedforward = _make_replay_feedforward(args, effective_hz)
    print(f"[INFO] compensation={args.compensation}; gravity=assumed level base; finger angle=-30 deg")
    if feedforward is not None:
        import json
        print("[INFO] Feedforward model:", json.dumps(feedforward.model.description(), ensure_ascii=False))

    if warnings and args.enable_motors and not args.allow_warnings:
        raise SystemExit("Refusing to enable motors while trajectory warnings exist. Add --allow_warnings to override.")

    if not args.enable_motors:
        print("[DRY-RUN] Motors disabled. Add --enable_motors to transmit commands.")
        return

    _run_hardware_replay(commands, effective_hz, args, feedforward)


if __name__ == "__main__":
    main()
