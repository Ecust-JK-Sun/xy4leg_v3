from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - runtime fallback for minimal deployments
    yaml = None

from motor_control import DaMiaoMotorController


THIS_DIR = Path(__file__).resolve().parent
LEG_PORTS = {
    "LB": "/dev/ttyUSB0",
    "LF": "/dev/ttyUSB1",
    "RB": "/dev/ttyUSB2",
    "RF": "/dev/ttyUSB3",
}
DEFAULT_LEG_ORDER = ("LB", "LF", "RB", "RF")
MOTOR_IDS = (1, 2, 3, 4, 5)

# Copied from XY4LegTrussV3TeacherEnvCfg.robot.init_state.joint_pos.
INIT_JOINT_POS_DEG = {
    "LB": (65.0, 45.0, -20.0, 24.0, 24.0),
    "LF": (-65.0, -85.0, -20.0, 24.0, 24.0),
    "RB": (-65.0, -85.0, -20.0, 24.0, 24.0),
    "RF": (65.0, 45.0, -20.0, 24.0, 24.0),
}

# Same sim-to-real sign convention as play_real.py. Joint5 is inverted.
SIM_TO_REAL_SIGN = (1.0, 1.0, 1.0, 1.0, -1.0)
DEFAULT_KP = (20.0, 20.0, 20.0, 20.0, 2.0)
DEFAULT_KD = (1.0, 1.0, 1.0, 1.0, 0.5)


@dataclass
class LegController:
    leg: str
    ctrl: DaMiaoMotorController


def _read_motor_cfg(path: str | None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not path or yaml is None:
        return DEFAULT_KP, DEFAULT_KD
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    kp = tuple(float(x) for x in cfg.get("kp_truss", DEFAULT_KP)[:5])
    kd = tuple(float(x) for x in cfg.get("kd", DEFAULT_KD)[:5])
    if len(kp) != 5 or len(kd) != 5:
        raise ValueError(f"Expected at least five kp_truss/kd values in {path}")
    return kp, kd


def _target_rad_for_leg(leg: str, joint5_sign_flip: bool) -> list[float]:
    signs = list(SIM_TO_REAL_SIGN)
    if not joint5_sign_flip:
        signs[4] = 1.0
    return [math.radians(deg) * sign for deg, sign in zip(INIT_JOINT_POS_DEG[leg], signs)]


def _format_deg(values_rad: list[float]) -> str:
    return "[" + ", ".join(f"{math.degrees(x):7.2f}" for x in values_rad) + "] deg"


def _enable_and_read_current(
    ctrl: DaMiaoMotorController,
    motor_id: int,
) -> float | None:
    feedback = None
    try:
        feedback = ctrl.enable_motor(motor_id)
    except Exception as exc:
        print(f"[WARN] enable motor {motor_id} failed: {exc}")
    if feedback and "position" in feedback:
        return float(feedback["position"])

    try:
        response = ctrl.receive(8)
        feedback = ctrl.damiao_feedback(response) if response else None
    except Exception as exc:
        print(f"[WARN] read motor {motor_id} feedback failed: {exc}")
        feedback = None
    if feedback and "position" in feedback:
        return float(feedback["position"])
    return None


def _send_leg_position_targets(
    item: LegController,
    targets_rad: list[float],
    current_rad: list[float],
    kp: tuple[float, ...],
    kd: tuple[float, ...],
    max_step_rad: float,
    hz: float,
    inter_command_sleep: float,
) -> None:
    max_delta = max(abs(t - c) for t, c in zip(targets_rad, current_rad))
    steps = max(1, math.ceil(max_delta / max(max_step_rad, 1.0e-6)))
    period = 1.0 / max(hz, 1.0e-6)
    print(
        f"[INFO] {item.leg}: current={_format_deg(current_rad)} "
        f"target={_format_deg(targets_rad)} steps={steps}"
    )

    for step in range(1, steps + 1):
        loop_start = time.monotonic()
        alpha = step / steps
        command_rad = [
            current + alpha * (target - current)
            for current, target in zip(current_rad, targets_rad)
        ]
        for motor_id, pos, kp_i, kd_i in zip(MOTOR_IDS, command_rad, kp, kd):
            cmd = item.ctrl.damiao_mit_command(
                pos=float(pos),
                vel=0.0,
                kp=float(kp_i),
                kd=float(kd_i),
                tor=0.0,
                motor_id=motor_id,
            )
            item.ctrl.send(cmd)
            time.sleep(inter_command_sleep)
        elapsed = time.monotonic() - loop_start
        time.sleep(max(0.0, period - elapsed))


def _open_leg_controller(leg: str) -> LegController:
    return LegController(leg=leg, ctrl=DaMiaoMotorController(LEG_PORTS[leg]))


def _close_leg_controller(item: LegController) -> None:
    try:
        item.ctrl.close()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move real robot leg motors to XY4LegTrussV3TeacherEnvCfg init joint pose."
    )
    parser.add_argument("--motor_config", type=str, default=str(THIS_DIR / "damiao.yaml"), help="DaMiao gain yaml.")
    parser.add_argument("--leg_order", nargs="+", default=list(DEFAULT_LEG_ORDER), choices=DEFAULT_LEG_ORDER)
    parser.add_argument("--max_step_deg", type=float, default=2.0, help="Largest position change per ramp step.")
    parser.add_argument("--hz", type=float, default=10.0, help="Ramp command frequency for one leg.")
    parser.add_argument("--inter_command_sleep", type=float, default=0.02, help="Delay between motors on one leg.")
    parser.add_argument(
        "--fallback_current_deg",
        type=float,
        default=None,
        help="Use this current angle for every joint if feedback is unavailable.",
    )
    parser.add_argument(
        "--no_joint5_sign_flip",
        action="store_true",
        help="Do not apply the Joint5 sim-to-real negative sign used by play_real.py.",
    )
    args = parser.parse_args()

    kp, kd = _read_motor_cfg(args.motor_config)
    max_step_rad = math.radians(max(args.max_step_deg, 0.1))

    for leg in args.leg_order:
        item = _open_leg_controller(leg)
        try:
            targets_rad = _target_rad_for_leg(leg, joint5_sign_flip=not args.no_joint5_sign_flip)
            current_rad: list[float] = []
            for motor_id in MOTOR_IDS:
                pos = _enable_and_read_current(item.ctrl, motor_id)
                if pos is None:
                    if args.fallback_current_deg is None:
                        raise RuntimeError(
                            f"{leg} motor {motor_id} feedback unavailable. "
                            "Use --fallback_current_deg only after manually confirming the current pose."
                        )
                    pos = math.radians(args.fallback_current_deg)
                current_rad.append(pos)
                time.sleep(0.1)

            _send_leg_position_targets(
                item=item,
                targets_rad=targets_rad,
                current_rad=current_rad,
                kp=kp,
                kd=kd,
                max_step_rad=max_step_rad,
                hz=args.hz,
                inter_command_sleep=args.inter_command_sleep,
            )
        finally:
            _close_leg_controller(item)


if __name__ == "__main__":
    main()
