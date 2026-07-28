from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from motor_control import DaMiaoMotorController


LEG_PORTS = (
    ("LB", "/dev/ttyUSB0"),
    ("LF", "/dev/ttyUSB1"),
    ("RB", "/dev/ttyUSB2"),
    ("RF", "/dev/ttyUSB3"),
)
GRIPPER_MOTOR_ID = 6


@dataclass
class LegController:
    leg: str
    ctrl: DaMiaoMotorController


def _open_controllers(ports: tuple[tuple[str, str], ...]) -> list[LegController]:
    controllers: list[LegController] = []
    for leg, port in ports:
        ctrl = DaMiaoMotorController(port)
        controllers.append(LegController(leg=leg, ctrl=ctrl))
    return controllers


def _enable_grippers(controllers: list[LegController]) -> None:
    for item in controllers:
        try:
            item.ctrl.enable_motor(GRIPPER_MOTOR_ID)
        except Exception as exc:
            print(f"[WARN] {item.leg} enable motor {GRIPPER_MOTOR_ID} failed: {exc}")
        time.sleep(0.1)


def _send_open_command(item: LegController, torque: float, receive_feedback: bool) -> None:
    cmd = item.ctrl.damiao_mit_command(
        pos=0.0,
        vel=0.0,
        kp=0.0,
        kd=0.0,
        tor=torque,
        motor_id=GRIPPER_MOTOR_ID,
    )
    item.ctrl.send(cmd)
    if item.ctrl.ser is not None:
        item.ctrl.ser.flush()
    if receive_feedback:
        response = item.ctrl.receive(8)
        feedback = item.ctrl.damiao_feedback(response) if response else None
        print(f"{item.leg} MIT反馈: {feedback}")
    else:
        time.sleep(0.01)


def _close_controllers(controllers: list[LegController]) -> None:
    for item in controllers:
        try:
            item.ctrl.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Open all four grippers once with MIT torque command 6 1.")
    parser.add_argument("--torque", type=float, default=1.0, help="MIT torque for motor 6. Default matches '6 1'.")
    parser.add_argument("--no_feedback", action="store_true", help="Skip receive() after each command.")
    args = parser.parse_args()

    controllers = _open_controllers(LEG_PORTS)
    receive_feedback = not args.no_feedback

    try:
        _enable_grippers(controllers)
        for item in controllers:
            _send_open_command(item, torque=args.torque, receive_feedback=receive_feedback)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted while sending gripper open commands.")
    finally:
        _close_controllers(controllers)


if __name__ == "__main__":
    main()
