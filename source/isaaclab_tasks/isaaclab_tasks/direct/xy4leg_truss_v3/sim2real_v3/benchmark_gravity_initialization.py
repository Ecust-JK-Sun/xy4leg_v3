"""Offline timing benchmark. Never opens a serial port or enables a motor.

Runs the actual gravity limiter, initialization ramp/hold, MIT encoder and
feedback decoder against an in-memory serial transport. Optional blocking
write delay is synthetic and cannot certify USB/CAN/firmware performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from gravity_compensation import DEFAULT_CONFIG, DEFAULT_POSE_DEG, LEGS, GravityFeedforward, GravityModel
from init_default_joint_pose import (
    DEFAULT_KD, DEFAULT_KP, InitializationGravity, InitializationTiming, LegController,
    _hold_and_verify_leg, _send_leg_position_targets, _target_rad_for_leg,
)
from motor_control import DaMiaoMotorController


class InMemorySerial:
    """10-byte writes generate 8-byte feedback; position instantly follows target."""

    def __init__(self, driver, write_delay_s):
        self.driver, self.write_delay_s = driver, write_delay_s
        self.buffer = bytearray()
        self.out_waiting = 0
        self.open = True

    def isOpen(self):
        return self.open

    @property
    def in_waiting(self):
        return len(self.buffer)

    def write(self, command):
        if len(command) != 10 or command[1] != 0xF2:
            raise ValueError("Benchmark accepts only MIT position commands, never enable/disable/zero commands")
        if self.write_delay_s:
            time.sleep(self.write_delay_s)
        mid = command[0]
        pos = (command[2] << 8) | command[3]
        velocity = self.driver.float_to_uint(0., -10., 10., 12)
        torque = ((command[8] & 0xF) << 8) | command[9]
        self.buffer.extend(bytes([0x10 | mid, pos >> 8, pos & 0xFF, velocity >> 4,
                                  ((velocity & 0xF) << 4) | (torque >> 8), torque & 0xFF, 25, 25]))
        return len(command)

    def read(self, count):
        response = bytes(self.buffer[:count])
        del self.buffer[:count]
        return response

    def close(self):
        self.open = False


class SimulatedMotorController(DaMiaoMotorController):
    def __init__(self, write_delay_s):
        # Intentionally do not call the base constructor or serial.Serial().
        self.port = "IN-MEMORY-BENCHMARK"
        self._rx_buffer = bytearray()
        self.ser = InMemorySerial(self, write_delay_s)


def _summary(values):
    return dict(n=len(values), mean_ms=float(np.mean(values)), p95_ms=float(np.percentile(values, 95)),
                max_ms=float(np.max(values)), over_40ms=int(np.count_nonzero(np.asarray(values) > 40.)))


def benchmark_computation(model, iterations):
    rng = np.random.default_rng(20261030)
    base = np.deg2rad(DEFAULT_POSE_DEG).T.reshape(20)
    poses = base + rng.uniform(-.5, .5, (iterations+100, 20))
    single, full = GravityFeedforward(model), GravityFeedforward(model)
    results = {}
    for name, operation in (
        ("single_leg_with_limiter", lambda q, stamp: single.update_leg("LB", q[0::4], now=stamp)),
        ("four_legs_with_limiter", lambda q, stamp: full.update(q, [0., 0., -9.81], now=stamp)),
    ):
        elapsed = []
        for index, q in enumerate(poses):
            started = time.perf_counter()
            operation(q, index*.04)
            duration = (time.perf_counter()-started)*1000
            if index >= 100:
                elapsed.append(duration)
        results[name] = _summary(elapsed)
    return results


def benchmark_pipeline(model, duration_s, write_delay_s, leg="LB"):
    ctrl = SimulatedMotorController(write_delay_s)
    item = LegController(leg, ctrl)
    timing = InitializationTiming(leg, 25., source="SIMULATED in-memory serial; no hardware")
    gravity = InitializationGravity(leg, GravityFeedforward(model), .5, (0., 0., -9.81), -math.pi/6, timing)
    target = _target_rad_for_leg(leg, True)
    current = [value-math.radians(4.) for value in target]
    try:
        _send_leg_position_targets(item, target, current, DEFAULT_KP, DEFAULT_KD,
                                   math.radians(1.), 25., .0002, gravity, timing)
        # Deliberately run for the requested time, even if the instantaneous mock
        # pose follows targets. This measures cadence, never physical tracking.
        _hold_and_verify_leg(item, target, DEFAULT_KP, DEFAULT_KD, 25., .0002,
                             duration_s, duration_s+1., math.radians(1.), math.radians(2.), .5,
                             gravity_compensator=gravity, timing=timing)
        result = timing.report(final=True)
        result.update(write_delay_ms=write_delay_s*1000, simulated_motion="instant target following",
                      benchmark_stop="duration limit; NOT_REACHED is expected and is not a motion-accuracy result")
        return result
    finally:
        ctrl.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--duration_s", type=float, default=6.)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--serial_write_delay_ms", type=float, default=10*10/115200*1000,
                        help="Synthetic blocking delay for each 10-byte write; default equals nominal 115200-baud airtime")
    parser.add_argument("--output", type=Path, help="Save a new JSON artifact; existing files are not overwritten")
    args = parser.parse_args()
    if (not math.isfinite(args.duration_s) or args.duration_s <= 0 or args.iterations < 1
            or not math.isfinite(args.serial_write_delay_ms) or args.serial_write_delay_ms < 0):
        parser.error("Positive duration/iterations and finite nonnegative write delay are required")
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new artifact path")
    model = GravityModel(args.config, args.urdf)
    report = dict(model=model.description(), cpu_count=os.cpu_count(), platform=platform.platform(),
                  source_sha256={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                 for name in ("benchmark_gravity_initialization.py", "init_default_joint_pose.py",
                                              "gravity_compensation.py", "motor_control.py", "runtime_timing.py")},
                  python=sys.version, computation=benchmark_computation(model, args.iterations),
                  pipeline=benchmark_pipeline(model, args.duration_s, args.serial_write_delay_ms/1000),
                  hardware_verified=False,
                  interpretation="Offline software timing only. No real serial, converter, CAN or motor execution measured.")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
