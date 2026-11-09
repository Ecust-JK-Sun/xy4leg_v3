"""Discrete bidirectional static-load measurements; preview unless --enable_motors.

The measured torque is an effective holding load at the recorded full-leg pose.
Direction differences include static friction/hysteresis and are not identified gravity.
No existing compensation table is read or modified. Host timestamps are not device ACKs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from init_default_joint_pose import (
    DEFAULT_KP, LEG_PORTS, MOTOR_IDS, THIS_DIR, IntegralTorqueCompensator,
    _checked_feedback, _enable_and_read_current, _read_motor_cfg, _target_rad_for_leg,
)
from motor_control import DaMiaoMotorController
from runtime_timing import PeriodicDeadline


FEEDBACK_FIELDS = ("time_s", "visit", "phase", "motor_id", "target_deg", "position_deg",
                   "velocity_deg_s", "torque_nm", "backlog_bytes")
COMMAND_FIELDS = ("time_s", "write_done_s", "visit", "phase", "motor_id", "target_deg", "ff_nm",
                  "lateness_ms", "skipped_ticks")
STATIC_DEFAULT_KP = DEFAULT_KP
STATIC_DEFAULT_INTEGRAL_LIMIT = (7., 4.5, 3.5, 2.5, 1.)


def make_plan(initial, low, high, step, cycles):
    """Use one grid anchored at initial, including exact bounds in both directions."""
    if not all(math.isfinite(x) for x in (initial, low, high, step)):
        raise ValueError("Angles and step must be finite")
    if not low <= initial <= high or low >= high or step <= 0 or cycles < 1:
        raise ValueError("Require min <= default initial <= max, min < max, step > 0, cycles >= 1")
    grid = sorted({round(low, 10), round(high, 10), round(initial, 10)} | {
        round(initial + k * step, 10)
        for k in range(math.ceil((low - initial) / step), math.floor((high - initial) / step) + 1)
    })
    plan = [dict(visit=0, target_deg=initial, direction=0, sweep="initial")]

    def append(points, direction, sweep):
        for angle in points:
            if abs(angle - plan[-1]["target_deg"]) > 1.e-8:
                plan.append(dict(visit=len(plan), target_deg=angle, direction=direction, sweep=sweep))

    append([x for x in reversed(grid) if x < initial - 1.e-8], -1, "initial_to_min")
    for cycle in range(cycles):
        append(grid, 1, f"up_{cycle + 1}")
        if cycle + 1 < cycles:
            append(list(reversed(grid)), -1, f"down_{cycle + 1}")
    append([x for x in reversed(grid) if x >= initial - 1.e-8], -1, "return_initial")
    return plan


def validate(args):
    if args.step_deg not in (1.5, 2., 2.5) or not 1 <= args.cycles <= 3:
        raise ValueError("Use step_deg 1.5/2/2.5 and cycles 1/2/3")
    initial = round(math.degrees(_target_rad_for_leg(args.leg, True)[args.joint - 1]), 10)
    if not (-math.degrees(12.5) <= args.min_deg < args.max_deg <= math.degrees(12.5)):
        raise ValueError("Command bounds must be ordered and inside the MIT position encoding range")
    plan = make_plan(initial, args.min_deg, args.max_deg, args.step_deg, args.cycles)
    positive = ("tx_hz", "poll_hz", "point_timeout_s", "stable_s", "sample_s", "tolerance_deg",
                "other_tolerance_deg", "velocity_tolerance_deg_s", "feedback_timeout_s", "init_speed_deg_s",
                "position_window_s", "position_stability_deg", "init_range_deg")
    if any(not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0 for key in positive):
        raise ValueError("Rates, durations, tolerances and initial speed must be positive and finite")
    if not math.isfinite(args.discard_s) or args.discard_s < 0:
        raise ValueError("discard_s must be finite and nonnegative")
    if not math.isfinite(args.monitor_margin_deg) or args.monitor_margin_deg < 0:
        raise ValueError("monitor_margin_deg must be finite and nonnegative")
    if not 10 <= args.tx_hz <= 100 or args.poll_hz < args.tx_hz:
        raise ValueError("Require 10 <= tx_hz <= 100 and poll_hz >= tx_hz")
    if args.point_timeout_s <= args.stable_s + args.discard_s + args.sample_s:
        raise ValueError("point_timeout_s must exceed stable_s + discard_s + sample_s")
    if args.tolerance_deg <= .25 or args.other_tolerance_deg <= .25:
        raise ValueError("Position tolerances must exceed the integral's 0.25 deg deadband")
    if args.return_tolerance_deg is not None and (
            not math.isfinite(args.return_tolerance_deg) or args.return_tolerance_deg <= .25):
        raise ValueError("return_tolerance_deg must be finite and exceed the integral's 0.25 deg deadband")
    if args.min_samples < 2 or args.min_samples > args.tx_hz * args.sample_s:
        raise ValueError("Require 2 <= min_samples <= tx_hz * sample_s")
    if args.position_window_s < 2 / args.tx_hz or args.position_window_s >= args.point_timeout_s:
        raise ValueError("position_window_s must cover >=3 nominal packets and be shorter than point_timeout_s")
    if any(abs(math.degrees(target)) + args.init_range_deg + args.monitor_margin_deg > math.degrees(12.5)
           for target in _target_rad_for_leg(args.leg, True)):
        raise ValueError("Initialization monitoring bounds exceed the MIT position encoding range")
    if any(not math.isfinite(value) or not 0 <= value <= 28 for value in args.integral_torque_limit):
        raise ValueError("integral_torque_limit must be finite and in [0,28] N*m (protocol encoding bounds)")
    kp, kd = _read_motor_cfg(args.motor_config)
    if args.motor_config is None:
        kp = STATIC_DEFAULT_KP
    args.kp, args.kd = tuple(args.kp or kp), tuple(args.kd or kd)
    if any(not math.isfinite(x) or not 0 <= x <= cap
           for values, cap in ((args.kp, 500), (args.kd, 5)) for x in values):
        raise ValueError("Require kp in [0,500] and kd in [0,5]")
    return plan


def _make_integral(args):
    return IntegralTorqueCompensator(torque_limit=tuple(args.integral_torque_limit), max_dt_s=.05)


def distribution(values):
    middle = statistics.median(values)
    return dict(median=middle, mad=statistics.median(abs(x - middle) for x in values),
                minimum=min(values), maximum=max(values), samples=len(values))


def _approach_direction(point):
    # Initial pose holds are not directional samples. Old runs retain their
    # command labels; new runs must have observed motion to form a branch.
    return point.get("observed_approach_direction", point["direction"]) if point["direction"] else 0


def summarize_points(points):
    """Equal weight per visit, retaining directional branches and every raw visit."""
    angles = sorted({p["target_deg"] for p in points if p.get("valid") and _approach_direction(p)})
    result = []
    for angle in angles:
        row = dict(target_deg=angle, branches={})
        for direction, name in ((-1, "decreasing"), (1, "increasing")):
            visits = [p for p in points if p.get("valid") and p["target_deg"] == angle
                      and _approach_direction(p) == direction]
            if not visits:
                continue
            torque = distribution([p["selected_torque_nm"]["median"] for p in visits])
            threshold = max(.1, 3 * 1.4826 * torque["mad"])
            row["branches"][name] = dict(
                torque_nm=torque, visits=[p["visit"] for p in visits],
                direction_basis=sorted({"observed_position" if "observed_approach_direction" in p
                                        else "legacy_command" for p in visits}),
                position_deg=distribution([p["selected_position_deg"]["median"] for p in visits]),
                # A small visit count cannot identify which branch observation is an outlier.
                possible_outlier_visits=[p["visit"] for p in visits
                                        if len(visits) >= 3 and
                                        abs(p["selected_torque_nm"]["median"] - torque["median"]) > threshold],
            )
        row["bidirectional"] = len(row["branches"]) == 2
        if row["bidirectional"]:
            down = row["branches"]["decreasing"]["torque_nm"]["median"]
            up = row["branches"]["increasing"]["torque_nm"]["median"]
            row.update(load_midpoint_nm=(up + down) / 2, signed_half_difference_nm=(up - down) / 2)
        result.append(row)
    return result


class ApproachTracker:
    """Last significant encoder motion; retain its sign during a stationary hold."""

    def __init__(self, initial_deg=None, resolution_deg=.1):
        self.anchor = initial_deg
        self.resolution_deg = resolution_deg
        self.direction = 0
        self.reversals = []

    def observe(self, position_deg, time_s):
        if self.anchor is None:
            self.anchor = position_deg
            return
        delta = position_deg - self.anchor
        if abs(delta) < self.resolution_deg:
            return
        direction = 1 if delta > 0 else -1
        if self.direction and direction != self.direction:
            self.reversals.append(dict(time_s=time_s, position_deg=position_deg,
                                       from_direction=self.direction, to_direction=direction))
        self.direction, self.anchor = direction, position_deg


class PositionHistory:
    """Short-window position span, independent of the motor-reported velocity."""

    def __init__(self, window_s):
        self.window_s = window_s
        self.samples = {motor: deque() for motor in MOTOR_IDS}

    def observe(self, row):
        history = self.samples[row["motor_id"]]
        history.append((row["time_s"], row["position_deg"]))
        # Retain one anchor just before the cutoff, so jitter does not continually
        # invalidate a nominally complete window. Times remain host receipt times.
        while len(history) > 1 and history[1][0] <= row["time_s"] - self.window_s:
            history.popleft()

    def metrics(self, motor, now):
        history = self.samples[motor]
        if not history:
            return dict(ready=False, span_deg=None, span_rate_deg_s=None, samples=0)
        duration = history[-1][0] - history[0][0]
        span = max(p for _, p in history) - min(p for _, p in history)
        return dict(ready=(len(history) >= 3 and duration >= self.window_s - 1.e-6
                           and now - history[-1][0] <= self.window_s),
                    span_deg=span, span_rate_deg_s=span / duration if duration > 0 else None,
                    samples=len(history), duration_s=duration)


class Session:
    """One serial reader, measured resends, per-motor feedback freshness and integration."""

    def __init__(self, args, ctrl, command_writer, feedback_writer):
        self.args, self.ctrl = args, ctrl
        self.command_writer, self.feedback_writer = command_writer, feedback_writer
        self.started = time.monotonic()
        self.next_tx = self.started
        self.latest, self.received_at, self.integrated_at = {}, {}, {}
        self.integral = _make_integral(args)
        self.position_history = PositionHistory(args.position_window_s)
        self.schedule = PeriodicDeadline(args.poll_hz, clock=time.monotonic, sleep=time.sleep)
        self.packet_times = {motor: [] for motor in MOTOR_IDS}
        self.tx_times = []
        self.skipped_tx = 0
        self.monitor_bounds_deg = {
            args.joint: (args.min_deg - args.monitor_margin_deg, args.max_deg + args.monitor_margin_deg)
        }

    def _fresh(self, now):
        stale = [motor for motor in MOTOR_IDS
                 if now - self.received_at.get(motor, self.started) > self.args.feedback_timeout_s]
        if stale:
            raise RuntimeError(f"Missing/stale feedback from motors {stale}")

    def tick(self, targets, visit, phase, integrate):
        self.schedule.wait()
        now = time.monotonic()
        self._fresh(now)
        if now >= self.next_tx - 1.e-9:
            late = max(0., now - self.next_tx)
            skipped = int(late * self.args.tx_hz)
            self.next_tx += (skipped + 1) / self.args.tx_hz
            self.skipped_tx += skipped
            self.tx_times.append(now - self.started)
            for index, motor in enumerate(MOTOR_IDS):
                sent = time.monotonic()
                self.ctrl.send(self.ctrl.damiao_mit_command(
                    pos=targets[index], vel=0., kp=self.args.kp[index], kd=self.args.kd[index],
                    tor=self.integral.torque[index], motor_id=motor))
                self.command_writer.writerow(dict(
                    time_s=sent - self.started, write_done_s=time.monotonic() - self.started,
                    visit=visit, phase=phase, motor_id=motor, target_deg=math.degrees(targets[index]),
                    ff_nm=self.integral.torque[index], lateness_ms=late * 1000, skipped_ticks=skipped))
                time.sleep(.0002)
        frames, _, backlog, overflow = self.ctrl.receive_available()
        if overflow:
            raise RuntimeError("Feedback queue overflow; timestamps cannot be used")
        rows, updates = [], {}
        for frame in frames:
            data = _checked_feedback(self.ctrl.damiao_feedback(frame))
            if data is None:
                continue
            motor, received = data["motor_id"], time.monotonic()
            row = dict(time_s=received - self.started, visit=visit, phase=phase, motor_id=motor,
                       target_deg=math.degrees(targets[motor - 1]), position_deg=math.degrees(data["position"]),
                       velocity_deg_s=math.degrees(data["velocity"]), torque_nm=data["torque"], backlog_bytes=backlog)
            self.feedback_writer.writerow(row)
            rows.append(row)
            self.position_history.observe(row)
            self.latest[motor], self.received_at[motor], updates[motor] = data, received, data
            self.packet_times[motor].append(received - self.started)
            bounds = self.monitor_bounds_deg.get(motor)
            if bounds is not None and not bounds[0] <= row["position_deg"] <= bounds[1]:
                raise RuntimeError(f"Measured J{motor} angle {row['position_deg']:.3f} exceeded monitoring bounds "
                                   f"[{bounds[0]:.3f}, {bounds[1]:.3f}] deg during {phase}")
        now = time.monotonic()
        self._fresh(now)
        if integrate and len(self.latest) == len(MOTOR_IDS):
            for motor, data in updates.items():
                position_motion = self.position_history.metrics(motor, now - self.started)
                if position_motion["ready"]:
                    # Keep raw velocity in the log; use position-derived motion for
                    # the integrator's growth gate when velocity feedback is noisy.
                    estimated = dict(data, velocity=math.radians(position_motion["span_rate_deg_s"]))
                    self.integral.update({motor: estimated}, targets, now - self.integrated_at.get(motor, now))
        for motor in updates:
            self.integrated_at[motor] = now
        return now, rows

    def diagnostics(self, targets, rows, tolerance_deg=None):
        blockers, motors = [], {}
        now = time.monotonic() - self.started
        for motor in MOTOR_IDS:
            if motor not in self.latest:
                blockers.append(f"J{motor}:missing_feedback")
                continue
            data = self.latest[motor]
            tolerance = tolerance_deg if tolerance_deg is not None else (
                self.args.tolerance_deg if motor == self.args.joint else self.args.other_tolerance_deg)
            error = math.degrees(targets[motor - 1] - data["position"])
            incoming = [r for r in rows if r["motor_id"] == motor]
            if abs(error) > tolerance or any(abs(r["target_deg"] - r["position_deg"]) > tolerance for r in incoming):
                blockers.append(f"J{motor}:position_error")
            motion = self.position_history.metrics(motor, now)
            if not motion["ready"]:
                blockers.append(f"J{motor}:position_window_pending")
            elif motion["span_deg"] > self.args.position_stability_deg:
                blockers.append(f"J{motor}:position_unstable")
            raw_velocity = math.degrees(data["velocity"])
            if self.args.require_velocity_feedback and (abs(raw_velocity) > self.args.velocity_tolerance_deg_s
                    or any(abs(r["velocity_deg_s"]) > self.args.velocity_tolerance_deg_s for r in incoming)):
                blockers.append(f"J{motor}:reported_velocity")
            motors[str(motor)] = dict(error_deg=error, reported_velocity_deg_s=raw_velocity,
                                      position_window=motion, integral_ff_nm=self.integral.torque[motor - 1],
                                      integral_at_limit=abs(self.integral.torque[motor - 1]) >=
                                      self.integral.torque_limit[motor - 1] - 1.e-9)
        return dict(blockers=blockers, motors=motors)

    def good(self, targets, rows):
        return not self.diagnostics(targets, rows)["blockers"]


def _print_diagnostics(point, phase):
    state = point.get("last_diagnostics", {})
    values = []
    for motor, data in state.get("motors", {}).items():
        span = data["position_window"]["span_deg"]
        span_text = f"{span:.3f}" if span is not None else "pending"
        values.append(f"J{motor}: error={data['error_deg']:+.3f}deg span={span_text}deg "
                      f"raw_v={data['reported_velocity_deg_s']:+.2f}deg/s ff={data['integral_ff_nm']:+.3f}Nm")
    print(f"[WAIT] visit={point['visit']} phase={phase} blockers={state.get('blockers', [])}; " + "; ".join(values))


def measure_point(session, point, base_targets, phase_prefix="", tolerance_deg=None):
    """Retry unstable windows within one bounded timeout, preserving all raw packets."""
    args = session.args
    targets = list(base_targets)
    targets[args.joint - 1] = math.radians(point["target_deg"])
    started, stable_since, frozen_at, sample_started = time.monotonic(), None, None, None
    phase, samples, rejected_windows = "settle", [], 0
    blocker_counts, rejection_counts = Counter(), Counter()
    previous_position = session.latest.get(args.joint, {}).get("position")
    approach = ApproachTracker(math.degrees(previous_position) if previous_position is not None else None)
    next_report = started + 5.
    point.update(valid=False, status="in_progress", blocker_counts=blocker_counts, rejection_counts=rejection_counts,
                 command_direction=point.get("direction", 0), approach_direction_resolution_deg=approach.resolution_deg)
    while time.monotonic() - started < args.point_timeout_s:
        now, rows = session.tick(targets, point["visit"], phase_prefix + phase, integrate=(phase == "settle"))
        for row in rows:
            if row["motor_id"] == args.joint:
                approach.observe(row["position_deg"], row["time_s"])
        point.update(observed_approach_direction=approach.direction, motion_reversals=approach.reversals,
                     approach_matches_command=(approach.direction == point["command_direction"])
                     if approach.direction and point["command_direction"] else None)
        diagnostics = session.diagnostics(targets, rows, tolerance_deg=tolerance_deg)
        point["last_diagnostics"] = diagnostics
        blockers = diagnostics["blockers"]
        # Also constrain the entire accepted sample, preventing slow creep from
        # passing a succession of individually quiet 0.2 s windows.
        if phase == "sample":
            for motor in MOTOR_IDS:
                angles = [r["position_deg"] for r in samples + rows if r["motor_id"] == motor]
                if angles and max(angles) - min(angles) > args.position_stability_deg:
                    blockers.append(f"J{motor}:sample_position_drift")
        blocker_counts.update(blockers)
        if now >= next_report:
            _print_diagnostics(point, phase_prefix + phase)
            next_report = now + 5.
        if blockers:
            if phase != "settle":
                rejected_windows += 1
                rejection_counts.update(f"{phase}:{reason}" for reason in blockers)
            phase, stable_since, frozen_at, sample_started, samples = "settle", None, None, None, []
            continue
        if phase == "settle":
            if stable_since is None:
                stable_since = now
            elif now - stable_since >= args.stable_s and all(
                session.received_at.get(motor, -math.inf) > stable_since for motor in MOTOR_IDS
            ):
                frozen_at, phase = now, "discard"
        elif phase == "discard" and now - frozen_at >= args.discard_s:
            sample_started, phase = now, "sample"
        elif phase == "sample":
            samples.extend(rows)
            if now - sample_started >= args.sample_s:
                by_motor = {motor: [r for r in samples if r["motor_id"] == motor] for motor in MOTOR_IDS}
                valid_counts = all(len(rows_i) >= args.min_samples and
                                   rows_i[-1]["time_s"] - rows_i[0]["time_s"] >= .8 * args.sample_s
                                   for rows_i in by_motor.values())
                if valid_counts:
                    selected = by_motor[args.joint]
                    point.update(
                        valid=True, status="sampled", elapsed_s=now - started,
                        rejected_windows=rejected_windows,
                        sample_start_s=sample_started - session.started, sample_end_s=now - session.started,
                        selected_position_deg=distribution([r["position_deg"] for r in selected]),
                        selected_torque_nm=distribution([r["torque_nm"] for r in selected]),
                        max_abs_velocity_deg_s=max(abs(r["velocity_deg_s"]) for r in samples),
                        position_span_deg_by_motor={str(motor): max(r["position_deg"] for r in rows_i) -
                                                    min(r["position_deg"] for r in rows_i)
                                                    for motor, rows_i in by_motor.items()},
                        position_deg_by_motor={str(motor): distribution([r["position_deg"] for r in rows_i])
                                               for motor, rows_i in by_motor.items()},
                        torque_nm_by_motor={str(motor): distribution([r["torque_nm"] for r in rows_i])
                                            for motor, rows_i in by_motor.items()},
                        frozen_integral_ff_nm=session.integral.torque.copy(),
                        integral_at_limit_motors=[motor for motor in MOTOR_IDS if
                            abs(session.integral.torque[motor - 1]) >= session.integral.torque_limit[motor - 1] - 1.e-9],
                    )
                    return
                rejected_windows += 1
                rejection_counts.update(["sample:insufficient_feedback"])
                phase, stable_since, samples = "settle", None, []
    point.update(status="timeout", elapsed_s=time.monotonic() - started, rejected_windows=rejected_windows)
    _print_diagnostics(point, phase_prefix + phase)
    raise RuntimeError(f"Visit {point['visit']} target {point['target_deg']} deg did not yield a valid static window; "
                       f"final blockers={point['last_diagnostics']['blockers']}; "
                       f"blocking tick counts={dict(blocker_counts)}; rejections={dict(rejection_counts)}")


def cadence(times):
    gaps = sorted((b - a) * 1000 for a, b in zip(times, times[1:]))
    return dict(samples=len(times), hz=(len(times) - 1) / (times[-1] - times[0])
                if len(times) > 1 and times[-1] > times[0] else None,
                gap_max_ms=max(gaps) if gaps else None,
                gap_p95_ms=gaps[math.ceil(.95 * len(gaps)) - 1] if gaps else None)


def run(args):
    plan = validate(args)
    base_targets = _target_rad_for_leg(args.leg, True)
    init_bounds = {motor: (math.degrees(target) - args.init_range_deg - args.monitor_margin_deg,
                           math.degrees(target) + args.init_range_deg + args.monitor_margin_deg)
                   for motor, target in zip(MOTOR_IDS, base_targets)}
    description = dict(
        leg=args.leg, joint=args.joint, default_pose_deg=[math.degrees(x) for x in base_targets],
        command_range_deg=[args.min_deg, args.max_deg],
        feedback_abort_bounds_deg=[args.min_deg - args.monitor_margin_deg, args.max_deg + args.monitor_margin_deg],
        initialization=dict(enabled=True, monitor_bounds_deg=init_bounds,
                            target_deg=[math.degrees(x) for x in base_targets],
                            retain_integral_for_scan=True),
        return_to_initial=dict(enabled=args.return_tolerance_deg is not None,
                               all_joint_tolerance_deg=args.return_tolerance_deg),
        integral_settings=vars(_make_integral(args)),
        visits=len(plan), plan=plan,
        stability=dict(position_window_s=args.position_window_s, position_span_limit_deg=args.position_stability_deg,
                       sample_span_limit_deg=args.position_stability_deg,
                       require_reported_velocity=args.require_velocity_feedback),
        minimum_hold_total_s=len(plan) * (args.stable_s + args.discard_s + args.sample_s),
        point_timeout_total_s=len(plan) * args.point_timeout_s,
        note="Hold totals exclude the initial ramp and time needed to reach each point; feedback rate is measured.")
    print(json.dumps(description, ensure_ascii=False, indent=2))
    if not args.enable_motors:
        print("[PREVIEW] No serial port opened. Add --enable_motors to execute.")
        return description
    root = THIS_DIR.parents[5] / "logs/real_v3"
    directory = Path(args.output_dir) if args.output_dir else root / (
        f"static_load_{args.leg}_j{args.joint}_" + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir(parents=True, exist_ok=False)
    metadata = dict(settings=vars(args), run_type="hardware", complete=False, error=None,
                    procedure=description, points=[], initialization=dict(status="not_started"),
                    return_to_initial=dict(status="not_started" if args.return_tolerance_deg is not None else "disabled"),
                    interpretation="Effective holding torque at the recorded pose; branch differences are not pure friction. "
                    "No load model or directional feedforward is applied. Integral is frozen during accepted sampling. "
                    "Position stability is required over both a recent window and the complete sample window. "
                    "Reported velocity is diagnostic unless require_velocity_feedback is explicitly enabled. "
                    "Host receipt timestamps do not certify device sampling time or 40 ms dynamic settling.",
                    integral_settings=vars(_make_integral(args)))
    ctrl, session = None, None
    try:
        with (directory / "commands.csv").open("x", newline="") as commands, (
            directory / "feedback.csv").open("x", newline="") as feedback:
            command_writer = csv.DictWriter(commands, COMMAND_FIELDS)
            feedback_writer = csv.DictWriter(feedback, FEEDBACK_FIELDS)
            command_writer.writeheader()
            feedback_writer.writeheader()
            (directory / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            ctrl = DaMiaoMotorController(args.port or LEG_PORTS[args.leg])
            current = [_enable_and_read_current(ctrl, motor) for motor in MOTOR_IDS]
            metadata["initialization"].update(
                status="read_current", measured_start_deg=[math.degrees(x) if x is not None else None for x in current],
                monitor_bounds_deg=init_bounds)
            if any(x is None for x in current):
                raise RuntimeError("Cannot initialize without measured positions for all five joints")
            print(f"[INIT] {args.leg}: current_deg={metadata['initialization']['measured_start_deg']} "
                  f"target_deg={[math.degrees(x) for x in base_targets]}")
            outside = [f"J{motor}={math.degrees(position):.3f} deg outside "
                       f"[{init_bounds[motor][0]:.3f}, {init_bounds[motor][1]:.3f}]"
                       for motor, position in zip(MOTOR_IDS, current)
                       if not init_bounds[motor][0] <= math.degrees(position) <= init_bounds[motor][1]]
            if outside:
                raise RuntimeError("Initial pose is outside initialization monitoring bounds; "
                                   + "; ".join(outside) + "; no pose command sent")
            session = Session(args, ctrl, command_writer, feedback_writer)
            scan_bounds = session.monitor_bounds_deg.copy()
            session.monitor_bounds_deg = init_bounds
            metadata["initialization"]["status"] = "ramping"
            # First ramp all five joints to the default pose; do not jump from unknown initial angles.
            duration = max(abs(math.degrees(b - a)) for a, b in zip(current, base_targets)) / args.init_speed_deg_s
            ramp_started = time.monotonic()
            while time.monotonic() - ramp_started < duration:
                fraction = min(1., (time.monotonic() - ramp_started) / duration)
                session.tick([a + fraction * (b - a) for a, b in zip(current, base_targets)], -1, "initial_ramp", False)
            initial_point = dict(visit=-1, target_deg=math.degrees(base_targets[args.joint - 1]),
                                 direction=0, sweep="initialization")
            metadata["initialization"].update(status="verifying", pose_verification=initial_point)
            measure_point(session, initial_point, base_targets, phase_prefix="initialization_")
            # Verify the scan bounds on fresh initialized feedback before starting
            # the first scan point. Keep the learned torque and the open serial link.
            for motor, (low, high) in scan_bounds.items():
                actual = math.degrees(session.latest[motor]["position"])
                if not low <= actual <= high:
                    raise RuntimeError(f"Initialized J{motor}={actual:.3f} deg is outside scan monitoring bounds "
                                       f"[{low:.3f}, {high:.3f}]; scan not started")
            session.monitor_bounds_deg = scan_bounds
            metadata["initialization"].update(status="complete", integral_ff_nm=session.integral.torque.copy())
            commands.flush()
            feedback.flush()
            (directory / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            print("[INIT] Default pose verified; retaining compensation and starting scan.")
            for planned in plan:
                point = dict(planned)
                metadata["points"].append(point)
                print(f"[POINT] {point['visit'] + 1}/{len(plan)} J{args.joint} target={point['target_deg']:.3f} "
                      f"direction={point['direction']:+d} {point['sweep']}")
                measure_point(session, point, base_targets)
                print(f"[SAMPLED] angle={point['selected_position_deg']['median']:.3f} deg "
                      f"torque={point['selected_torque_nm']['median']:.3f} N*m "
                      f"MAD={point['selected_torque_nm']['mad']:.3f} elapsed={point['elapsed_s']:.2f}s "
                      f"observed_direction={point['observed_approach_direction']:+d} "
                      f"reversals={len(point['motion_reversals'])}")
                commands.flush()
                feedback.flush()
                (directory / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.return_tolerance_deg is not None:
                # The scan already ends at the default target. Verify every joint
                # with a shared tolerance before allowing the next joint/run.
                session.monitor_bounds_deg = init_bounds
                return_point = dict(visit=-2, target_deg=math.degrees(base_targets[args.joint - 1]),
                                    direction=0, sweep="return_to_initial")
                metadata["return_to_initial"].update(status="verifying", pose_verification=return_point)
                print(f"[RETURN] Verifying all five joints within {args.return_tolerance_deg:g} deg of default.")
                measure_point(session, return_point, base_targets, phase_prefix="return_",
                              tolerance_deg=args.return_tolerance_deg)
                metadata["return_to_initial"]["status"] = "complete"
                print("[RETURN] Full-leg default pose verified.")
            metadata["complete"] = True
    except BaseException as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        if metadata["initialization"]["status"] != "complete":
            metadata["initialization"].update(status="failed", error=metadata["error"])
            verification = metadata["initialization"].get("pose_verification")
            if verification is not None and verification.get("status") == "in_progress":
                verification.update(status="interrupted", error=metadata["error"])
        if metadata["points"] and metadata["points"][-1].get("status") == "in_progress":
            metadata["points"][-1].update(status="interrupted", error=metadata["error"])
        if metadata["return_to_initial"]["status"] == "verifying":
            metadata["return_to_initial"].update(status="failed", error=metadata["error"])
            verification = metadata["return_to_initial"]["pose_verification"]
            if verification.get("status") == "in_progress":
                verification.update(status="interrupted", error=metadata["error"])
        raise
    finally:
        try:
            if ctrl is not None:
                ctrl.close()  # No disable/zero commands; normal completion already returns to initial pose.
        finally:
            summary = dict(complete=metadata["complete"], error=metadata["error"],
                           initialization_status=metadata["initialization"]["status"],
                           return_to_initial_status=metadata["return_to_initial"]["status"],
                           valid_visits=sum(p.get("valid", False) for p in metadata["points"]),
                           planned_visits=len(plan), angle_summary=summarize_points(metadata["points"]),
                           unclassified_direction_visits=[p["visit"] for p in metadata["points"] if
                               p.get("valid") and p["direction"] and not _approach_direction(p)])
            if session is not None:
                summary.update(command_cadence=cadence(session.tx_times),
                               feedback_cadence_by_motor={str(k): cadence(v) for k, v in session.packet_times.items()},
                               skipped_command_ticks=session.skipped_tx)
            (directory / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[INFO] Results: {directory.resolve()}")
    return directory


def build_parser(*, batch=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leg", choices=tuple(LEG_PORTS), default="LB")
    if batch:
        parser.add_argument("--rounds", type=int, default=5, help="Repeat J1->J2->J3->J4->J5 (default 5 rounds)")
        parser.add_argument("--range_deg", type=float, default=30., help="Scan each joint +/- this angle from default")
    else:
        parser.add_argument("--joint", type=int, choices=MOTOR_IDS, required=True)
        parser.add_argument("--min_deg", type=float, required=True, help="Command minimum in hardware feedback coordinates")
        parser.add_argument("--max_deg", type=float, required=True, help="Command maximum in hardware feedback coordinates")
    parser.add_argument("--step_deg", type=float, default=2., help="1.5, 2 or 2.5 deg; exact boundaries use a shorter step")
    if batch:
        parser.set_defaults(cycles=1)
    else:
        parser.add_argument("--cycles", type=int, default=2, help="1: initial->min->max->initial; 2/3 add full round trips")
    parser.add_argument("--kp", type=float, nargs=5)
    parser.add_argument("--kd", type=float, nargs=5)
    parser.add_argument("--integral_torque_limit", type=float, nargs=5, default=STATIC_DEFAULT_INTEGRAL_LIMIT,
                        metavar=("J1", "J2", "J3", "J4", "J5"),
                        help="Per-joint integral feedforward limits in N*m (default 7 4.5 3.5 2.5 1); PD is additional")
    parser.add_argument("--motor_config", help="Optional YAML gains; default static KP is 35 25 25 25 2")
    parser.add_argument("--port")
    parser.add_argument("--tx_hz", type=float, default=100.)
    parser.add_argument("--poll_hz", type=float, default=200.)
    parser.add_argument("--init_speed_deg_s", type=float, default=5.)
    parser.add_argument("--init_range_deg", type=float, default=30.,
                        help="Initialization monitor half-range around each default joint angle (default 30 deg)")
    parser.add_argument("--point_timeout_s", type=float, default=30.)
    parser.add_argument("--stable_s", type=float, default=.5)
    parser.add_argument("--discard_s", type=float, default=.3, help="Frozen-integral wait before sampling")
    parser.add_argument("--sample_s", type=float, default=1.)
    parser.add_argument("--min_samples", type=int, default=50, help="Minimum fresh samples per motor per accepted window")
    parser.add_argument("--tolerance_deg", type=float, default=.8, help="Selected joint position tolerance (default 0.8 deg)")
    parser.add_argument("--other_tolerance_deg", type=float, default=5., help="Other four joint position tolerance (default 5 deg)")
    parser.add_argument("--return_tolerance_deg", type=float,
                        default=2. if batch else None,
                        help=("All-five-joint default-pose verification tolerance (default 2 deg)" if batch else
                              "Verify all five joints at the default pose after scanning; omitted disables this extra check"))
    parser.add_argument("--velocity_tolerance_deg_s", type=float, default=2.)
    parser.add_argument("--require_velocity_feedback", action="store_true",
                        help="Also require raw reported velocity within tolerance; default uses position stability")
    parser.add_argument("--position_window_s", type=float, default=.2, help="Recent position stability window")
    parser.add_argument("--position_stability_deg", type=float, default=.1,
                        help="Max position peak-to-peak per motor over recent window AND whole accepted sample")
    parser.add_argument("--feedback_timeout_s", type=float, default=.2)
    parser.add_argument("--monitor_margin_deg", type=float, default=1., help="Feedback abort bounds = command bounds +/- margin")
    parser.add_argument("--output_dir", help="Must not already exist; default timestamp under logs/real_v3")
    parser.add_argument("--enable_motors", action="store_true", help="Execute; omitted means preview without serial")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
