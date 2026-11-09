"""Single-leg tracking measurements. Default run is a command preview without hardware.

Host receipt timestamps are NOT device sample timestamps or motor command ACKs.
This tool reports observed tracking; it cannot certify a 40 ms motor settling bound.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import html
import json
import math
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from init_default_joint_pose import (
    DEFAULT_LEG_ORDER, LEG_PORTS, MOTOR_IDS, THIS_DIR, IntegralTorqueCompensator, LegController,
    _checked_feedback, _enable_and_read_current, _hold_and_verify_leg, _read_motor_cfg,
    _send_leg_position_targets, _target_rad_for_leg,
)
from motor_control import DaMiaoMotorController
from runtime_timing import PeriodicDeadline
from tracking_compensation import LoadFrictionController, LoadFrictionModel


COMMAND_FIELDS = ("time_s", "write_done_s", "lateness_ms", "skipped_ticks", "target_deg", "ff_nm",
                  "target_sequence", "target_update_s", "target_lateness_ms", "target_skipped_ticks",
                  "requested_load_nm", "requested_friction_nm", "requested_ff_nm", "load_clamped",
                  "ff_saturated", "motion_direction", "dynamic_active")
FEEDBACK_FIELDS = ("time_s", "motor_id", "position_deg", "velocity_deg_s", "torque_nm", "backlog_bytes")


def target_deg(args, elapsed):
    if args.mode == "sine":
        offset = args.amplitude_deg * math.sin(2 * math.pi * args.frequency_hz * elapsed)
    elif args.mode == "ramp":
        distance = max(0., elapsed - args.hold_s) * args.ramp_speed_deg_s
        phase = distance % (4 * args.amplitude_deg)
        if phase <= args.amplitude_deg:
            offset = phase
        elif phase <= 3 * args.amplitude_deg:
            offset = 2 * args.amplitude_deg - phase
        else:
            offset = phase - 4 * args.amplitude_deg
    else:
        offset = (0., args.amplitude_deg, 0., -args.amplitude_deg)[int((elapsed + 1.e-9) / args.hold_s) % 4]
    return args.center_deg + offset


def compensation_direction(args, elapsed, model):
    if args.mode == "step":
        # Retain the latest target-change direction through the hold. This is
        # feedforward, not an extra position-error feedback gain or integrator.
        segment = int((elapsed + 1.e-9) / args.hold_s)
        return 0. if segment == 0 else (1. if segment % 4 in (0, 1) else -1.)
    if args.mode == "sine":
        velocity = args.amplitude_deg * 2 * math.pi * args.frequency_hz * math.cos(2 * math.pi * args.frequency_hz * elapsed)
    elif elapsed <= args.hold_s:
        velocity = 0.
    else:
        phase = ((elapsed - args.hold_s) * args.ramp_speed_deg_s) % (4 * args.amplitude_deg)
        velocity = args.ramp_speed_deg_s * (1 if phase < args.amplitude_deg or phase >= 3 * args.amplitude_deg else -1)
    return math.tanh(velocity / model.velocity_soft_deg_s)


def _fixed_ff_fields(value):
    return dict(requested_load_nm=0., requested_friction_nm=0., requested_ff_nm=value,
                load_clamped=0, ff_saturated=0, motion_direction=0., dynamic_active=0)


def _validate(args):
    # Preserve compatibility with older run.json files and programmatic callers.
    args.warmup_tolerance_deg = getattr(args, "warmup_tolerance_deg", 1.)
    args.tx_hz = args.hz if getattr(args, "tx_hz", None) is None else args.tx_hz
    args.ramp_speed_deg_s = getattr(args, "ramp_speed_deg_s", 1.)
    args.compensation_model = getattr(args, "compensation_model", None)
    args.friction_scale = getattr(args, "friction_scale", 0.)
    args.coverage_target = getattr(args, "coverage_target", .75)
    if not math.isfinite(args.friction_scale) or not 0 <= args.friction_scale <= 1:
        raise ValueError("friction_scale must be finite and in [0,1]")
    if not math.isfinite(args.coverage_target) or not 0 < args.coverage_target <= 1:
        raise ValueError("coverage_target must be finite and in (0,1]")
    if args.friction_scale and not args.compensation_model:
        raise ValueError("friction_scale requires --compensation_model")
    positive = ("amplitude_deg", "hz", "poll_hz", "duration_s", "hold_s", "frequency_hz",
                "tolerance_deg", "feedback_timeout_s", "warmup_timeout_s", "warmup_tolerance_deg",
                "tx_hz", "ramp_speed_deg_s")
    if any(not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0 for key in positive):
        raise ValueError("Frequencies, durations, amplitude and tolerances must be positive and finite")
    if any(not math.isfinite(getattr(args, key)) for key in ("center_deg", "min_deg", "max_deg")):
        raise ValueError("Center and angle bounds must be finite")
    if not (args.min_deg <= args.center_deg - args.amplitude_deg
            < args.center_deg + args.amplitude_deg <= args.max_deg):
        raise ValueError("The entire center +/- amplitude trajectory must lie inside min_deg/max_deg")
    if args.amplitude_deg <= args.tolerance_deg:
        raise ValueError("Amplitude must exceed tolerance; otherwise a stationary joint could appear to pass")
    if args.hz != 25.:
        raise ValueError("This test targets the current policy's 25 Hz command interface")
    if args.tx_hz < args.hz or args.poll_hz < args.tx_hz or args.duration_s > 120:
        raise ValueError("Require poll_hz >= tx_hz >= hz and duration_s <= 120")
    if abs(args.tx_hz / args.hz - round(args.tx_hz / args.hz)) > 1.e-9:
        raise ValueError("tx_hz must be an integer multiple of the 25 Hz target rate")
    if args.mode == "sine" and (args.frequency_hz > args.hz / 10 or args.duration_s * args.frequency_hz < 3):
        raise ValueError("Sine tests require >=10 commands/cycle and >=3 cycles")
    if args.mode == "step" and args.duration_s < 5 * args.hold_s:
        raise ValueError("Step duration must cover at least five holds (center,+A,center,-A,center)")
    if args.mode == "ramp" and args.duration_s < args.hold_s + 4 * args.amplitude_deg / args.ramp_speed_deg_s:
        raise ValueError("Ramp duration must cover the initial hold and center->upper->lower->center")
    if args.warmup_timeout_s <= .5 or (args.integral_compensation and args.warmup_tolerance_deg <= .25):
        raise ValueError("Warmup needs >0.5 s; integral warmup tolerance must exceed its 0.25 deg deadband")
    kp, kd = _read_motor_cfg(args.motor_config)
    args.kp, args.kd = tuple(args.kp or kp), tuple(args.kd or kd)
    if any(not math.isfinite(value) or not 0 <= value <= bound
           for gains, bound in ((args.kp, 500), (args.kd, 5)) for value in gains):
        raise ValueError("Expected finite kp in [0,500] and kd in [0,5]")
    if args.compensation_model:
        model = LoadFrictionModel.read(args.compensation_model)
        model.validate_identity(args.leg, args.joint)
        if not args.integral_compensation:
            raise ValueError("Model comparison requires --integral_compensation for the same static warmup")
        return model
    return None


def capture(args, ctrl, base_targets, feedforward, commands, feedback, model=None):
    """25 Hz target updates with optional faster resends; log each received frame once."""
    started = time.monotonic()
    next_tx = started
    next_target = started
    target_sequence = -1
    reference = args.center_deg
    target_info = {}
    tx_hz = getattr(args, "tx_hz", None) or args.hz
    if model is None and getattr(args, "compensation_model", None):
        model = LoadFrictionModel.read(args.compensation_model)
        model.validate_identity(args.leg, args.joint)
    dynamic = LoadFrictionController(model, feedforward[args.joint - 1], 1. / tx_hz,
                                    getattr(args, "friction_scale", 0.)) if model else None
    latest_selected = None
    last_compensation_at = started
    received_at = {}
    schedule = PeriodicDeadline(args.poll_hz, clock=time.monotonic, sleep=time.sleep)
    while time.monotonic() - started < args.duration_s - 1.e-9:
        schedule.wait()
        now = time.monotonic()
        if now - started >= args.duration_s - 1.e-9:
            break
        # Equivalent absolute grids can differ by sub-nanosecond float rounding.
        # Do not wait an extra 5 ms poll solely because of that rounding.
        if now >= next_target - 1.e-9:
            target_late = max(0., now - next_target)
            target_skipped = int(target_late * args.hz)
            next_target += (target_skipped + 1) / args.hz
            target_sequence += target_skipped + 1
            reference = target_deg(args, now - started)
            target_info = dict(target_sequence=target_sequence, target_update_s=now - started,
                               target_lateness_ms=target_late * 1000, target_skipped_ticks=target_skipped)
        if now >= next_tx - 1.e-9:
            stale = [motor_id for motor_id in MOTOR_IDS
                     if now - received_at.get(motor_id, started) > args.feedback_timeout_s]
            if stale:
                raise RuntimeError(f"Missing/stale feedback from motors {stale}")
            late = max(0., now - next_tx)
            skipped = int(late * tx_hz)
            next_tx += (skipped + 1) / tx_hz
            positions = list(base_targets)
            positions[args.joint - 1] = math.radians(reference)
            torques = list(feedforward)
            components = _fixed_ff_fields(torques[args.joint - 1])
            if dynamic is not None and latest_selected is not None:
                angle = math.degrees(latest_selected["position"])
                direction = compensation_direction(args, now - started, model)
                state = dynamic.update(angle, direction, now - last_compensation_at)
                torques[args.joint - 1] = state.pop("ff_nm")
                components = state
            last_compensation_at = now
            row = None
            for index, motor_id in enumerate(MOTOR_IDS):
                sent = time.monotonic()
                packet = ctrl.damiao_mit_command(pos=positions[index], vel=0., kp=args.kp[index],
                                               kd=args.kd[index], tor=torques[index], motor_id=motor_id)
                ctrl.send(packet)
                if motor_id == args.joint:
                    row = dict(time_s=sent - started, write_done_s=time.monotonic() - started,
                               lateness_ms=late * 1000, skipped_ticks=skipped, target_deg=reference,
                               ff_nm=torques[index], **target_info, **components)
                    commands.append(row)
                time.sleep(.0002)  # Same per-command spacing as play_real's motor session.
        frames, _, backlog, overflow = ctrl.receive_available()
        if overflow:
            raise RuntimeError("Feedback queue overflow; tracking timestamps are unreliable")
        for frame in frames:
            data = _checked_feedback(ctrl.damiao_feedback(frame))
            if data is None:
                continue
            received = time.monotonic()
            received_at[data["motor_id"]] = received
            if data["motor_id"] == args.joint:
                latest_selected = data
            position = math.degrees(data["position"])
            feedback.append(dict(time_s=received - started, motor_id=data["motor_id"], position_deg=position,
                                 velocity_deg_s=math.degrees(data["velocity"]), torque_nm=data["torque"],
                                 backlog_bytes=backlog))
            if data["motor_id"] == args.joint and not args.min_deg <= position <= args.max_deg:
                raise RuntimeError(f"Measured joint angle {position:.2f} deg exceeded the specified angle bounds")
        now = time.monotonic()
        stale = [motor_id for motor_id in MOTOR_IDS
                 if now - received_at.get(motor_id, started) > args.feedback_timeout_s]
        if stale:
            raise RuntimeError(f"Missing/stale feedback from motors {stale}")


def _cadence(times):
    intervals = np.diff(times)
    return dict(samples=len(times), hz=(len(times) - 1) / (times[-1] - times[0]) if len(times) > 1
                and times[-1] > times[0] else None,
                gap_p95_ms=float(np.percentile(intervals, 95) * 1000) if len(intervals) else None,
                gap_max_ms=float(max(intervals) * 1000) if len(intervals) else None)


def _target_updates(commands):
    if not commands or "target_sequence" not in commands[0]:
        return commands  # Old recordings transmitted once per target update.
    seen, updates = set(), []
    for row in commands:
        if row["target_sequence"] not in seen:
            seen.add(row["target_sequence"])
            updates.append(dict(row, time_s=row["target_update_s"]))
    return updates


def _ramp_analysis(args, commands, samples):
    updates = _target_updates(commands)
    times = [row["time_s"] for row in updates]
    directions = [0] + [int(np.sign(updates[i]["target_deg"] - updates[i-1]["target_deg"]))
                        for i in range(1, len(updates))]
    segments = []
    for index, direction in enumerate(directions):
        if direction and (index == 0 or direction != directions[index - 1]):
            segments.append((index, direction))
    events = []
    for number, (index, direction) in enumerate(segments):
        begin = times[index]
        end = times[segments[number + 1][0]] if number + 1 < len(segments) else args.duration_s
        window = [row for row in samples if begin <= row["time_s"] < end]
        before = [row["position_deg"] for row in samples if begin - .2 <= row["time_s"] < begin]
        baseline = float(np.median(before)) if before else (window[0]["position_deg"] if window else None)
        onset = None
        for sample_index in range(max(0, len(window) - 2)):
            confirmation = window[sample_index:sample_index + 3]
            if (confirmation[-1]["time_s"] - confirmation[0]["time_s"] >= .02 - 1.e-9
                    and all(direction * (row["position_deg"] - baseline) >= .1 for row in confirmation)):
                row = confirmation[0]
                update_index = max(0, bisect.bisect_right(times, row["time_s"]) - 1)
                reference = updates[update_index]["target_deg"]
                onset = dict(time_s=row["time_s"], delay_from_direction_change_ms=(row["time_s"] - begin) * 1000,
                             target_deg=reference, position_deg=row["position_deg"],
                             error_deg=reference - row["position_deg"], torque_nm=row["torque_nm"])
                break
        events.append(dict(direction="increasing" if direction > 0 else "decreasing",
                           segment_start_s=begin, baseline_position_deg=baseline, observed_movement_onset=onset))
    # Compare opposite-direction torque at approximately the same measured angle.
    bins = {}
    for row in samples:
        index = max(0, bisect.bisect_right(times, row["time_s"]) - 1)
        direction = directions[index]
        if direction and direction * row["velocity_deg_s"] >= .5:
            bucket = math.floor(row["position_deg"])
            bins.setdefault(bucket, {1: [], -1: []})[direction].append(row["torque_nm"])
    paired_bins = []
    for bucket, values in sorted(bins.items()):
        if min(len(values[1]), len(values[-1])) >= 3:
            up, down = float(np.median(values[1])), float(np.median(values[-1]))
            paired_bins.append(dict(position_bin_deg=[bucket, bucket + 1], increasing_torque_nm=up,
                                    decreasing_torque_nm=down, torque_difference_nm=up-down,
                                    increasing_samples=len(values[1]), decreasing_samples=len(values[-1])))
    return dict(movement_onsets=events, directional_torque_bins=paired_bins,
                note="Onset requires >=0.1 deg displacement in 3 packets spanning >=20 ms; "
                     "these are observed thresholds, not calibrated friction constants")


def analyze(args, commands, feedback, complete=True, error=None, warmup=None, model_snapshot=None):
    commands = sorted(commands, key=lambda row: row["time_s"])
    updates = _target_updates(commands)
    samples = sorted((row for row in feedback if int(row["motor_id"]) == args.joint),
                     key=lambda row: row["time_s"])
    result = dict(complete=complete, error=error, mode=args.mode, leg=args.leg, joint=args.joint,
                  command_timing="measured" if args.enable_motors else "planned_preview_only",
                  command_cadence=_cadence([row["time_s"] for row in commands]),
                  target_update_cadence=_cadence([row["time_s"] for row in updates]),
                  feedback_cadence=_cadence([row["time_s"] for row in samples]),
                  skipped_command_ticks=sum(row["skipped_ticks"] for row in commands),
                  skipped_target_ticks=sum(row.get("target_skipped_ticks", row["skipped_ticks"]) for row in updates),
                  timing_basis="Host serial-write/receipt timestamps; no device timestamp or command ACK",
                  certified_40ms=False,
                  interpretation="Observed host timing only. Sparse feedback cannot establish sub-cycle settling.",
                  warmup=warmup,
                  tracking=None)
    active = [row for row in commands if row.get("dynamic_active", 0)]
    result["compensation"] = dict(
        mode=("load_friction" if getattr(args, "friction_scale", 1.) else "load_only")
             if getattr(args, "compensation_model", None) else
             "fixed" if args.integral_compensation else "zero_feedforward",
        friction_scale=getattr(args, "friction_scale", 1. if getattr(args, "compensation_model", None) else 0.),
        dynamic_packets=len(active),
        clamped_load_packets=sum(row["load_clamped"] for row in active),
        clamped_load_fraction=sum(row["load_clamped"] for row in active) / len(active) if active else None,
        saturated_packets=sum(row["ff_saturated"] for row in active))
    result["coverage"] = dict(target_fraction=getattr(args, "coverage_target", .75),
                               requested_angle_span_fraction=None, feedback_sample_fraction=None,
                               span_target_met=None, sample_target_met=None,
                               note="Span and received-packet occupancy are different metrics; this experiment does not establish policy-wide coverage")
    model = LoadFrictionModel(model_snapshot) if model_snapshot else None
    if model:
        span = model.span_coverage(args.center_deg - args.amplitude_deg, args.center_deg + args.amplitude_deg)
        sample_fraction = (sum(not model.load_at(row["position_deg"])[1] for row in samples) / len(samples)
                           if samples else None)
        result["coverage"].update(requested_angle_span_fraction=span, feedback_sample_fraction=sample_fraction,
                                  span_target_met=span >= result["coverage"]["target_fraction"],
                                  sample_target_met=sample_fraction >= result["coverage"]["target_fraction"]
                                  if sample_fraction is not None and complete else None)
        result["compensation"]["effective_friction_limit_nm"] = (
            model.friction_torque_nm * result["compensation"]["friction_scale"])
    if not commands or not samples:
        return result
    tx_times = [row["time_s"] for row in commands]
    paired = [(row, max(0, bisect.bisect_right(tx_times, row["time_s"]) - 1)) for row in samples
              if row["time_s"] >= tx_times[0]]
    if not paired:
        return result
    errors = np.array([commands[index]["target_deg"] - row["position_deg"] for row, index in paired])
    result["tracking"] = dict(rmse_deg=float(np.sqrt(np.mean(errors ** 2))),
                              abs_error_p95_deg=float(np.percentile(np.abs(errors), 95)),
                              abs_error_max_deg=float(max(np.abs(errors))), bias_deg=float(np.mean(errors)))
    if model:
        mask = np.array([not model.load_at(row["position_deg"])[1] for row, _ in paired])
        calibrated_errors = errors[mask]
        result["tracking_calibrated_only"] = dict(
            samples=len(calibrated_errors), rmse_deg=float(np.sqrt(np.mean(calibrated_errors ** 2))) if mask.any() else None,
            abs_error_p95_deg=float(np.percentile(np.abs(calibrated_errors), 95)) if mask.any() else None,
            note="Diagnostic subset within the fixed table's measured support; full-trajectory metrics remain primary")
    if args.mode == "step":
        changes = [index for index in range(1, len(commands))
                   if commands[index]["target_deg"] != commands[index - 1]["target_deg"]]
        baseline_end = commands[changes[0]]["time_s"] if changes else args.duration_s
        baseline_samples = [row for row in samples
                            if max(tx_times[0], baseline_end - .2) <= row["time_s"] < baseline_end]
        baseline_error = (float(np.mean([commands[0]["target_deg"] - row["position_deg"]
                                         for row in baseline_samples])) if baseline_samples else None)
        result["initial_baseline"] = dict(error_deg=baseline_error, samples=len(baseline_samples),
                                          source="Last 0.2 s of the initial center hold")
        if baseline_error is not None:
            corrected = errors - baseline_error
            result["baseline_corrected_tracking"] = dict(
                rmse_deg=float(np.sqrt(np.mean(corrected ** 2))),
                note="Initial static offset removed for incremental-response diagnosis only; not absolute accuracy")
        steps = []
        for number, index in enumerate(changes):
            start, target = commands[index]["time_s"], commands[index]["target_deg"]
            end = commands[changes[number + 1]]["time_s"] if number + 1 < len(changes) else args.duration_s
            window = [row for row in samples if start <= row["time_s"] < end]
            bad = [i for i, row in enumerate(window)
                   if abs(target - row["position_deg"]) > args.tolerance_deg or abs(row["velocity_deg_s"]) > 2.]
            first_tail = bad[-1] + 1 if bad else 0
            tail = window[first_tail:]
            stable = len(tail) >= 3 and tail[-1]["time_s"] - tail[0]["time_s"] >= .2
            settling = (tail[0]["time_s"] - start) * 1000 if stable else None
            incremental_settling = None
            if baseline_error is not None:
                incremental_bad = [i for i, row in enumerate(window)
                                   if abs(target - row["position_deg"] - baseline_error) > args.tolerance_deg
                                   or abs(row["velocity_deg_s"]) > 2.]
                incremental_tail = window[incremental_bad[-1] + 1 if incremental_bad else 0:]
                if len(incremental_tail) >= 3 and incremental_tail[-1]["time_s"] - incremental_tail[0]["time_s"] >= .2:
                    incremental_settling = (incremental_tail[0]["time_s"] - start) * 1000
            after_40 = next((row for row in window if row["time_s"] - start >= .04), None)
            direction = math.copysign(1., target - commands[index - 1]["target_deg"])
            steps.append(dict(start_s=start, target_deg=target, observed_settling_ms=settling,
                              last_outside_sample_ms=(window[bad[-1]]["time_s"] - start) * 1000 if bad else None,
                              observed_settling_le_40ms=settling <= 40 if settling is not None else None,
                              baseline_corrected_settling_ms=incremental_settling,
                              first_sample_at_or_after_40ms_ms=(after_40["time_s"] - start) * 1000 if after_40 else None,
                              error_at_that_sample_deg=target - after_40["position_deg"] if after_40 else None,
                              overshoot_deg=max([0.] + [direction * (row["position_deg"] - target) for row in window]),
                              feedback_samples=len(window)))
        result["steps"] = steps
    elif args.mode == "sine":
        # Skip one cycle. Compare sinusoidal fits of sent ZOH targets and measured positions.
        usable = [(row, index) for row, index in paired if row["time_s"] >= 1. / args.frequency_hz]
        if len(usable) >= 12:
            omega = 2 * math.pi * args.frequency_hz
            times = np.array([row["time_s"] for row, _ in usable])
            design = np.column_stack((np.ones(len(times)), np.sin(omega * times), np.cos(omega * times)))
            ref = np.array([commands[index]["target_deg"] for _, index in usable])
            measured = np.array([row["position_deg"] for row, _ in usable])
            sent_fit = np.linalg.lstsq(design, ref, rcond=None)[0]
            measured_fit = np.linalg.lstsq(design, measured, rcond=None)[0]
            sent_amp, measured_amp = float(np.hypot(*sent_fit[1:])), float(np.hypot(*measured_fit[1:]))
            # A flat output still has tiny floating-point sine coefficients. Its
            # atan2 angle is arbitrary and must not be reported as a real delay.
            minimum_amplitude = max(.02, .01 * sent_amp)
            phase_identifiable = sent_amp > 1.e-9 and measured_amp >= minimum_amplitude
            phase = None
            if phase_identifiable:
                phase = math.atan2(measured_fit[2], measured_fit[1]) - math.atan2(sent_fit[2], sent_fit[1])
                phase = (phase + math.pi) % (2 * math.pi) - math.pi
            result["sine_fit"] = dict(frequency_hz=args.frequency_hz,
                                      amplitude_ratio=measured_amp / sent_amp if sent_amp > 1.e-9 else None,
                                      sent_amplitude_deg=sent_amp, measured_amplitude_deg=measured_amp,
                                      measured_peak_to_peak_deg=float(np.ptp(measured)),
                                      phase_identifiable=phase_identifiable,
                                      phase_amplitude_threshold_deg=minimum_amplitude,
                                      phase_lag_deg=-math.degrees(phase) if phase is not None else None,
                                      fitted_lag_ms=-phase / omega * 1000 if phase is not None else None,
                                      fitted_bias_deg=float(measured_fit[0] - sent_fit[0]),
                                      note=("Single-frequency host-observed fit, not a device latency measurement"
                                            if phase_identifiable else
                                            "Output amplitude is below the phase-identification threshold; delay is undefined"))
    else:
        result["ramp"] = _ramp_analysis(args, commands, samples)
    return result


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _report(path, args, commands, feedback, summary):
    """Dependency-free diagnostic HTML; raw CSV remains the source for quantitative analysis."""
    samples = [row for row in feedback if int(row["motor_id"]) == args.joint]
    charts = []
    series = [("Command and measured angle (deg)",
               [("target", "#b45309", [(row["time_s"], row["target_deg"]) for row in commands]),
                ("measured", "#0369a1", [(row["time_s"], row["position_deg"]) for row in samples])]),
              ("Measured torque (N*m)",
               [("torque", "#047857", [(row["time_s"], row["torque_nm"]) for row in samples])])]
    if any(row.get("dynamic_active", 0) for row in commands):
        series.append(("Feedforward (N*m): requested components and applied filtered output", [
            ("requested load", "#7c3aed", [(r["time_s"], r["requested_load_nm"]) for r in commands]),
            ("requested friction", "#b45309", [(r["time_s"], r["requested_friction_nm"]) for r in commands]),
            ("applied feedforward", "#0369a1", [(r["time_s"], r["ff_nm"]) for r in commands])]))
    for title, lines in series:
        points = [point for _, _, data in lines for point in data]
        if not points:
            continue
        xmax = max(max(point[0] for point in points), .1)
        ymin, ymax = min(point[1] for point in points), max(point[1] for point in points)
        margin = max((ymax - ymin) * .1, .1)
        ymin, ymax = ymin - margin, ymax + margin
        svg = ['<svg viewBox="0 0 900 270" role="img">']
        for tick in range(6):
            x, y = 65 + tick * 160, 225 - tick * 40
            svg.append(f'<path d="M{x} 25V225 M65 {y}H865" stroke="#ddd"/>')
            svg.append(f'<text x="{x}" y="248" text-anchor="middle">{xmax * tick / 5:.2f}s</text>')
            svg.append(f'<text x="58" y="{y + 4}" text-anchor="end">{ymin + (ymax-ymin)*tick/5:.2f}</text>')
        for name, color, data in lines:
            if name == "target":
                data = [point for index, (stamp, value) in enumerate(data)
                        for point in ([(stamp, data[index - 1][1]), (stamp, value)] if index else [(stamp, value)])]
            # Downsample display only; the metrics above use every logged sample.
            stride = max(1, len(data) // 4000)
            coords = " ".join(f'{65+t/xmax*800:.2f},{225-(value-ymin)/(ymax-ymin)*200:.2f}'
                              for t, value in data[::stride])
            svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{coords}"/>')
        svg.append('</svg>')
        legend = " / ".join(f'<span style="color:{color}">{name}</span>' for name, color, _ in lines)
        charts.append(f'<h2>{title}</h2><p>{legend}</p>' + "".join(svg))
    path.write_text('<!doctype html><meta charset="utf-8"><title>Joint tracking</title>'
                    '<style>body{font:15px system-ui;max-width:1050px;margin:30px auto;padding:0 20px}'
                    'svg{width:100%}text{font:12px system-ui}pre{white-space:pre-wrap;background:#f4f4f4;padding:16px}</style>'
                    f'<h1>{args.leg} Joint{args.joint}: {args.mode}</h1>'
                    '<p>Host timestamps only. This report does not certify 40 ms motor settling.</p>'
                    '<p><a href="commands.csv">Commands CSV</a> · <a href="feedback.csv">Feedback CSV</a>'
                    ' · <a href="summary.json">Summary JSON</a></p>' + "".join(charts)
                    + '<pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)) + '</pre>',
                    encoding="utf-8")


def save_results(directory, args, commands, feedback, complete, error, warmup=None, model_snapshot=None):
    _write_csv(directory / "commands.csv", COMMAND_FIELDS, commands)
    _write_csv(directory / "feedback.csv", FEEDBACK_FIELDS, feedback)
    summary = analyze(args, commands, feedback, complete, error, warmup, model_snapshot)
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _report(directory / "report.html", args, commands, feedback, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"[INFO] Results: {directory.resolve()}")


def run(args):
    model = _validate(args)  # All validation occurs before opening hardware.
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    tag = ("loadonly_" if args.friction_scale == 0 else f"loadfriction_s{args.friction_scale:g}_") if model else ""
    directory = Path(args.output_dir or f"logs/real_v3/tracking_{args.leg}_j{args.joint}_{args.mode}_{tag}{stamp}")
    directory.mkdir(parents=True, exist_ok=False)
    metadata = dict(settings=vars(args), complete=False, error=None,
                    run_type="hardware" if args.enable_motors else "preview", fixed_feedforward_nm=None,
                    compensation_model_snapshot=model.data if model else None)
    metadata["model_snapshot_sha256"] = (hashlib.sha256(json.dumps(model.data, sort_keys=True, allow_nan=False).encode()).hexdigest()
                                          if model else None)
    if model:
        print(f"[INFO] Experimental {args.leg} Joint{args.joint} load table: "
              f"coverage={model.coverage_intervals()} deg; outside coverage uses nearest endpoint, no extrapolation")
        print(f"[INFO] Feedforward limit={model.feedforward_limit_nm:g} N*m (PD additional), "
              f"slew={model.slew_rate_nm_s:g} N*m/s; selected warmup torque is replaced, not added")
        print(f"[INFO] Fixed table: friction scale={args.friction_scale:g}, "
              f"effective friction <= {args.friction_scale * model.friction_torque_nm:g} N*m; "
              f"coverage target={args.coverage_target:.0%}; no online parameter updates")
    commands, feedback = [], []
    ctrl = None
    try:
        if not args.enable_motors:
            commands = [dict(time_s=index / args.tx_hz, write_done_s=index / args.tx_hz, lateness_ms=0.,
                             skipped_ticks=0, target_deg=target_deg(args, math.floor(index / args.tx_hz * args.hz + 1.e-9) / args.hz), ff_nm=0.,
                             target_sequence=math.floor(index / args.tx_hz * args.hz + 1.e-9),
                             target_update_s=math.floor(index / args.tx_hz * args.hz + 1.e-9) / args.hz,
                             target_lateness_ms=0., target_skipped_ticks=0, **_fixed_ff_fields(0.))
                        for index in range(math.ceil(args.duration_s * args.tx_hz))]
            print("[INFO] PREVIEW ONLY: no serial ports opened; planned commands are not measurements")
        else:
            ctrl = DaMiaoMotorController(args.port or LEG_PORTS[args.leg])
            item = LegController(args.leg, ctrl)
            current = [_enable_and_read_current(ctrl, motor_id) for motor_id in MOTOR_IDS]
            if any(value is None for value in current):
                raise RuntimeError("Missing initial joint feedback")
            targets = _target_rad_for_leg(args.leg, True)
            targets[args.joint - 1] = math.radians(args.center_deg)
            metadata["other_joint_targets_deg"] = [None if index == args.joint - 1 else math.degrees(value)
                                                   for index, value in enumerate(targets)]
            _send_leg_position_targets(item, targets, current, args.kp, args.kd, math.radians(1.), args.hz, .0002)
            compensator = IntegralTorqueCompensator(max_dt_s=1. / args.hz) if args.integral_compensation else None
            warmup = dict(tolerance_deg=args.warmup_tolerance_deg)
            metadata["warmup"] = warmup
            print(f"[INFO] Warmup tolerance={args.warmup_tolerance_deg:g} deg for Joint1-5; "
                  f"dynamic measurement tolerance={args.tolerance_deg:g} deg for Joint{args.joint}")
            reached = _hold_and_verify_leg(item, targets, args.kp, args.kd, args.hz, .0002,
                                           args.warmup_timeout_s, .5, math.radians(args.warmup_tolerance_deg),
                                           math.radians(2.), args.feedback_timeout_s, compensator, result_out=warmup)
            metadata["warmup_reached"] = reached
            if not reached and args.integral_compensation:
                raise RuntimeError(f"Warmup did not settle within {args.warmup_tolerance_deg:g} deg; "
                                   "dynamic test was not started")
            if not reached:
                print("[WARN] PD-only warmup has steady error; tracking metrics will include that bias")
            fixed = compensator.torque.copy() if compensator is not None else [0.] * 5
            metadata["fixed_feedforward_nm"] = fixed
            metadata["warmup_feedforward_nm"] = fixed
            metadata["selected_feedforward_mode"] = ("load_friction_replaces_warmup" if args.friction_scale else
                                                     "load_only_replaces_warmup") if model else "fixed"
            print(f"[INFO] Starting {args.mode}: target updates={args.hz:g} Hz, TX={args.tx_hz:g} Hz, "
                  f"RX polls={args.poll_hz:g} Hz (actual packet rate is measured, not configured), "
                  f"fixed feedforward={fixed}; integral is frozen during motion")
            capture(args, ctrl, targets, fixed, commands, feedback, model=model)
        metadata["complete"] = True
    except BaseException as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if ctrl is not None:
            ctrl.close()  # No extra stop/disable/zero commands.
        (directory / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        save_results(directory, args, commands, feedback,
                     metadata["complete"] and args.enable_motors, metadata["error"], metadata.get("warmup"),
                     metadata.get("compensation_model_snapshot"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="operation", required=True)
    start = subs.add_parser("run", help="Preview or execute one-leg Joint1-5 tracking measurements")
    start.add_argument("--leg", choices=DEFAULT_LEG_ORDER, default="LB")
    start.add_argument("--joint", type=int, choices=MOTOR_IDS, default=1)
    start.add_argument("--port", help="Override serial path for this leg")
    for name in ("center_deg", "min_deg", "max_deg"):
        start.add_argument(f"--{name}", type=float, required=True)
    start.add_argument("--amplitude_deg", type=float, default=2.)
    start.add_argument("--mode", choices=("step", "sine", "ramp"), default="step")
    start.add_argument("--ramp_speed_deg_s", type=float, default=1.)
    start.add_argument("--frequency_hz", type=float, default=.3)
    start.add_argument("--hold_s", type=float, default=1.)
    start.add_argument("--duration_s", type=float, default=12.)
    start.add_argument("--hz", type=float, default=25.)
    start.add_argument("--tx_hz", type=float, default=None,
                       help="MIT resend rate, default equals --hz. 100 repeats each 25 Hz target four times.")
    start.add_argument("--poll_hz", type=float, default=200.,
                       help="Host receive polling rate only; does not configure motor feedback generation.")
    start.add_argument("--tolerance_deg", type=float, default=.5)
    start.add_argument("--feedback_timeout_s", type=float, default=.2)
    start.add_argument("--warmup_timeout_s", type=float, default=20.)
    start.add_argument("--warmup_tolerance_deg", type=float, default=1.,
                       help="Joint1-5 static warmup tolerance; independent of dynamic --tolerance_deg.")
    start.add_argument("--motor_config", default=str(THIS_DIR / "damiao.yaml"))
    start.add_argument("--kp", type=float, nargs=5)
    start.add_argument("--kd", type=float, nargs=5)
    start.add_argument("--integral_compensation", action="store_true", help="Learn static feedforward during warmup, then freeze it")
    start.add_argument("--compensation_model", help="Experimental load/friction JSON; replaces selected joint's fixed torque")
    start.add_argument("--friction_scale", type=float, default=0.,
                       help="Fixed friction ablation scale in [0,1]; default 0 applies load table only.")
    start.add_argument("--coverage_target", type=float, default=.75,
                       help="Reporting target for calibrated angle-span and sample coverage; does not expand motion bounds.")
    start.add_argument("--enable_motors", action="store_true", help="Open serial port, enable Joint1-5 and send commands")
    start.add_argument("--output_dir")
    offline = subs.add_parser("analyze", help="Recompute a saved measurement without opening hardware")
    offline.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.operation == "run":
        try:
            run(args)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        metadata = json.loads((args.directory / "run.json").read_text())
        settings = argparse.Namespace(**metadata["settings"])
        rows = []
        for name in ("commands", "feedback"):
            with (args.directory / f"{name}.csv").open() as stream:
                rows.append([{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)])
        save_results(args.directory, settings, *rows,
                     metadata["complete"] and metadata["run_type"] == "hardware", metadata["error"], metadata.get("warmup"),
                     metadata.get("compensation_model_snapshot"))


if __name__ == "__main__":
    main()
