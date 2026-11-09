"""Static calibration with fake serial/clock; no simulator or motor access."""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import math
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def static(monkeypatch):
    real = Path(__file__).parents[1] / "isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3"
    monkeypatch.syspath_prepend(str(real))
    spec = importlib.util.spec_from_file_location("static_load_test", real / "calibrate_joint_load.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def args_for(module, *extra):
    args = module.build_parser().parse_args(["--joint", "1", "--min_deg", "45", "--max_deg", "65", *extra])
    module.validate(args)
    return args


def clock_for(module):
    clock = types.SimpleNamespace(now=100.)
    module.time = types.SimpleNamespace(monotonic=lambda: clock.now,
                                       sleep=lambda dt: setattr(clock, "now", clock.now + dt))
    return clock


class FakeSerial:
    def __init__(self, missing=(), offsets=None, velocities=None):
        self.missing, self.offsets = missing, offsets or {}
        self.velocities = velocities or {}
        self.commands, self.pending = [], []
        self.closed = False

    def damiao_mit_command(self, **kwargs):
        return kwargs

    def send(self, command):
        self.commands.append(command.copy())
        motor = command["motor_id"]
        if motor not in self.missing:
            self.pending.append(dict(motor_id=motor, error_code=1,
                                     position=command["pos"] + math.radians(self.offsets.get(motor, 0.)),
                                     velocity=math.radians(self.velocities.get(motor, 0.)), torque=2. + command["tor"]))

    def receive_available(self):
        result, self.pending = self.pending, []
        return result, 0, 0, False

    def damiao_feedback(self, frame):
        return frame

    def close(self):
        self.closed = True


def session_for(module, args, ctrl):
    streams = [io.StringIO(), io.StringIO()]
    session = module.Session(args, ctrl, csv.DictWriter(streams[0], module.COMMAND_FIELDS),
                             csv.DictWriter(streams[1], module.FEEDBACK_FIELDS))
    return session, streams


@pytest.mark.parametrize("step", [1.5, 2., 2.5])
def test_plan_matches_angles_both_directions_and_returns_to_default(static, step):
    plan = static.make_plan(65., 45., 65., step, 2)
    assert plan[0]["target_deg"] == plan[-1]["target_deg"] == 65.
    down = {p["target_deg"] for p in plan if p["direction"] == -1}
    up = {p["target_deg"] for p in plan if p["direction"] == 1}
    assert down - up == {45.}
    assert up - down == {65.}
    assert all(0 < abs(b["target_deg"] - a["target_deg"]) <= step + 1.e-8 for a, b in zip(plan, plan[1:]))
    assert all(45 <= p["target_deg"] <= 65 for p in plan)
    if step == 2:
        assert len(plan) == 41


def test_plan_with_initial_inside_range_starts_down_and_completes_repeats(static):
    plan = static.make_plan(65., 45., 85., 2.5, 3)
    assert [p["target_deg"] for p in plan[:3]] == [65., 62.5, 60.]
    assert plan[-1]["target_deg"] == 65
    assert sum(p["target_deg"] == 45 for p in plan) == 3
    assert sum(p["target_deg"] == 85 for p in plan) == 3


@pytest.mark.parametrize("joint,low,high", [(1, 35, 95), (2, 15, 75), (3, -50, 10), (4, -6, 54), (5, -54, 6)])
def test_requested_all_joint_ranges_have_three_visits_per_direction_at_internal_angles(static, joint, low, high):
    args = args_for(static, "--joint", str(joint), "--min_deg", str(low), "--max_deg", str(high), "--cycles", "3")
    plan = static.validate(args)
    assert len(plan) == 181
    assert plan[0]["target_deg"] == plan[-1]["target_deg"] == (low + high) / 2
    for angle in {p["target_deg"] for p in plan} - {low, high}:
        for direction in (-1, 1):
            assert sum(p["target_deg"] == angle and p["direction"] == direction for p in plan) == 3


@pytest.mark.parametrize("joint,low,high", [(4, -6, 24), (5, -24, 6)])
def test_degree_radian_roundoff_does_not_reject_default_at_endpoint(static, joint, low, high):
    args_for(static, "--joint", str(joint), "--min_deg", str(low), "--max_deg", str(high))


def test_preview_does_not_open_serial_or_create_logs(static, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Preview opened hardware")
    monkeypatch.setattr(static, "DaMiaoMotorController", forbidden)
    out = tmp_path / "preview"
    args = args_for(static, "--output_dir", str(out))
    result = static.run(args)
    assert result["visits"] == 41
    assert not out.exists()


def test_requested_position_tolerances_and_gains_with_frozen_sampling(static):
    args = args_for(static)
    clock_for(static)
    ctrl = FakeSerial(offsets={1: .7, 2: 4.5})
    session, streams = session_for(static, args, ctrl)
    point = dict(visit=0, target_deg=65., direction=0)
    static.measure_point(session, point, static._target_rad_for_leg("LB", True))
    assert point["valid"]
    assert point["position_deg_by_motor"]["1"]["median"] == pytest.approx(65.7)
    assert point["position_deg_by_motor"]["2"]["median"] == pytest.approx(49.5)
    commands = list(csv.DictReader(io.StringIO(
        ",".join(static.COMMAND_FIELDS) + "\n" + streams[0].getvalue())))
    sampled = [r for r in commands if r["phase"] == "sample" and r["motor_id"] == "2"]
    assert len(sampled) >= args.min_samples
    assert len({r["ff_nm"] for r in sampled}) == 1
    assert point["selected_torque_nm"]["median"] == pytest.approx(2. + point["frozen_integral_ff_nm"][0])
    assert {c["motor_id"] for c in ctrl.commands} == {1, 2, 3, 4, 5}
    assert tuple(c["kp"] for c in ctrl.commands[:5]) == (30., 25., 20., 20., 2.)


def test_noisy_reported_velocity_does_not_block_stationary_positions_or_integral(static):
    args = args_for(static)
    clock_for(static)
    ctrl = FakeSerial(offsets={2: .7}, velocities={2: 20., 5: -10.})
    session, _ = session_for(static, args, ctrl)
    point = dict(visit=0, target_deg=65., direction=0)
    static.measure_point(session, point, static._target_rad_for_leg("LB", True))
    assert point["valid"]
    assert point["max_abs_velocity_deg_s"] == 20.
    assert point["frozen_integral_ff_nm"][1] < 0  # Position-derived motion permits correcting the static error.
    assert all(span == 0 for span in point["position_span_deg_by_motor"].values())


@pytest.mark.parametrize("limit", [4., 7.])
def test_configured_integral_limit_caps_actual_commands_and_can_exceed_old_four_nm_cap(static, limit):
    args = args_for(static, "--integral_torque_limit", str(limit), "1.5", "1.5", "1.5", "0.2")
    clock = clock_for(static)
    ctrl = FakeSerial(offsets={1: -1.})
    session, _ = session_for(static, args, ctrl)
    session.integral.torque[0] = limit - .01
    targets = static._target_rad_for_leg("LB", True)
    started = clock.now
    while clock.now - started < 1.:
        session.tick(targets, 0, "settle", True)
    sent = [c["tor"] for c in ctrl.commands if c["motor_id"] == 1]
    assert max(sent) == pytest.approx(limit)
    assert all(0 <= value <= limit for value in sent)
    assert session.integral.torque_rate == (.3, .15, .15, .15, .02)
    if limit == 7.:
        assert max(sent) > 4.


def test_small_motor_default_integral_limit_clamps_sent_j5_feedforward_at_one_nm(static):
    args = args_for(static)
    clock = clock_for(static)
    ctrl = FakeSerial(offsets={5: -2.})
    session, _ = session_for(static, args, ctrl)
    session.integral.torque[4] = .999
    started = clock.now
    while clock.now - started < 1.:
        session.tick(static._target_rad_for_leg("LB", True), 0, "settle", True)
    j5_sent = [c["tor"] for c in ctrl.commands if c["motor_id"] == 5]
    assert max(j5_sent) == pytest.approx(1.)
    assert all(0 <= value <= 1. for value in j5_sent)
    assert session.integral.torque_limit == (7., 4.5, 3.5, 2.5, 1.)


def test_reported_velocity_can_be_required_explicitly_and_timeout_explains_blocker(static):
    args = args_for(static, "--point_timeout_s", "3", "--require_velocity_feedback")
    clock_for(static)
    session, _ = session_for(static, args, FakeSerial(velocities={5: 3.}))
    point = dict(visit=0, target_deg=65.)
    with pytest.raises(RuntimeError, match="J5:reported_velocity"):
        static.measure_point(session, point, static._target_rad_for_leg("LB", True))
    assert not point["valid"] and point["blocker_counts"]["J5:reported_velocity"] > 0


def test_position_oscillation_is_rejected_even_with_zero_reported_velocity(static):
    args = args_for(static, "--point_timeout_s", "3")
    clock = clock_for(static)
    ctrl = FakeSerial()
    send = ctrl.send

    def moving(command):
        ctrl.offsets[1] = .2 * math.sin(2 * math.pi * 4 * clock.now)
        send(command)

    ctrl.send = moving
    session, _ = session_for(static, args, ctrl)
    point = dict(visit=0, target_deg=65.)
    with pytest.raises(RuntimeError, match="J1:position_unstable"):
        static.measure_point(session, point, static._target_rad_for_leg("LB", True))
    assert not point["valid"]
    assert point["last_diagnostics"]["motors"]["1"]["reported_velocity_deg_s"] == 0
    assert "J1:position_error" not in point["blocker_counts"]  # Absolute tolerance alone would have passed.


def test_slow_creep_cannot_pass_a_sequence_of_quiet_short_windows(static):
    args = args_for(static, "--point_timeout_s", "3")
    clock = clock_for(static)
    ctrl = FakeSerial()
    send, started = ctrl.send, clock.now

    def moving(command):
        ctrl.offsets[2] = .25 * (clock.now - started)
        send(command)

    ctrl.send = moving
    session, _ = session_for(static, args, ctrl)
    point = dict(visit=0, target_deg=65.)
    with pytest.raises(RuntimeError, match="sample_position_drift"):
        static.measure_point(session, point, static._target_rad_for_leg("LB", True))
    assert not point["valid"] and point["rejected_windows"] > 0
    assert point["blocker_counts"]["J2:sample_position_drift"] > 0
    assert "J2:position_unstable" not in point["blocker_counts"]
    assert "J2:position_error" not in point["blocker_counts"]


def test_invalid_sampling_window_is_rejected_then_retried_without_integral_updates_in_sample(static):
    args = args_for(static)
    clock = clock_for(static)
    ctrl = FakeSerial()
    session, _ = session_for(static, args, ctrl)
    original_tick = session.tick
    disturbed, first_frozen = False, None

    def tick(targets, visit, phase, integrate):
        nonlocal disturbed, first_frozen
        if phase == "sample":
            assert integrate is False
            if not disturbed:
                first_frozen = clock.now
                disturbed = True
        if disturbed:
            # Keep the disturbance through at least one 100 Hz TX, including intervening RX polls.
            ctrl.offsets[1] = .9 if clock.now - first_frozen < .02 else 0.
        return original_tick(targets, visit, phase, integrate)

    session.tick = tick
    point = dict(visit=0, target_deg=65., direction=0)
    static.measure_point(session, point, static._target_rad_for_leg("LB", True))
    assert point["valid"] and point["rejected_windows"] == 1
    assert point["sample_start_s"] > first_frozen - session.started + args.stable_s
    assert point["selected_position_deg"]["maximum"] == pytest.approx(65.)


def test_missing_feedback_and_monitor_bound_crossing_abort(static):
    args = args_for(static)
    clock_for(static)
    session, _ = session_for(static, args, FakeSerial(missing=(3,)))
    with pytest.raises(RuntimeError, match="stale.*3"):
        static.measure_point(session, dict(visit=0, target_deg=65), static._target_rad_for_leg("LB", True))
    clock_for(static)
    session, _ = session_for(static, args, FakeSerial(offsets={1: 2.}))
    with pytest.raises(RuntimeError, match="monitoring bounds"):
        static.measure_point(session, dict(visit=0, target_deg=65), static._target_rad_for_leg("LB", True))


def test_failed_run_preserves_raw_data_invalid_visit_and_closes_only(static, monkeypatch, tmp_path):
    args = args_for(static, "--enable_motors", "--output_dir", str(tmp_path / "run"))
    clock_for(static)
    ctrl = FakeSerial(missing=(3,))
    monkeypatch.setattr(static, "DaMiaoMotorController", lambda port: ctrl)
    monkeypatch.setattr(static, "_enable_and_read_current",
                        lambda ctrl, motor: static._target_rad_for_leg("LB", True)[motor - 1])
    with pytest.raises(RuntimeError, match="stale"):
        static.run(args)
    metadata = json.loads((tmp_path / "run/run.json").read_text())
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert not metadata["complete"] and not summary["complete"]
    assert metadata["points"] == []
    assert metadata["initialization"]["status"] == "failed"
    assert metadata["initialization"]["pose_verification"]["status"] == "interrupted"
    assert summary["valid_visits"] == 0
    assert (tmp_path / "run/feedback.csv").stat().st_size > 300
    assert ctrl.closed and all(isinstance(c, dict) for c in ctrl.commands)


def test_complete_run_captures_both_directions_and_finishes_at_default_pose(static, monkeypatch, tmp_path):
    args = args_for(static, "--enable_motors", "--cycles", "1", "--output_dir", str(tmp_path / "run"),
                    "--integral_torque_limit", "5.5", "1.5", "1.5", "1.5", "0.2")
    clock_for(static)
    ctrl = FakeSerial()
    monkeypatch.setattr(static, "DaMiaoMotorController", lambda port: ctrl)
    # Start two degrees below default, exercising the initial ramp before static holds.
    monkeypatch.setattr(static, "_enable_and_read_current", lambda ctrl, motor:
                        static._target_rad_for_leg("LB", True)[motor - 1] - math.radians(2.))
    static.run(args)
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["complete"] and summary["valid_visits"] == 21
    assert sum(p["bidirectional"] for p in summary["angle_summary"]) == 9
    assert summary["command_cadence"]["hz"] == pytest.approx(100., abs=.05)
    assert summary["feedback_cadence_by_motor"]["1"]["hz"] == pytest.approx(100., abs=.05)
    assert summary["skipped_command_ticks"] == 0
    metadata = json.loads((tmp_path / "run/run.json").read_text())
    assert metadata["integral_settings"]["torque_limit"] == [5.5, 1.5, 1.5, 1.5, .2]
    assert metadata["procedure"]["integral_settings"]["torque_limit"] == [5.5, 1.5, 1.5, 1.5, .2]
    assert [c["pos"] for c in ctrl.commands[-5:]] == static._target_rad_for_leg("LB", True)
    assert ctrl.closed


def test_initialize_from_outside_scan_bounds_then_verify_before_scan_without_reopening_serial(static, monkeypatch, tmp_path):
    args = args_for(static, "--min_deg", "63", "--max_deg", "65", "--cycles", "1",
                    "--enable_motors", "--output_dir", str(tmp_path / "run"))
    clock_for(static)
    ctrl = FakeSerial()
    opened = []

    def open_controller(port):
        opened.append(port)
        return ctrl

    monkeypatch.setattr(static, "DaMiaoMotorController", open_controller)
    monkeypatch.setattr(static, "_enable_and_read_current", lambda ctrl, motor:
                        math.radians(55.) if motor == 1 else static._target_rad_for_leg("LB", True)[motor - 1])
    static.run(args)
    metadata = json.loads((tmp_path / "run/run.json").read_text())
    assert metadata["complete"] and metadata["initialization"]["status"] == "complete"
    assert metadata["initialization"]["measured_start_deg"][0] == pytest.approx(55.)
    assert metadata["initialization"]["pose_verification"]["valid"]
    assert len(opened) == 1
    assert [p["target_deg"] for p in metadata["points"]] == [65., 63., 65.]
    with (tmp_path / "run/commands.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    first_scan = next(i for i, row in enumerate(rows) if row["visit"] == "0")
    assert any(row["phase"] == "initial_ramp" for row in rows[:first_scan])
    assert any(row["phase"] == "initialization_sample" for row in rows[:first_scan])
    assert min(float(row["target_deg"]) for row in rows[:first_scan] if row["motor_id"] == "1") == pytest.approx(55.)


def test_initialization_outside_its_own_bounds_logs_actual_pose_before_any_position_command(static, monkeypatch, tmp_path):
    args = args_for(static, "--enable_motors", "--output_dir", str(tmp_path / "run"))
    clock_for(static)
    ctrl = FakeSerial()
    monkeypatch.setattr(static, "DaMiaoMotorController", lambda port: ctrl)
    monkeypatch.setattr(static, "_enable_and_read_current", lambda ctrl, motor:
                        math.radians(20.) if motor == 1 else static._target_rad_for_leg("LB", True)[motor - 1])
    with pytest.raises(RuntimeError, match="J1=20.000"):
        static.run(args)
    metadata = json.loads((tmp_path / "run/run.json").read_text())
    assert metadata["initialization"]["status"] == "failed"
    assert metadata["initialization"]["measured_start_deg"][0] == pytest.approx(20.)
    assert not metadata["points"] and not ctrl.commands
    assert ctrl.closed


def test_narrow_scan_monitoring_is_restored_after_initialization(static, monkeypatch, tmp_path):
    args = args_for(static, "--enable_motors", "--output_dir", str(tmp_path / "run"))
    clock_for(static)
    ctrl = FakeSerial()
    monkeypatch.setattr(static, "DaMiaoMotorController", lambda port: ctrl)
    monkeypatch.setattr(static, "_enable_and_read_current", lambda ctrl, motor:
                        static._target_rad_for_leg("LB", True)[motor - 1])
    measure = static.measure_point

    def disturb_after_initialization(*args, **kwargs):
        measure(*args, **kwargs)
        if kwargs.get("phase_prefix") == "initialization_":
            ctrl.offsets[1] = 2.  # Allowed in initialization band, disallowed in scan band.

    monkeypatch.setattr(static, "measure_point", disturb_after_initialization)
    with pytest.raises(RuntimeError, match="exceeded monitoring bounds.*during settle"):
        static.run(args)
    metadata = json.loads((tmp_path / "run/run.json").read_text())
    assert metadata["initialization"]["status"] == "complete"
    assert metadata["points"][0]["status"] == "interrupted"


def test_summary_is_equal_weight_per_visit_and_does_not_label_hysteresis_as_friction(static):
    points = []
    for visit, (direction, value) in enumerate([(-1, 1.), (-1, 1.), (-1, 2.), (1, 3.)]):
        points.append(dict(visit=visit, valid=True, direction=direction, target_deg=55.,
                           selected_position_deg=static.distribution([55.]),
                           selected_torque_nm=static.distribution([value])))
    points.append(dict(visit=4, valid=False, direction=1, target_deg=57.))
    result, = static.summarize_points(points)
    assert result["bidirectional"] and result["load_midpoint_nm"] == 2.
    assert result["signed_half_difference_nm"] == 1.
    assert result["branches"]["decreasing"]["possible_outlier_visits"] == [2]
    assert len(points) == 5  # No destructive outlier removal.


def test_approach_tracker_detects_reversal_without_treating_encoder_jitter_as_motion(static):
    tracker = static.ApproachTracker(65.)
    tracker.observe(65.022, 0.)
    assert tracker.direction == 0
    tracker.observe(64., 1.)
    tracker.observe(62.303, 2.)
    tracker.observe(62.325, 3.)
    assert tracker.direction == -1 and not tracker.reversals
    tracker.observe(62.45, 4.)
    assert tracker.direction == 1
    assert tracker.reversals == [dict(time_s=4., position_deg=62.45, from_direction=-1, to_direction=1)]


def test_summary_groups_actual_arrival_after_overshoot_not_original_command_direction(static):
    points = [dict(visit=i, valid=True, direction=-1, target_deg=63., observed_approach_direction=direction,
                   selected_position_deg=static.distribution([angle]),
                   selected_torque_nm=static.distribution([torque]))
              for i, (direction, angle, torque) in enumerate([(-1, 63.2, 1.), (1, 62.8, 3.), (0, 63., 9.)])]
    result, = static.summarize_points(points)
    assert result["bidirectional"] and result["load_midpoint_nm"] == 2.
    assert result["branches"]["increasing"]["visits"] == [1]
    assert result["branches"]["decreasing"]["visits"] == [0]
    assert result["branches"]["increasing"]["direction_basis"] == ["observed_position"]


@pytest.fixture
def batch(static, monkeypatch):
    monkeypatch.setitem(sys.modules, "calibrate_joint_load", static)
    spec = importlib.util.spec_from_file_location("static_leg_load_test", Path(static.__file__).with_name("calibrate_leg_load.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_five_round_batch_preview_has_joint_order_ranges_and_no_hardware_or_logs(batch, static, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Preview opened hardware")

    monkeypatch.setattr(static, "DaMiaoMotorController", forbidden)
    output = tmp_path / "batch"
    args = batch.build_parser().parse_args(["--output_dir", str(output)])
    preview = batch.run(args)
    assert preview["scans"] == 25 and preview["planned_visits"] == 1525
    assert preview["all_joint_return_tolerance_deg"] == 2.
    assert [(entry["round"], entry["joint"]) for entry in preview["plan"]] == [
        (repeat, joint) for repeat in range(1, 6) for joint in range(1, 6)]
    assert [(entry["min_deg"], entry["max_deg"]) for entry in preview["plan"][:5]] == [
        (35, 95), (15, 75), (-50, 10), (-6, 54), (-54, 6)]
    assert not output.exists()


def test_batch_verifies_whole_leg_before_each_next_joint_and_repeats_rounds(batch, static, monkeypatch, tmp_path):
    args = batch.build_parser().parse_args(["--rounds", "2", "--range_deg", "2", "--enable_motors",
                                            "--output_dir", str(tmp_path / "batch")])
    clock_for(static)
    controllers, returns = [], []
    measure = static.measure_point

    def open_controller(port):
        assert len(returns) == len(controllers)  # Previous run must have verified its full return.
        ctrl = FakeSerial(offsets={2: 1.5})  # Accepted by the relaxed 2 deg return threshold.
        controllers.append(ctrl)
        return ctrl

    def measured(session, point, targets, **kwargs):
        measure(session, point, targets, **kwargs)
        if kwargs.get("phase_prefix") == "return_":
            assert point["valid"]
            assert kwargs["tolerance_deg"] == 2.
            assert all(abs(point["position_deg_by_motor"][str(motor)]["median"] - math.degrees(targets[motor - 1])) <= 2.
                       for motor in static.MOTOR_IDS)
            returns.append(session.args.joint)

    # J2 is the selected scan joint in its own run, so remove the offset there.
    def read_current(ctrl, motor):
        if len(controllers) % 5 == 2:
            ctrl.offsets = {}
        return static._target_rad_for_leg("LB", True)[motor - 1]

    monkeypatch.setattr(static, "DaMiaoMotorController", open_controller)
    monkeypatch.setattr(static, "_enable_and_read_current", read_current)
    monkeypatch.setattr(static, "measure_point", measured)
    directory = batch.run(args)
    metadata = json.loads((directory / "batch.json").read_text())
    assert metadata["complete"] and len(controllers) == 10
    assert returns == [1, 2, 3, 4, 5] * 2
    assert all(ctrl.closed for ctrl in controllers)
    assert all(entry["status"] == "complete" for entry in metadata["runs"])
    with (directory / "round_01_j1/commands.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert any(row["phase"] == "return_sample" and row["visit"] == "-2" for row in rows)


def test_failed_whole_leg_return_preserves_data_and_prevents_next_joint(batch, static, monkeypatch, tmp_path):
    args = batch.build_parser().parse_args(["--rounds", "5", "--range_deg", "2", "--point_timeout_s", "3",
                                            "--enable_motors", "--output_dir", str(tmp_path / "batch")])
    clock_for(static)
    opened = []

    def open_controller(port):
        ctrl = FakeSerial(offsets={2: 2.5})  # Within scan's 5 deg, outside return's 2 deg.
        opened.append(ctrl)
        return ctrl

    monkeypatch.setattr(static, "DaMiaoMotorController", open_controller)
    monkeypatch.setattr(static, "_enable_and_read_current", lambda ctrl, motor:
                        static._target_rad_for_leg("LB", True)[motor - 1])
    with pytest.raises(RuntimeError, match="J2:position_error"):
        batch.run(args)
    directory = Path(args.output_dir)
    metadata = json.loads((directory / "batch.json").read_text())
    single = json.loads((directory / "round_01_j1/run.json").read_text())
    assert len(opened) == 1 and opened[0].closed
    assert not metadata["complete"] and metadata["runs"][0]["status"] == "failed"
    assert single["return_to_initial"]["status"] == "failed"
    assert not single["return_to_initial"]["pose_verification"]["valid"]
    assert all(point["valid"] for point in single["points"])  # Scan succeeded; return is a separate gate.
    assert not (directory / "round_01_j2").exists()


@pytest.mark.parametrize("extra", [["--rounds", "0"], ["--range_deg", "nan"], ["--range_deg", "1000"],
                                   ["--return_tolerance_deg", "nan"], ["--return_tolerance_deg", "0.2"]])
def test_invalid_batch_rejected_before_hardware_or_log_creation(batch, static, monkeypatch, tmp_path, extra):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid batch opened hardware")

    monkeypatch.setattr(static, "DaMiaoMotorController", forbidden)
    output = tmp_path / "batch"
    args = batch.build_parser().parse_args(["--enable_motors", "--output_dir", str(output), *extra])
    with pytest.raises(ValueError):
        batch.run(args)
    assert not output.exists()


def test_resume_skips_four_verified_scans_retries_j5_and_finishes_second_round(batch, static, monkeypatch, tmp_path):
    directory = tmp_path / "batch"
    args = batch.build_parser().parse_args(["--rounds", "2", "--range_deg", "2", "--point_timeout_s", "90",
                                            "--enable_motors", "--output_dir", str(directory)])
    clock_for(static)
    opened = []

    def initial_controller(port):
        ctrl = FakeSerial(missing=(5,) if len(opened) == 4 else ())
        opened.append(ctrl)
        return ctrl

    monkeypatch.setattr(static, "DaMiaoMotorController", initial_controller)
    monkeypatch.setattr(static, "_enable_and_read_current", lambda ctrl, motor:
                        static._target_rad_for_leg("LB", True)[motor - 1])
    with pytest.raises(RuntimeError, match="stale"):
        batch.run(args)
    failed_bytes = (directory / "round_01_j5/run.json").read_bytes()
    batch_bytes = (directory / "batch.json").read_bytes()
    successful_bytes = (directory / "round_01_j1/run.json").read_bytes()
    resume_args = batch.build_parser().parse_args(["--resume_batch", str(directory)])
    preview = batch.run(resume_args)
    assert preview["skipped_complete_scans"] == 4 and preview["pending_scans"] == 6
    assert preview["plan"][0]["directory"] == "round_01_j5_retry_01"
    assert len(opened) == 5 and (directory / "batch.json").read_bytes() == batch_bytes

    def clean_controller(port):
        ctrl = FakeSerial()
        opened.append(ctrl)
        return ctrl

    monkeypatch.setattr(static, "DaMiaoMotorController", clean_controller)
    resume_args.enable_motors = True
    batch.run(resume_args)
    result = json.loads((directory / "batch.json").read_text())
    assert result["complete"] and len(result["runs"]) == 10 and len(opened) == 11
    assert result["runs"][4]["directory"] == "round_01_j5_retry_01"
    assert result["runs"][4]["previous_attempts"][0]["directory"] == "round_01_j5"
    assert all(record["status"] == "complete" for record in result["runs"])
    assert (directory / "round_01_j5/run.json").read_bytes() == failed_bytes
    assert (directory / "round_01_j1/run.json").read_bytes() == successful_bytes
    retry = json.loads((directory / "round_01_j5_retry_01/run.json").read_text())
    assert retry["settings"]["point_timeout_s"] == 90.
    assert result["resume_history"][-1]["status"] == "complete"
    assert len(list(directory.glob("batch_before_resume_*.json"))) == 1


@pytest.mark.parametrize("extra", [["--min_deg", "66"], ["--step_deg", "0"],
                                   ["--sample_s", "0.1"], ["--cycles", "4"],
                                   ["--integral_torque_limit", "29", "1.5", "1.5", "1.5", ".2"],
                                   ["--integral_torque_limit", "nan", "1.5", "1.5", "1.5", ".2"],
                                   ["--integral_torque_limit", "-1", "1.5", "1.5", "1.5", ".2"]])
def test_invalid_plan_or_impossible_sampling_rejected_before_hardware(static, extra):
    with pytest.raises(ValueError):
        args_for(static, *extra)


@pytest.fixture
def merger(static, monkeypatch):
    monkeypatch.setitem(sys.modules, "calibrate_joint_load", static)
    spec = importlib.util.spec_from_file_location("static_load_merge_test", Path(static.__file__).with_name("merge_static_load_batches.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_batch(static, directory, rounds, timeout=30.):
    directory.mkdir()
    entries = []
    for repeat in range(1, rounds + 1):
        for joint in static.MOTOR_IDS:
            initial = round(math.degrees(static._target_rad_for_leg("LB", True)[joint - 1]), 10)
            plan = static.make_plan(initial, initial - 2, initial + 2, 2, 1)
            points = [dict(p, valid=True, observed_approach_direction=p["direction"],
                           selected_position_deg=static.distribution([p["target_deg"]]),
                           selected_torque_nm=static.distribution([float(repeat)])) for p in plan]
            name = f"round_{repeat:02d}_j{joint}"
            path = directory / name
            path.mkdir()
            metadata = dict(complete=True, return_to_initial=dict(status="complete"),
                            settings=dict(leg="LB", joint=joint, kp=[35, 25, 25, 25, 2], point_timeout_s=timeout),
                            procedure=dict(plan=plan), points=points)
            (path / "run.json").write_text(json.dumps(metadata))
            (path / "summary.json").write_text(json.dumps(dict(complete=True, return_to_initial_status="complete")))
            entries.append(dict(round=repeat, joint=joint, status="complete", directory=name))
    (directory / "batch.json").write_text(json.dumps(dict(procedure=dict(plan=entries), runs=entries)))
    return directory


def test_merge_three_plus_two_rounds_excludes_failed_fourth_and_preserves_sources(merger, static, tmp_path):
    old = source_batch(static, tmp_path / "old", 3)
    new = source_batch(static, tmp_path / "new", 2, timeout=90.)
    original = json.loads((old / "batch.json").read_text())
    original["runs"].append(dict(round=4, joint=1, status="failed", directory="round_04_j1"))
    (old / "batch.json").write_text(json.dumps(original))
    original_bytes = (old / "batch.json").read_bytes()
    directory = merger.merge([old, new], tmp_path / "merged")
    result = json.loads((directory / "batch.json").read_text())
    assert result["complete"] and result["procedure"]["rounds"] == 5
    assert len(result["runs"]) == 25 and len(result["excluded_rounds"]) == 1
    assert result["excluded_rounds"][0]["source_round"] == 4
    assert (directory / "round_04_j1").resolve() == new / "round_01_j1"
    assert (directory / "round_05_j5").resolve() == new / "round_02_j5"
    assert (old / "batch.json").read_bytes() == original_bytes
    assert result["runs"][15]["point_timeout_s"] == 90.
    with (directory / "points.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["joint"] == "1"]
    assert [int(r["visit"]) for r in rows] == list(range(25))
    assert {r["source_round"] for r in rows if r["round"] == "4"} == {"1"}
    summary = json.loads((directory / "summary_j1.json").read_text())
    assert summary["valid_visits"] == 25
    bidirectional = [row for row in summary["angle_summary"] if row["bidirectional"]]
    assert len(bidirectional) == 1
    assert len(bidirectional[0]["branches"]["decreasing"]["visits"]) == 5


@pytest.mark.parametrize("problem", ["missing_rounds", "duplicate_batch", "duplicate_scan", "gain_mismatch", "failed_return"])
def test_merge_rejects_incomplete_duplicate_or_incompatible_data_before_output(merger, static, tmp_path, problem):
    old = source_batch(static, tmp_path / "old", 3)
    new = source_batch(static, tmp_path / "new", 2)
    batches = [old, new]
    if problem == "missing_rounds":
        batches = [old]
    elif problem == "duplicate_batch":
        batches = [old, old]
    elif problem == "duplicate_scan":
        manifest = json.loads((new / "batch.json").read_text())
        manifest["runs"][0]["directory"] = str(old / "round_01_j1")
        (new / "batch.json").write_text(json.dumps(manifest))
    else:
        path = new / "round_01_j1/run.json"
        run = json.loads(path.read_text())
        if problem == "gain_mismatch":
            run["settings"]["kp"][0] = 40
        else:
            run["return_to_initial"]["status"] = "failed"
        path.write_text(json.dumps(run))
    output = tmp_path / "merged"
    with pytest.raises(ValueError):
        merger.merge(batches, output)
    assert not output.exists()
