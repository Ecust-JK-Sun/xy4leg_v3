"""Tracking measurement tests with synthetic responses and a fake clock/serial link."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def tracking(monkeypatch):
    real = Path(__file__).parents[1] / "isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3"
    monkeypatch.syspath_prepend(str(real))
    spec = importlib.util.spec_from_file_location("tracking_measurement_test", real / "measure_joint_tracking.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _args(module, **overrides):
    values = dict(operation="run", leg="LB", joint=1, port=None, center_deg=62., min_deg=55., max_deg=65.,
                  amplitude_deg=2., mode="step", frequency_hz=.3, hold_s=1., duration_s=12., hz=25., poll_hz=200.,
                  tolerance_deg=.5, feedback_timeout_s=.2, warmup_timeout_s=20.,
                  motor_config=str(module.THIS_DIR / "damiao.yaml"), kp=None, kd=None,
                  integral_compensation=True, enable_motors=True, output_dir=None)
    values.update(overrides)
    return argparse.Namespace(**values)


def _commands(args):
    rows = []
    for index in range(round(args.duration_s * args.hz)):
        stamp = index / args.hz
        if args.mode == "sine":
            target = args.center_deg + args.amplitude_deg * math.sin(2 * math.pi * args.frequency_hz * stamp)
        else:
            target = args.center_deg + (0., args.amplitude_deg, 0., -args.amplitude_deg)[int(stamp / args.hold_s) % 4]
        rows.append(dict(time_s=stamp, write_done_s=stamp + .001, lateness_ms=0., skipped_ticks=0,
                         target_deg=target, ff_nm=3.))
    return rows


def _sample(stamp, position, velocity=0.):
    return dict(time_s=stamp, motor_id=1, position_deg=position, velocity_deg_s=velocity,
                torque_nm=3., backlog_bytes=8)


@pytest.mark.parametrize("dt,tau,expected_ms", [(.005, .02, 80.), (.005, .005, 30.), (.04, .005, 40.)])
def test_step_analysis_measures_settling_without_claiming_device_40ms_certification(
    tracking, dt, tau, expected_ms,
):
    args = _args(tracking, duration_s=3., tolerance_deg=.1)
    commands = _commands(args)
    feedback = []
    for index in range(round(args.duration_s / dt)):
        stamp = index * dt
        if stamp < 1.:
            position, velocity = 62., 0.
        elif stamp < 2.:
            exponential = math.exp(-(stamp - 1.) / tau)
            position, velocity = 64. - 2 * exponential, 2 / tau * exponential
        else:
            exponential = math.exp(-(stamp - 2.) / tau)
            position, velocity = 62. + 2 * exponential, -2 / tau * exponential
        feedback.append(_sample(stamp, position, velocity))
    summary = tracking.analyze(args, commands, feedback)
    assert summary["steps"][0]["observed_settling_ms"] == pytest.approx(expected_ms)
    assert summary["feedback_cadence"]["hz"] == pytest.approx(1 / dt)
    assert summary["certified_40ms"] is False
    assert summary["steps"][0]["first_sample_at_or_after_40ms_ms"] >= 40.


def test_sine_analysis_recovers_known_gain_bias_and_lag(tracking):
    args = _args(tracking, mode="sine", duration_s=8., frequency_hz=.5)
    commands = _commands(args)
    feedback = [_sample(row["time_s"], args.center_deg + .2 + .8 * args.amplitude_deg * math.sin(
        2 * math.pi * args.frequency_hz * (row["time_s"] - .08))) for row in commands]
    summary = tracking.analyze(args, commands, feedback)
    assert summary["sine_fit"]["amplitude_ratio"] == pytest.approx(.8)
    assert summary["sine_fit"]["fitted_lag_ms"] == pytest.approx(80.)
    assert summary["sine_fit"]["fitted_bias_deg"] == pytest.approx(.2)
    assert summary["sine_fit"]["phase_identifiable"]
    assert summary["command_cadence"]["hz"] == pytest.approx(25.)


def test_flat_sine_response_has_no_identifiable_phase_or_delay(tracking):
    args = _args(tracking, mode="sine", duration_s=20., frequency_hz=.3)
    commands = _commands(args)
    feedback = [_sample(row["time_s"], 63.13376788110866) for row in commands]
    result = tracking.analyze(args, commands, feedback)["sine_fit"]
    assert result["amplitude_ratio"] == pytest.approx(0., abs=1.e-10)
    assert result["measured_peak_to_peak_deg"] == 0.
    assert not result["phase_identifiable"]
    assert result["fitted_lag_ms"] is None and result["phase_lag_deg"] is None


def test_step_static_bias_is_reported_separately_without_relaxing_absolute_accuracy(tracking):
    args = _args(tracking, duration_s=3., tolerance_deg=.5)
    commands = _commands(args)
    feedback = []
    for index in range(600):
        stamp = index * .005
        if stamp < 1.:
            position, velocity = 62., 0.
        elif stamp < 2.:
            exponential = math.exp(-(stamp - 1.) / .005)
            position, velocity = 64. - 2 * exponential, 400 * exponential
        else:
            exponential = math.exp(-(stamp - 2.) / .005)
            position, velocity = 62. + 2 * exponential, -400 * exponential
        feedback.append(_sample(stamp, position - .74, velocity))
    summary = tracking.analyze(args, commands, feedback)
    assert summary["initial_baseline"]["error_deg"] == pytest.approx(.74)
    assert summary["steps"][0]["observed_settling_ms"] is None
    assert summary["steps"][0]["baseline_corrected_settling_ms"] == pytest.approx(30.)
    assert summary["baseline_corrected_tracking"]["rmse_deg"] < summary["tracking"]["rmse_deg"]
    assert not summary["certified_40ms"]


class FakeSerial:
    def __init__(self, missing=(), outside=False):
        self.commands, self.pending = [], []
        self.missing, self.outside, self.closed = missing, outside, False

    def damiao_mit_command(self, **kwargs):
        return kwargs

    def send(self, command):
        self.commands.append(command.copy())
        motor_id = command["motor_id"]
        if motor_id not in self.missing:
            position = math.radians(66.) if self.outside and motor_id == 1 else command["pos"]
            self.pending.append(dict(motor_id=motor_id, error_code=1, position=position,
                                     velocity=0., torque=command["tor"]))

    def receive_available(self):
        rows, self.pending = self.pending, []
        return rows, 0, len(rows) * 8, False

    def damiao_feedback(self, frame):
        return frame

    def close(self):
        self.closed = True


def _fake_clock(module):
    clock = types.SimpleNamespace(now=100.)
    module.time = types.SimpleNamespace(monotonic=lambda: clock.now,
                                        sleep=lambda dt: setattr(clock, "now", clock.now + dt))
    return clock


def test_capture_uses_25hz_commands_frozen_bias_and_only_fresh_received_packets(tracking):
    args = _args(tracking, duration_s=1., hold_s=.1)
    tracking._validate(args)
    _fake_clock(tracking)
    ctrl, commands, feedback = FakeSerial(), [], []
    fixed = [3., .5, -.7, .8, .05]
    tracking.capture(args, ctrl, [math.radians(62.)] * 5, fixed, commands, feedback)
    assert len(commands) == 25
    assert len(feedback) == len(ctrl.commands) == 125
    assert {row["motor_id"] for row in ctrl.commands} == {1, 2, 3, 4, 5}
    assert all(row["tor"] == fixed[row["motor_id"] - 1] for row in ctrl.commands)
    assert max(abs(row["lateness_ms"]) for row in commands) < .01
    assert tracking.analyze(args, commands, feedback)["feedback_cadence"]["hz"] == pytest.approx(25.)


def test_100hz_resends_preserve_25hz_targets_and_measure_actual_feedback_rate(tracking):
    args = _args(tracking, duration_s=1., hold_s=.1, tx_hz=100.)
    tracking._validate(args)
    _fake_clock(tracking)
    ctrl, commands, feedback = FakeSerial(), [], []
    tracking.capture(args, ctrl, [math.radians(62.)] * 5, [0.] * 5, commands, feedback)
    assert len(commands) == 100 and len(feedback) == 500
    for index in range(0, 100, 4):
        batch = commands[index:index+4]
        assert len({row["target_sequence"] for row in batch}) == 1
        assert len({row["target_deg"] for row in batch}) == 1
    result = tracking.analyze(args, commands, feedback)
    assert result["command_cadence"]["hz"] == pytest.approx(100.)
    assert result["target_update_cadence"]["hz"] == pytest.approx(25.)
    assert result["feedback_cadence"]["hz"] == pytest.approx(100.)
    assert result["skipped_command_ticks"] == result["skipped_target_ticks"] == 0


@pytest.mark.parametrize("friction_scale", [0., .25, 1.])
def test_dynamic_feedforward_changes_only_selected_joint_and_preserves_clocks(tracking, friction_scale):
    path = tracking.THIS_DIR / "calibration/lb_j1_load_friction_v1.json"
    args = _args(tracking, duration_s=1., hold_s=.1, tx_hz=100., center_deg=55., amplitude_deg=10.,
                 min_deg=44., max_deg=66.5, compensation_model=str(path), friction_scale=friction_scale)
    original = path.read_bytes()
    model = tracking._validate(args)
    _fake_clock(tracking)
    ctrl, commands, feedback = FakeSerial(), [], []
    fixed = [2.2, .5, -.7, .8, .05]
    tracking.capture(args, ctrl, [math.radians(55.)] * 5, fixed, commands, feedback, model=model)
    assert len(commands) == 100
    assert all(row["tor"] == fixed[row["motor_id"] - 1] for row in ctrl.commands if row["motor_id"] != 1)
    assert commands[0]["ff_nm"] == 2.2
    assert any(row["ff_nm"] != 2.2 for row in commands[1:])
    assert all(abs(row["ff_nm"]) <= 4. for row in commands)
    assert max(abs(b["ff_nm"]-a["ff_nm"]) for a, b in zip(commands, commands[1:])) <= .1 + 1.e-9
    result = tracking.analyze(args, commands, feedback, model_snapshot=model.data)
    assert result["target_update_cadence"]["hz"] == pytest.approx(25.)
    assert result["command_cadence"]["hz"] == pytest.approx(100.)
    assert result["compensation"]["mode"] == ("load_friction" if friction_scale else "load_only")
    assert result["compensation"]["effective_friction_limit_nm"] == pytest.approx(model.friction_torque_nm * friction_scale)
    assert max(abs(row["requested_friction_nm"]) for row in commands) <= model.friction_torque_nm * friction_scale
    assert result["compensation"]["clamped_load_packets"] > 0
    assert result["compensation"]["dynamic_packets"] == 99
    assert result["coverage"]["requested_angle_span_fraction"] == pytest.approx(.7)
    assert not result["coverage"]["span_target_met"]
    assert result["tracking_calibrated_only"]["samples"] > 0
    assert path.read_bytes() == original


def test_model_for_another_leg_is_rejected_before_hardware(tracking, monkeypatch):
    def forbidden(*args):
        pytest.fail("Wrong compensation profile must not open hardware")
    monkeypatch.setattr(tracking, "DaMiaoMotorController", forbidden)
    path = tracking.THIS_DIR / "calibration/lb_j1_load_friction_v1.json"
    with pytest.raises(ValueError, match="belongs to LB Joint1"):
        tracking.run(_args(tracking, leg="RF", compensation_model=str(path)))


def test_step_friction_uses_target_change_direction_without_extra_error_feedback(tracking):
    model = tracking.LoadFrictionModel.read(tracking.THIS_DIR / "calibration/lb_j1_load_friction_v1.json")
    args = _args(tracking, hold_s=2.)
    for stamp, expected in ((0., 0.), (2.1, 1.), (4.1, -1.), (6.1, -1.), (8.1, 1.)):
        assert tracking.compensation_direction(args, stamp, model) == expected


def test_no_dynamic_update_without_selected_joint_feedback(tracking):
    path = tracking.THIS_DIR / "calibration/lb_j1_load_friction_v1.json"
    args = _args(tracking, compensation_model=str(path), tx_hz=100.)
    model = tracking._validate(args)
    _fake_clock(tracking)
    commands = []
    with pytest.raises(RuntimeError, match="Missing/stale.*1"):
        tracking.capture(args, FakeSerial(missing=(1,)), [math.radians(62.)] * 5, [2.2]*5,
                         commands, [], model=model)
    assert all(row["dynamic_active"] == 0 and row["ff_nm"] == 2.2 for row in commands)


def test_angle_span_and_feedback_occupancy_have_separate_coverage_targets(tracking):
    path = tracking.THIS_DIR / "calibration/lb_j1_load_friction_v1.json"
    model = tracking.LoadFrictionModel.read(path)
    args = _args(tracking, center_deg=55., amplitude_deg=10., min_deg=44., max_deg=66.5,
                 compensation_model=str(path), friction_scale=0., coverage_target=.75)
    rows = [_sample(index*.01, value) for index, value in enumerate((55., 60., 55., 45.))]
    result = tracking.analyze(args, _commands(args), rows, model_snapshot=model.data)
    assert result["coverage"]["requested_angle_span_fraction"] == pytest.approx(.7)
    assert result["coverage"]["feedback_sample_fraction"] == .75
    assert result["coverage"]["sample_target_met"]
    assert not result["coverage"]["span_target_met"]
    assert not result["certified_40ms"]
    interrupted = tracking.analyze(args, _commands(args), rows, complete=False, model_snapshot=model.data)
    assert interrupted["coverage"]["sample_target_met"] is None


def test_10_degree_amplitude_ramp_has_continuous_bounded_round_trip(tracking):
    args = _args(tracking, mode="ramp", center_deg=55., amplitude_deg=10., min_deg=45., max_deg=65.,
                 duration_s=41., ramp_speed_deg_s=1., tx_hz=100.)
    tracking._validate(args)
    for stamp, expected in ((0., 55.), (1., 55.), (11., 65.), (31., 45.), (41., 55.)):
        assert tracking.target_deg(args, stamp) == pytest.approx(expected)
    values = [tracking.target_deg(args, index / 25) for index in range(1026)]
    assert min(values) == 45. and max(values) == 65.
    assert max(abs(b-a) for a, b in zip(values, values[1:])) <= .04 + 1.e-9


def test_ramp_analysis_detects_directional_deadband_and_torque_difference(tracking):
    args = _args(tracking, mode="ramp", center_deg=55., amplitude_deg=10., min_deg=45., max_deg=65.,
                 duration_s=41., ramp_speed_deg_s=1.)
    commands = [dict(time_s=index / 25, target_deg=tracking.target_deg(args, index / 25),
                     skipped_ticks=0, ff_nm=3.) for index in range(1025)]
    position, feedback = 55., []
    for index in range(4100):
        stamp = index / 100
        target = commands[min(1024, math.floor(stamp * 25 + 1.e-9))]["target_deg"]
        previous = position
        if target-position > 1.:
            position = target-1.
        elif target-position < -1.:
            position = target+1.
        row = _sample(stamp, position, (position-previous) * 100)
        row["torque_nm"] = 3. + 30. * math.radians(target-position)
        feedback.append(row)
    result = tracking.analyze(args, commands, feedback)["ramp"]
    onsets = result["movement_onsets"]
    assert [row["direction"] for row in onsets] == ["increasing", "decreasing", "increasing"]
    assert onsets[0]["observed_movement_onset"]["error_deg"] == pytest.approx(1.)
    assert onsets[1]["observed_movement_onset"]["error_deg"] == pytest.approx(-1.)
    assert result["directional_torque_bins"]
    assert all(row["torque_difference_nm"] > .9 for row in result["directional_torque_bins"])


@pytest.mark.parametrize("missing,outside,match", [((3,), False, "Missing/stale.*3"), ((), True, "angle.*bounds")])
def test_capture_stops_on_missing_feedback_or_out_of_bounds_actual_angle(tracking, missing, outside, match):
    args = _args(tracking)
    tracking._validate(args)
    _fake_clock(tracking)
    with pytest.raises(RuntimeError, match=match):
        tracking.capture(args, FakeSerial(missing, outside), [math.radians(62.)] * 5, [0.] * 5, [], [])


def test_preview_and_offline_reanalysis_never_open_hardware_or_report_tracking_success(tracking, monkeypatch, tmp_path):
    def forbidden(*args):
        pytest.fail("Preview or offline analysis must not open a serial port")
    monkeypatch.setattr(tracking, "DaMiaoMotorController", forbidden)
    directory = tmp_path / "preview"
    args = _args(tracking, enable_motors=False, output_dir=str(directory))
    tracking.run(args)
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["command_timing"] == "planned_preview_only"
    assert not summary["complete"] and summary["tracking"] is None and not summary["certified_40ms"]
    assert (directory / "report.html").is_file()
    monkeypatch.setattr(sys, "argv", ["measure_joint_tracking.py", "analyze", str(directory)])
    tracking.main()
    assert json.loads((directory / "summary.json").read_text())["tracking"] is None


@pytest.mark.parametrize("overrides", [dict(max_deg=63.), dict(amplitude_deg=.2), dict(hz=50.),
                                      dict(duration_s=-1.), dict(mode="sine", frequency_hz=5.),
                                      dict(tx_hz=0.), dict(tx_hz=26.), dict(tx_hz=100., poll_hz=50.),
                                      dict(mode="ramp", ramp_speed_deg_s=.1),
                                      dict(friction_scale=-1.), dict(friction_scale=1.1),
                                      dict(friction_scale=.25), dict(coverage_target=0.)])
def test_invalid_trajectories_are_rejected_before_opening_hardware(tracking, monkeypatch, overrides):
    def forbidden(*args):
        pytest.fail("Invalid trajectory must not open a serial port")
    monkeypatch.setattr(tracking, "DaMiaoMotorController", forbidden)
    with pytest.raises(ValueError):
        tracking.run(_args(tracking, **overrides))


def test_fast_resend_preview_and_saved_csv_keep_target_clock_distinct(tracking, monkeypatch, tmp_path):
    def forbidden(*args):
        pytest.fail("Preview must not open hardware")
    monkeypatch.setattr(tracking, "DaMiaoMotorController", forbidden)
    directory = tmp_path / "fast_preview"
    args = _args(tracking, enable_motors=False, tx_hz=100., output_dir=str(directory))
    tracking.run(args)
    result = json.loads((directory / "summary.json").read_text())
    assert result["command_cadence"]["samples"] == 1200
    assert result["target_update_cadence"]["samples"] == 300
    assert result["feedback_cadence"]["samples"] == 0
    monkeypatch.setattr(sys, "argv", ["measure_joint_tracking.py", "analyze", str(directory)])
    tracking.main()
    result = json.loads((directory / "summary.json").read_text())
    assert result["target_update_cadence"]["hz"] == pytest.approx(25.)
    assert result["command_cadence"]["hz"] == pytest.approx(100.)


def test_aborted_capture_keeps_partial_data_and_closes_port(tracking, monkeypatch, tmp_path):
    args = _args(tracking, output_dir=str(tmp_path / "aborted"))
    _fake_clock(tracking)
    ctrl = FakeSerial(missing=(3,))
    monkeypatch.setattr(tracking, "DaMiaoMotorController", lambda port: ctrl)
    monkeypatch.setattr(tracking, "_enable_and_read_current", lambda *a: math.radians(62.))
    monkeypatch.setattr(tracking, "_send_leg_position_targets", lambda *a: None)
    monkeypatch.setattr(tracking, "_hold_and_verify_leg", lambda *a, **kw: True)
    with pytest.raises(RuntimeError, match="Missing/stale"):
        tracking.run(args)
    assert ctrl.closed
    metadata = json.loads((Path(args.output_dir) / "run.json").read_text())
    assert not metadata["complete"] and "Missing/stale" in metadata["error"]
    assert (Path(args.output_dir) / "commands.csv").read_text().count("\n") > 1
    summary = json.loads((Path(args.output_dir) / "summary.json").read_text())
    assert not summary["complete"] and summary["error"]


@pytest.mark.parametrize("dynamic_tolerance,errors,expected_start", [
    (.5, [.51, -.74, .03, -.36, .18], True),
    (.1, [.51, -.74, .03, -.36, .18], True),
    (.5, [1.5, -.74, .03, -.36, .18], False),
])
def test_warmup_gate_is_independent_and_preserves_diagnostics(
    tracking, monkeypatch, tmp_path, dynamic_tolerance, errors, expected_start,
):
    args = _args(tracking, output_dir=str(tmp_path / "warmup"), tolerance_deg=dynamic_tolerance)
    _fake_clock(tracking)
    ctrl = FakeSerial()
    monkeypatch.setattr(tracking, "DaMiaoMotorController", lambda port: ctrl)
    monkeypatch.setattr(tracking, "_enable_and_read_current", lambda *a: math.radians(62.))
    monkeypatch.setattr(tracking, "_send_leg_position_targets", lambda *a: None)
    def warmup(*positional, result_out):
        assert math.degrees(positional[8]) == pytest.approx(1.)
        reached = all(abs(value) <= 1. for value in errors)
        result_out.update(reached=reached, error_deg=errors,
                          position_deg=[math.degrees(target) - error for target, error in zip(positional[1], errors)],
                          velocity_deg_s=[0.] * 5, torque_nm=[0.] * 5, integral_ff_nm=[0.] * 5)
        return reached
    monkeypatch.setattr(tracking, "_hold_and_verify_leg", warmup)
    if expected_start:
        tracking.run(args)
    else:
        with pytest.raises(RuntimeError, match="Warmup did not settle within 1 deg"):
            tracking.run(args)
    summary = json.loads((Path(args.output_dir) / "summary.json").read_text())
    assert summary["warmup"]["error_deg"] == errors
    assert summary["warmup"]["tolerance_deg"] == 1.
    assert bool(summary["command_cadence"]["samples"]) == expected_start
    assert summary["complete"] == expected_start
    assert args.tolerance_deg == dynamic_tolerance
    assert ctrl.closed
