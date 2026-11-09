"""No-hardware checks for load-table coverage, smoothing, fitting and A/B reports."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


@pytest.fixture
def modules(monkeypatch):
    real = Path(__file__).parents[1] / "isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3"
    monkeypatch.syspath_prepend(str(real))
    loaded = []
    for name in ("tracking_compensation", "compare_joint_tracking"):
        spec = importlib.util.spec_from_file_location(name, real / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


def model_data():
    return dict(version=1, leg="LB", joint=1, load_angle_deg=[45., 55., 65.],
                load_torque_nm=[.5, 1.5, 2.5], friction_torque_nm=.95,
                velocity_soft_deg_s=.5, filter_tau_s=.05,
                slew_rate_nm_s=10., feedforward_limit_nm=4., max_interpolation_gap_deg=10.)


def test_lookup_clamps_outside_support_and_refuses_to_bridge_unmeasured_gaps(modules):
    model = modules[0].LoadFrictionModel(model_data())
    assert model.load_at(50.) == pytest.approx((1., False))
    assert model.load_at(40.) == (.5, True)
    assert model.load_at(70.) == (2.5, True)
    data = model_data()
    data.update(load_angle_deg=[45., 46., 60., 61.], load_torque_nm=[.5, .6, 2., 2.1],
                max_interpolation_gap_deg=2.)
    model = modules[0].LoadFrictionModel(data)
    assert model.coverage_intervals() == [[45., 46.], [60., 61.]]
    assert model.load_at(50.) == (.6, True)
    assert model.load_at(59.) == (2., True)
    assert model.load_at(60.5) == pytest.approx((2.05, False))
    assert model.span_coverage(45., 61.) == pytest.approx(2/16)
    with pytest.raises(ValueError, match="belongs to LB Joint1"):
        model.validate_identity("RF", 1)


@pytest.mark.parametrize("field,value", [("load_angle_deg", [45., 45., 65.]),
                                        ("load_torque_nm", [.5, math.nan, 2.5]),
                                        ("friction_torque_nm", -1.), ("filter_tau_s", 0.),
                                        ("feedforward_limit_nm", 29.)])
def test_invalid_models_are_rejected(modules, field, value):
    data = model_data()
    data[field] = value
    with pytest.raises(ValueError):
        modules[0].LoadFrictionModel(data)


def test_feedforward_replaces_warmup_bias_with_rate_limited_reversal(modules):
    model = modules[0].LoadFrictionModel(model_data())
    controller = modules[0].LoadFrictionController(model, initial_torque=2.2, max_dt_s=.01)
    outputs = [controller.update(55., 1., .01)["ff_nm"] for _ in range(100)]
    assert outputs[-1] == pytest.approx(1.5 + .95, abs=1.e-6)  # Not plus the old 2.2 Nm.
    previous = outputs[-1]
    reverse = controller.update(55., -1., 10.)
    assert abs(reverse["ff_nm"] - previous) <= .1 + 1.e-12
    assert reverse["requested_friction_nm"] == -.95
    outputs += [reverse["ff_nm"]] + [controller.update(55., -1., .01)["ff_nm"] for _ in range(100)]
    assert outputs[-1] == pytest.approx(1.5 - .95, abs=1.e-6)
    assert max(abs(b-a) for a, b in zip(outputs, outputs[1:])) <= .1 + 1.e-12


@pytest.mark.parametrize("scale", [0., .25, .5])
def test_friction_ablation_leaves_the_fixed_load_table_unchanged(modules, scale):
    data = model_data()
    original = json.dumps(data, sort_keys=True)
    model = modules[0].LoadFrictionModel(data)
    controller = modules[0].LoadFrictionController(model, 1.5, .01, friction_scale=scale)
    for _ in range(200):
        state = controller.update(55., -1., .01)
    assert state["requested_load_nm"] == 1.5
    assert state["requested_friction_nm"] == pytest.approx(-.95*scale)
    assert state["ff_nm"] == pytest.approx(1.5-.95*scale, abs=1.e-6)
    assert json.dumps(model.data, sort_keys=True) == original


def test_a_permanently_zero_friction_model_is_valid(modules):
    data = model_data()
    data["friction_torque_nm"] = 0.
    model = modules[0].LoadFrictionModel(data)
    controller = modules[0].LoadFrictionController(model, 1.5, .01)
    assert controller.update(55., 1., .01)["requested_friction_nm"] == 0.


def test_model_output_limit_bounds_total_feedforward_and_reports_saturation(modules):
    data = model_data()
    data["load_torque_nm"] = [3.5, 3.5, 3.5]
    model = modules[0].LoadFrictionModel(data)
    controller = modules[0].LoadFrictionController(model, 0., .01)
    for _ in range(200):
        state = controller.update(55., 1., .01)
        assert abs(state["ff_nm"]) <= 4.
    assert state["ff_saturated"] == 1
    assert state["requested_ff_nm"] > 4.


def test_load_and_friction_reduce_error_in_a_loaded_coulomb_friction_plant(modules):
    model = modules[0].LoadFrictionModel(model_data())
    def simulate(dynamic):
        controller = modules[0].LoadFrictionController(model, 2.2, .01)
        position, velocity, target, ff, squared_error = math.radians(55.), 0., 55., 2.2, []
        for tick in range(30000):
            t = tick * .001
            if tick % 40 == 0:
                target = 55 + 5 * math.sin(2 * math.pi * .1 * t)
            if dynamic and tick % 10 == 0:
                intent = math.tanh((5 * 2 * math.pi * .1 * math.cos(2 * math.pi * .1 * t)) / .5)
                ff = controller.update(math.degrees(position), intent, .01)["ff_nm"]
            load = 1.5 + .1 * (math.degrees(position) - 55.)
            effort = 30 * (math.radians(target) - position) - velocity + ff - load
            friction = .95 * math.copysign(1., velocity) if abs(velocity) > .001 else max(-.95, min(.95, effort))
            velocity += (effort - friction) / .025 * .001
            position += velocity * .001
            if t >= 10.:
                squared_error.append((target - math.degrees(position)) ** 2)
        return math.sqrt(sum(squared_error) / len(squared_error))
    baseline, corrected = simulate(False), simulate(True)
    assert baseline > 1.
    assert corrected < baseline * .5


def _ramp(directory, low):
    directory.mkdir()
    settings = dict(leg="LB", joint=1, mode="ramp", kp=[30., 20., 20., 20., 2.], kd=[1., 1., 1., 1., .5])
    (directory / "run.json").write_text(json.dumps(dict(complete=True, run_type="hardware", settings=settings)))
    targets = [low-1, low, low+1, low+2, low+1, low, low-1]
    with (directory / "commands.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["time_s", "target_deg"])
        writer.writeheader()
        writer.writerows(dict(time_s=i, target_deg=value) for i, value in enumerate(targets))
    with (directory / "feedback.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["time_s", "motor_id", "position_deg", "velocity_deg_s", "torque_nm"])
        writer.writeheader()
        for i, position in ((1, low), (2, low+1), (3, low+2), (4, low+2), (5, low+1), (6, low)):
            direction = 1 if i <= 3 else -1
            for offset in (.01, .02, .03):
                writer.writerow(dict(time_s=i+offset, motor_id=1, position_deg=position, velocity_deg_s=direction,
                                     torque_nm=1 + .1*(position-50) + .8*direction))


def test_multiple_ramps_extend_table_while_preserving_unmeasured_holes(modules, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _ramp(first, 49.5)
    _ramp(second, 60.5)
    data = modules[0].fit_ramps([first, second])
    model = modules[0].LoadFrictionModel(data)
    assert model.coverage_intervals() == [[49.5, 51.5], [60.5, 62.5]]
    assert model.friction_torque_nm == pytest.approx(.8)
    assert model.load_at(55.)[1]


def test_fitting_does_not_mix_different_recorded_other_joint_postures(modules, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _ramp(first, 49.5)
    _ramp(second, 60.5)
    for directory, joint2 in ((first, 45.), (second, 60.)):
        metadata = json.loads((directory / "run.json").read_text())
        metadata["other_joint_targets_deg"] = [None, joint2, -20., 24., -24.]
        (directory / "run.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="different other-joint"):
        modules[0].fit_ramps([first, second])


def _comparison_run(path, dynamic, rmse, **overrides):
    path.mkdir(parents=True)
    settings = dict(leg="LB", joint=1, mode="sine", center_deg=55., amplitude_deg=10., min_deg=44., max_deg=66.5,
                    duration_s=20., hz=25., tx_hz=100., poll_hz=200., tolerance_deg=.5, integral_compensation=True,
                    frequency_hz=.3, warmup_timeout_s=30., kp=[30., 20., 20., 20., 2.], kd=[1., 1., 1., 1., .5],
                    compensation_model="model.json" if dynamic else None)
    settings.update(overrides)
    (path / "run.json").write_text(json.dumps(dict(complete=True, run_type="hardware", settings=settings)))
    summary = dict(complete=True, tracking=dict(rmse_deg=rmse, abs_error_p95_deg=rmse*2, abs_error_max_deg=rmse*3,
                                               bias_deg=rmse/4), target_update_cadence=dict(hz=25.),
                   command_cadence=dict(hz=100.), feedback_cadence=dict(hz=100.))
    (path / "summary.json").write_text(json.dumps(summary))


def test_comparison_pairs_only_equal_settings_and_rejects_preview(modules, tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    _comparison_run(before / "run", False, 2.)
    _comparison_run(after / "run", True, 1.)
    result = modules[1].compare(before, after)
    assert result["pairs"][0]["metrics"]["rmse_deg"]["reduction_percent"] == 50.
    modules[1].write_report(result, tmp_path / "report")
    assert (tmp_path / "report/comparison.md").is_file()
    _comparison_run(after / "changed", True, 1., kp=[50., 20., 20., 20., 2.])
    assert len(modules[1].compare(before, after)["unmatched"]) == 1
    meta = json.loads((after / "run/run.json").read_text())
    meta["run_type"] = "preview"
    (after / "run/run.json").write_text(json.dumps(meta))
    assert len(modules[1].compare(before, after)["pairs"]) == 0


def test_ablation_comparison_keeps_scales_and_requires_identical_model(modules, tmp_path):
    before, after = tmp_path / "load_only", tmp_path / "variants"
    _comparison_run(before / "run", True, 2., friction_scale=0.)
    for name, scale in (("quarter", .25), ("half", .5)):
        _comparison_run(after / name, True, 1., friction_scale=scale)
    for directory in (before / "run", after / "quarter", after / "half"):
        metadata = json.loads((directory / "run.json").read_text())
        metadata["compensation_model_snapshot"] = model_data()
        (directory / "run.json").write_text(json.dumps(metadata))
    result = modules[1].compare(before, after, baseline_mode="load_only")
    assert sorted(pair["friction_scale"] for pair in result["pairs"]) == [.25, .5]
    metadata = json.loads((after / "half/run.json").read_text())
    metadata["compensation_model_snapshot"]["load_torque_nm"][1] += .1
    (after / "half/run.json").write_text(json.dumps(metadata))
    result = modules[1].compare(before, after, baseline_mode="load_only")
    assert len(result["pairs"]) == len(result["unmatched"]) == 1
