"""Offline checks for direction-dependent candidates and honest round validation."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def fitter():
    path = Path(__file__).parents[1] / "isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/fit_static_directional_compensation.py"
    spec = importlib.util.spec_from_file_location("directional_fit_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observations():
    rows = []
    for repeat in range(1, 6):
        for direction in (-1, 1):
            for angle in np.linspace(-30, 30, 31):
                x = angle/30
                rows.append(dict(round=repeat, q=angle, direction=direction,
                                 torque=.5+2*x+.2*x*x+direction*(.3+.2*x)+(repeat-3)*.003))
    return rows


def test_recovers_angle_dependent_direction_difference(fitter):
    rows = observations()
    model = fitter.fit(rows, 2, "directional_curves", 0, 30)
    q = np.linspace(-30, 30, 61)
    up, down = fitter.predict(model, q, np.ones(61)), fitter.predict(model, q, -np.ones(61))
    assert model["converged"]
    assert np.allclose((up+down)/2, .5+2*q/30+.2*(q/30)**2, atol=1.e-7)
    assert np.allclose((up-down)/2, .3+.2*q/30, atol=1.e-7)


def test_robust_fit_retains_but_downweights_a_large_single_point_error(fitter):
    rows = observations()
    location = next(i for i, row in enumerate(rows) if row["round"] == 2 and row["direction"] == -1 and row["q"] == 20)
    rows[location]["torque"] += 3.
    model = fitter.fit(rows, 2, "directional_curves", 0, 30)
    assert model["robust_weights"][location] < .1
    assert len(model["robust_weights"]) == len(rows)
    truth = .5+2*(20/30)+.2*(20/30)**2-(.3+.2*(20/30))
    assert fitter.predict(model, [20], [-1])[0] == pytest.approx(truth, abs=.01)


def test_feedback_count_or_repeated_visits_do_not_overweight_one_round_direction(fitter):
    rows = observations()
    repeated = rows + [dict(row) for row in rows if row["round"] == 1 and row["direction"] == 1]*20
    first = fitter.fit(rows, 2, "directional_curves", 0, 30)
    second = fitter.fit(repeated, 2, "directional_curves", 0, 30)
    assert second["coefficients"] == pytest.approx(first["coefficients"], abs=1.e-8)


def test_nested_holdout_excludes_the_whole_round_and_preserves_direction_benefit(fitter):
    rows = observations()
    grid = np.arange(-28., 29.)
    plain, _ = fitter.nested_validation(rows, "angle_only", 0, 30, grid)
    directional, records = fitter.nested_validation(rows, "directional_curves", 0, 30, grid)
    assert directional["mean_fold_balanced_mae_nm"] < .1*plain["mean_fold_balanced_mae_nm"]
    assert directional["excluded_boundary_points"] == 20
    assert len(records) == 290
    for fold in directional["folds"]:
        assert fold["held_out_round"] not in fold["training_rounds"]
        assert len(fold["training_rounds"]) == 4
        assert all(len(item["fold_mae_nm"]) == 4 for item in fold["inner_selection"])


def test_validation_rejects_extrapolation_and_ambiguous_training_direction(fitter):
    rows = observations()
    model = fitter.fit(rows, 1, "directional_curves", 0, 30)
    with pytest.raises(ValueError, match="extrapolate"):
        fitter.table_predict(model, [dict(q=31, direction=1)], np.arange(-30.,31.))
    rows[0]["direction"] = 0
    with pytest.raises(ValueError, match="observed"):
        fitter.fit(rows, 1, "directional_curves", 0, 30)


def test_three_round_nested_validation_uses_two_inner_folds_without_degree_three(fitter):
    rows = [row for row in observations() if row["round"] <= 3]
    result, records = fitter.nested_validation(rows, "directional_curves", 0, 30,
                                               np.arange(-28., 29.), degrees=(1, 2))
    assert len(result["folds"]) == 3 and len(records) == 174
    for fold in result["folds"]:
        assert fold["held_out_round"] not in fold["training_rounds"]
        assert len(fold["training_rounds"]) == 2
        assert {item["degree"] for item in fold["inner_selection"]} == {1, 2}
        assert all(len(item["fold_mae_nm"]) == 2 for item in fold["inner_selection"])


def test_negative_directional_residual_is_not_forced_to_positive_friction(fitter):
    rows = observations()
    for row in rows:
        row["torque"] -= 2*row["direction"]*(.3+.2*row["q"]/30)
    model = fitter.fit(rows, 2, "directional_curves", 0, 30)
    grid = np.arange(-30., 31.)
    up, down = fitter.predict(model, grid, np.ones(len(grid))), fitter.predict(model, grid, -np.ones(len(grid)))
    assert np.all(up < down)
    assert np.allclose((up-down)/2, -.3-.2*grid/30, atol=1.e-7)


def make_merged_source(directory, leg):
    directory.mkdir()
    defaults = [-65, -85, -20, 24, -24] if leg in ("LF", "RB") else [65, 45, -20, 24, -24]
    entries = []
    for repeat in range(1, 4):
        for joint in range(1, 6):
            path = directory / f"round_{repeat:02d}_j{joint}"
            path.mkdir()
            points = []
            for visit, (offset, direction) in enumerate(((0, 0), (-2, -1), (-4, -1), (-2, 1), (0, 1))):
                points.append(dict(visit=visit, valid=True, direction=direction, observed_approach_direction=direction,
                                   selected_position_deg=dict(median=defaults[joint-1]+offset),
                                   selected_torque_nm=dict(median=float(offset)), motion_reversals=[],
                                   position_deg_by_motor={str(m):dict(median=q+(offset if m==joint else 0))
                                                          for m,q in enumerate(defaults, 1)}))
            run = dict(complete=True, return_to_initial=dict(status="complete"),
                       settings=dict(leg=leg, joint=joint, point_timeout_s=90.),
                       points=points, procedure=dict(plan=points, default_pose_deg=defaults))
            (path/"run.json").write_text(json.dumps(run))
            entries.append(dict(round=repeat, joint=joint, directory=path.name))
    (directory/"batch.json").write_text(json.dumps(dict(complete=True, procedure=dict(leg=leg), runs=entries)))
    return defaults


@pytest.mark.parametrize("leg", ["LF", "RB", "RF"])
def test_new_leg_sources_keep_identity_pose_and_saturation_flags(fitter, tmp_path, leg):
    directory = tmp_path/leg
    defaults = make_merged_source(directory, leg)
    annotations = tmp_path/"saturation.csv"
    with annotations.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, ["source_directory", "visit", "motor", "phase", "packets"])
        writer.writeheader()
        writer.writerow(dict(source_directory=str(directory/"round_01_j1"), visit=2, motor=1, phase="settle", packets=123))
    rows, settings, sources = fitter.load_rows(directory, annotations)
    assert len(rows) == 60 and {row["leg"] for row in rows} == {leg}
    assert {source["leg"] for source in sources} == {leg}
    assert all(source["default_pose_deg"] == defaults for source in sources)
    flagged = [row for row in rows if row["settle_integral_saturated"]]
    assert len(flagged) == 1 and flagged[0]["settle_saturation_packets"] == 123
    assert flagged[0]["round"] == 1 and flagged[0]["joint"] == 1 and flagged[0]["source_visit"] == 2
    assert set(settings) == {1, 2, 3, 4, 5}


def test_mixed_leg_scan_is_rejected_before_fitting(fitter, tmp_path):
    directory = tmp_path/"mixed"
    make_merged_source(directory, "LF")
    path = directory/"round_01_j1/run.json"
    run = json.loads(path.read_text())
    run["settings"]["leg"] = "RB"
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="same leg"):
        fitter.load_rows(directory)
