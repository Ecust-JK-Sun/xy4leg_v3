"""Whole-leg comparison checks, using an in-memory controller and clock."""
import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest

from test_xy4leg_truss_v3_random_positions import fake_hardware, experiment_args


@pytest.fixture
def leg_test(monkeypatch):
    root=Path(__file__).parents[1]/"isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3"
    monkeypatch.syspath_prepend(str(root))
    spec=importlib.util.spec_from_file_location("random_leg_position_test",root/"measure_random_leg_positions.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hardware(module,monkeypatch,**kwargs):
    controllers=fake_hardware(module,monkeypatch,**kwargs)
    monkeypatch.setattr(module.single,"time",module.time)
    return controllers


@pytest.mark.parametrize("leg",["LB","LF","RB","RF"])
def test_preview_and_paired_vector_plans(leg_test,monkeypatch,tmp_path,leg):
    def forbidden(*args,**kwargs):raise AssertionError("Preview accessed hardware")
    monkeypatch.setattr(leg_test.static,"DaMiaoMotorController",forbidden)
    args=leg_test.build_parser().parse_args(["--leg",leg,"--output_dir",str(tmp_path/"unused")])
    bundle,_=leg_test.resolve_model(args)
    _,cases,description=leg_test.prepare(args,bundle)
    _,again,_=leg_test.prepare(args,bundle)
    assert cases==again
    assert description["scope"]=="whole_leg"
    for trial in range(1,4):
        selected=[c for c in cases if c["trial"]==trial]
        assert len({c["plan_sha256"] for c in selected})==1
        assert selected[0]["plan"]==selected[1]["plan"]==selected[2]["plan"]
        for point in selected[0]["plan"]:
            assert len(point["target_deg"])==5
            for motor in range(1,6):
                q=point["target_deg"][motor-1]
                lo,hi=description["target_bounds_deg"][str(motor)]
                assert lo<=q<=hi
                assert abs(q-description["default_pose_deg"][motor-1])<=10.000001
                assert 2<=abs(point["delta_deg"][motor-1])<=10
        assert any(len(set(p["direction"]))>1 for p in selected[0]["plan"])
    assert [c["variant"] for c in cases]==["fixed","midpoint","directional","midpoint","directional","fixed","directional","fixed","midpoint"]
    leg_test.run(args)
    assert not (tmp_path/"unused").exists()


def test_end_to_end_all_five_targets_and_compensation_are_replayed(leg_test,monkeypatch,tmp_path):
    controllers=hardware(leg_test,monkeypatch)
    root=tmp_path/"whole_leg"
    leg_test.run(experiment_args(leg_test,root))
    experiment=json.loads((root/"experiment.json").read_text())
    comparison=json.loads((root/"comparison.json").read_text())
    assert experiment["complete"] and len(controllers)==3 and all(c.closed for c in controllers)
    assert comparison["matched_poses"]==3 and not comparison["certified_40ms"]
    for mode in comparison["variants"].values():
        assert mode["whole_leg"]["all_five_steady_pass_fraction"]==1
        assert mode["whole_leg"]["mean_axis_steady_mae_deg"]==pytest.approx(.1)
        for axis in mode["joints"].values():assert axis["steady_mae_deg"]==pytest.approx(.1)
    vectors=[]
    for case in experiment["cases"]:
        directory=root/case["directory"]
        run=json.loads((directory/"run.json").read_text())
        assert run["return_to_initial"]["status"]=="complete"
        with (directory/"commands.csv").open() as stream:
            rows=[r for r in csv.DictReader(stream) if r["phase"]=="random_hold"]
        vectors.append([[next(float(r["target_deg"]) for r in rows if int(r["motor_id"])==m and int(r["visit"])==step)
                         for m in range(1,6)] for step in range(3)])
        for m in range(1,6):
            axis=[r for r in rows if int(r["motor_id"])==m]
            assert len({r["target_deg"] for r in axis})==3
            assert all(float(r["requested_ff_nm"])==pytest.approx(sum(float(r[k]) for k in ("requested_fixed_nm","requested_midpoint_nm","requested_directional_nm"))) for r in axis)
            if case["variant"]=="fixed":assert len({r["ff_nm"] for r in axis})==1
            else:assert len({r["ff_nm"] for r in axis})>1
        summary=json.loads((directory/"summary.json").read_text())
        assert all(p["first_axis_send_skew_ms"]>0 for p in summary["poses"])
    assert vectors[0]==vectors[1]==vectors[2]
    assert (root/"paired_joint_steps.csv").exists() and (root/"COMPARISON.md").exists()
    assert (root/"comparison_trial_01.png").exists()
    leg_test.run(leg_test.build_parser().parse_args(["--analyze_dir",str(root)]))


def test_whole_pose_requires_all_five_axes_and_overlapping_settling(leg_test):
    args=leg_test.build_parser().parse_args(["--points","2"])
    joints={str(m):dict(steps=[dict(valid=True,target_deg=0.,first_command_s=m*.001,end_s=2.,sustained_settling_ms=100.,
                                  steady_all_within_tolerance=True,steady_mae_deg=.1) for _ in range(2)]) for m in range(1,6)}
    joints["5"]["steps"][0].update(steady_all_within_tolerance=False,sustained_settling_ms=None,steady_mae_deg=1.)
    joints["4"]["steps"][1]["sustained_settling_ms"]=1900.
    poses=leg_test.pose_steps(joints,args)
    assert poses[0]["valid"] and not poses[0]["all_five_steady_within"]
    assert all(p["sustained_settling_ms"] is None for p in poses)
    assert leg_test.aggregate_poses(poses)["all_five_steady_pass_fraction"]==.5


def test_failure_closes_controller_and_resume_preserves_vectors_and_failed_attempt(leg_test,monkeypatch,tmp_path):
    controllers=hardware(leg_test,monkeypatch,missing_first=True)
    root=tmp_path/"resume"
    with pytest.raises(RuntimeError,match="stale"):leg_test.run(experiment_args(leg_test,root))
    assert len(controllers)==1 and controllers[0].closed
    failed=(root/"trial_01_fixed/run.json").read_bytes()
    before=json.loads((root/"experiment.json").read_text())
    assert not (root/"trial_01_midpoint").exists()
    args=leg_test.build_parser().parse_args(["--resume_dir",str(root),"--enable_motors"])
    leg_test.run(args)
    after=json.loads((root/"experiment.json").read_text())
    assert after["complete"] and len(controllers)==4
    assert after["all_plans_sha256"]==before["all_plans_sha256"]
    assert after["model_snapshot_sha256"]==before["model_snapshot_sha256"]
    assert after["cases"][0]["directory"]=="trial_01_fixed_retry_01"
    assert (root/"trial_01_fixed/run.json").read_bytes()==failed


@pytest.mark.parametrize("extra",[["--pose_range_deg","31"],["--pose_range_deg","nan"],["--pose_range_deg",".5"],
                                ["--integral_torque_limit","7","4.5","3.5","2.5","2"]])
def test_bad_range_or_axis_limit_rejected_before_opening_hardware(leg_test,monkeypatch,tmp_path,extra):
    def forbidden(*args,**kwargs):raise AssertionError("Invalid input opened hardware")
    monkeypatch.setattr(leg_test.static,"DaMiaoMotorController",forbidden)
    root=tmp_path/"unused"
    args=leg_test.build_parser().parse_args(["--enable_motors","--output_dir",str(root),*extra])
    with pytest.raises(ValueError):leg_test.run(args)
    assert not root.exists()
