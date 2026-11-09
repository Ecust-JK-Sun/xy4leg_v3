"""Random-position comparison checks with no simulator or physical motor access."""
import csv
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def random_test(monkeypatch):
    root=Path(__file__).parents[1]/"isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3"
    monkeypatch.syspath_prepend(str(root))
    spec=importlib.util.spec_from_file_location("random_position_test",root/"measure_random_joint_positions.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plans_are_reproducible_bounded_and_replay_identically_with_rotated_order(random_test):
    plan=random_test.make_plan(65,39,90,200,2,10,123)
    assert plan==random_test.make_plan(65,39,90,200,2,10,123)
    assert plan!=random_test.make_plan(65,39,90,200,2,10,124)
    assert all(39<=point["target_deg"]<=90 and 2<=abs(point["delta_deg"])<=10 for point in plan)
    args=random_test.build_parser().parse_args([])
    assert args.min_step_deg==2 and args.max_step_deg==10
    model,_=random_test.resolve_model(args)
    _,cases,_=random_test.prepare(args,model)
    for trial in range(1,4):
        selected=[case for case in cases if case["trial"]==trial]
        assert len({case["plan_sha256"] for case in selected})==1
        assert selected[0]["plan"]==selected[1]["plan"]==selected[2]["plan"]
    assert [case["variant"] for case in cases]==["fixed","midpoint","directional","midpoint","directional","fixed","directional","fixed","midpoint"]


@pytest.mark.parametrize("leg", ["LB","LF","RB","RF"])
def test_all_catalog_models_prepare_without_hardware_or_log_creation(random_test,monkeypatch,tmp_path,leg):
    def forbidden(*args,**kwargs):raise AssertionError("Preview opened serial")
    monkeypatch.setattr(random_test.static,"DaMiaoMotorController",forbidden)
    for joint in range(1,6):
        args=random_test.build_parser().parse_args(["--leg",leg,"--joint",str(joint),"--trials","1","--output_dir",str(tmp_path/"unused")])
        preview=random_test.run(args)
        assert preview["groups"]==3
    assert not (tmp_path/"unused").exists()


def test_directional_reader_keeps_signed_gap_and_controller_holds_a_stationary_branch(random_test):
    data=dict(schema_version=1,model_type="directional_static_holding_torque_candidate",leg="LF",joint=3,
              angle_deg=[0.,10.],increasing_torque_nm=[.9,1.9],decreasing_torque_nm=[1.1,2.1],
              midpoint_torque_nm=[1.,2.],signed_half_difference_nm=[-.1,-.1],feedforward_limit_nm=3.)
    table=random_test.DirectionalTable(data)
    assert table.at(5)["half_difference_nm"]==pytest.approx(-.1)
    assert table.at(-1)["clamped"] and table.at(-1)["midpoint_nm"]==1.
    with pytest.raises(ValueError):table.validate_identity("RB",3)
    from directional_compensation import DirectionalController
    controller=DirectionalController(table,"directional",1.5,filter_tau_s=.05,slew_rate_nm_s=1,max_dt_s=.01)
    first=controller.update(5,1,10)
    assert abs(first["ff_nm"]-1.5)<=.010001  # A scheduler pause cannot accumulate a large FF jump.
    for _ in range(100):last=controller.update(5,1,.01)
    assert last["requested_ff_nm"]==pytest.approx(1.4)
    assert last["ff_nm"]==pytest.approx(1.4,abs=1.e-6)
    assert last["requested_ff_nm"]==pytest.approx(last["requested_fixed_nm"]+last["requested_midpoint_nm"]+last["requested_directional_nm"])
    for _ in range(100):reversed_state=controller.update(5,-1,.01)
    assert reversed_state["ff_nm"]==pytest.approx(1.6,abs=1.e-6)


def test_step_analysis_distinguishes_first_entry_from_sustained_settling(random_test):
    args=random_test.build_parser().parse_args([])
    plan=[dict(step=0,target_deg=10.,previous_deg=5.,direction=1),dict(step=1,target_deg=8.,previous_deg=10.,direction=-1)]
    commands=[dict(time_s=float(step),visit=step,phase="random_hold") for step in range(2)]
    feedback=[]
    for i in range(200):
        t=i/100
        if t<.05:q=9.
        elif t<.10:q=10.1
        elif t<.20:q=10.8
        elif t<1:q=10.2
        else:q=8.2
        feedback.append(dict(time_s=t,position_deg=q))
    result=random_test.analyze_steps(args,plan,commands,feedback,2.)
    assert all(point["valid"] for point in result)
    assert result[0]["first_arrival_ms"]==pytest.approx(50.)
    assert result[0]["sustained_settling_ms"]==pytest.approx(200.)
    assert result[0]["overshoot_deg"]==pytest.approx(.8)
    assert result[0]["steady_mae_deg"]==pytest.approx(.2)
    assert result[0]["error_at_40ms_deg"]==pytest.approx(1.)
    assert result[0]["observed_time_at_40ms_ms"]>=40


class FakeSerial:
    def __init__(self,missing=()):
        self.missing=missing
        self.commands,self.pending=[],[]
        self.closed=False

    def damiao_mit_command(self,**kwargs):return kwargs

    def send(self,command):
        self.commands.append(command.copy())
        motor=command["motor_id"]
        if motor not in self.missing:
            self.pending.append(dict(motor_id=motor,error_code=1,position=command["pos"]+math.radians(.1),
                                     velocity=0.,torque=command["tor"]))

    def receive_available(self):
        result,self.pending=self.pending,[]
        return result,0,0,False

    def damiao_feedback(self,frame):return frame
    def close(self):self.closed=True


def fake_hardware(module,monkeypatch,missing_first=False):
    clock=types.SimpleNamespace(now=100.)
    fake_time=types.SimpleNamespace(monotonic=lambda:clock.now,sleep=lambda dt:setattr(clock,"now",clock.now+dt))
    monkeypatch.setattr(module,"time",fake_time)
    monkeypatch.setattr(module.static,"time",fake_time)
    controllers=[]
    def open_controller(port):
        ctrl=FakeSerial(missing=(3,) if missing_first and not controllers else ())
        controllers.append(ctrl)
        return ctrl
    monkeypatch.setattr(module.static,"DaMiaoMotorController",open_controller)
    monkeypatch.setattr(module.static,"_enable_and_read_current",lambda ctrl,motor:module.static._target_rad_for_leg("LB",True)[motor-1])
    return controllers


def experiment_args(module,path):
    return module.build_parser().parse_args(["--trials","1","--points","3","--hold_s",".6","--initial_hold_s",".2",
                                            "--steady_s",".2","--settle_s",".1","--output_dir",str(path),"--enable_motors"])


def test_complete_paired_experiment_freezes_motion_integration_and_does_not_invent_improvement(random_test,monkeypatch,tmp_path):
    controllers=fake_hardware(random_test,monkeypatch)
    root=tmp_path/"comparison"
    random_test.run(experiment_args(random_test,root))
    experiment=json.loads((root/"experiment.json").read_text())
    summary=json.loads((root/"comparison.json").read_text())
    assert experiment["complete"] and len(controllers)==3 and all(c.closed for c in controllers)
    assert summary["matched_steps"]==3 and not summary["certified_40ms"]
    for value in summary["variants"].values():assert value["steady_mae_deg"]==pytest.approx(.1)
    assert summary["improvements_vs_fixed"]["directional"]["steady_mae_reduction_deg"]==pytest.approx(0.,abs=1.e-10)
    plans=[]
    for case in experiment["cases"]:
        directory=root/case["directory"]
        run=json.loads((directory/"run.json").read_text())
        assert run["return_to_initial"]["status"]=="complete"
        assert all(point["observed_approach_direction"]==point["direction"] for point in run["steps"])
        with (directory/"commands.csv").open() as stream:rows=list(csv.DictReader(stream))
        active=[row for row in rows if row["phase"]=="random_hold" and row["motor_id"]=="1"]
        assert all(float(row["requested_ff_nm"])==pytest.approx(float(row["requested_fixed_nm"])+float(row["requested_midpoint_nm"])+float(row["requested_directional_nm"])) for row in active)
        plans.append([next(float(row["target_deg"]) for row in active if int(row["visit"])==i) for i in range(3)])
        for motor,target in zip(range(2,6),(45,-20,24,-24)):
            assert all(float(row["target_deg"])==pytest.approx(target) for row in rows if row["motor_id"]==str(motor) and row["phase"]=="random_hold")
        if case["variant"]=="fixed":assert len({row["ff_nm"] for row in active})==1
    assert plans[0]==plans[1]==plans[2]
    assert (root/"COMPARISON.md").exists() and (root/"paired_steps.csv").exists()


def test_feedback_failure_stops_next_variant_and_resume_preserves_original_attempt(random_test,monkeypatch,tmp_path):
    controllers=fake_hardware(random_test,monkeypatch,missing_first=True)
    root=tmp_path/"comparison"
    with pytest.raises(RuntimeError,match="stale"):
        random_test.run(experiment_args(random_test,root))
    assert len(controllers)==1 and controllers[0].closed
    failed=(root/"trial_01_fixed/run.json").read_bytes()
    before=json.loads((root/"experiment.json").read_text())
    assert not before["complete"] and before["cases"][0]["status"]=="failed"
    assert not (root/"trial_01_midpoint").exists()
    resume=random_test.build_parser().parse_args(["--resume_dir",str(root),"--enable_motors"])
    random_test.run(resume)
    after=json.loads((root/"experiment.json").read_text())
    assert after["complete"] and len(controllers)==4
    assert after["cases"][0]["directory"]=="trial_01_fixed_retry_01"
    assert after["all_plans_sha256"]==before["all_plans_sha256"]
    assert (root/"trial_01_fixed/run.json").read_bytes()==failed
    assert len(list(root.glob("experiment_before_resume_*.json")))==1


def test_other_joint_drift_during_motion_aborts_without_starting_next_variant(random_test,monkeypatch,tmp_path):
    original=FakeSerial.send
    def disturbed(self,command):
        if command["motor_id"]==1 and abs(math.degrees(command["pos"])-65)>1:
            self.disturbed=True
        original(self,command)
        if command["motor_id"]==2 and getattr(self,"disturbed",False):
            self.pending[-1]["position"]+=math.radians(6.)
    monkeypatch.setattr(FakeSerial,"send",disturbed)
    controllers=fake_hardware(random_test,monkeypatch)
    root=tmp_path/"comparison"
    with pytest.raises(RuntimeError,match="Other joint J2"):
        random_test.run(experiment_args(random_test,root))
    assert len(controllers)==1 and controllers[0].closed
    assert not (root/"trial_01_midpoint").exists()
    metadata=json.loads((root/"trial_01_fixed/run.json").read_text())
    assert not metadata["complete"] and metadata["return_to_initial"]["status"]=="not_started"


@pytest.mark.parametrize("extra",[["--min_step_deg",".1"],["--max_deg","1000"],["--hold_s",".1"],["--filter_tau_s","nan"]])
def test_invalid_configuration_is_rejected_before_serial_or_logs(random_test,monkeypatch,tmp_path,extra):
    def forbidden(*args,**kwargs):raise AssertionError("Invalid experiment opened serial")
    monkeypatch.setattr(random_test.static,"DaMiaoMotorController",forbidden)
    output=tmp_path/"unused"
    args=random_test.build_parser().parse_args(["--enable_motors","--output_dir",str(output),*extra])
    with pytest.raises(ValueError):random_test.run(args)
    assert not output.exists()
