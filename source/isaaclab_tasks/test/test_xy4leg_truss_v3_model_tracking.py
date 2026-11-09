"""No-hardware validation of zero-based model tests and command rate guards."""
import importlib.util
import csv
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    real = Path(__file__).parents[1]/"isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3"
    monkeypatch.syspath_prepend(str(real))
    monkeypatch.setitem(sys.modules,"motor_control",types.SimpleNamespace(DaMiaoMotorController=object))
    for name in ("init_default_joint_pose","measure_model_leg_tracking"):
        spec = importlib.util.spec_from_file_location(name,real/f"{name}.py")
        m = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules,name,m)
        spec.loader.exec_module(m)
    clock = types.SimpleNamespace(now=100.)
    m.time = types.SimpleNamespace(monotonic=lambda:clock.now,
                                  sleep=lambda dt:setattr(clock,"now",clock.now+dt))
    monkeypatch.setattr(sys.modules["init_default_joint_pose"],"time",m.time)
    return m


def fake_model():
    return types.SimpleNamespace(torque_limits=np.array([5.,3.,2.,1.5,.3]),
                                 slew_rates=np.array([.5,.3,.2,.15,.03]),
                                 leg_torque=lambda *a:np.zeros(5),description=lambda:dict(test=True))


class Motors:
    def __init__(self):
        self.positions = np.zeros(5)
        self.pending,self.commands = [],[]
        self.closed = False
        self.drop = False

    def enable_motor(self,motor):
        return dict(motor_id=motor,error_code=1,position=self.positions[motor-1],velocity=0.,torque=0.)

    def damiao_mit_command(self,**kw):return kw

    def damiao_feedback(self,frame):return frame

    def send(self,command):
        self.commands.append(command)
        self.positions[command["motor_id"]-1] = command["pos"]
        if not self.drop:self.pending.append(self.enable_motor(command["motor_id"]))

    def receive_available(self):
        data,self.pending = self.pending,[]
        return data,0,len(data)*8,False

    def close(self):self.closed=True


def test_seeded_random_targets_use_hardware_zero_and_requested_ranges(module):
    a = module.build_parser().parse_args([]);module.validate(a)
    assert a.kp == [30.,25.,20.,20.,2.]
    plan = module.make_plan(a)
    assert plan == module.make_plan(a) and plan["zero_pose_deg"] == [0.]*5
    center = np.array([65,45,-20,24,-24])
    values = np.array(plan["targets_deg"])
    assert np.all(abs(values-center) <= [35,20,20,20,20])


def test_quintic_and_windowed_sine_obey_speed_and_acceleration_bounds(module):
    a = module.build_parser().parse_args(["--mode","sine"]);module.validate(a)
    q = np.array([65,45,-20,24,-24])
    duration = module.move_duration(np.zeros(5),q,15,30)
    t = np.linspace(0,duration,10001)
    values = np.array([module.smooth_pose(np.zeros(5),q,v/duration) for v in t])
    dt=t[1]-t[0]
    assert np.max(abs(np.diff(values,axis=0)/dt)) <= 15+1e-5
    assert np.max(abs(np.diff(values,n=2,axis=0)/dt**2)) <= 30+1e-3
    t=np.linspace(0,a.cycles/a.frequency_hz,10001);dt=t[1]-t[0]
    values=np.array([module.sine_pose(q,[35,20,20,20,20],v,a.frequency_hz,a.cycles) for v in t])
    assert np.max(abs(np.diff(values,axis=0)/dt)) <= 15
    assert np.max(abs(np.diff(values,n=2,axis=0)/dt**2)) <= 30
    np.testing.assert_allclose(values[[0,-1]],np.tile(q,(2,1)))


def test_unsafe_sine_rejected_and_preview_never_opens_serial(module,monkeypatch,tmp_path):
    monkeypatch.setattr(module,"DaMiaoMotorController",lambda *a:pytest.fail("Opened serial"))
    a=module.build_parser().parse_args(["--mode","sine","--frequency_hz",".1"])
    with pytest.raises(ValueError,match="speed or acceleration"):module.validate(a)
    a=module.build_parser().parse_args(["--output_dir",str(tmp_path/"preview")])
    module.run(a)
    assert json.loads((a.output_dir/"run.json").read_text())["run_type"] == "preview"


def test_per_tick_rate_and_acceleration_limit_survive_scheduling_pause(module):
    a=module.build_parser().parse_args([]);module.validate(a)
    ctrl=Motors();s=module.Session(a,ctrl,fake_model(),np.zeros(5),module.make_plan(a))
    s.tick(np.array([65,45,-20,24,-24]),"test",0,0.)
    previous=s.command.copy();old_delta=s.previous_delta.copy()
    module.time.sleep(1.)
    s.tick(np.array([65,45,-20,24,-24]),"test",0,1.)
    assert max(abs(s.command-previous)) <= 15/25
    assert max(abs(s.previous_delta-old_delta)) <= 30/25**2+1e-12
    assert s.target_clips == 2


def test_feedback_dropout_stops_without_reenable_or_fallback(module):
    a=module.build_parser().parse_args([]);module.validate(a)
    ctrl=Motors();s=module.Session(a,ctrl,fake_model(),np.zeros(5),module.make_plan(a))
    ctrl.drop=True
    s.tick(np.zeros(5),"test",0,0.)
    sent=len(ctrl.commands);module.time.sleep(.51)
    with pytest.raises(RuntimeError,match="Missing/stale"):s.tick(np.zeros(5),"test",0,.51)
    assert len(ctrl.commands)==sent


@pytest.mark.parametrize("mode",["random","sine"])
@pytest.mark.parametrize("leg",["LB","LF","RB","RF"])
def test_full_zero_random_return_sequence_records_data_and_acceptance(module,monkeypatch,tmp_path,mode,leg):
    ctrl=Motors()
    monkeypatch.setattr(module,"DaMiaoMotorController",lambda *a:ctrl)
    monkeypatch.setattr(module,"make_model",lambda *a:fake_model())
    monkeypatch.setitem(sys.modules,"matplotlib",None)  # Exercise the minimal deployment report path.
    a=module.build_parser().parse_args(["--leg",leg,"--mode",mode,"--enable_motors","--output_dir",str(tmp_path/"run")])
    module.run(a)
    meta=json.loads((a.output_dir/"run.json").read_text());result=json.loads((a.output_dir/"summary.json").read_text())
    assert meta["complete"] and ctrl.closed
    assert result["accepted_random_points"] == (8 if mode=="random" else 0)
    assert [h["phase"] for h in result["holds"]] == (["verify_zero"]+["target_hold","verify_zero"]*8
                                                     if mode=="random" else ["verify_zero","verify_center","verify_zero"])
    np.testing.assert_allclose(np.rad2deg(ctrl.positions),np.zeros(5),atol=1e-6)
    with (a.output_dir/"commands.csv").open() as stream:
        commands = list(csv.DictReader(stream))
    for joint in range(1,6):
        values=np.array([float(c["target_deg"]) for c in commands if int(c["joint"])==joint])
        assert np.max(abs(np.diff(values))) <= 15/25+1e-9
        assert np.max(abs(np.diff(values,n=2))) <= 30/25**2+1e-9
    assert {c["motor_id"] for c in ctrl.commands} == {1,2,3,4,5}
    assert all(c["kp"] == [30.,25.,20.,20.,2.][c["motor_id"]-1] for c in ctrl.commands)
    assert all(c["tor"]==0. for c in ctrl.commands)
    report=(a.output_dir/"REPORT.md").read_text()
    assert "实际发送Hz" in report
    if mode=="sine":
        assert "观测相位延迟" in report
        assert all(m["amplitude_gain"]==pytest.approx(1.,abs=.01) for m in result["metrics"] if m["phase"]=="sine")


def test_reached_absolute_deadline_sends_even_when_subtraction_rounds_down(module):
    a=module.build_parser().parse_args([]);module.validate(a)
    module.time.sleep(28.)
    s=module.Session(a,Motors(),fake_model(),np.zeros(5),module.make_plan(a))
    s.last_tx=module.time.monotonic()
    module.time.sleep(1/a.hz)
    assert module.time.monotonic()-s.last_tx < 1/a.hz
    assert s.tick(np.zeros(5),"test",0,0.)


def test_feedforward_saturation_and_slew_survive_pause_and_reversal(module):
    a=module.build_parser().parse_args([]);module.validate(a)
    model=fake_model();sign=[1.]
    model.leg_torque=lambda *args:sign[0]*np.full(5,100.)
    s=module.Session(a,Motors(),model,np.zeros(5),module.make_plan(a))
    for k in range(40):
        if k==20:sign[0]=-1.
        module.time.sleep(.3 if k==21 else .04)
        s.tick(np.zeros(5),"test",0,k/a.hz)
    for joint in range(1,6):
        ff=np.array([c["ff_nm"] for c in s.commands if c["joint"]==joint])
        assert np.max(abs(ff)) <= model.torque_limits[joint-1]
        assert np.max(abs(np.diff(ff))) <= model.slew_rates[joint-1]/a.hz+1e-12
        assert np.max(ff)>0 and ff[-1]<np.max(ff)


@pytest.mark.parametrize("fault,pattern",[("drop","Missing/stale"),
                                        ("overflow","overflow"),("nan","invalid"),("range","outside")])
def test_failed_run_stops_and_persists_partial_data(module,monkeypatch,tmp_path,fault,pattern):
    class FaultyMotors(Motors):
        def receive_available(self):
            frames,pending,backlog,overflow=super().receive_available()
            if len(self.commands)>=100:
                if fault=="drop":frames=[f for f in frames if f["motor_id"]!=3]
                elif fault=="overflow":overflow=True
                elif frames:frames[0]["position"]=math.nan if fault=="nan" else math.radians(200.)
            return frames,pending,backlog,overflow

    ctrl=FaultyMotors()
    monkeypatch.setattr(module,"DaMiaoMotorController",lambda *a:ctrl)
    monkeypatch.setattr(module,"make_model",lambda *a:fake_model())
    monkeypatch.setitem(sys.modules,"matplotlib",None)
    a=module.build_parser().parse_args(["--enable_motors","--output_dir",str(tmp_path/"run")])
    with pytest.raises(RuntimeError,match=pattern):module.run(a)
    meta=json.loads((a.output_dir/"run.json").read_text())
    summary=json.loads((a.output_dir/"summary.json").read_text())
    assert not meta["complete"] and meta["error"] and ctrl.closed
    assert not summary["complete"] and summary["error"]==meta["error"]
    assert summary["random_points_tested"]==0
    assert (a.output_dir/"commands.csv").is_file() and (a.output_dir/"REPORT.md").is_file()


@pytest.mark.parametrize("mode",["random","sine"])
@pytest.mark.parametrize("failure",["position","velocity"])
def test_unmet_accuracy_or_stability_is_recorded_and_full_plan_continues(module,monkeypatch,tmp_path,mode,failure):
    class BiasedMotors(Motors):
        def send(self,command):
            super().send(command)
            if command["motor_id"]==2:
                self.pending[-1]["position"] += math.radians(2.24) if failure=="position" else 0.
                self.pending[-1]["velocity"] = math.radians(3.) if failure=="velocity" else 0.

    ctrl=BiasedMotors()
    monkeypatch.setattr(module,"DaMiaoMotorController",lambda *a:ctrl)
    monkeypatch.setattr(module,"make_model",lambda *a:fake_model())
    monkeypatch.setitem(sys.modules,"matplotlib",None)
    a=module.build_parser().parse_args(["--mode",mode,"--points","2","--enable_motors","--output_dir",str(tmp_path/"run")])
    module.run(a)
    meta=json.loads((a.output_dir/"run.json").read_text())
    summary=json.loads((a.output_dir/"summary.json").read_text())
    assert meta["complete"] and meta["error"] is None and ctrl.closed
    assert summary["random_points_tested"]==(2 if mode=="random" else 0)
    assert summary["accepted_random_points"]==0
    holds=summary["holds"]
    assert [h["phase"] for h in holds]==(["verify_zero"]+["target_hold","verify_zero"]*2 if mode=="random"
                                        else ["verify_zero","verify_center","verify_zero"])
    assert all(not h["accepted"] and h[f"{failure}_outside_tolerance_joints"]==[2] for h in holds)
    assert all(h["elapsed_s"]>=a.zero_timeout_s for h in holds if h["phase"] in ("verify_zero","verify_center"))
    assert all(h["error_deg"][1]==pytest.approx(-2.24 if failure=="position" else 0.) for h in holds)
    np.testing.assert_allclose(np.rad2deg(ctrl.positions),np.zeros(5),atol=1e-6)
    report=(a.output_dir/"REPORT.md").read_text()
    assert "精度或稳定性未达标只记录" in report and "| False |" in report


def test_sine_analysis_estimates_known_phase_lag(module):
    a=module.build_parser().parse_args(["--mode","sine"]);module.validate(a)
    commands,feedback=[],[]
    delay=.2;omega=2*math.pi*a.frequency_hz
    for t in np.arange(0,a.cycles/a.frequency_hz,.04):
        q=module.sine_pose([0]*5,[35]*5,t,a.frequency_hz,a.cycles)[0]
        commands.append(dict(time_s=float(t),phase="sine",point=-1,logical_s=float(t),joint=1,
                             target_deg=float(q),requested_deg=float(q)))
        feedback.append(dict(time_s=float(t),phase="sine",joint=1,
                             position_deg=35*math.sin(omega*(t-delay))))
    result=module.analyze(a,commands,feedback,[])["metrics"][0]
    assert result["observed_phase_delay_ms"] == pytest.approx(200,abs=1.)
