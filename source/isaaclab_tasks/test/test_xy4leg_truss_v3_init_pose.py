"""Exercise initialization with simulated motor lag, steady error and missing feedback."""

from __future__ import annotations

import importlib.util
import math
import sys
import threading
import time as wall_time
import types
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def init_pose(monkeypatch):
    path = (Path(__file__).parents[1] / "isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3"
            / "init_default_joint_pose.py")
    monkeypatch.syspath_prepend(str(path.parent))
    # Loading the helper must never construct a real serial controller.
    monkeypatch.setitem(sys.modules, "motor_control", types.SimpleNamespace(DaMiaoMotorController=object))
    spec = importlib.util.spec_from_file_location("v3_init_pose_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    # Single-loop tests use a deterministic clock. Concurrent workers share a
    # scaled wall clock: independent virtual clocks cannot model peer freshness
    # once running workers no longer synchronize at every control cycle.
    clock = threading.local()
    origin = wall_time.monotonic()
    scale = .1
    def concurrent():
        return threading.current_thread().name.startswith("init-pose")
    def monotonic():
        return 100.+(wall_time.monotonic()-origin)/scale if concurrent() else getattr(clock, "now", 100.)
    def sleep(duration):
        if concurrent():
            wall_time.sleep(max(0., duration)*scale)
        else:
            clock.now = getattr(clock, "now", 100.)+duration
    module.time = types.SimpleNamespace(monotonic=monotonic, sleep=sleep)
    return module


class Motors:
    def __init__(self, response=.3, bias=0., velocity=0., missing=(), state=1):
        self.response, self.bias, self.velocity = response, bias, velocity
        self.missing, self.state = missing, state
        self.positions = {motor_id: 0. for motor_id in range(1, 6)}
        self.pending, self.commands = [], []
        self.closed = False

    def _feedback(self, motor_id, torque=0.):
        return dict(motor_id=motor_id, error_code=self.state, position=self.positions[motor_id],
                    velocity=self.velocity, torque=torque)

    def enable_motor(self, motor_id):
        return self._feedback(motor_id)

    def damiao_mit_command(self, **kwargs):
        return kwargs

    def send(self, command):
        self.commands.append(command.copy())
        motor_id = command["motor_id"]
        self.positions[motor_id] += self.response * (
            command["pos"] - self.bias - self.positions[motor_id])
        if motor_id not in self.missing:
            self.pending.append(self._feedback(motor_id, command["kp"] * (command["pos"] - self.positions[motor_id])))

    def receive_available(self):
        frames, self.pending = self.pending, []
        return frames, 0, len(frames) * 8, False

    def damiao_feedback(self, frame):
        return frame

    def close(self):
        self.closed = True


class LoadedMotors(Motors):
    """Quasi-static load model: PD alone needs a position error to support each load."""

    def __init__(self, targets):
        super().__init__()
        self.loads = (2.5, .5, -.7, .8, .03)
        self.positions = {motor_id: target - load / kp for motor_id, target, load, kp
                          in zip(range(1, 6), targets, self.loads, (20., 20., 20., 20., 2.))}

    def send(self, command):
        self.commands.append(command.copy())
        motor_id = command["motor_id"]
        previous = self.positions[motor_id]
        equilibrium = command["pos"] + (command["tor"] - self.loads[motor_id - 1]) / command["kp"]
        self.positions[motor_id] += .3 * (equilibrium - previous)
        self.velocity = (self.positions[motor_id] - previous) / .1
        self.pending.append(self._feedback(motor_id, command["kp"] * (
            command["pos"] - self.positions[motor_id]) + command["tor"]))


def _hold(module, ctrl, **overrides):
    args = dict(item=module.LegController("LB", ctrl), targets_rad=[.3] * 5,
                kp=module.DEFAULT_KP, kd=module.DEFAULT_KD, hz=10., inter_command_sleep=0.,
                settle_timeout_s=3., settle_time_s=.3, tolerance_rad=.01,
                velocity_tolerance_rad_s=.035, feedback_timeout_s=.5)
    args.update(overrides)
    return module._hold_and_verify_leg(**args)


def test_lagging_joints_keep_receiving_final_target_until_stable(init_pose, capsys):
    ctrl = Motors()
    target = [.3] * 5
    init_pose._send_leg_position_targets(
        init_pose.LegController("LB", ctrl), target, [0.] * 5,
        init_pose.DEFAULT_KP, init_pose.DEFAULT_KD, math.radians(2.), 10., 0.)
    ramp_commands = len(ctrl.commands)
    assert max(abs(value - .3) for value in ctrl.positions.values()) > .01
    assert _hold(init_pose, ctrl)
    assert len(ctrl.commands) > ramp_commands + 5
    assert all(abs(value - .3) <= .01 for value in ctrl.positions.values())
    assert {cmd["motor_id"] for cmd in ctrl.commands} == {1, 2, 3, 4, 5}
    assert all(cmd["tor"] == cmd["vel"] == 0. for cmd in ctrl.commands)
    assert "REACHED" in capsys.readouterr().out


@pytest.mark.parametrize("bias,velocity", [(.15, 0.), (0., .2)])
def test_steady_error_or_motion_times_out_without_adjusting_targets_or_gains(init_pose, capsys, bias, velocity):
    ctrl = Motors(response=1., bias=bias, velocity=velocity)
    assert not _hold(init_pose, ctrl, settle_timeout_s=1.)
    output = capsys.readouterr().out
    assert "NOT_REACHED" in output and "error(target-actual)" in output and "torque=" in output
    assert all(cmd["pos"] == .3 and cmd["kp"] == init_pose.DEFAULT_KP[cmd["motor_id"] - 1]
               for cmd in ctrl.commands)


def test_missing_individual_motor_feedback_does_not_pass_on_other_motor_updates(init_pose):
    with pytest.raises(RuntimeError, match=r"missing/stale.*\[3\]"):
        _hold(init_pose, Motors(response=1., missing=(3,)))


def test_fault_feedback_stops_verification(init_pose):
    with pytest.raises(RuntimeError, match="not enabled: status=8"):
        _hold(init_pose, Motors(state=8))


@pytest.mark.parametrize("compensate", [False, True])
def test_integral_removes_loaded_joint_error_without_increasing_kp(init_pose, capsys, compensate):
    ctrl = LoadedMotors([.3] * 5)
    compensator = init_pose.IntegralTorqueCompensator() if compensate else None
    result = {}
    reached = _hold(init_pose, ctrl, settle_timeout_s=20., integral_compensator=compensator, result_out=result)
    assert reached == compensate
    assert result["reached"] == reached
    assert result["position_deg"][0] == pytest.approx(math.degrees(ctrl.positions[1]))
    assert result["integral_ff_nm"][0] == [cmd["tor"] for cmd in ctrl.commands if cmd["motor_id"] == 1][-1]
    if compensate:
        assert all(abs(value - .3) <= .01 for value in ctrl.positions.values())
        assert any(cmd["tor"] != 0 for cmd in ctrl.commands)
        assert "integral_ff=" in capsys.readouterr().out
        for motor_id in range(1, 6):
            torques = [cmd["tor"] for cmd in ctrl.commands if cmd["motor_id"] == motor_id]
            limit = compensator.torque_limit[motor_id - 1]
            change = compensator.torque_rate[motor_id - 1] * .1
            assert all(abs(torque) <= limit for torque in torques)
            assert all(abs(b - a) <= change + 1.e-9 for a, b in zip(torques, torques[1:]))
    else:
        assert abs(ctrl.positions[1] - .3) == pytest.approx(ctrl.loads[0] / init_pose.DEFAULT_KP[0])
        assert all(cmd["tor"] == 0 for cmd in ctrl.commands)
    assert all(cmd["kp"] == init_pose.DEFAULT_KP[cmd["motor_id"] - 1] for cmd in ctrl.commands)


def test_integral_saturates_without_windup_and_unwinds_after_error_reverses(init_pose):
    compensator = init_pose.IntegralTorqueCompensator(
        ki=(100.,) * 5, torque_limit=(.2,) * 5, torque_rate=(.1,) * 5)
    for _ in range(100):
        compensator.update({1: dict(position=-1., velocity=0.)}, [0.] * 5, .1)
    assert compensator.torque == [.2, 0., 0., 0., 0.]
    compensator.update({1: dict(position=1., velocity=0.)}, [0.] * 5, .1)
    assert compensator.torque[0] == pytest.approx(.19)
    # A long scheduling interval cannot accumulate a large hidden torque step.
    compensator.update({1: dict(position=1., velocity=0.)}, [0.] * 5, 10.)
    assert compensator.torque[0] == pytest.approx(.18)


def test_integral_holds_in_deadband_and_freezes_growth_during_fast_motion(init_pose):
    compensator = init_pose.IntegralTorqueCompensator()
    compensator.torque[0] = .2
    compensator.update({1: dict(position=0., velocity=0.)}, [0.] * 5, .1)
    assert compensator.torque[0] == .2
    compensator.update({1: dict(position=-.1, velocity=1.)}, [0.] * 5, .1)
    assert compensator.torque[0] == .2
    compensator.update({1: dict(position=.1, velocity=1.)}, [0.] * 5, .1)
    assert compensator.torque[0] < .2  # Unwind even while moving after an overshoot.


def test_integral_does_not_grow_when_motor_feedback_is_incomplete(init_pose):
    compensator = init_pose.IntegralTorqueCompensator()
    ctrl = Motors(missing=(3,))
    with pytest.raises(RuntimeError, match="missing/stale"):
        _hold(init_pose, ctrl, integral_compensator=compensator)
    assert compensator.torque == [0.] * 5
    assert all(cmd["tor"] == 0 for cmd in ctrl.commands)


def test_integral_limit_does_not_turn_an_unreached_pose_into_success(init_pose, capsys):
    ctrl = LoadedMotors([.3] * 5)
    compensator = init_pose.IntegralTorqueCompensator(torque_limit=(.1,) * 5)
    assert not _hold(init_pose, ctrl, settle_timeout_s=5., integral_compensator=compensator)
    output = capsys.readouterr().out
    assert "NOT_REACHED" in output and "compensation_limit_motors=[1, 2, 3, 4]" in output
    assert all(abs(cmd["tor"]) <= .1 for cmd in ctrl.commands)


def test_enable_feedback_must_match_requested_id(init_pose):
    wrong = dict(motor_id=2, error_code=1, position=2., velocity=0., torque=0.)
    correct = dict(wrong, motor_id=1, position=.5)
    replies = iter([wrong, correct])
    ctrl = types.SimpleNamespace(enable_motor=lambda motor_id: wrong, receive=lambda size: next(replies),
                                 damiao_feedback=lambda reply: reply)
    assert init_pose._enable_and_read_current(ctrl, 1) == .5


def test_enable_retries_only_missing_reply_and_recovers_without_motion_commands(init_pose):
    class EnableAfterRetry(Motors):
        enable_calls = 0

        def enable_motor(self, motor):
            self.enable_calls += 1
            return None if self.enable_calls == 1 else super().enable_motor(motor)

        def receive(self, size):
            return b""

    ctrl, diagnostic = EnableAfterRetry(), {}
    assert init_pose._enable_and_read_current(ctrl, 1, attempts=3, feedback_timeout_s=.01,
                                              diagnostics=diagnostic) == 0.
    assert ctrl.enable_calls == 2 and len(diagnostic["attempts"]) == 2
    assert diagnostic["received_position"] and not ctrl.commands


def test_enable_exhaustion_records_wrong_motor_ids_without_accepting_them(init_pose):
    class WrongMotor(Motors):
        enable_calls = 0

        def enable_motor(self, motor):
            self.enable_calls += 1
            return self._feedback(2)

        def receive(self, size):
            return self._feedback(2)

    ctrl, diagnostic = WrongMotor(), {}
    assert init_pose._enable_and_read_current(ctrl, 1, attempts=3, feedback_timeout_s=.01,
                                              diagnostics=diagnostic) is None
    assert ctrl.enable_calls == 3 and not diagnostic["received_position"]
    assert all(a["observed_motor_ids"] == [2] and a["post_enable_reply_count"] > 0
               for a in diagnostic["attempts"])
    assert not ctrl.commands


def test_enable_fault_is_not_retried(init_pose):
    class Fault(Motors):
        enable_calls = 0

        def enable_motor(self, motor):
            self.enable_calls += 1
            return super().enable_motor(motor)

    ctrl = Fault(state=8)
    with pytest.raises(RuntimeError, match="status=8"):
        init_pose._enable_and_read_current(ctrl, 1, attempts=3)
    assert ctrl.enable_calls == 1 and not ctrl.commands


def test_driver_enable_diagnostics_include_consumed_disabled_and_wrong_id_frames(init_pose, monkeypatch):
    path = Path(init_pose.__file__).parent / "motor_control.py"
    monkeypatch.setitem(sys.modules, "serial", types.SimpleNamespace())
    spec = importlib.util.spec_from_file_location("v3_enable_driver_diagnostics_test", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.time = init_pose.time
    controller = driver.DaMiaoMotorController.__new__(driver.DaMiaoMotorController)
    controller.port, controller._rx_buffer = "/dev/test-RB", bytearray()
    replies = [bytes.fromhex("02 00 00 00 00 00 00 00"),  # Wrong motor, status=0.
               bytes.fromhex("01 00 00 00 00 00 00 00")]  # Requested motor still disabled.
    commands = []
    controller.send = commands.append
    def receive(size=8):
        driver.time.sleep(.01)
        return replies.pop(0) if replies else b""
    controller.receive = receive
    diagnostic = {}
    assert init_pose._enable_and_read_current(controller, 1, diagnostics=diagnostic) is None
    internal = diagnostic["attempts"][0]["driver_enable_receive"]
    assert internal["reply_count"] == 2
    assert internal["observed_motor_ids"] == [2, 1]
    assert internal["observed_status_codes"] == [0]
    assert len(internal["recent_reply_hex"]) == 2
    assert diagnostic["attempts"][0]["observed_motor_ids"] == [2, 1]
    assert len(commands) == 1 and commands[0][-1] == 0xFC


def test_parallel_startup_no_reply_cancels_before_any_leg_ramps(init_pose, monkeypatch, tmp_path):
    import json

    class Silent(Motors):
        def enable_motor(self, motor):
            return None

        def receive(self, size):
            return b""

    controllers = {}
    def open_leg(leg):
        ctrl = controllers[leg] = Silent() if leg == "RB" else Motors()
        return init_pose.LegController(leg, ctrl)

    path = tmp_path / "startup.json"
    monkeypatch.setattr(init_pose, "_open_leg_controller", open_leg)
    monkeypatch.setattr(sys, "argv", ["init_default_joint_pose.py", "--result_json", str(path),
                                     "--enable_feedback_timeout_s", ".01"])
    with pytest.raises(RuntimeError, match="RB motor 1.*ttyUSB2.*3 enable attempts"):
        init_pose.main()
    assert all(ctrl.closed and not ctrl.commands for ctrl in controllers.values())
    result = json.loads(path.read_text())["legs"]["RB"]["startup_feedback"]["1"]
    assert len(result["attempts"]) == 3 and not result["received_position"]


@pytest.mark.parametrize("state,position", [(0, 0.), (8, 0.), (1, math.nan)])
def test_disabled_faulty_or_nonfinite_startup_feedback_is_rejected(init_pose, state, position):
    ctrl = Motors(state=state)
    ctrl.positions[1] = position
    with pytest.raises(RuntimeError):
        init_pose._enable_and_read_current(ctrl, 1)


def test_main_reports_unsatisfied_legs_and_closes_ports(init_pose, monkeypatch):
    controllers = []
    def open_leg(leg):
        ctrl = Motors(response=1., bias=.15)
        controllers.append(ctrl)
        return init_pose.LegController(leg, ctrl)
    monkeypatch.setattr(init_pose, "_open_leg_controller", open_leg)
    monkeypatch.setattr(sys, "argv", ["init_default_joint_pose.py", "--leg_order", "LB", "LF",
                                     "--settle_timeout_s", "1", "--settle_time_s", ".2"])
    with pytest.raises(SystemExit, match=r"Pose not reached.*LB.*LF"):
        init_pose.main()
    assert len(controllers) == 2 and all(ctrl.closed for ctrl in controllers)


@pytest.mark.parametrize("options,expected_kp,expected_kd", [
    ([], (20., 20., 20., 20., 2.), (1., 1., 1., 1., .5)),
    (["--kp", "30", "20", "20", "20", "2"], (30., 20., 20., 20., 2.), (1., 1., 1., 1., .5)),
    (["--kd", "1.5", "1", "1", "1", ".5"], (20., 20., 20., 20., 2.), (1.5, 1., 1., 1., .5)),
])
def test_gain_overrides_reach_ramp_and_hold_without_rewriting_config(
    init_pose, monkeypatch, tmp_path, options, expected_kp, expected_kd,
):
    cfg_path = tmp_path / "damiao.yaml"
    original = "kp_truss: [20, 20, 20, 20, 2, 0]\nkd: [1, 1, 1, 1, 0.5, 0]\n"
    cfg_path.write_text(original)
    ctrl = Motors(response=1.)
    monkeypatch.setattr(init_pose, "_open_leg_controller", lambda leg: init_pose.LegController(leg, ctrl))
    monkeypatch.setattr(sys, "argv", ["init_default_joint_pose.py", "--leg_order", "LB",
                                     "--motor_config", str(cfg_path), "--compensation", "none", *options])
    init_pose.main()
    assert len(ctrl.commands) > 5  # Includes interpolation and final-target resends.
    assert all(command["kp"] == expected_kp[command["motor_id"] - 1]
               and command["kd"] == expected_kd[command["motor_id"] - 1]
               and command["tor"] == 0. for command in ctrl.commands)
    assert cfg_path.read_text() == original
    assert ctrl.closed


def test_integral_is_reset_between_legs_and_never_used_during_ramp(init_pose, monkeypatch):
    controllers, compensators = [], []
    hold = init_pose._hold_and_verify_leg
    def open_leg(leg):
        ctrl = LoadedMotors(init_pose._target_rad_for_leg(leg, True))
        controllers.append(ctrl)
        return init_pose.LegController(leg, ctrl)
    def verify(*args, **kwargs):
        compensator = kwargs["integral_compensator"]
        assert compensator.torque == [0.] * 5
        assert all(cmd["tor"] == 0 for cmd in args[0].ctrl.commands)
        compensators.append(compensator)
        return hold(*args, **kwargs)
    monkeypatch.setattr(init_pose, "_open_leg_controller", open_leg)
    monkeypatch.setattr(init_pose, "_hold_and_verify_leg", verify)
    monkeypatch.setattr(sys, "argv", ["init_default_joint_pose.py", "--leg_order", "LB", "LF",
                                     "--integral_compensation", "--settle_timeout_s", "20"])
    init_pose.main()
    assert len(compensators) == 2 and compensators[0] is not compensators[1]
    assert all(ctrl.closed for ctrl in controllers)
    assert all(any(cmd["tor"] != 0 for cmd in ctrl.commands) for ctrl in controllers)


@pytest.mark.parametrize("options", [
    ["--kp", "nan", "20", "20", "20", "2"],
    ["--kp", "501", "20", "20", "20", "2"],
    ["--kd", "-1", "1", "1", "1", ".5"],
    ["--kd", "6", "1", "1", "1", ".5"],
    ["--kp", "30", "20"],
    ["--integral_compensation", "--ki", "nan", "8", "8", "8", "1"],
    ["--integral_compensation", "--integral_torque_limit", "29", "1", "1", "1", ".2"],
    ["--integral_compensation", "--integral_torque_rate", "-1", "1", "1", "1", ".2"],
    ["--integral_compensation", "--integral_deadband_deg", "2"],
    ["--integral_compensation", "--integral_velocity_gate_deg_s", "0"],
])
def test_invalid_gain_overrides_are_rejected_before_opening_ports(init_pose, monkeypatch, options):
    def open_leg(leg):
        pytest.fail("Invalid gains must not open motor ports")
    monkeypatch.setattr(init_pose, "_open_leg_controller", open_leg)
    monkeypatch.setattr(sys, "argv", ["init_default_joint_pose.py", *options])
    with pytest.raises(SystemExit) as exc:
        init_pose.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("option,value", [("--hz", "nan"), ("--settle_timeout_s", "0"),
                                         ("--tolerance_deg", "-1"), ("--inter_command_sleep", "-1")])
def test_invalid_timing_or_tolerance_is_rejected_before_opening_ports(init_pose, monkeypatch, option, value):
    def open_leg(leg):
        pytest.fail("Invalid arguments must not open motor ports")
    monkeypatch.setattr(init_pose, "_open_leg_controller", open_leg)
    monkeypatch.setattr(sys, "argv", ["init_default_joint_pose.py", option, value])
    with pytest.raises(SystemExit) as exc:
        init_pose.main()
    assert exc.value.code == 2


def _gravity(module, leg="LB", loads=(2.5, .5, -.7, .8, .03)):
    from gravity_compensation import GravityFeedforward

    model = types.SimpleNamespace(
        torque_limits=np.array([4., 1.5, 1.5, 1.5, .2]),
        slew_rates=np.array([.3, .15, .15, .15, .02]),
        leg_torque=lambda *args: np.array(loads))
    controller = GravityFeedforward(model, max_dt_s=.04)
    return module.InitializationGravity(leg, controller, .5, (0., 0., -9.81), -math.pi/6)


def test_gravity_compensation_supports_load_without_error_integration(init_pose, capsys):
    ctrl = LoadedMotors([.3] * 5)
    compensation = _gravity(init_pose)
    result = {}
    assert _hold(init_pose, ctrl, hz=25., settle_timeout_s=20.,
                 gravity_compensator=compensation, result_out=result)
    assert all(abs(value-.3) <= .01 for value in ctrl.positions.values())
    assert result["integral_ff_nm"] == [0.] * 5
    assert result["gravity_ramp_complete"]
    assert result["gravity_raw_ff_nm"] == list(ctrl.loads)
    for motor in init_pose.MOTOR_IDS:
        sent = [cmd["tor"] for cmd in ctrl.commands if cmd["motor_id"] == motor]
        assert sent[0] == 0.
        assert result["gravity_ff_nm"][motor-1] == sent[-1]
        assert max(abs(value) for value in sent) <= compensation.controller.model.torque_limits[motor-1]
        step = compensation.controller.model.slew_rates[motor-1] * .04
        assert all(abs(b-a) <= step+1e-10 for a, b in zip(sent, sent[1:]))
    assert "gravity_sent_ff=" in capsys.readouterr().out


def test_gravity_cannot_report_reached_before_feedforward_has_finished_ramping(init_pose):
    ctrl = Motors(response=1.)  # Position follows instantly even while FF is still ramping.
    result = {}
    assert not _hold(init_pose, ctrl, hz=25., settle_timeout_s=1., gravity_compensator=_gravity(init_pose),
                     result_out=result)
    assert all(abs(value-.3) < 1e-12 for value in ctrl.positions.values())
    assert not result["gravity_ramp_complete"]


def test_gravity_uses_latest_measured_pose_not_interpolation_targets(init_pose):
    ctrl = Motors(response=.1)
    compensation = _gravity(init_pose, loads=(.1, .1, .1, .1, .01))
    poses = []
    def record(leg, positions, gravity, finger):
        assert leg == "LB" and tuple(gravity) == (0., 0., -9.81)
        poses.append(list(positions))
        return np.zeros(5)
    compensation.controller.model.leg_torque = record
    init_pose._send_leg_position_targets(
        init_pose.LegController("LB", ctrl), [1.] * 5, [0.] * 5,
        init_pose.DEFAULT_KP, init_pose.DEFAULT_KD, .1, 25., 0., gravity_compensator=compensation)
    assert poses and poses[0] == pytest.approx([.01]*5)  # Actual response to the first 0.1-rad target.
    assert poses[-1][0] < .9  # The final planned ramp target is 1 rad.


def test_gravity_missing_feedback_stops_ramp_before_any_nonzero_feedforward(init_pose):
    ctrl = Motors(missing=(3,))
    compensation = _gravity(init_pose)
    with pytest.raises(RuntimeError, match=r"missing/stale.*\[3\]"):
        init_pose._send_leg_position_targets(
            init_pose.LegController("LB", ctrl), [1.] * 5, [0.] * 5,
            init_pose.DEFAULT_KP, init_pose.DEFAULT_KD, .01, 25., 0., gravity_compensator=compensation)
    assert ctrl.commands and all(cmd["tor"] == 0 for cmd in ctrl.commands)


def test_gravity_fault_feedback_stops_ramp(init_pose):
    ctrl = Motors(state=8)
    with pytest.raises(RuntimeError, match="not enabled: status=8"):
        init_pose._send_leg_position_targets(
            init_pose.LegController("LB", ctrl), [1.] * 5, [0.] * 5,
            init_pose.DEFAULT_KP, init_pose.DEFAULT_KD, .1, 25., 0., gravity_compensator=_gravity(init_pose))


def test_gravity_feedback_age_is_not_reset_at_ramp_to_hold_transition(init_pose):
    ctrl = Motors(response=1.)
    compensation = _gravity(init_pose)
    init_pose._send_leg_position_targets(
        init_pose.LegController("LB", ctrl), [.3] * 5, [0.] * 5,
        init_pose.DEFAULT_KP, init_pose.DEFAULT_KD, .1, 25., 0., gravity_compensator=compensation)
    compensation.observe(init_pose._read_leg_feedback(ctrl), init_pose.time.monotonic())
    init_pose.time.sleep(.6)
    count = len(ctrl.commands)
    with pytest.raises(RuntimeError, match="missing/stale"):
        _hold(init_pose, ctrl, gravity_compensator=compensation)
    assert len(ctrl.commands) == count


def test_feedback_consumed_before_model_calculation_also_counts_for_hold_verification(init_pose):
    class BeforeModelOnly(Motors):
        polls = 0

        def receive_available(self):
            self.polls += 1
            # Endpoint responses arrive just after the post-send poll; the next
            # cycle's prepare() consumes them. Both are valid receipt paths.
            if self.polls % 2 == 0:
                return [], 0, 0, False
            return super().receive_available()

    ctrl = BeforeModelOnly(response=1.)
    result = {}
    assert _hold(init_pose, ctrl, hz=25., settle_time_s=.7,
                 gravity_compensator=_gravity(init_pose, loads=(0.,)*5), result_out=result)
    assert result["position_deg"] == pytest.approx([math.degrees(.3)]*5)
    assert result["missing_feedback_motors"] == result["stale_feedback_motors"] == []
    assert all(0 <= age < 500 for age in result["feedback_age_ms"].values())


def test_feedback_from_both_poll_sites_is_merged_per_motor(init_pose):
    class SplitPolls(Motors):
        polls = 0

        def receive_available(self):
            self.polls += 1
            selected = {1, 3, 5} if self.polls % 2 else {2, 4}
            frames = [frame for frame in self.pending if frame["motor_id"] in selected]
            self.pending = [frame for frame in self.pending if frame["motor_id"] not in selected]
            return frames, 0, len(frames)*8, False

    assert _hold(init_pose, SplitPolls(response=1.), hz=25., settle_time_s=.7,
                 gravity_compensator=_gravity(init_pose, loads=(0.,)*5))


def test_cached_in_tolerance_feedback_cannot_pass_without_new_receipts(init_pose):
    ctrl = Motors(response=1., missing=(1, 2, 3, 4, 5))
    gravity = _gravity(init_pose, loads=(0.,)*5)
    gravity.observe({motor: dict(motor_id=motor, error_code=1, position=.3, velocity=0., torque=0.)
                     for motor in init_pose.MOTOR_IDS}, init_pose.time.monotonic())
    with pytest.raises(RuntimeError, match="missing/stale"):
        _hold(init_pose, ctrl, hz=25., gravity_compensator=gravity)


def test_model_feedback_path_still_rejects_one_missing_motor(init_pose):
    class BeforeModelOnly(Motors):
        polls = 0

        def receive_available(self):
            self.polls += 1
            if self.polls % 2 == 0:
                return [], 0, 0, False
            return super().receive_available()

    with pytest.raises(RuntimeError, match=r"missing/stale.*\[3\]"):
        _hold(init_pose, BeforeModelOnly(response=1., missing=(3,)), hz=25.,
              gravity_compensator=_gravity(init_pose, loads=(0.,)*5))


def test_gravity_mode_main_keeps_one_controller_across_ramp_and_hold_and_resets_between_legs(
    init_pose, monkeypatch, capsys,
):
    controllers, instances, ramp_ff = [], {}, {}
    ramp, hold = init_pose._send_leg_position_targets, init_pose._hold_and_verify_leg
    def open_leg(leg):
        ctrl = Motors(response=1.)
        controllers.append(ctrl)
        return init_pose.LegController(leg, ctrl)
    def send_ramp(*args, **kwargs):
        instance = kwargs["gravity_compensator"]
        instances[instance.leg] = instance
        assert kwargs["hz"] == 25. and kwargs["inter_command_sleep"] == .0002
        assert instance.last_sent == [0.]*5
        ramp(*args, **kwargs)
        ramp_ff[instance.leg] = instance.last_sent.copy()
    def verify(*args, **kwargs):
        instance = kwargs["gravity_compensator"]
        assert instance is instances[instance.leg] and instance.last_sent == ramp_ff[instance.leg]
        assert kwargs["integral_compensator"] is None
        return hold(*args, **kwargs)
    monkeypatch.setattr(init_pose, "_open_leg_controller", open_leg)
    monkeypatch.setattr(init_pose, "_send_leg_position_targets", send_ramp)
    monkeypatch.setattr(init_pose, "_hold_and_verify_leg", verify)
    monkeypatch.setattr(sys, "argv", ["init_default_joint_pose.py", "--leg_order", "LB", "LF",
                                     "--enable_gravity_compensation"])
    init_pose.main()
    assert len(instances) == 2 and instances["LB"] is not instances["LF"]
    assert all(ctrl.closed for ctrl in controllers)
    for ctrl in controllers:
        assert {cmd["motor_id"] for cmd in ctrl.commands} == {1, 2, 3, 4, 5}
        assert all(cmd["tor"] == 0. for cmd in ctrl.commands[:5])
        assert any(cmd["tor"] != 0. for cmd in ctrl.commands[5:])
    output = capsys.readouterr().out
    assert "gravity_sent_ff=" in output and "final-target timeout=20s" in output


def test_gravity_preview_opens_no_ports_and_uses_hardware_j5_direction(init_pose, monkeypatch, capsys):
    monkeypatch.setattr(init_pose, "_open_leg_controller", lambda _: pytest.fail("Preview opened a port"))
    monkeypatch.setattr(sys, "argv", ["init_default_joint_pose.py", "--leg_order", "LB",
                                     "--enable_gravity_compensation", "--preview"])
    init_pose.main()
    output = capsys.readouterr().out
    assert "[PREVIEW]" in output and '"gravity_raw_ff_nm"' in output and "-24.0" in output


@pytest.mark.parametrize("options", [
    ["--enable_gravity_compensation", "--integral_compensation"],
    ["--enable_gravity_compensation", "--gravity_scale", "nan"],
    ["--enable_gravity_compensation", "--gravity_scale", "1.1"],
    ["--enable_gravity_compensation", "--gravity_b", "0", "nan", "-9.81"],
    ["--enable_gravity_compensation", "--gravity_finger_angle_deg", "nan"],
    ["--enable_gravity_compensation", "--fallback_current_deg", "0"],
])
def test_invalid_gravity_options_never_open_ports(init_pose, monkeypatch, options):
    monkeypatch.setattr(init_pose, "_open_leg_controller", lambda _: pytest.fail("Invalid arguments opened a port"))
    monkeypatch.setattr(sys, "argv", ["init_default_joint_pose.py", *options])
    with pytest.raises(SystemExit) as exc:
        init_pose.main()
    assert exc.value.code == 2


def test_gravity_model_load_failure_precedes_opening_ports(init_pose, monkeypatch, tmp_path):
    monkeypatch.setattr(init_pose, "_open_leg_controller", lambda _: pytest.fail("Invalid model opened a port"))
    monkeypatch.setattr(sys, "argv", ["init_default_joint_pose.py", "--enable_gravity_compensation",
                                     "--gravity_config", str(tmp_path / "missing.yaml")])
    with pytest.raises(FileNotFoundError):
        init_pose.main()


@pytest.mark.parametrize("write_delay,expected_hz,overruns", [(0., 25., False), (.012, 1/.06, True)])
def test_timing_measures_actual_cadence_and_detects_blocking_writes(init_pose, write_delay, expected_hz, overruns):
    class Transport(Motors):
        def send(self, command):
            init_pose.time.sleep(write_delay)
            super().send(command)
    ctrl = Transport(response=1.)
    timing = init_pose.InitializationTiming("LB", 25., interval_s=100., source="test simulated I/O")
    compensation = _gravity(init_pose)
    compensation.timing = timing
    init_pose._send_leg_position_targets(
        init_pose.LegController("LB", ctrl), [.2]*5, [0.]*5,
        init_pose.DEFAULT_KP, init_pose.DEFAULT_KD, .01, 25., 0., compensation, timing)
    result = timing.snapshot()["phases"]["ramp"]
    assert result["cycle_hz"] == pytest.approx(expected_hz)
    assert result["tx_cycles_hz"] == pytest.approx(expected_hz)
    assert bool(result["over_budget_cycles"]) == overruns
    assert result["samples"]["tx_ms"]["max"] == pytest.approx(write_delay*5*1000)
    assert result["samples"]["gravity_ms"]["n"] > 0
    # A later quiet interval must lower the reported rate instead of keeping an old rate.
    init_pose.time.sleep(1.)
    assert timing.snapshot()["phases"]["ramp"]["cycle_hz"] < expected_hz


def test_timing_counts_real_feedback_frames_before_latest_value_deduplication(init_pose, tmp_path):
    import json

    ctrl = Motors()
    timing = init_pose.InitializationTiming("LB", 25., tmp_path / "timing.jsonl", source="test simulated I/O")
    timing.begin("hold")
    ctrl.pending.extend([ctrl._feedback(1), ctrl._feedback(1), ctrl._feedback(2)])
    latest = init_pose._read_leg_feedback(ctrl, timing)
    assert len(latest) == 2
    init_pose.time.sleep(.04)
    timing.end()
    report = timing.report(final=True)
    phase = report["phases"]["hold"]
    assert phase["rx_frames_by_motor"][1] == 2
    assert phase["rx_frames_by_motor"][2] == 1
    assert phase["rx_min_motor_hz"] == 0.
    saved = json.loads((tmp_path / "timing.jsonl").read_text())
    assert saved["final"] and saved["source"] == "test simulated I/O"
    assert saved["phases"]["hold"]["rx_frames_by_motor"]["1"] == 2


def test_absolute_schedule_does_not_accumulate_post_wait_feedback_processing_time(init_pose):
    starts = []
    class DelayedFeedback(Motors):
        def receive_available(self):
            init_pose.time.sleep(.003)
            return super().receive_available()
        def send(self, command):
            if command["motor_id"] == 1:
                starts.append(init_pose.time.monotonic())
            super().send(command)
    ctrl = DelayedFeedback(response=1.)
    timing = init_pose.InitializationTiming("LB", 25., interval_s=100.)
    compensation = _gravity(init_pose, loads=(0.,)*5)
    compensation.timing = timing
    assert _hold(init_pose, ctrl, hz=25., gravity_compensator=compensation, timing=timing)
    assert len(starts) >= 5
    # The initial extra receive is a phase boundary effect; steady periods stay 40 ms.
    np.testing.assert_allclose(np.diff(starts)[1:], .04, atol=1e-10)
    result = timing.snapshot()["phases"]["hold"]
    assert result["schedule_mode"] == "absolute" and result["skipped_deadlines"] == 0


def test_four_legs_move_together_keep_holding_and_emit_one_final_report(init_pose, monkeypatch, tmp_path, capsys):
    import json

    controllers, enabled, thread_ids = {}, set(), set()
    opening = threading.Barrier(4)

    class ParallelMotors(Motors):
        def __init__(self, leg):
            super().__init__(response=1.)
            self.leg = leg
            self.targets = init_pose._target_rad_for_leg(leg, True)
            self.hold_cycles = 0

        def enable_motor(self, motor):
            enabled.add((self.leg, motor))
            return super().enable_motor(motor)

        def send(self, command):
            assert len(enabled) == 20  # No leg ramps while another is still enabling.
            motor = command["motor_id"]
            if motor == 1 and command["pos"] == self.targets[0]:
                self.hold_cycles += 1
            self.bias = .1 if self.leg == "RF" and self.hold_cycles < 8 else 0.
            super().send(command)

    def open_leg(leg):
        thread_ids.add(threading.get_ident())
        opening.wait(timeout=2.)  # A sequential implementation cannot pass.
        ctrl = controllers[leg] = ParallelMotors(leg)
        return init_pose.LegController(leg, ctrl)

    summary = init_pose._print_initialization_summary
    def final_summary(legs, results):
        assert all(ctrl.closed for ctrl in controllers.values())
        assert all(result["reached"] for result in results.values())
        summary(legs, results)

    output = tmp_path / "result.json"
    monkeypatch.setattr(init_pose, "_open_leg_controller", open_leg)
    monkeypatch.setattr(init_pose, "_print_initialization_summary", final_summary)
    monkeypatch.setattr(sys, "argv", ["init_default_joint_pose.py", "--max_step_deg", "10",
                                     "--inter_command_sleep", "0", "--result_json", str(output)])
    init_pose.main()
    assert len(thread_ids) == 4
    # All use nine interpolation steps, then independently keep sending holds.
    for leg, ctrl in controllers.items():
        positions = [c["pos"] for c in ctrl.commands if c["motor_id"] == 1]
        np.testing.assert_allclose(positions[:9], np.arange(1,10)/9*ctrl.targets[0], atol=1e-12)
    assert all(ctrl.hold_cycles >= 8 for ctrl in controllers.values())
    data = json.loads(output.read_text())
    assert list(data["legs"]) == ["LB", "LF", "RB", "RF"]
    assert all(result["reached"] and result["error_deg"] == pytest.approx([0.]*5)
               and len(result["torque_nm"]) == 5 for result in data["legs"].values())
    text = capsys.readouterr().out
    assert text.count("[SUMMARY]") == 1 and text.count("[RESULT]") == 4


@pytest.mark.parametrize("phase", ["enable", "ramp", "hold"])
def test_one_leg_fault_cancels_parallel_peers_and_closes_every_port(init_pose, monkeypatch, phase, capsys):
    controllers = {}

    class FailingMotors(Motors):
        def enable_motor(self, motor):
            if phase == "enable":
                raise RuntimeError("test leg transport fault")
            return super().enable_motor(motor)

        def send(self, command):
            if (phase == "ramp" and len(self.commands) >= 5 or
                    phase == "hold" and len(self.commands) >= 5
                    and command["pos"] == init_pose._target_rad_for_leg("RF", True)[command["motor_id"]-1]):
                raise RuntimeError("test leg transport fault")
            super().send(command)

    def open_leg(leg):
        ctrl = controllers[leg] = FailingMotors(response=1.) if leg == "RF" else Motors(response=1.)
        return init_pose.LegController(leg, ctrl)

    monkeypatch.setattr(init_pose, "_open_leg_controller", open_leg)
    monkeypatch.setattr(sys, "argv", ["init_default_joint_pose.py", "--max_step_deg", "10",
                                     "--inter_command_sleep", "0"])
    with pytest.raises(RuntimeError, match="test leg transport fault"):
        init_pose.main()
    assert len(controllers) == 4 and all(ctrl.closed for ctrl in controllers.values())
    assert "[SUMMARY]" in capsys.readouterr().out


def test_duplicate_leg_ports_are_rejected_before_hardware_access(init_pose, monkeypatch):
    monkeypatch.setattr(init_pose, "_open_leg_controller", lambda _: pytest.fail("Opened duplicate port"))
    monkeypatch.setattr(sys, "argv", ["init_default_joint_pose.py", "--leg_order", "LB", "LB"])
    with pytest.raises(SystemExit) as exc:
        init_pose.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("mode", ["none", "nominal", "fitted", "fitted_j1", "table"])
def test_four_compensation_choices_preview_without_hardware(init_pose, monkeypatch, capsys, mode):
    monkeypatch.setattr(init_pose, "_open_leg_controller", lambda _: pytest.fail("Preview opened hardware"))
    monkeypatch.setattr(sys, "argv", ["init_default_joint_pose.py", "--compensation", mode, "--preview"])
    init_pose.main()
    output = capsys.readouterr().out
    assert f"compensation={mode}" in output and "[PREVIEW]" in output
    assert ('"gravity_raw_ff_nm"' in output) == (mode != "none")


def test_fitted_changes_only_node_masses_and_has_correct_potential_gradient(init_pose):
    from gravity_compensation import GravityModel
    from initialization_models import FittedNodeGravityModel

    nominal = GravityModel()
    fitted = FittedNodeGravityModel(nominal.config_path, None, init_pose.DEFAULT_NODE_MASS_FILE)
    for leg in init_pose.DEFAULT_LEG_ORDER:
        angles = init_pose._target_rad_for_leg(leg, True)
        state = fitted.mass_state(leg, angles)
        np.testing.assert_allclose(state.masses[:6], fitted.fit["node_masses_kg"][leg])
        np.testing.assert_array_equal(state.masses[6:], nominal.mass_state(leg, angles).masses[6:])
        np.testing.assert_array_equal(fitted.torque_limits, nominal.torque_limits)
        np.testing.assert_array_equal(fitted.slew_rates, nominal.slew_rates)
        q = np.array(angles)
        numerical = [(fitted.potential_energy(leg, q+np.eye(5)[j]*1e-6)
                      - fitted.potential_energy(leg, q-np.eye(5)[j]*1e-6))/2e-6 for j in range(5)]
        np.testing.assert_allclose(fitted.leg_torque(leg, angles), numerical, atol=1e-8)
    assert not fitted.description()["nuisance_terms_applied"]


@pytest.mark.parametrize("mode", ["nominal", "fitted", "fitted_j1", "table"])
def test_selected_model_runs_through_parallel_ramp_hold_and_is_recorded(
    init_pose, monkeypatch, tmp_path, mode,
):
    import json

    controllers = []
    def open_leg(leg):
        ctrl = Motors(response=1.)
        controllers.append(ctrl)
        return init_pose.LegController(leg, ctrl)

    output = tmp_path / "result.json"
    monkeypatch.setattr(init_pose, "_open_leg_controller", open_leg)
    monkeypatch.setattr(sys, "argv", ["init_default_joint_pose.py", "--compensation", mode,
                                     "--result_json", str(output)])
    init_pose.main()
    data = json.loads(output.read_text())
    assert data["configuration"]["compensation"] == mode
    assert data["configuration"]["hz"] == 25.
    assert all(result["reached"] and result["compensation"] == mode for result in data["legs"].values())
    assert len(controllers) == 4 and all(ctrl.closed for ctrl in controllers)
    assert all(any(cmd["tor"] != 0 for cmd in ctrl.commands) for ctrl in controllers)


@pytest.mark.parametrize("legacy_enable", [False, True])
def test_default_initialization_loads_fitted_masses_and_j1_residual_without_ports(init_pose, monkeypatch, capsys, legacy_enable):
    import json
    from initialization_models import make_compensation_model

    monkeypatch.setattr(init_pose, "_open_leg_controller", lambda _: pytest.fail("Preview opened hardware"))
    monkeypatch.setattr(sys, "argv", ["init_default_joint_pose.py", "--preview"]
                        + (["--enable_gravity_compensation"] if legacy_enable else []))
    init_pose.main()
    output = capsys.readouterr().out
    expected = make_compensation_model().description()
    assert Path(expected["node_mass_file"]).parent == init_pose.THIS_DIR / "calibration"
    assert Path(expected["j1_residual"]["path"]).parent == init_pose.THIS_DIR / "calibration"
    assert "compensation=fitted_j1" in output and "command rate=25 Hz" in output
    assert expected["node_mass_sha256"] in output and expected["j1_residual"]["sha256"] in output
    assert json.loads(output.split("[PREVIEW]")[1].split('\n', 1)[1])[0]["compensation"] == "fitted_j1"


def test_default_initialization_missing_residual_fails_before_ports(init_pose, monkeypatch, tmp_path):
    monkeypatch.setattr(init_pose, "_open_leg_controller", lambda _: pytest.fail("Invalid model opened hardware"))
    monkeypatch.setattr(sys, "argv", ["init_default_joint_pose.py", "--j1_residual_file", str(tmp_path/"missing.json")])
    with pytest.raises(FileNotFoundError):
        init_pose.main()


@pytest.mark.parametrize("extra", [
    ["--compensation", "fitted", "--integral_compensation"],
    ["--compensation", "nominal", "--enable_gravity_compensation"],
    ["--compensation", "typo"],
    ["--compensation", "fitted", "--fallback_current_deg", "0"],
])
def test_conflicting_comparison_options_fail_before_ports(init_pose, monkeypatch, extra):
    monkeypatch.setattr(init_pose, "_open_leg_controller", lambda _: pytest.fail("Invalid options opened hardware"))
    monkeypatch.setattr(sys, "argv", ["init_default_joint_pose.py", *extra])
    with pytest.raises(SystemExit) as exc:
        init_pose.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("change", ["negative", "geometry"])
def test_invalid_node_mass_file_fails_before_ports(init_pose, monkeypatch, tmp_path, change):
    import json

    data = json.loads(init_pose.DEFAULT_NODE_MASS_FILE.read_text())
    if change == "negative":
        data["node_masses_kg"]["LB"][1] = -1.
    else:
        data["geometry"]["urdf_sha256"] = "wrong"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(init_pose, "_open_leg_controller", lambda _: pytest.fail("Invalid mass file opened hardware"))
    monkeypatch.setattr(sys, "argv", ["init_default_joint_pose.py", "--compensation", "fitted",
                                     "--node_mass_file", str(path)])
    with pytest.raises(ValueError):
        init_pose.main()


def _residual_candidate(init_pose, tmp_path, bias=.1):
    import json
    from initialization_models import FittedNodeGravityModel
    base = FittedNodeGravityModel(init_pose.THIS_DIR/"gravity_compensation.yaml", None,
                                 init_pose.DEFAULT_NODE_MASS_FILE)
    data = dict(schema_version=1, model_type="j1_pose_direction_residual", base_model=base.description(),
                direction_deadband_deg=.25, scope="test", legs={leg: dict(
                    gain=1., bias_nm=bias, gap_nm=.5, gap_slope_nm=0., center_deg=0., scale_deg=30.,
                    angle_range_deg=[-180., 180.]) for leg in init_pose.DEFAULT_LEG_ORDER})
    path = tmp_path/"residual.json"
    path.write_text(json.dumps(data))
    return base, path


def test_j1_residual_uses_motion_direction_retains_at_rest_and_never_accumulates_error(init_pose, tmp_path):
    from j1_residual import J1ResidualModel
    base, path = _residual_candidate(init_pose, tmp_path)
    model = J1ResidualModel(base, path)
    q = np.array(init_pose._target_rad_for_leg("LB", True))
    np.testing.assert_allclose(model.leg_torque("LB", q)-base.leg_torque("LB", q), [.1,0,0,0,0], atol=1e-12)
    q[0] += math.radians(.4)
    expected = model.leg_torque("LB", q)
    assert model.leg_diagnostics("LB")["j1_residual_direction"] == 1
    assert expected[0]-base.leg_torque("LB", q)[0] == pytest.approx(.6)
    for _ in range(100):
        np.testing.assert_array_equal(model.leg_torque("LB", q), expected)
    q[0] -= math.radians(.5)
    value = model.leg_torque("LB", q)
    assert value[0]-base.leg_torque("LB", q)[0] == pytest.approx(-.4)
    np.testing.assert_array_equal(value[1:], base.leg_torque("LB", q)[1:])


def test_j1_residual_caps_output_and_shared_controller_keeps_slew_limit_on_reversal(init_pose, tmp_path):
    from j1_residual import J1ResidualModel
    from gravity_compensation import GravityFeedforward
    base, path = _residual_candidate(init_pose, tmp_path, bias=2.)
    model = J1ResidualModel(base, path, limit_nm=.8)
    q = np.array(init_pose._target_rad_for_leg("LB", True))
    assert model.leg_torque("LB", q)[0]-base.leg_torque("LB", q)[0] == pytest.approx(.8)
    controller = GravityFeedforward(model)
    outputs = []
    for index in range(50):
        q[0] += math.radians(.4 if index < 25 else -.4)
        outputs.append(controller.update_leg("LB", q, now=index*.04))
    assert np.all(abs(np.diff(outputs, axis=0)) <= base.slew_rates*.04+1e-12)


def test_j1_residual_joint_major_api_corrects_only_j1_and_zero_scale_is_baseline(init_pose, tmp_path):
    from j1_residual import J1ResidualModel
    base, path = _residual_candidate(init_pose, tmp_path)
    q = np.array([init_pose._target_rad_for_leg(leg, True) for leg in init_pose.DEFAULT_LEG_ORDER]).T.ravel()
    model = J1ResidualModel(base, path)
    np.testing.assert_allclose(model.torque(q)-base.torque(q), [.1]*4+[0.]*16, atol=1e-12)
    np.testing.assert_array_equal(J1ResidualModel(base, path, scale=0).torque(q), base.torque(q))


def test_j1_residual_cli_is_explicit_and_logged_in_parallel_results(init_pose, monkeypatch, tmp_path):
    import json
    _, path = _residual_candidate(init_pose, tmp_path)
    monkeypatch.setattr(init_pose, "_open_leg_controller", lambda leg: init_pose.LegController(leg, Motors(response=1.)))
    output = tmp_path/"output.json"
    monkeypatch.setattr(sys, "argv", ["init_default_joint_pose.py", "--compensation", "fitted",
                                     "--j1_residual_file", str(path), "--result_json", str(output)])
    init_pose.main()
    result = json.loads(output.read_text())
    assert result["configuration"]["model"]["j1_residual"]["sha256"]
    assert all(v["reached"] and "j1_residual_nm" in v and v["integral_ff_nm"] == [0.]*5
               for v in result["legs"].values())


def test_j1_residual_rejects_different_base_mass_file(init_pose, tmp_path):
    import json
    from j1_residual import J1ResidualModel
    base, path = _residual_candidate(init_pose, tmp_path)
    data = json.loads(path.read_text())
    data["base_model"]["node_mass_sha256"] = "different"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="node_mass_sha256"):
        J1ResidualModel(base, path)


def test_j1_residual_fit_recovers_known_gravity_gain_and_direction_terms(init_pose):
    from fit_j1_residual import fit, design
    rows, values = [], []
    for repeat in (1, 2, 3):
        for direction in (-1, 1):
            for q in np.linspace(35., 95., 31):
                rows.append(dict(leg="LB", joint=1, round=repeat, direction=direction,
                                 angles=np.deg2rad([q,45,-20,24,-24])))
                values.append(4*np.sin(np.deg2rad(q))-2)
    base = np.array(values)
    coefficients = np.array([1.03,.1,.6,.2])
    target = design(rows, base, True)@coefficients
    for row, torque in zip(rows, target):
        row["torque"] = torque
    np.testing.assert_allclose(fit(rows, base, True), coefficients, atol=1e-8)


def test_default_timing_does_not_print_or_write_periodic_reports_inside_control(init_pose, monkeypatch):
    timing = init_pose.InitializationTiming("LB", 25.)
    monkeypatch.setattr(timing, "report", lambda *a, **k: pytest.fail("Control loop emitted synchronous report"))
    timing.begin("hold")
    init_pose.time.sleep(3.)
    timing.end()
    assert timing.snapshot()["phases"]["hold"]["cycles"] == 1


def test_receive_audit_distinguishes_empty_polls_partial_frames_and_unknown_ids(init_pose):
    class AuditedTransport(Motors):
        _rx_buffer = b"12"

        def receive_available(self):
            # A complete frame with unknown ID must be visible even if ignored.
            return [dict(motor_id=9, error_code=1, position=0., velocity=0., torque=0.)], 0, 8, False

    timing = init_pose.InitializationTiming("LB", 25.)
    assert init_pose._read_leg_feedback(AuditedTransport(), timing) == {}
    snapshot = timing.io_snapshot()
    assert snapshot["rx_complete_frames"] == 1 and snapshot["valid_frames"] == 0
    assert snapshot["recent_events"][-1]["partial_buffer_bytes"] == 2
    for _ in range(100):
        timing.received([], 0, 0, 0, False)
    assert len(timing.io_snapshot()["recent_events"]) == 64


def test_real_feedback_loss_still_aborts_with_io_history_and_no_timeout_extension(init_pose, monkeypatch, tmp_path):
    import json
    class DropsFeedback(Motors):
        def send(self, command):
            super().send(command)
            self.pending.clear()

    def open_leg(leg):
        return init_pose.LegController(leg, DropsFeedback(response=1.) if leg == "LB" else Motors(response=1.))

    # Non-model mode checks dropout during hold, after all ramp steps.
    path = tmp_path/"dropout.json"
    monkeypatch.setattr(init_pose, "_open_leg_controller", open_leg)
    monkeypatch.setattr(sys, "argv", ["init_default_joint_pose.py", "--compensation", "none", "--result_json", str(path)])
    with pytest.raises(RuntimeError, match=r"LB: missing/stale.*\[1, 2, 3, 4, 5\]"):
        init_pose.main()
    result = json.loads(path.read_text())["legs"]["LB"]
    assert all(age is None for age in result["feedback_age_ms"].values())
    assert result["duration_s"] <= .7
    io = result["io_diagnostics"]
    assert io["tx_cycles"] > 0 and io["rx_polls"] > 0 and io["valid_frames"] == 0
    assert any(e["kind"] == "tx" for e in io["recent_events"])


def test_parallel_readiness_does_not_block_a_leg_waiting_for_peer(init_pose):
    group = init_pose.ParallelInitialization(("LB", "LF"))
    finished = threading.Event()
    answer = []
    def worker():
        answer.append(group.verify("LB", True, False))
        finished.set()
    thread = threading.Thread(target=worker)
    thread.start()
    completed_without_peer = finished.wait(.5)
    if not completed_without_peer:
        group.abort()
    thread.join(timeout=1.)
    assert completed_without_peer and answer == [False]
    assert group.verify("LF", True, False)


def test_old_peer_ready_state_cannot_complete_group_after_feedback_timeout(init_pose):
    group = init_pose.ParallelInitialization(("LB", "LF"))
    assert not group.verify("LB", True, False)
    init_pose.time.sleep(.6)
    assert not group.verify("LF", True, False, .5)
    assert group.verify("LB", True, False, .5)


def test_peer_fault_cancellation_remains_effective_without_cycle_barriers(init_pose):
    group = init_pose.ParallelInitialization(("LB", "LF"))
    group.abort()
    with pytest.raises(threading.BrokenBarrierError):
        group.check()
