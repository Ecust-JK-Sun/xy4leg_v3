"""Gravity mathematics and command-path checks; no simulator or hardware is opened."""

from __future__ import annotations

import importlib.util
import math
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest
import yaml


REAL = Path(__file__).parents[1] / "isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gravity = _module("v3_gravity_test", REAL / "gravity_compensation.py")
shared_mod = _module("v3_gravity_shared_test", REAL / "threads_shared.py")
motor_mod = _module("v3_gravity_motor_test", REAL / "motor_session_gripper.py")


@pytest.fixture
def model():
    return gravity.GravityModel()


@pytest.mark.parametrize("leg", gravity.LEGS)
def test_endpoint_equivalence_mass_conservation_and_j6_hinge_midpoint(model, leg):
    rng = np.random.default_rng(18)
    for _ in range(8):
        q, g = rng.uniform(-2., 2., 5), rng.uniform(-9., 9., 3)
        finger = rng.uniform(-.6, -.2)
        endpoints = model.mass_state(leg, q, finger)
        midpoints = model.mass_state(leg, q, finger, "midpoints")
        assert endpoints.masses.sum() == pytest.approx(3.35)
        np.testing.assert_allclose(endpoints.masses[:6], [.525, .55, .55, .55, .55, .525])
        np.testing.assert_allclose(
            endpoints.masses @ endpoints.positions, midpoints.masses @ midpoints.positions, atol=1e-14)
        np.testing.assert_allclose(model.leg_torque(leg, q, g, finger),
                                   model.leg_torque(leg, q, g, finger, "midpoints"), atol=1e-12)
        assert model.potential_energy(leg, q, g, finger) == pytest.approx(
            model.potential_energy(leg, q, g, finger, "midpoints"), abs=1e-12)
        # J6 belongs to Link5: opening the fingers cannot move the motor mass.
        other = model.mass_state(leg, q, finger + .4)
        np.testing.assert_allclose(endpoints.motor_positions, other.motor_positions, atol=1e-14)
        transform = np.eye(4)
        for angle, (origin, axis) in zip(model.signs*q + model.offsets[gravity.LEGS.index(leg)], model.chains[leg]):
            motion = np.eye(4)
            motion[:3, :3] = gravity._rotation(axis, angle)
            transform = transform @ origin @ motion
        expected = sum((transform @ item[0])[:3, 3] for item in model.fingers[leg]) / 2
        np.testing.assert_allclose(endpoints.motor_positions[-1], expected, atol=1e-14)


@pytest.mark.parametrize("leg", gravity.LEGS)
def test_gravity_torque_matches_potential_energy_derivative_in_hardware_coordinates(model, leg):
    # Nonzero offsets and nonvertical gravity exercise coordinate/sign conversions.
    model.offsets[gravity.LEGS.index(leg)] = np.deg2rad([3., -5., 2., 7., -9.])
    rng = np.random.default_rng(91)
    for _ in range(5):
        q, g = rng.uniform(-1.8, 1.8, 5), rng.normal(size=3)
        g *= 9.81 / np.linalg.norm(g)
        numerical = np.empty(5)
        for index in range(5):
            delta = np.eye(5)[index] * 1e-6
            numerical[index] = (model.potential_energy(leg, q+delta, g)
                                - model.potential_energy(leg, q-delta, g)) / 2e-6
        np.testing.assert_allclose(model.leg_torque(leg, q, g), numerical, atol=1e-8, rtol=1e-7)


def test_two_distinct_other_joint_postures_change_upstream_compensation(model):
    q1, q2 = np.deg2rad([65., 45., -20., 24., -24.]), np.deg2rad([65., 0., 30., -10., -24.])
    assert abs(model.leg_torque("LB", q1)[0] - model.leg_torque("LB", q2)[0]) > .1
    np.testing.assert_array_equal(model.leg_torque("LB", q1, [0., 0., 0.]), np.zeros(5))


def test_straight_horizontal_chain_has_analytically_known_gravity_torque(tmp_path):
    # Six motors at x=0..5 m and five 50 g uniform rods; all axes are +y.
    parts = ['<robot name="test"><link name="base_link"/>']
    for leg in gravity.LEGS:
        parent = "base_link"
        for joint in range(1, 6):
            child = f"{leg}_Link{joint}"
            parts.append(f'<link name="{child}"/><joint name="{leg}_Joint{joint}" type="revolute">'
                         f'<parent link="{parent}"/><child link="{child}"/>'
                         f'<origin xyz="{0 if joint == 1 else 1} 0 0"/><axis xyz="0 1 0"/></joint>')
            parent = child
        for finger in (1, 2):
            child = f"{leg}_Finger{finger}"
            parts.append(f'<link name="{child}"><inertial><origin xyz="0 0 0"/></inertial></link>'
                         f'<joint name="{leg}_{finger}out_Joint" type="revolute">'
                         f'<parent link="{parent}"/><child link="{child}"/>'
                         '<origin xyz="1 0 0"/><axis xyz="0 1 0"/></joint>')
    urdf = tmp_path / "chain.urdf"
    urdf.write_text("".join(parts) + "</robot>")
    cfg = yaml.safe_load(gravity.DEFAULT_CONFIG.read_text())
    cfg.update(urdf_path=str(urdf), joint_signs=[1]*5, finger_mass_kg=0.)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    simple = gravity.GravityModel(config)
    expected = []
    for joint in range(5):
        motor_moment = .5 * sum(k-joint for k in range(joint+1, 6))
        rod_moment = .05 * sum(k+.5-joint for k in range(joint, 5))
        expected.append(-9.81 * (motor_moment + rod_moment))
    np.testing.assert_allclose(simple.leg_torque("LB", np.zeros(5)), expected, atol=1e-12)


def test_all_legs_interleaved_order_and_gravity_reversal(model):
    q = np.deg2rad(gravity.DEFAULT_POSE_DEG).T.reshape(20)
    torque = model.torque(q)
    for index, leg in enumerate(gravity.LEGS):
        np.testing.assert_allclose(torque[index::4], model.leg_torque(leg, q[index::4]))
    np.testing.assert_allclose(model.torque(q, [0., 0., 9.81]), -torque, atol=1e-12)


@pytest.mark.parametrize("leg", gravity.LEGS)
def test_single_leg_controller_matches_playback_full_body_controller(model, leg):
    single, full = gravity.GravityFeedforward(model), gravity.GravityFeedforward(model)
    q = np.deg2rad(gravity.DEFAULT_POSE_DEG).T.reshape(20)
    index = gravity.LEGS.index(leg)
    g, fingers = [1.2, -2.3, -9.4], np.array([-.3, -.4, -.5, -.25])
    for stamp in (1., 1.04, 1.08, 1.12, 2., 2.04):
        q[index+4] += .05
        expected = full.update(q, g, fingers, now=stamp)[index::4]
        actual = single.update_leg(leg, q[index::4], g, fingers[index], now=stamp)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(single.raw[index::4], full.raw[index::4], atol=1e-12)


def test_rate_limit_zero_start_saturation_and_long_pause(model):
    model.torque = lambda *args: np.full(20, 100.)
    controller = gravity.GravityFeedforward(model, scale=.5, max_dt_s=.04)
    np.testing.assert_array_equal(controller.update(np.zeros(20), [0, 0, -9.81], now=1.), np.zeros(20))
    step = np.repeat(model.slew_rates, 4) * .04
    np.testing.assert_allclose(controller.update(np.zeros(20), [0, 0, -9.81], now=1.04), step)
    np.testing.assert_allclose(controller.update(np.zeros(20), [0, 0, -9.81], now=100.), 2*step)
    assert controller.saturated.all()
    for index in range(1000):
        controller.update(np.zeros(20), [0, 0, -9.81], now=100.04+index*.04)
    np.testing.assert_allclose(controller.output, np.repeat(model.torque_limits, 4))
    with pytest.raises(ValueError, match="monotonic"):
        controller.update(np.zeros(20), [0, 0, -9.81], now=0.)


@pytest.mark.parametrize("q,g", [(np.zeros(19), [0, 0, -9.81]), (np.full(20, np.nan), [0, 0, -9.81]),
                                  (np.zeros(20), [0, math.inf, 0])])
def test_invalid_sensor_inputs_are_rejected(model, q, g):
    with pytest.raises(ValueError):
        model.torque(q, g)


def test_shared_command_keeps_feedforward_atomic_and_resets_it_for_legacy_publications():
    shared = shared_mod.SharedData()
    action, ff = np.arange(24, dtype=float), np.arange(20, dtype=float)
    shared.set_action(action, observed_at=1., feedforward=ff)
    action[:] = ff[:] = -1
    act, torque, stamp, sequence, observed = shared.command_snapshot()
    np.testing.assert_array_equal(act, np.arange(24))
    np.testing.assert_array_equal(torque, np.arange(20))
    assert sequence == 1 and observed == 1.
    with pytest.raises(RuntimeError, match="expired"):
        shared.command_snapshot(max_age_s=.1, now=stamp+1.)
    shared.set_action(np.zeros(24))
    assert shared.command_snapshot()[1] is None


def _session():
    class Controller:
        def __init__(self):
            self.commands = []
        def damiao_mit_command(self, **kwargs):
            return kwargs
        def send(self, command):
            self.commands.append(command)
    controllers = [Controller() for _ in range(4)]
    return motor_mod.MotorSession(*controllers, *([list(range(1, 7))]*4),
                                  [20, 20, 20, 20, 2, 0], [1, 1, 1, 1, .5, 0],
                                  np.zeros(24), np.zeros(24), enabled=True)


def test_mit_transmits_feedforward_by_joint_and_preserves_gripper_torque(monkeypatch):
    monkeypatch.setattr(motor_mod.time, "sleep", lambda _: None)
    session = _session()
    target = np.arange(24) * .01
    torque = np.arange(20) * .1
    session.send_targets(target, feedforward_torque=torque)
    for offset, (ctrl, _, _) in enumerate(session._legs()):
        for joint, command in enumerate(ctrl.commands[:5]):
            assert command["tor"] == torque[offset+joint*4]
            assert command["pos"] == pytest.approx(target[offset+joint*4])
        assert ctrl.commands[5]["tor"] == pytest.approx(target[20+offset])
        assert ctrl.commands[5]["kp"] == ctrl.commands[5]["kd"] == 0.


@pytest.mark.parametrize("bad", [np.zeros(24), np.full(20, np.nan), np.full(20, 29.)])
def test_invalid_feedforward_rejected_before_any_motor_is_written(bad):
    session = _session()
    with pytest.raises(ValueError):
        session.send_targets(np.zeros(24), feedforward_torque=bad)
    assert all(not ctrl.commands for ctrl, _, _ in session._legs())


@pytest.mark.parametrize("mode", ["periodic", "on_update"])
def test_transmit_thread_sends_position_and_feedforward_from_one_snapshot(monkeypatch, mode):
    monkeypatch.syspath_prepend(str(REAL))
    monkeypatch.setitem(sys.modules, "threads_shared", shared_mod)
    serial = _module(f"v3_gravity_serial_{mode}", REAL / "serial_threads.py")
    shared, stop, sent = shared_mod.SharedData(), threading.Event(), threading.Event()
    publications = []
    def send(action, feedforward_torque):
        publications.append((action.copy(), feedforward_torque.copy()))
        sent.set()
        stop.set()
    shared.set_action(np.ones(24), feedforward=np.full(20, .25))
    thread = serial.start_serial_tx_thread(shared, True, types.SimpleNamespace(send_targets=send),
                                          stop_event=stop, mode=mode)
    try:
        assert sent.wait(1.)
    finally:
        stop.set()
        thread.join(timeout=1.)
    shared.raise_if_failed()
    assert len(publications) == 1
    np.testing.assert_array_equal(publications[0][0], np.ones(24))
    np.testing.assert_array_equal(publications[0][1], np.full(20, .25))


def test_static_comparison_uses_full_measured_pose_and_does_not_fit(model, tmp_path):
    q_deg = gravity.DEFAULT_POSE_DEG[0]
    predicted = model.leg_torque("LB", np.deg2rad(q_deg))[0]
    path = tmp_path / "points.csv"
    path.write_text("leg,joint,q1_deg,q2_deg,q3_deg,q4_deg,q5_deg,torque_nm\n"
                    + "LB,1," + ",".join(str(value) for value in q_deg) + f",{predicted+.3}\n")
    result = gravity.compare_static_points(model, path)
    assert result["samples"] == 1
    assert result["groups"][0]["mae_nm"] == pytest.approx(.3)
    assert result["groups"][0]["bias_nm"] == pytest.approx(-.3)
    assert model.leg_torque("LB", np.deg2rad(q_deg))[0] == predicted
