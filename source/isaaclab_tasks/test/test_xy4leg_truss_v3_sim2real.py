"""CPU-only regression checks; no Isaac Sim, serial ports or cameras are opened."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3"
REAL = TASK / "sim2real_v3"
LOCAL_BASELINE = ROOT / (
    "logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar_sim2real/"
    "model_68500.pt"
)
FIXTURE_BASELINE = Path(__file__).parent / "fixtures/xy4leg_truss_v3/model_68500.pt"
# These tests load params/ only; no model weights are needed. A clean checkout
# uses the committed interface snapshot instead of depending on ignored logs/.
BASELINE = LOCAL_BASELINE if all(
    (LOCAL_BASELINE.parent / "params" / name).is_file() for name in ("env.yaml", "agent.yaml")
) else FIXTURE_BASELINE


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _module("v3_deployment_config_test", REAL / "deployment_config.py")
shared_mod = _module("v3_shared_test", REAL / "threads_shared.py")
motor = _module("v3_motor_test", REAL / "motor_session_gripper.py")
radar = _module("v3_radar_test", REAL / "radar_receiver.py")
timing_mod = _module("runtime_timing", REAL / "runtime_timing.py")


@pytest.fixture
def play(monkeypatch):
    # Run the actual helper/main definitions without the module's AppLauncher side effects.
    monkeypatch.syspath_prepend(str(REAL))
    tree = ast.parse((REAL / "play_real.py").read_text())
    module = types.ModuleType("v3_play_test")
    sys.modules[module.__name__] = module
    constants = {"POLICY_LEGS", "HARDWARE_LEGS", "GRAVITY_W", "REAL_TO_SIM_SIGN", "SIM_TO_REAL_SIGN"}
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            node.decorator_list = [d for d in node.decorator_list
                                   if not isinstance(d, ast.Call) or not isinstance(d.func, ast.Name)
                                   or d.func.id != "hydra_task_config"]
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
            nodes.append(node)
    module.__dict__.update(np=np, torch=torch, yaml=yaml, math=math, dataclass=dataclass, time=time,
                           json=json, Path=Path, LatestCamera=timing_mod.LatestCamera,
                           PeriodicDeadline=timing_mod.PeriodicDeadline, TimingStats=timing_mod.TimingStats,
                           os=__import__("os"), signal=__import__("signal"), threading=threading,
                           restore_training_contract=contract.restore_training_contract,
                           resolve_depth_bar_estimator_checkpoint=contract.resolve_depth_bar_estimator_checkpoint,
                           resolve_loop_hz=contract.resolve_loop_hz)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "play_real.py", "exec"), module.__dict__)
    return module


def _args(extra=()):
    tree = ast.parse((REAL / "play_real.py").read_text())
    parser = argparse.ArgumentParser()
    for node in tree.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == "parser"
                and node.value.func.attr == "add_argument"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "arguments", "exec"),
                 {"parser": parser, "argparse": argparse, "THIS_DIR": REAL,
                  "DEFAULT_TASK": "depthbar", "DEFAULT_DEPTH_BAR_ESTIMATOR": "estimator.pt"})
    args = parser.parse_args(extra)
    for key, value in {"device": "cpu", "checkpoint": str(BASELINE), "max_steps": 2,
                       "print_interval": 0, "startup_hold_s": 0., "load_run": ".*"}.items():
        setattr(args, key, value)
    return args


def _configs():
    saved_env = contract.load_training_snapshot(BASELINE.parent / "params/env.yaml")
    saved_agent = contract.load_training_snapshot(BASELINE.parent / "params/agent.yaml")
    env = types.SimpleNamespace(**{k: saved_env.get(k) for k in contract.ENV_INTERFACE_FIELDS})
    env.sim = types.SimpleNamespace(dt=.02, device="cpu")
    env.robot = types.SimpleNamespace(
        soft_joint_pos_limit_factor=.95,
        spawn=types.SimpleNamespace(usd_path=saved_env["robot"]["spawn"]["usd_path"]),
        init_state=types.SimpleNamespace(joint_pos={}))
    env.scene = types.SimpleNamespace(num_envs=1)
    env.events = None
    env.domain_randomization_enabled = False
    env.max_action_delay_steps = 0
    env.enable_depth_bar_camera = True
    env.depth_bar_estimator_checkpoint = ""
    agent = types.SimpleNamespace(**saved_agent)
    agent.to_dict = lambda: {k: getattr(agent, k) for k in saved_agent}
    return env, agent


@pytest.mark.parametrize("mode", ["fitted_j1", "fitted", "nominal", "none"])
def test_runtime_default_and_explicit_compensation_modes(play, mode):
    from initialization_models import make_compensation_model

    args = _args([] if mode == "fitted_j1" else ["--compensation", mode])
    args.loop_hz = 25.
    env, agent = _configs()
    contract.restore_training_contract(env, agent, str(BASELINE))
    play._resolve_runtime_defaults(args, env)
    assert args.compensation == mode and args.enable_gravity_compensation == (mode != "none")
    model = make_compensation_model(args.compensation)
    if mode == "none":
        assert model is None
    else:
        assert ("j1_residual" in model.description()) == (mode == "fitted_j1")


def test_disabled_compensation_rejects_explicit_residual(play):
    args = _args(["--compensation", "none", "--j1_residual_file", "candidate.json"])
    args.loop_hz = 25.
    env, agent = _configs()
    contract.restore_training_contract(env, agent, str(BASELINE))
    with pytest.raises(ValueError, match="require"):
        play._resolve_runtime_defaults(args, env)


def test_saved_baseline_contract_restores_old_gait_scales_and_latch():
    env, agent = _configs()
    env.gait_frequency, env.bar_estimate_dy_scale, env.bar_estimate_dz_scale = .18, .6, 1.
    env.gripper_action_latch_open_deadband = .06
    result = contract.restore_training_contract(env, agent, str(BASELINE))
    assert (env.gait_frequency, env.bar_estimate_dy_scale, env.bar_estimate_dz_scale) == (.3, .3, .5)
    assert env.gripper_action_latch_open_deadband == .1
    assert env.gripper_action_latch_close_deadband == .1
    assert result["policy_period_s"] == .04
    assert contract.resolve_loop_hz(None, .04) == 25.
    with pytest.raises(ValueError, match="match training"):
        contract.resolve_loop_hz(20, .04)


def test_missing_snapshots_and_executable_yaml_tags_are_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        contract.load_training_snapshot(tmp_path / "missing.yaml")
    path = tmp_path / "unsafe.yaml"
    path.write_text("!!python/object/apply:builtins.str [123]")
    with pytest.raises(yaml.constructor.ConstructorError):
        contract.load_training_snapshot(path)


def test_wrong_runner_is_rejected_before_policy_construction():
    env, agent = _configs()
    agent.class_name = "OnPolicyRunner"
    with pytest.raises(ValueError, match="does not match"):
        contract.restore_training_contract(env, agent, str(BASELINE))


def test_flattened_baseline_params_are_exclusive_and_other_models_need_own_params(tmp_path, monkeypatch):
    baseline_path = tmp_path / "model_68500.pt"
    params = tmp_path / "params"
    params.mkdir()
    for name in ("env.yaml", "agent.yaml"):
        (params / name).write_text((BASELINE.parent / "params" / name).read_text())
    monkeypatch.setattr(contract, "DEFAULT_CHECKPOINT", str(baseline_path))
    monkeypatch.setattr(contract, "BASELINE_PARAMS_DIR", params)
    env, agent = _configs()
    assert contract.restore_training_contract(env, agent, str(baseline_path))["policy_period_s"] == .04
    other_path = tmp_path / "model_98400.pt"
    for explicit_dir in (None, str(params)):
        with pytest.raises(ValueError, match="belongs only to model_68500.pt"):
            contract.restore_training_contract(env, agent, str(other_path), explicit_dir)
    own_params = tmp_path / "model_98400_params"
    own_params.mkdir()
    for name in ("env.yaml", "agent.yaml"):
        (own_params / name).write_text((params / name).read_text())
    result = contract.restore_training_contract(env, agent, str(other_path), str(own_params))
    assert result["params_dir"] == str(own_params)


def test_saved_estimator_path_is_relocated_without_changing_other_estimators(tmp_path, monkeypatch):
    monkeypatch.setattr(contract, "REPO_ROOT", tmp_path)
    local_estimator = tmp_path / "models/depth_crop_estimator_realcam_xmax060.pt"
    local_estimator.parent.mkdir()
    local_estimator.touch()
    monkeypatch.setattr(contract, "DEFAULT_DEPTH_BAR_ESTIMATOR", str(local_estimator))
    legacy = "logs/rsl_rl/xy4leg_gripper_truss_v3_depth_crop_realcam/" + local_estimator.name
    assert contract.resolve_depth_bar_estimator_checkpoint(legacy) == str(local_estimator)
    assert contract.resolve_depth_bar_estimator_checkpoint(None) == str(local_estimator)
    own_estimator = tmp_path / "own_estimator.pt"
    own_estimator.touch()
    assert contract.resolve_depth_bar_estimator_checkpoint(own_estimator.name) == str(own_estimator)
    with pytest.raises(FileNotFoundError, match="--depth_bar_estimator_checkpoint"):
        contract.resolve_depth_bar_estimator_checkpoint("missing_other_estimator.pt")


class FakeController:
    def __init__(self, acknowledge=True, state=1):
        self.acknowledge, self.state = acknowledge, state
        self.commands, self.closed = [], False

    def enable_motor(self, mid):
        return {"motor_id": mid, "error_code": self.state, "position": .1, "velocity": 0.} if self.acknowledge else None

    def damiao_mit_command(self, **kwargs):
        return kwargs

    def send(self, command):
        self.commands.append(command)

    def receive(self, count):
        return b""

    def receive_available(self):
        return [], 0, 0, False

    def close(self):
        self.closed = True


def _session(acknowledge=True, state=1):
    controllers = [FakeController(acknowledge, state) for _ in range(4)]
    return motor.MotorSession(*controllers, *([list(range(1, 7))] * 4),
                              [20, 20, 20, 20, 2, 0], [1, 1, 1, 1, .5, 0],
                              np.zeros(24), np.zeros(24))


@pytest.mark.parametrize("state,ack", [(1, False), (0, True), (8, True)])
def test_enable_requires_per_motor_enabled_acknowledgement(monkeypatch, state, ack):
    monkeypatch.setattr(motor.time, "sleep", lambda _: None)
    session = _session(ack, state)
    with pytest.raises(RuntimeError):
        session.enable_all()
    assert not session.enabled
    assert not any(c.commands for c, _, _ in session._legs())


def test_disabled_or_nonfinite_commands_cannot_be_sent():
    session = _session()
    with pytest.raises(RuntimeError, match="not enabled"):
        session.send_targets(np.zeros(24))
    session.enabled = True
    with pytest.raises(ValueError):
        session.send_targets(np.full(24, np.nan))
    assert not any(c.commands for c, _, _ in session._legs())


def test_fault_feedback_invalidates_the_motor_without_refreshing_its_age():
    session = _session()
    normal = {"motor_id": 1, "error_code": 1, "position": .3, "velocity": .2}
    assert session._store_feedback(normal, 0)
    timestamp = session.feedback_times[0]
    assert not session._store_feedback({**normal, "error_code": 8, "position": 4.}, 0)
    assert session.feedback_times[0] == timestamp
    assert session.pos_buf[0] == .3 and not session.feedback_valid[0]


def test_missing_individual_motor_feedback_is_not_hidden_by_fresh_others():
    shared = shared_mod.SharedData()
    stamps = np.full(24, 10.); stamps[17] = 0.
    shared.set_obs(np.zeros(48), sample_times=stamps, valid=np.ones(24, dtype=bool))
    with pytest.raises(RuntimeError, match="17"):
        shared.get_obs(max_age_s=.5, now=10.1)
    stamps[17] = 10.
    shared.set_obs(np.zeros(48), sample_times=stamps, valid=np.ones(24, dtype=bool))
    assert shared.get_obs(max_age_s=.5, now=10.1).shape == (48,)


def test_close_adds_no_shutdown_or_other_motor_commands():
    session = _session(); session.enabled = True
    session.close()
    assert all(c.closed and not c.commands for c, _, _ in session._legs())


def test_feedback_diagnostics_distinguish_states_ages_and_hardware_names(monkeypatch):
    monkeypatch.setattr(motor.time, "monotonic", lambda: 10.)
    session = _session()
    enabled = {"motor_id": 1, "error_code": 1, "position": .1, "velocity": 0.}
    for index in (0, 4, 8, 12):
        session._store_feedback(enabled, index)
    session.feedback_times[4] = 9.
    session._store_feedback({**enabled, "error_code": 8}, 8)
    session._store_feedback({**enabled, "error_code": 0}, 12)
    report = session.feedback_diagnostics(now=10.1)
    motors = {row["index"]: row for row in report["motors"]}
    assert (motors[0]["leg"], motors[0]["joint"], motors[0]["motor_id"]) == ("LB", 1, 1)
    assert motors[0]["problems"] == []
    assert motors[4]["problems"] == ["stale"]
    assert motors[8]["problems"] == ["status=8"]
    assert motors[12]["problems"] == ["status=0"]
    assert motors[1]["leg"] == "LF" and motors[1]["age_ms"] is None
    assert "missing" in motors[1]["problems"]
    json.dumps(report, allow_nan=False)
    assert not any(ctrl.commands for ctrl, _, _ in session._legs())


def test_motor_failure_report_preserves_failure_snapshot_and_saves_json(play, tmp_path):
    session, shared = _session(), shared_mod.SharedData()
    captured = session.feedback_diagnostics(now=10.)
    shared.set_error("motor command thread", "feedback expired", details=captured)
    shared.set_error("later thread", "must not overwrite", details={})
    session.feedback_valid[:] = True
    session.feedback_times[:] = time.monotonic()
    args = types.SimpleNamespace(feedback_timeout_s=.5, timing_jsonl=str(tmp_path / "timing.jsonl"))
    report = play._report_motor_feedback(session, shared, args, "outer failure")
    assert report["captured_at_failure"]
    assert report["error"] == "motor command thread: feedback expired"
    assert not report["feedback"]["motors"][0]["valid"]
    saved = json.loads((tmp_path / "timing.jsonl.motors.jsonl").read_text())
    assert saved == report
    assert not any(ctrl.commands for ctrl, _, _ in session._legs())


def test_startup_current_command_requires_feedback_and_keeps_grippers_at_zero_torque(play):
    session = _session()
    with pytest.raises(RuntimeError, match="valid enable feedback"):
        play._startup_current_command(session)
    for index in range(24):
        session._store_feedback(dict(error_code=1, position=.01 * index, velocity=0.), index)
    command = play._startup_current_command(session)
    np.testing.assert_allclose(command[:20], session.pos_buf[:20])
    assert not command[20:].any()
    assert not any(ctrl.commands for ctrl, _, _ in session._legs())


def test_default_transmit_mode_is_on_update():
    assert _args().tx_mode == "on_update"


def test_startup_interpolation_eases_both_ends_without_overshoot(play):
    start = np.linspace(-1., 1., 20)
    target = -start
    duration = 6.
    interpolate = play._startup_interpolated_positions
    np.testing.assert_array_equal(interpolate(start, target, 0., duration), start)
    np.testing.assert_array_equal(interpolate(start, target, duration, duration), target)
    np.testing.assert_array_equal(interpolate(start, target, duration + 1., duration), target)
    np.testing.assert_allclose(interpolate(start, target, duration / 2., duration), (start + target) / 2.)
    linear_step = np.abs(target - start) * (.04 / duration)
    first_step = np.abs(interpolate(start, target, .04, duration) - start)
    last_step = np.abs(target - interpolate(start, target, duration - .04, duration))
    assert np.all(first_step < linear_step * .01)
    np.testing.assert_allclose(first_step, last_step, atol=1e-12)
    positions = np.array([interpolate(start, target, t, duration) for t in np.linspace(0., duration, 101)])
    fractions = (positions - start) / (target - start)
    assert np.all(np.diff(fractions, axis=0) >= -1e-12)
    assert np.all((fractions >= -1e-12) & (fractions <= 1. + 1e-12))


def test_command_watchdog_reports_expiry_without_sending_an_action(monkeypatch):
    monkeypatch.setitem(sys.modules, "threads_shared", shared_mod)
    serial_threads = _module("v3_serial_threads_test", REAL / "serial_threads.py")
    shared = shared_mod.SharedData()
    shared.set_action(np.ones(24))
    shared.action_time = time.monotonic() - 2.
    session = _session(); session.enabled = True
    stop = threading.Event()
    thread = serial_threads.start_serial_tx_thread(shared, True, session, stop_event=stop, command_timeout_s=.5)
    thread.join(timeout=1.)
    assert not thread.is_alive()
    with pytest.raises(RuntimeError, match="expired"):
        shared.raise_if_failed()
    assert not any(c.commands for c, _, _ in session._legs())


def test_feedback_guard_rejects_faulted_motor_before_transmission(monkeypatch):
    monkeypatch.setitem(sys.modules, "threads_shared", shared_mod)
    serial_threads = _module("v3_serial_feedback_guard_test", REAL / "serial_threads.py")
    shared = shared_mod.SharedData()
    shared.set_action(np.zeros(24))
    valid = np.ones(24, dtype=bool); valid[6] = False
    shared.set_obs(np.zeros(48), np.full(24, time.monotonic()), valid)
    ready = threading.Event(); ready.set()
    session = _session(); session.enabled = True
    thread = serial_threads.start_serial_tx_thread(shared, True, session, feedback_ready_event=ready)
    thread.join(timeout=1.)
    assert not thread.is_alive()
    with pytest.raises(RuntimeError, match="6"):
        shared.raise_if_failed()
    assert not any(c.commands for c, _, _ in session._legs())


def test_partial_serial_frames_are_buffered_and_out_of_range_commands_rejected():
    tree = ast.parse((REAL / "motor_control.py").read_text())
    driver_node = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    scope = {"math": math, "time": time}
    exec(compile(ast.Module(body=[driver_node], type_ignores=[]), "motor_control.py", "exec"), scope)
    driver = object.__new__(scope["DaMiaoMotorController"])
    chunks = iter([b"\x11\x22\x33", b"\x44\x55\x66\x77\x88"])
    driver.ser = types.SimpleNamespace(isOpen=lambda: True, read=lambda n: next(chunks))
    driver._rx_buffer = bytearray()
    assert driver.receive(8) == b""
    assert driver.receive(8) == bytes.fromhex("1122334455667788")
    assert driver.damiao_feedback(None) is None
    with pytest.raises(ValueError, match="outside"):
        driver.damiao_mit_command(0., 0., 600., 1., 0., 1)


def test_policy_history_uses_the_same_clipped_action_as_execution(play):
    action = play._validated_policy_action(np.full(24, 1.8))
    obs = play._build_policy_observation(
        np.zeros(20), np.zeros(20), np.array([0., 0., -1.]), np.zeros(3), np.zeros(3),
        0., action, np.full(4, -.4), np.zeros(20), np.ones(20), np.zeros(4),
        torch.device("cpu"), "deployable_bar", 76)
    assert torch.equal(obs["deployable"][0, 48:72], torch.ones(24))
    with pytest.raises(ValueError, match="finite"):
        play._validated_policy_action(np.full(24, np.nan))


def test_nan_estimator_is_a_miss_and_memory_eventually_expires(play):
    class Camera:
        depth_scale_m = .001
        def read(self, img_size):
            return None, np.full((*img_size, 1), 500, dtype=np.uint16)
    class Estimator:
        def __call__(self, depth):
            return torch.full((1, 3), float("nan"))
    fallback = np.array([2.4, 0., -.5])
    runtime = play.DepthBarRuntime(Estimator(), np.array([.4, 0., -.5]), 0., 1., fallback, fallback.copy(), True)
    def observation(now):
        return play._depth_bar_observation_from_camera(
            Camera(), runtime, torch.device("cpu"), now, 64, .1, 1.5,
            (-.05, .65), (-.45, .45), (-.85, .25), .001, .3, 1.6, .035, 1.2, .3, .5)
    bar, _, visible, memory_used, _ = observation(1.5)
    assert not visible and memory_used and bar[3] == 1
    bar, _, visible, memory_used, _ = observation(3.)
    assert not visible and not memory_used and bar[3] == 0
    assert np.isfinite(bar).all()


@pytest.mark.parametrize("update", [{"q": [0, 0, 0, 0]}, {"w": [1, 2]}, {"w": [0, float("nan"), 0]}])
def test_malformed_radar_packets_are_rejected(update):
    data = {"q": [0, 0, 0, 1], "w": [0, 0, 0], **update}
    with pytest.raises(ValueError):
        radar.RadarReceiver._decode_packet(json.dumps(data).encode())


def test_radar_freshness_uses_local_receipt_time():
    receiver = object.__new__(radar.RadarReceiver)
    receiver.lock, receiver.error = threading.Lock(), None
    receiver.latest_data = {"timestamp": 0., "quat": np.array([0., 0., 0., 1.])}
    receiver.last_received_at = 10.
    assert receiver.get_data(max_age_s=.5, now=10.1)["timestamp"] == 0.
    with pytest.raises(RuntimeError, match="stale"):
        receiver.get_data(max_age_s=.5, now=11.)


@pytest.mark.parametrize("fail_second", [False, True])
def test_actual_main_dry_run_restores_config_and_uses_clipped_history(play, monkeypatch, tmp_path, fail_second):
    env, agent = _configs()
    env.gait_frequency, env.bar_estimate_dy_scale, env.bar_estimate_dz_scale = .18, .6, 1.
    args = _args()
    args.timing_jsonl = str(tmp_path / "main-timing.jsonl")
    play.args_cli = args
    play.DEFAULT_LOG_ROOT, play.DEFAULT_DEPTH_BAR_ESTIMATOR = "logs", "estimator.pt"
    play.INSTALLED_RSL_RL_VERSION = "local"
    play.cli_args = types.SimpleNamespace(update_rsl_rl_cfg=lambda cfg, args: cfg)
    play.handle_deprecated_rsl_rl_cfg = lambda cfg, version: cfg
    play.retrieve_file_path = str
    play.SharedData = shared_mod.SharedData
    play.simulation_app = types.SimpleNamespace(is_running=lambda: True)
    raw = types.SimpleNamespace(
        cfg=env, device="cpu", nominal_action=torch.zeros(24), _action_offset=torch.zeros(20),
        _action_scale=torch.ones(20), _joint_lower=-torch.ones(20), _joint_upper=torch.ones(20),
        _main_joint_ids=list(range(20)),
        robot=types.SimpleNamespace(data=types.SimpleNamespace(default_joint_pos=torch.zeros((1, 20)))))
    raw.unwrapped = raw
    closed = []
    raw.close = lambda: closed.append(True)
    play.gym = types.SimpleNamespace(make=lambda *a, **k: raw)
    play.DirectMARLEnv = type("UnusedMultiAgent", (), {})
    play.RslRlVecEnvWrapper = lambda env, **kw: env
    inputs = []
    class Runner:
        def __init__(self, *a, **kw): pass
        def load(self, path): assert path == str(BASELINE)
        def get_inference_policy(self, device):
            def policy(obs):
                inputs.append(obs["deployable"].clone())
                if fail_second and len(inputs) == 2:
                    raise RuntimeError("inference failure")
                return torch.full((1, 24), 1.8)
            return policy
    play.DistillationRunner = Runner
    # Never alter the test process's signal handlers or open hardware.
    monkeypatch.setattr(play.signal, "signal", lambda *args: None)
    if fail_second:
        with pytest.raises(RuntimeError, match="inference failure"):
            play.main(env, agent)
    else:
        play.main(env, agent)
    assert (env.gait_frequency, env.bar_estimate_dy_scale, env.bar_estimate_dz_scale) == (.3, .3, .5)
    assert len(inputs) == 2 and torch.equal(inputs[1][0, 48:72], torch.ones(24))
    assert closed == [True]
    report = json.loads(Path(args.timing_jsonl).read_text())
    assert report["counts"]["policy"] == (1 if fail_second else 2)


@pytest.mark.parametrize("enable_radar", [False, True])
@pytest.mark.parametrize("gravity_enabled", [False, True])
@pytest.mark.parametrize("fail_startup", [False, True])
@pytest.mark.parametrize("tx_mode", [None, "periodic"])
def test_actual_main_motor_startup_has_one_reader_and_no_extra_shutdown_action(
    play, monkeypatch, tmp_path, enable_radar, gravity_enabled, fail_startup, tx_mode,
):
    monkeypatch.syspath_prepend(str(REAL))
    env, agent = _configs()
    args = _args()
    if tx_mode is not None:
        args.tx_mode = tx_mode
    args.enable_motors = args.enable_camera = True
    args.enable_radar = enable_radar
    args.enable_gravity_compensation = gravity_enabled
    args.compensation = "fitted_j1" if gravity_enabled else "none"
    args.startup_ramp_s = .08
    args.timing_jsonl = str(tmp_path / "timing.jsonl")
    play.args_cli = args
    play.DEFAULT_LOG_ROOT, play.DEFAULT_DEPTH_BAR_ESTIMATOR = "logs", "estimator.pt"
    play.INSTALLED_RSL_RL_VERSION = "local"
    play.cli_args = types.SimpleNamespace(update_rsl_rl_cfg=lambda cfg, args: cfg)
    play.handle_deprecated_rsl_rl_cfg = lambda cfg, version: cfg
    play.retrieve_file_path = str
    publications = []
    class RecordedShared(shared_mod.SharedData):
        def set_action(self, *args, **kwargs):
            super().set_action(*args, **kwargs)
            publications.append(self.command_snapshot())
    play.SharedData = RecordedShared
    play.simulation_app = types.SimpleNamespace(is_running=lambda: True)
    raw = types.SimpleNamespace(
        cfg=env, device="cpu", nominal_action=torch.zeros(24), _action_offset=torch.zeros(20),
        _action_scale=torch.ones(20), _joint_lower=-torch.ones(20), _joint_upper=torch.ones(20),
        _main_joint_ids=list(range(20)),
        robot=types.SimpleNamespace(data=types.SimpleNamespace(default_joint_pos=torch.zeros((1, 20)))))
    raw.unwrapped = raw
    events, commands = [], []
    raw.close = lambda: events.append("env_closed")
    play.gym = types.SimpleNamespace(make=lambda *a, **k: raw)
    play.DirectMARLEnv = type("UnusedMultiAgent", (), {})
    play.RslRlVecEnvWrapper = lambda env, **kw: env
    class Runner:
        def __init__(self, *a, **kw): pass
        def load(self, path): pass
        def get_inference_policy(self, device):
            def policy(obs):
                if not enable_radar:
                    assert torch.equal(obs["deployable"][0, 40:43], torch.tensor([0., 0., -1.]))
                    assert torch.equal(obs["deployable"][0, 43:46], torch.zeros(3))
                return torch.zeros((1, 24))
            return policy
    play.DistillationRunner = Runner
    class Radar:
        def __init__(self, **kwargs): assert enable_radar, "Radar must stay closed when disabled"
        def start(self): events.append("radar_ready")
        def stop(self): events.append("radar_closed")
        def get_data(self, **kwargs):
            return {"quat": [0., 0., 0., 1.], "lin_vel": [0., 0., 0.], "ang_vel": [0., 0., 0.]}
    play.RadarReceiver = Radar
    class Camera:
        depth_scale_m = .001
        def __init__(self, **kwargs): pass
        def read(self, img_size): return None, np.full((*img_size, 1), 500, dtype=np.uint16)
        def stop(self): events.append("camera_closed")
    monkeypatch.setitem(sys.modules, "realsense_camera", types.SimpleNamespace(RealSenseCamera=Camera))
    fallback = np.array([2.4, 0., -.5])
    play._load_depth_bar_runtime = lambda *a, **k: play.DepthBarRuntime(
        lambda x: torch.tensor([[.4, 0., -.5]]), fallback.copy(), 3., -1., fallback, fallback.copy(), False)
    session = _session()
    def enable(cancelled):
        assert events == (["radar_ready"] if enable_radar else [])
        assert not cancelled()
        events.append("enabled")
        for index in range(24):
            session._store_feedback(dict(error_code=1, position=.01 * (index + 1), velocity=0.), index)
        session.enabled = True
    session.enable_all = enable
    play.create_motor_session = lambda *a: (True, session)
    thread = types.SimpleNamespace(join=lambda timeout: None)
    def start_tx(shared, motor_enabled, session_arg, **kwargs):
        assert session_arg.enabled
        assert kwargs["mode"] == (tx_mode or "on_update") and kwargs["loop_hz"] == 25.
        events.append("tx_started")
        commands.append(shared.get_action())
        if fail_startup:
            shared.set_error("motor command thread", "startup feedback rejected",
                             details=session.feedback_diagnostics())
        return thread
    def start_obs(shared, *a, **kwargs):
        assert "enabled" in events
        assert kwargs["loop_hz"] == 200.
        events.append("obs_started")
        fresh_positions = session.pos_buf.copy()
        fresh_positions[:20] += .1  # Later receipt must not step the already-published target.
        shared.set_obs(np.concatenate((fresh_positions, session.vel_buf)),
                       np.full(24, time.monotonic()), np.ones(24, dtype=bool))
        return thread
    play.start_serial_tx_thread, play.start_motor_obs_thread = start_tx, start_obs
    monkeypatch.setattr(play.signal, "signal", lambda *args: None)
    if fail_startup:
        with pytest.raises(RuntimeError, match="startup feedback rejected"):
            play.main(env, agent)
        assert (tmp_path / "timing.jsonl.motors.jsonl").exists()
        report = json.loads((tmp_path / "timing.jsonl").read_text())
        assert report["counts"].get("policy", 0) == 0
    else:
        play.main(env, agent)
    startup_events = (["radar_ready"] if enable_radar else []) + ["enabled", "tx_started", "obs_started"]
    assert events[:len(startup_events)] == startup_events
    assert ("radar_closed" in events) == enable_radar
    assert all(c.closed and not c.commands for c, _, _ in session._legs())
    assert len(commands) == 1 and commands[0].shape == (24,)
    np.testing.assert_allclose(commands[0][:20], .01 * np.arange(1, 21))
    assert not commands[0][20:].any()
    if not fail_startup:
        # The first fresh feedback cannot change the published ramp origin.
        startup_targets = np.array([publication[0][:20] for publication in publications])
        np.testing.assert_allclose(startup_targets[1], startup_targets[0], atol=1e-7)
        # All four legs use the same smooth trajectory, ending at nominal.
        fractions = startup_targets / startup_targets[0]
        np.testing.assert_allclose(fractions, np.repeat(fractions[:, :1], 20, axis=1), atol=1e-6)
        assert np.all(np.diff(fractions, axis=0) <= 1e-6)
        assert np.any((fractions > .01) & (fractions < .99))
        np.testing.assert_allclose(startup_targets[-1], 0., atol=1e-7)
    if gravity_enabled:
        assert all(item[1].shape == (20,) for item in publications)
        assert np.array_equal(publications[0][1], np.zeros(20))
        if not fail_startup:
            assert np.any(np.abs(publications[-1][1]) > 0.)
    else:
        assert all(item[1] is None for item in publications)


def test_gravity_helper_uses_measured_hardware_joints_and_reorders_fingers(play):
    measured = np.arange(20, dtype=np.float32) * .01
    pos, _, _ = play._hardware_obs_to_policy_joints(
        np.concatenate((measured, np.zeros(28))),
        play._read_gripper_calibration(str(REAL / "gripper_calibration.yaml"), 1., .05))
    fingers = np.array([-.25, -.3, -.4, -.5])  # RF LF RB LB
    calls = []
    class Controller:
        def update(self, q, gravity, finger_angles, now):
            calls.append((q.copy(), gravity.copy(), finger_angles.copy(), now))
            return np.zeros(20)
    play._gravity_feedforward(Controller(), pos, fingers, np.array([0., -1., 0.]), now=4.)
    q, gravity, hw_fingers, stamp = calls[0]
    np.testing.assert_allclose(q, measured)
    np.testing.assert_allclose(gravity, [0., -9.81, 0.])
    np.testing.assert_allclose(hw_fingers, fingers[[3, 1, 2, 0]])
    assert stamp == 4.


def _driver():
    tree = ast.parse((REAL / "motor_control.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    scope = {"math": math, "time": time}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "motor_control.py", "exec"), scope)
    driver = object.__new__(scope["DaMiaoMotorController"])
    driver._rx_buffer = bytearray()
    return driver


class BufferedSerial:
    def __init__(self, payload=b"", arriving=b""):
        self.payload = bytearray(payload)
        self.arriving = arriving
        self.read_sizes = []

    def isOpen(self):
        return True

    @property
    def in_waiting(self):
        return len(self.payload)

    def read(self, count):
        assert count > 0, "An empty port must never call blocking read()"
        self.read_sizes.append(count)
        result = bytes(self.payload[:count])
        del self.payload[:count]
        self.payload.extend(self.arriving)
        self.arriving = b""
        return result


def test_empty_port_never_performs_a_blocking_read():
    driver = _driver(); driver.ser = BufferedSerial()
    assert driver.receive_available() == ([], 0, 0, False)
    assert driver.ser.read_sizes == []


def test_serial_drain_preserves_fragments_and_reads_all_complete_frames():
    driver = _driver(); driver.ser = BufferedSerial(bytes(range(19)))
    frames, pending, backlog, overflow = driver.receive_available()
    assert frames == [bytes(range(8)), bytes(range(8, 16))]
    assert (pending, backlog, overflow) == (0, 19, False)
    assert driver._rx_buffer == bytes(range(16, 19))
    driver.ser.payload.extend(bytes(range(19, 24)))
    assert driver.receive_available()[0] == [bytes(range(16, 24))]
    assert not driver._rx_buffer


def test_bytes_arriving_during_read_are_distinguished_from_overflow():
    driver = _driver(); driver.ser = BufferedSerial(b"1" * 8, arriving=b"2" * 8)
    assert driver.receive_available() == ([b"1" * 8], 8, 8, False)
    driver.ser.payload = bytearray(b"3" * 24)
    assert driver.receive_available(max_bytes=8) == ([b"3" * 8], 16, 24, True)


def test_missing_leg_does_not_delay_other_legs_and_latest_per_id_wins():
    session = _session()
    for ctrl, _, _ in session._legs():
        ctrl.receive = lambda *a: pytest.fail("Steady RX must not use blocking receive()")
        ctrl.damiao_feedback = lambda value: value
    first, *others = session._legs()
    # An empty LB port must not prevent the remaining three legs from being updated.
    for ctrl, ids, offset in others:
        packets = [{"motor_id": mid, "error_code": 1, "position": float(mid), "velocity": 0.}
                   for mid in reversed(ids)]
        packets.append({"motor_id": 2, "error_code": 1, "position": 9., "velocity": .5})
        ctrl.receive_available = lambda p=packets: (p, 0, len(p) * 8, False)
    session.recv_feedback()
    assert not session.feedback_valid[0::4].any()
    for _, _, offset in others:
        assert session.feedback_valid[offset::4].all()
        assert session.pos_buf[offset + 4] == 9.
    assert session.communication_snapshot()["rx_frames"] == 21
    assert session.communication_snapshot()["rx_complete"] == 0


def test_overflow_is_not_accepted_as_fresh_feedback():
    session = _session()
    session.feedback_valid[:] = True
    session.feedback_times[:] = 10.
    ctrl = session.ctrl_lb
    ctrl.receive_available = lambda: ([b"x" * 8], 16, 8208, True)
    ctrl.damiao_feedback = lambda _: pytest.fail("Overflowed packets must not refresh timestamps")
    session.recv_feedback()
    assert not session.feedback_valid[0::4].any()
    assert np.all(session.feedback_times[0::4] == 10.)


def test_duplicate_serial_paths_are_rejected_before_opening_ports(tmp_path):
    alias = tmp_path / "alias"
    original = tmp_path / "port"
    alias.symlink_to(original)
    with pytest.raises(ValueError, match="distinct"):
        motor.create_motor_session([1] * 6, [1] * 6,
                                   [str(original), str(alias), "unused-rb", "unused-rf"])


def test_new_arrivals_do_not_invalidate_a_successfully_drained_leg():
    session = _session()
    packets = [{"motor_id": mid, "error_code": 1, "position": .1, "velocity": 0.}
               for mid in range(1, 7)]
    session.ctrl_lb.receive_available = lambda: (packets, 8, 48, False)
    session.ctrl_lb.damiao_feedback = lambda packet: packet
    session.recv_feedback()
    assert session.feedback_valid[0::4].all()


def test_unknown_serial_id_is_counted_without_changing_joint_feedback():
    session = _session()
    session.ctrl_lb.receive_available = lambda: ([b"x" * 8], 0, 8, False)
    session.ctrl_lb.damiao_feedback = lambda _: {"motor_id": 15, "error_code": 1, "position": .3, "velocity": 0.}
    session.recv_feedback()
    assert session.communication_snapshot()["rx_invalid_frames"] == 1
    assert session.feedback_diagnostics()["ports"][0]["last_frames_hex"] == [(b"x" * 8).hex()]
    assert not session.feedback_valid.any()


def test_transmit_queue_measurement_keeps_the_largest_port_queue():
    session = _session()
    for (ctrl, _, _), size in zip(session._legs(), (20, 70, 30, 10)):
        ctrl.ser = types.SimpleNamespace(out_waiting=size)
    assert session.transmit_queue_bytes() == 70


def test_receive_thread_propagates_serial_failures(monkeypatch):
    monkeypatch.setitem(sys.modules, "threads_shared", shared_mod)
    serial_threads = _module("v3_rx_failure_test", REAL / "serial_threads.py")
    def receive():
        raise OSError("serial disconnected")
    shared = shared_mod.SharedData()
    thread = serial_threads.start_motor_obs_thread(shared, types.SimpleNamespace(recv_feedback=receive), "cpu", 1)
    thread.join(timeout=1.)
    assert not thread.is_alive()
    with pytest.raises(RuntimeError, match="serial disconnected"):
        shared.raise_if_failed()


def test_absolute_deadlines_do_not_add_work_time_to_every_period():
    current = [0.]
    schedule = timing_mod.PeriodicDeadline(
        25., clock=lambda: current[0], sleep=lambda delay: current.__setitem__(0, current[0] + delay))
    assert schedule.wait()
    current[0] += .011
    assert schedule.wait() and current[0] == pytest.approx(.04)
    current[0] += .011
    assert schedule.wait() and current[0] == pytest.approx(.08)
    # A 100 ms operation skips the missed .12 tick; it cannot create a replay burst.
    current[0] += .1
    assert schedule.wait() and schedule.skipped_ticks == 1
    assert schedule.next_deadline == pytest.approx(.20)
    assert schedule.wait() and current[0] == pytest.approx(.20)


def test_frequency_uses_inter_event_span_and_report_resets(monkeypatch):
    current = [10.]
    monkeypatch.setattr(timing_mod.time, "monotonic", lambda: current[0])
    stats = timing_mod.TimingStats()
    stats.tick("policy"); stats.tick("rx_joint_frames", 24)
    current[0] += .04
    stats.tick("policy"); stats.tick("rx_joint_frames", 24)
    stats.observe("actor_ms", .2); stats.observe("actor_ms", float("nan"))
    report = stats.snapshot(reset=False)
    assert report["rates_hz"]["policy"] == pytest.approx(25.)
    assert report["rates_hz"]["rx_joint_frames"] == pytest.approx(600.)
    assert report["samples"]["actor_ms"]["n"] == 1
    current[0] += .04  # Silence must lower the reported rate rather than freeze it.
    assert stats.snapshot()["rates_hz"]["policy"] == pytest.approx(12.5)
    assert stats.snapshot()["counts"] == {}


def test_camera_consumer_does_not_wait_for_a_blocked_capture():
    second_read, release = threading.Event(), threading.Event()
    class Camera:
        depth_scale_m = .001
        calls = 0
        def read(self, img_size):
            self.calls += 1
            if self.calls == 2:
                second_read.set()
                release.wait(1.)
            return None, np.ones((*img_size, 1), dtype=np.uint16)
        def stop(self): pass
    camera = timing_mod.LatestCamera(Camera(), 64).start()
    result = []
    try:
        assert second_read.wait(1.)
        consumer = threading.Thread(target=lambda: result.append(camera.read(img_size=(64, 64))))
        consumer.start(); consumer.join(timeout=.1)
        assert not consumer.is_alive() and result[0][1].shape == (64, 64, 1)
    finally:
        release.set(); camera.stop()


@pytest.mark.parametrize("scenario,calls,frames,error", [
    ("slow_start", 3, 1, None),
    ("never_ready", 3, 0, "no first frame within 0.5s"),
    ("stream_stops", 2, 1, "frame timeout"),
    ("device_error", 1, 0, "camera unplugged"),
])
def test_camera_startup_retries_only_frame_timeouts_until_first_frame(
    monkeypatch, scenario, calls, frames, error
):
    current = [0.]
    monkeypatch.setattr(timing_mod.time, "monotonic", lambda: current[0])
    class Schedule:
        def __init__(self, hz): pass
        def wait(self, stop): return not stop.is_set()
    monkeypatch.setattr(timing_mod, "PeriodicDeadline", Schedule)
    class Camera:
        depth_scale_m = .001
        calls = 0
        def read(self, img_size):
            self.calls += 1
            current[0] += .2
            if scenario == "device_error":
                raise OSError("camera unplugged")
            if (scenario == "slow_start" and self.calls < 3
                    or scenario == "never_ready"
                    or scenario == "stream_stops" and self.calls > 1):
                raise TimeoutError("frame timeout")
            if scenario == "slow_start":
                cache.stop_event.set()
            return None, np.ones((*img_size, 1), dtype=np.uint16)
    stats = timing_mod.TimingStats()
    camera = Camera()
    cache = timing_mod.LatestCamera(camera, 64, stats=stats, startup_timeout_s=.5)
    cache._capture()
    assert camera.calls == calls
    assert cache.sequence == frames
    if error is None:
        assert cache.error is None
        assert cache.read()[1].shape == (64, 64, 1)
        assert stats.snapshot()["counts"]["camera_startup_timeouts"] == 2
    else:
        with pytest.raises(RuntimeError, match=error):
            cache.read()


@pytest.mark.parametrize("outcome", ["timeout", "device_error", "incomplete", "frame"])
def test_realsense_read_distinguishes_frame_timeout_from_sdk_failure(monkeypatch, outcome):
    monkeypatch.setitem(sys.modules, "camera", types.SimpleNamespace(CameraDriver=object))
    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace())
    realsense = _module("v3_realsense_test", TASK / "camera/realsense_camera.py")
    camera = realsense.RealSenseCamera.__new__(realsense.RealSenseCamera)
    camera._timeout_ms, camera._flip = 200, False
    color = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.uint16)
    frames = types.SimpleNamespace(
        get_color_frame=lambda: None if outcome == "incomplete" else types.SimpleNamespace(get_data=lambda: color),
        get_depth_frame=lambda: types.SimpleNamespace(get_data=lambda: depth),
    )
    def wait(timeout_ms):
        assert timeout_ms == 200
        if outcome == "device_error":
            raise RuntimeError("device disconnected")
        return outcome != "timeout", frames
    camera._pipeline = types.SimpleNamespace(try_wait_for_frames=wait)
    if outcome == "frame":
        _, received = camera.read()
        np.testing.assert_array_equal(received[..., 0], depth)
    else:
        exception, message = {
            "timeout": (TimeoutError, "within 200 ms"),
            "device_error": (RuntimeError, "device disconnected"),
            "incomplete": (RuntimeError, "both depth and color"),
        }[outcome]
        with pytest.raises(exception, match=message):
            camera.read()


def test_cached_camera_age_and_capture_errors_are_propagated():
    camera = timing_mod.LatestCamera(types.SimpleNamespace(depth_scale_m=.001), 64)
    with pytest.raises(RuntimeError, match="not ready"):
        camera.read()
    camera.latest = (None, np.ones((64, 64, 1)), time.monotonic() - 1., 1)
    with pytest.raises(RuntimeError, match="stale"):
        camera.read()
    camera.error = OSError("camera unplugged")
    with pytest.raises(RuntimeError, match="unplugged"):
        camera.read()


def test_reused_camera_frame_does_not_reapply_filter_or_reset_its_age(play):
    class Camera:
        depth_scale_m = .001
        last_read_sequence = 1
        last_read_capture_time = 10.
        def read(self, img_size):
            return None, np.full((*img_size, 1), 500, dtype=np.uint16)
    camera = Camera()
    estimate = [.4]
    fallback = np.array([2.4, 0., -.5])
    runtime = play.DepthBarRuntime(lambda _: torch.tensor([[estimate[0], 0., -.5]]),
                                  fallback.copy(), 3., -1., fallback, fallback.copy(), False)
    def observe(now):
        return play._depth_bar_observation_from_camera(
            camera, runtime, torch.device("cpu"), now, 64, .1, 1.5,
            (-.05, .65), (-.45, .45), (-.85, .25), .001, .5, 1.6, .035, 1.2, .3, .5)
    observe(10.1)
    estimate[0] = .6
    observe(10.2)
    assert runtime.filtered_estimate_b[0] == pytest.approx(.4)
    assert runtime.memory_age_s == pytest.approx(.2)
    camera.last_read_sequence = 2
    camera.last_read_capture_time = 10.25
    observe(10.3)
    assert runtime.filtered_estimate_b[0] == pytest.approx(.5)
    assert runtime.memory_age_s == pytest.approx(.05)


def test_on_update_transmits_new_actions_once_without_a_periodic_repeat(monkeypatch):
    monkeypatch.setitem(sys.modules, "threads_shared", shared_mod)
    serial_threads = _module("v3_on_update_test", REAL / "serial_threads.py")
    shared = shared_mod.SharedData(); stats = timing_mod.TimingStats()
    commands, first, second, stop = [], threading.Event(), threading.Event(), threading.Event()
    def send(action):
        commands.append(action.copy())
        (first if len(commands) == 1 else second).set()
    session = types.SimpleNamespace(send_targets=send)
    shared.set_action(np.ones(24), observed_at=time.monotonic())
    thread = serial_threads.start_serial_tx_thread(shared, True, session, stop_event=stop,
                                                  stats=stats, mode="on_update")
    try:
        assert first.wait(1.)
        assert not second.wait(.09)  # More than two nominal TX periods, without another action.
        shared.set_action(np.zeros(24), observed_at=time.monotonic())
        assert second.wait(1.)
    finally:
        stop.set(); thread.join(timeout=1.)
    shared.raise_if_failed()
    report = stats.snapshot()
    assert len(commands) == 2 and not np.array_equal(commands[0], commands[1])
    assert report["counts"]["tx_new"] == 2 and report["counts"].get("tx_repeat", 0) == 0
    assert report["samples"]["command_age_ms"]["mean"] >= 0


def test_timing_report_is_parseable_and_contains_measurements(play, tmp_path):
    stats = timing_mod.TimingStats()
    stats.tick("policy"); stats.observe("actor_ms", .06)
    path = tmp_path / "timing.jsonl"
    args = types.SimpleNamespace(loop_hz=25., tx_mode="periodic", torch_threads=1, timing_jsonl=str(path))
    report = play._report_timing(stats, args)
    saved = json.loads(path.read_text())
    assert saved == report and saved["target_policy_hz"] == 25.
    assert saved["counts"]["policy"] == 1
    assert saved["samples"]["actor_ms"]["p95"] == .06
