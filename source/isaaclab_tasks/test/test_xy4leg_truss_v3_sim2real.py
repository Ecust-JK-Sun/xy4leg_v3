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
    "2026-08-05_04-55-55_realcam_crop_noise_l2far_render12hz_from42000/model_68500.pt"
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
def play():
    # Run the actual helper/main definitions without the module's AppLauncher side effects.
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
                           resolve_loop_hz=contract.resolve_loop_hz)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "play_real.py", "exec"), module.__dict__)
    return module


def _args():
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
    args = parser.parse_args([])
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


def test_actual_main_motor_startup_has_one_reader_and_no_extra_shutdown_action(play, monkeypatch):
    env, agent = _configs()
    args = _args()
    args.enable_motors = args.enable_camera = args.enable_radar = True
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
    events, commands = [], []
    raw.close = lambda: events.append("env_closed")
    play.gym = types.SimpleNamespace(make=lambda *a, **k: raw)
    play.DirectMARLEnv = type("UnusedMultiAgent", (), {})
    play.RslRlVecEnvWrapper = lambda env, **kw: env
    class Runner:
        def __init__(self, *a, **kw): pass
        def load(self, path): pass
        def get_inference_policy(self, device):
            return lambda obs: torch.zeros((1, 24))
    play.DistillationRunner = Runner
    class Radar:
        def __init__(self, **kwargs): pass
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
        assert events == ["radar_ready"]
        assert not cancelled()
        events.append("enabled")
        session.enabled = True
    session.enable_all = enable
    play.create_motor_session = lambda *a: (True, session)
    thread = types.SimpleNamespace(join=lambda timeout: None)
    def start_tx(shared, motor_enabled, session_arg, **kwargs):
        assert session_arg.enabled
        assert kwargs["mode"] == "periodic" and kwargs["loop_hz"] == 25.
        events.append("tx_started")
        commands.append(shared.get_action())
        return thread
    def start_obs(shared, *a, **kwargs):
        assert "enabled" in events
        assert kwargs["loop_hz"] == 200.
        events.append("obs_started")
        shared.set_obs(np.zeros(48), np.full(24, time.monotonic()), np.ones(24, dtype=bool))
        return thread
    play.start_serial_tx_thread, play.start_motor_obs_thread = start_tx, start_obs
    monkeypatch.setattr(play.signal, "signal", lambda *args: None)
    play.main(env, agent)
    assert events[:4] == ["radar_ready", "enabled", "tx_started", "obs_started"]
    assert all(c.closed and not c.commands for c, _, _ in session._legs())
    assert len(commands) == 1 and commands[0].shape == (24,)


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
