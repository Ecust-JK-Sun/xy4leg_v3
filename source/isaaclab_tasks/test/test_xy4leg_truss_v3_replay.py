"""CPU-only checks for the common default model and measured replay feedforward."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


TASK = Path(__file__).parents[1] / "isaaclab_tasks/direct/xy4leg_truss_v3"
REAL = TASK / "sim2real_v3"


def load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.syspath_prepend(str(REAL))
    load(monkeypatch, "common", TASK / "simdata_replay/common.py")
    module = load(monkeypatch, "replay_compensation_test", TASK / "simdata_replay/replay_simdata_real.py")
    shared = load(monkeypatch, "threads_shared", REAL / "threads_shared.py")
    clock = types.SimpleNamespace(now=100.)
    def sleep(delay):
        clock.now += delay
    module.time = types.SimpleNamespace(time=lambda: clock.now, monotonic=lambda: clock.now, sleep=sleep)
    shared.time = module.time
    return module, shared, clock


@pytest.mark.parametrize("mode", ["fitted_j1", "fitted", "nominal", "none"])
def test_replay_default_and_comparison_models(replay, mode):
    module, _, _ = replay
    args = module.parser.parse_args(["--trajectory", "unused.npz"] + ([] if mode == "fitted_j1" else ["--compensation", mode]))
    controller = module._make_replay_feedforward(args, 25.)
    if mode == "none":
        assert controller is None
    else:
        from initialization_models import make_compensation_model
        assert controller.model.description() == make_compensation_model(mode).description()
        assert ("j1_residual" in controller.model.description()) == (mode == "fitted_j1")


@pytest.mark.parametrize("option,value", [
    ("--gravity_scale", "nan"), ("--feedback_timeout_s", "0"),
    ("--j1_residual_scale", "2"), ("--j1_residual_limit_nm", "-1"),
])
def test_invalid_replay_model_options_rejected(replay, option, value):
    module, _, _ = replay
    args = module.parser.parse_args(["--trajectory", "unused.npz", option, value])
    with pytest.raises(ValueError):
        module._make_replay_feedforward(args, 25.)


def test_missing_default_residual_does_not_enter_hardware_replay(replay, monkeypatch, tmp_path):
    module, _, _ = replay
    args = ["replay", "--trajectory", "unused.npz", "--enable_motors", "--j1_residual_file", str(tmp_path/"missing.json")]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(module, "_prepare_commands", lambda _: (np.zeros((2,24)), {}, 25., [], {}))
    monkeypatch.setattr(module, "_print_summary", lambda *args: None)
    monkeypatch.setattr(module, "_run_hardware_replay", lambda *args: pytest.fail("Opened hardware with missing model"))
    with pytest.raises(FileNotFoundError):
        module.main()


def test_replay_uses_measured_pose_and_preserves_j6_and_torque_limits(replay):
    module, shared_module, clock = replay
    args = module.parser.parse_args(["--trajectory", "unused.npz"])
    controller = module._make_replay_feedforward(args, 25.)
    shared = shared_module.SharedData()
    from gravity_compensation import DEFAULT_POSE_DEG
    measured = np.deg2rad(DEFAULT_POSE_DEG).T.reshape(20)
    command = np.r_[np.zeros(20), [.1, -.2, .3, -.4]]
    for _ in range(3):
        shared.set_obs(np.r_[measured, np.zeros(28)], np.full(24, clock.now), np.ones(24, dtype=bool))
        module._publish_replay_command(shared, command, controller, .5)
        clock.now += .04
    actual, torque, *_ = shared.command_snapshot()
    np.testing.assert_allclose(actual, command)
    assert torque.shape == (20,) and np.any(torque != 0)
    np.testing.assert_allclose(controller.raw, controller.model.torque(measured.astype(np.float32)), atol=1e-10)
    assert np.all(abs(torque) <= np.repeat(controller.model.slew_rates, 4)*.08+1e-8)
    clock.now += .51
    sequence = shared.action_sequence
    with pytest.raises(RuntimeError, match="stale"):
        module._publish_replay_command(shared, command, controller, .5)
    assert shared.action_sequence == sequence


@pytest.mark.parametrize("tx_error", [False, True])
def test_replay_all_motion_stages_publish_feedforward_and_cleanup(replay, monkeypatch, tx_error):
    module, shared_module, clock = replay
    args = module.parser.parse_args(["--trajectory", "unused.npz", "--startup_hold_s", ".08",
                                    "--hold_final_s", ".08", "--soft_start_s", ".08", "--use_feedback_soft_start"])
    events, publications = [], []
    class Shared(shared_module.SharedData):
        def set_action(self, *args, **kwargs):
            super().set_action(*args, **kwargs)
            publications.append(self.command_snapshot())
    session = types.SimpleNamespace(enable_all=lambda: events.append("enable"), close=lambda: events.append("close"))
    thread = types.SimpleNamespace(join=lambda **kwargs: None)
    def rx(shared, *args, **kwargs):
        assert events == ["enable"]
        events.append("rx")
        shared.set_obs(np.r_[np.full(20,.2), np.zeros(28)], np.full(24,clock.now), np.ones(24,dtype=bool))
        return thread
    def tx(shared, *args, **kwargs):
        assert events == ["enable", "rx"] and kwargs["feedback_ready_event"].is_set()
        events.append("tx")
        if tx_error:
            shared.set_error("tx", "failed")
        return thread
    monkeypatch.setitem(sys.modules, "threads_shared", types.SimpleNamespace(SharedData=Shared))
    monkeypatch.setitem(sys.modules, "motor_session_gripper", types.SimpleNamespace(create_motor_session=lambda *args:(True,session)))
    monkeypatch.setitem(sys.modules, "serial_threads", types.SimpleNamespace(start_motor_obs_thread=rx, start_serial_tx_thread=tx))
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    captured_poses = []
    def update(q, *args, **kwargs):
        captured_poses.append(q.copy())
        return np.full(20,.25)
    controller = types.SimpleNamespace(update=update)
    commands = np.vstack([np.r_[np.full(20,.3), np.ones(4)], np.r_[np.full(20,.4), -np.ones(4)]])
    if tx_error:
        with pytest.raises(RuntimeError, match="tx: failed"):
            module._run_hardware_replay(commands, 25., args, controller)
    else:
        module._run_hardware_replay(commands, 25., args, controller)
        assert len(publications) >= 9  # Startup, soft ramp, hold, trajectory, final hold, J6 exit.
        np.testing.assert_array_equal(publications[-1][0][20:], np.zeros(4))
        assert any(np.allclose(p[0], commands[1]) for p in publications)
    assert events[-1] == "close"
    assert all(np.allclose(q,.2) for q in captured_poses)
    assert all(np.array_equal(p[1],np.full(20,.25)) for p in publications)
