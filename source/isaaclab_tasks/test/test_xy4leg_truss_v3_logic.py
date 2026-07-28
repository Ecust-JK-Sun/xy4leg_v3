from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


_MODULE_PATH = (
    Path(__file__).parents[1]
    / "isaaclab_tasks"
    / "direct"
    / "xy4leg_truss_v3"
    / "crossing_logic.py"
)
_SPEC = importlib.util.spec_from_file_location("truss_v3_crossing_logic", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
logic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(logic)


def test_monotonic_progress_cannot_be_farmed_by_oscillation():
    previous = torch.tensor([0.4])
    gain, new_max = logic.monotonic_progress_gain(torch.tensor([0.2]), previous)
    assert torch.equal(gain, torch.zeros_like(gain))
    assert torch.equal(new_max, previous)

    gain, new_max = logic.monotonic_progress_gain(torch.tensor([0.6]), new_max)
    assert torch.allclose(gain, torch.tensor([0.2]))
    assert torch.allclose(new_max, torch.tensor([0.6]))


def test_safe_progress_does_not_advance_outside_valid_geometry():
    progress = torch.tensor([[0.8, 0.8]])
    previous = torch.tensor([[0.2, 0.2]])
    gain, new_max = logic.safe_monotonic_progress_gain(
        progress,
        previous,
        torch.tensor([[True, False]]),
    )
    assert torch.allclose(gain, torch.tensor([[0.6, 0.0]]))
    assert torch.allclose(new_max, torch.tensor([[0.8, 0.2]]))


def test_foot_must_clear_bar_in_rail_corridor_before_passing():
    state = torch.zeros((1, 2), dtype=torch.long)
    rail_valid = torch.tensor([[True, False]])

    state = logic.update_foot_crossing_state(
        state,
        torch.tensor([[0.20, 0.20]]),
        rail_valid,
        torch.tensor([[0.20, 0.20]]),
        clearance_x_window=0.14,
        min_clearance=0.04,
        pass_margin=0.05,
    )
    assert torch.equal(state, torch.zeros_like(state))

    state = logic.update_foot_crossing_state(
        state,
        torch.tensor([[0.00, 0.00]]),
        rail_valid,
        torch.tensor([[0.05, 0.05]]),
        clearance_x_window=0.14,
        min_clearance=0.04,
        pass_margin=0.05,
    )
    assert torch.equal(state, torch.tensor([[1, 0]]))

    state = logic.update_foot_crossing_state(
        state,
        torch.tensor([[0.10, 0.10]]),
        rail_valid,
        torch.tensor([[0.01, 0.20]]),
        clearance_x_window=0.14,
        min_clearance=0.04,
        pass_margin=0.05,
    )
    assert torch.equal(state, torch.tensor([[2, 0]]))


def test_reverse_curriculum_can_initialize_already_passed_feet():
    state = logic.update_foot_crossing_state(
        torch.zeros((1, 2), dtype=torch.long),
        torch.tensor([[0.10, 0.10]]),
        torch.tensor([[True, False]]),
        torch.zeros((1, 2)),
        clearance_x_window=0.14,
        min_clearance=0.04,
        pass_margin=0.05,
        initializing=torch.tensor([True]),
    )
    assert torch.equal(state, torch.tensor([[2, 0]]))


def test_crossing_stages_follow_front_then_hind_order():
    root = torch.tensor([-0.5, -0.2, 0.0, 0.2])
    feet = torch.tensor(
        [
            [-0.4, -0.4, -0.7, -0.7],
            [-0.1, -0.1, -0.4, -0.4],
            [0.1, 0.1, -0.2, -0.2],
            [0.2, 0.2, 0.1, 0.1],
        ]
    )
    stage = logic.crossing_stage(root, feet, pass_margin=0.05, attempt_distance=0.35)
    assert torch.equal(stage, torch.tensor([0, 1, 2, 3]))


def test_complete_success_requires_all_conditions():
    root = torch.tensor([0.2, 0.2, 0.2, 0.2])
    feet = torch.full((4, 4), 0.1)
    stable_steps = torch.tensor([10, 9, 10, 10])
    upright = torch.tensor([True, True, False, True])
    forbidden = torch.tensor([False, False, False, True])
    success = logic.crossing_success(
        root,
        feet,
        stable_steps,
        upright,
        forbidden,
        pass_margin=0.05,
        root_margin=0.15,
        required_stable_steps=10,
    )
    assert torch.equal(success, torch.tensor([True, False, False, False]))

    success = logic.crossing_success(
        root[:1],
        feet[:1],
        stable_steps[:1],
        upright[:1],
        forbidden[:1],
        pass_margin=0.05,
        root_margin=0.15,
        required_stable_steps=10,
        foot_passed=torch.ones((1, 4), dtype=torch.bool),
        foot_rail_valid=torch.tensor([[True, True, True, False]]),
    )
    assert not success.item()


def test_curriculum_landing_requirements_tighten_by_level():
    levels = torch.tensor([0, 1, 2, 3])
    passed = torch.tensor(
        [
            [False, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
        ]
    )
    rail_valid = torch.tensor(
        [
            [True, True, True, True],
            [True, True, True, False],
            [True, True, True, False],
            [True, True, True, False],
        ]
    )
    contacts = torch.tensor(
        [
            [True, True, True, True],
            [True, True, True, False],
            [True, True, True, False],
            [True, True, True, False],
        ]
    )
    landing = logic.curriculum_landing_condition(levels, passed, rail_valid, contacts)
    assert torch.equal(landing, torch.tensor([True, True, True, False]))

    five_level_landing = logic.curriculum_landing_condition(
        torch.tensor([3, 4]),
        passed[2:4],
        rail_valid[2:4],
        contacts[2:4],
        final_level=4,
    )
    assert torch.equal(five_level_landing, torch.tensor([True, False]))


def test_curriculum_success_requires_stable_root_pass_and_posture():
    success = logic.curriculum_crossing_success(
        root_passed=torch.tensor([True, True, False, True]),
        landing_condition=torch.tensor([True, True, True, True]),
        stable_contact_steps=torch.tensor([10, 9, 10, 10]),
        upright=torch.tensor([True, True, True, False]),
        forbidden_contact=torch.tensor([False, False, False, False]),
        required_stable_steps=10,
    )
    assert torch.equal(success, torch.tensor([True, False, False, False]))


def test_gripper_hysteresis_prevents_boundary_chatter():
    phase = torch.tensor([[0.72, 0.82, 0.74, 0.68]])
    previous_open = torch.tensor([[True, False, False, True]])
    state = logic.gripper_open_state(
        phase,
        previous_open,
        duty_factor=0.75,
        hysteresis=0.05,
    )
    assert torch.equal(state, torch.tensor([[True, True, False, False]]))


def test_curriculum_promotes_and_demotes_independently():
    levels = torch.tensor([0, 1, 2, 3])
    ema = torch.tensor([0.8, 0.8, 0.2, 0.9])
    episodes = torch.tensor([10, 10, 10, 10])
    failures = torch.tensor([0, 0, 8, 0])
    updated = logic.update_curriculum_levels(
        levels,
        ema,
        episodes,
        failures,
        min_episodes=10,
        promote_threshold=0.7,
        demote_threshold=0.3,
        max_level=3,
        failure_streak_limit=8,
    )
    assert torch.equal(updated, torch.tensor([1, 2, 1, 3]))
