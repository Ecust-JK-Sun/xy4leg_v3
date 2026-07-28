from __future__ import annotations

import torch


FRONT_FEET = (0, 1)
HIND_FEET = (2, 3)


def normalized_crossing_progress(
    signed_distance: torch.Tensor,
    approach_distance: float,
    pass_margin: float,
) -> torch.Tensor:
    """Map signed bar distance to a bounded crossing progress in [0, 1]."""
    scale = max(approach_distance + pass_margin, 1.0e-6)
    return ((signed_distance + approach_distance) / scale).clamp(0.0, 1.0)


def monotonic_progress_gain(
    progress: torch.Tensor,
    previous_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return only newly achieved progress, preventing oscillation reward farming."""
    new_max = torch.maximum(progress, previous_max)
    return (new_max - previous_max).clamp_min(0.0), new_max


def safe_monotonic_progress_gain(
    progress: torch.Tensor,
    previous_max: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update progress only where the crossing geometry and posture are valid."""
    candidate = torch.where(valid, progress, previous_max)
    return monotonic_progress_gain(candidate, previous_max)


def update_foot_crossing_state(
    previous_state: torch.Tensor,
    signed_distance: torch.Tensor,
    rail_valid: torch.Tensor,
    clearance: torch.Tensor,
    *,
    clearance_x_window: float,
    min_clearance: float,
    pass_margin: float,
    initializing: torch.Tensor | None = None,
) -> torch.Tensor:
    """Advance each foot through before-bar, cleared-bar and passed-bar states."""
    state = previous_state.clone()
    near_bar = torch.abs(signed_distance) <= clearance_x_window
    cleared_bar = rail_valid & near_bar & (clearance >= min_clearance)

    state = torch.where((state == 0) & cleared_bar, torch.ones_like(state), state)
    passed_bar = rail_valid & (state >= 1) & (signed_distance > pass_margin)
    state = torch.where(passed_bar, torch.full_like(state, 2), state)

    if initializing is not None:
        initializing = initializing[:, None].expand_as(state)
        initialized_state = torch.zeros_like(state)
        initialized_state = torch.where(cleared_bar, torch.ones_like(state), initialized_state)
        # Reverse-curriculum reset poses may legitimately begin with a foot
        # already behind the bar, so they do not need an observed clearance transition.
        initialized_state = torch.where(
            rail_valid & (signed_distance > pass_margin),
            torch.full_like(state, 2),
            initialized_state,
        )
        state = torch.where(initializing, initialized_state, state)
    return state


def crossing_stage(
    root_signed_distance: torch.Tensor,
    foot_signed_distance: torch.Tensor,
    pass_margin: float,
    attempt_distance: float,
    foot_passed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute approach/front/hind/landing stages as integer values 0..3."""
    if foot_passed is None:
        foot_passed = foot_signed_distance > pass_margin
    front_passed = torch.all(foot_passed[:, FRONT_FEET], dim=1)
    all_passed = torch.all(foot_passed, dim=1)
    near_bar = root_signed_distance > -attempt_distance

    stage = torch.zeros_like(root_signed_distance, dtype=torch.long)
    stage = torch.where(near_bar, torch.ones_like(stage), stage)
    stage = torch.where(front_passed, torch.full_like(stage, 2), stage)
    stage = torch.where(all_passed, torch.full_like(stage, 3), stage)
    return stage


def gripper_open_state(
    phase: torch.Tensor,
    previous_open: torch.Tensor,
    duty_factor: float,
    hysteresis: float,
) -> torch.Tensor:
    """Open during swing and use phase hysteresis around the stance/swing boundary."""
    open_threshold = min(duty_factor + hysteresis, 0.99)
    close_threshold = max(duty_factor - hysteresis, 0.0)
    return torch.where(previous_open, phase > close_threshold, phase > open_threshold)


def crossing_success(
    root_signed_distance: torch.Tensor,
    foot_signed_distance: torch.Tensor,
    stable_contact_steps: torch.Tensor,
    upright: torch.Tensor,
    forbidden_contact: torch.Tensor,
    pass_margin: float,
    root_margin: float,
    required_stable_steps: int,
    foot_passed: torch.Tensor | None = None,
    foot_rail_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the complete-crossing success mask."""
    if foot_passed is None:
        foot_passed = foot_signed_distance > pass_margin
    if foot_rail_valid is None:
        foot_rail_valid = torch.ones_like(foot_passed)
    feet_passed = torch.all(foot_passed & foot_rail_valid, dim=1)
    return (
        feet_passed
        & (root_signed_distance > root_margin)
        & (stable_contact_steps >= required_stable_steps)
        & upright
        & ~forbidden_contact
    )


def curriculum_landing_condition(
    curriculum_level: torch.Tensor,
    foot_passed: torch.Tensor,
    foot_rail_valid: torch.Tensor,
    foot_contacts: torch.Tensor,
    final_level: int = 3,
) -> torch.Tensor:
    """Return level-specific foot geometry and support requirements."""
    valid_support = foot_rail_valid & foot_contacts
    support_count = torch.sum(valid_support, dim=1)
    hind_ready = torch.all(
        foot_passed[:, HIND_FEET] & foot_rail_valid[:, HIND_FEET], dim=1
    )
    all_passed = torch.all(foot_passed, dim=1)

    level_zero = hind_ready & (support_count >= 3)
    intermediate = all_passed & (support_count >= 3)
    final = intermediate & torch.all(foot_rail_valid, dim=1)
    return torch.where(
        curriculum_level == 0,
        level_zero,
        torch.where(curriculum_level >= final_level, final, intermediate),
    )


def curriculum_crossing_success(
    root_passed: torch.Tensor,
    landing_condition: torch.Tensor,
    stable_contact_steps: torch.Tensor,
    upright: torch.Tensor,
    forbidden_contact: torch.Tensor,
    required_stable_steps: int,
) -> torch.Tensor:
    """Return success after the level-specific landing condition remains stable."""
    return (
        root_passed
        & landing_condition
        & (stable_contact_steps >= required_stable_steps)
        & upright
        & ~forbidden_contact
    )


def update_curriculum_levels(
    levels: torch.Tensor,
    success_ema: torch.Tensor,
    episode_counts: torch.Tensor,
    failure_streak: torch.Tensor,
    *,
    min_episodes: int,
    promote_threshold: float,
    demote_threshold: float,
    max_level: int,
    failure_streak_limit: int,
) -> torch.Tensor:
    """Promote or demote per-environment curriculum levels."""
    enough_data = episode_counts >= min_episodes
    promote = enough_data & (success_ema >= promote_threshold) & (levels < max_level)
    demote = (
        enough_data
        & ((success_ema <= demote_threshold) | (failure_streak >= failure_streak_limit))
        & (levels > 0)
    )

    new_levels = levels + promote.to(levels.dtype) - demote.to(levels.dtype)
    return new_levels.clamp(0, max_level)
