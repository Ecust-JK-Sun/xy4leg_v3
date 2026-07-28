# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a Truss V3 RSL-RL checkpoint without training."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args_v3 as cli_args  # isort: skip


DEFAULT_TASK = "Isaac-XY4Leg-Gripper-Truss-V3-Student-NoisyBar-Direct-v0"
DEFAULT_NUM_ENVS = 128
DEFAULT_HEADLESS = True
DEFAULT_LOG_ROOT = str(Path(__file__).resolve().parents[5] / "logs" / "rsl_rl")

parser = argparse.ArgumentParser(description="Evaluate a Truss V3 RSL-RL checkpoint.")
parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS, help="Number of parallel envs.")
parser.add_argument("--num_episodes", type=int, default=1000, help="Completed episodes to evaluate.")
parser.add_argument("--max_steps", type=int, default=200000, help="Safety limit on environment steps.")
parser.add_argument("--fixed_level", type=int, default=-1, help="Set 0--4 for fixed curriculum level, -1 to keep cfg.")
parser.add_argument(
    "--bar_distance_range",
    type=str,
    default=None,
    help="Override the bar distance range as 'min,max'. With --fixed_level, only that level is changed.",
)
parser.add_argument(
    "--depth_bar_estimator_checkpoint",
    type=str,
    default=None,
    help="TorchScript heightmap estimator checkpoint for DepthBar tasks.",
)
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Task name.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point."
)
parser.add_argument("--seed", type=int, default=0, help="Evaluation seed.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=DEFAULT_HEADLESS)
args_cli, hydra_args = parser.parse_known_args()
if "DepthBar" in args_cli.task:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def _scalar(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _print_metric(name: str, value: float | None) -> None:
    if value is None:
        return
    print(f"{name:42s}: {value:.6f}")


def _parse_range(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2 or parts[0] == "" or parts[1] == "":
        raise ValueError(f"Expected --bar_distance_range as 'min,max', got {value!r}")
    lower, upper = float(parts[0]), float(parts[1])
    if upper <= lower:
        raise ValueError(f"Expected upper > lower for --bar_distance_range, got {value!r}")
    return lower, upper


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    if args_cli.fixed_level >= 0:
        env_cfg.initial_curriculum_level = int(args_cli.fixed_level)
        env_cfg.curriculum_enabled = False
    if args_cli.bar_distance_range is not None:
        distance_range = _parse_range(args_cli.bar_distance_range)
        ranges = list(env_cfg.level_bar_distance_ranges)
        if args_cli.fixed_level >= 0:
            level = int(args_cli.fixed_level)
            if level < 0 or level >= len(ranges):
                raise ValueError(f"--fixed_level {level} outside level_bar_distance_ranges")
            ranges[level] = distance_range
        else:
            ranges = [distance_range for _ in ranges]
        env_cfg.level_bar_distance_ranges = tuple(ranges)
    if args_cli.depth_bar_estimator_checkpoint is not None:
        env_cfg.depth_bar_estimator_checkpoint = args_cli.depth_bar_estimator_checkpoint

    log_root_path = os.path.join(DEFAULT_LOG_ROOT, agent_cfg.experiment_name)
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env_cfg.log_dir = os.path.dirname(resume_path)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Checkpoint: {resume_path}")
    print(f"[INFO] Episodes target: {args_cli.num_episodes}")
    if args_cli.fixed_level >= 0:
        print(f"[INFO] Fixed level: {args_cli.fixed_level}")
    if args_cli.bar_distance_range is not None:
        print(f"[INFO] Bar distance range: {args_cli.bar_distance_range}")
    if args_cli.depth_bar_estimator_checkpoint is not None:
        print(f"[INFO] Depth bar estimator: {args_cli.depth_bar_estimator_checkpoint}")

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    if version.parse(installed_version) >= version.parse("4.0.0"):
        reset_policy = policy.reset
    else:
        policy_module = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
        reset_policy = policy_module.reset if hasattr(policy_module, "reset") else None

    obs = env.get_observations()
    completed = 0.0
    steps = 0
    metric_sums: dict[str, float] = defaultdict(float)
    metric_weights: dict[str, float] = defaultdict(float)
    level_episode_sums: dict[int, float] = defaultdict(float)
    level_success_sums: dict[int, float] = defaultdict(float)

    while completed < args_cli.num_episodes and steps < args_cli.max_steps:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            if reset_policy is not None:
                reset_policy(dones)
        steps += 1

        log = extras.get("log", {})
        if not log:
            continue

        level_counts = []
        for level in range(env.unwrapped.cfg.curriculum_max_level + 1):
            count_key = f"Curriculum/level_{level}_episodes"
            rate_key = f"Curriculum/level_{level}_success_rate"
            count = _scalar(log.get(count_key, 0.0))
            rate = _scalar(log.get(rate_key, 0.0))
            level_counts.append(count)
            level_episode_sums[level] += count
            level_success_sums[level] += rate * count

        reset_count = sum(level_counts)
        if reset_count <= 0.0:
            reset_count = float(dones.sum().item())
        if reset_count <= 0.0:
            continue

        completed += reset_count
        for key, value in log.items():
            if key.startswith("Curriculum/level_") and (
                key.endswith("_episodes")
                or key.endswith("_success_rate")
                or key.endswith("_success_count")
            ):
                continue
            metric_sums[key] += _scalar(value) * reset_count
            metric_weights[key] += reset_count

    env.close()

    print(f"[INFO] Completed episodes: {int(completed)}")
    print(f"[INFO] Environment steps: {steps}")
    print("\n=== Summary ===")
    summary_keys = [
        "Crossing/success_rate",
        "Crossing/front_complete_rate",
        "Crossing/hind_complete_rate",
        "Crossing/root_complete_rate",
        "Crossing/early_front_event_rate",
        "Curriculum/level_mean",
        "Curriculum/success_ema",
        "ResetRate/fall_or_tilt",
        "ResetRate/no_progress",
        "ResetRate/forbidden_contact",
        "Walk/forward_distance",
        "Walk/episode_duration",
        "Gait/stance_contact_score",
        "Gait/swing_no_contact_score",
        "Landing/max_stable_fraction",
        "Diagnostics/action_saturation",
        "Diagnostics/action_abs_gt_075",
        "BarEstimate/visible_rate",
        "BarEstimate/delay_steps",
        "Heightmap/valid_rate",
    ]
    for key in summary_keys:
        if metric_weights.get(key, 0.0) > 0.0:
            _print_metric(key, metric_sums[key] / metric_weights[key])

    print("\n=== Per-Level Success ===")
    for level in range(env.unwrapped.cfg.curriculum_max_level + 1):
        count = level_episode_sums[level]
        if count <= 0.0:
            continue
        print(
            f"level {level}: episodes={int(count):5d}, "
            f"success={level_success_sums[level] / count:.6f}"
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
