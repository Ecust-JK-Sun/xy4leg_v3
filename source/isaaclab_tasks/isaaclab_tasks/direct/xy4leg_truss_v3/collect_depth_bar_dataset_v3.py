from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args_v3 as cli_args  # isort: skip


DEFAULT_TASK = "Isaac-XY4Leg-Gripper-Truss-V3-Student-DepthBar-Direct-v0"
DEFAULT_NUM_ENVS = 128
DEFAULT_HEADLESS = True
DEFAULT_LOG_ROOT = str(Path(__file__).resolve().parents[5] / "logs" / "rsl_rl")

parser = argparse.ArgumentParser(description="Collect Truss V3 local heightmaps and bar-estimate labels.")
parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS)
parser.add_argument("--num_samples", type=int, default=50000)
parser.add_argument("--sample_stride", type=int, default=2)
parser.add_argument("--max_steps", type=int, default=200000)
parser.add_argument("--fixed_level", type=int, default=4, help="Set 0--4 for fixed curriculum level, -1 to keep cfg.")
parser.add_argument(
    "--levels",
    type=str,
    default=None,
    help="Comma-separated fixed levels for balanced collection, e.g. 0,1,2,3,4. Overrides --fixed_level.",
)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--task", type=str, default=DEFAULT_TASK)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=0)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=DEFAULT_HEADLESS)
args_cli, hydra_args = parser.parse_known_args()
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


def _parse_levels(levels: str | None) -> list[int] | None:
    if levels is None or levels.strip() == "":
        return None
    parsed = [int(item.strip()) for item in levels.split(",") if item.strip() != ""]
    if not parsed:
        raise ValueError("--levels did not contain any valid level")
    return parsed


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    requested_levels = _parse_levels(args_cli.levels)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    if requested_levels is not None:
        invalid_levels = [
            level for level in requested_levels if level < 0 or level > int(env_cfg.curriculum_max_level)
        ]
        if invalid_levels:
            raise ValueError(
                f"Requested levels {invalid_levels} outside valid range [0, {env_cfg.curriculum_max_level}]"
            )
        if args_cli.num_envs < len(requested_levels):
            raise ValueError(
                f"--num_envs ({args_cli.num_envs}) must be >= number of requested levels ({len(requested_levels)})"
            )
        env_cfg.initial_curriculum_level = int(requested_levels[0])
        env_cfg.curriculum_enabled = False
    elif args_cli.fixed_level >= 0:
        env_cfg.initial_curriculum_level = int(args_cli.fixed_level)
        env_cfg.curriculum_enabled = False

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

    if requested_levels is not None:
        assigned_levels = torch.tensor(
            [requested_levels[i % len(requested_levels)] for i in range(env.unwrapped.num_envs)],
            dtype=torch.long,
            device=env.unwrapped.device,
        )
        env.unwrapped._curriculum_level.copy_(assigned_levels)
        env.reset()
    else:
        assigned_levels = None

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
    heightmap_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    level_chunks: list[torch.Tensor] = []
    collected = 0
    steps = 0
    while collected < args_cli.num_samples and steps < args_cli.max_steps:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if reset_policy is not None:
                reset_policy(dones)
        steps += 1
        if steps % args_cli.sample_stride != 0:
            continue

        heightmap = obs["heightmap"].detach().cpu().to(dtype=torch.float16)
        label = obs["bar_estimate_label"].detach().cpu().to(dtype=torch.float32)
        levels = env.unwrapped._curriculum_level.detach().cpu().to(dtype=torch.long)
        remaining = args_cli.num_samples - collected
        if heightmap.shape[0] > remaining:
            heightmap = heightmap[:remaining]
            label = label[:remaining]
            levels = levels[:remaining]
        heightmap_chunks.append(heightmap)
        label_chunks.append(label)
        level_chunks.append(levels)
        collected += heightmap.shape[0]
        if collected % max(args_cli.num_envs * 10, 1) == 0:
            print(f"collected {collected}/{args_cli.num_samples} samples")

    env.close()
    output = Path(args_cli.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    levels = torch.cat(level_chunks, dim=0)
    torch.save(
        {
            "heightmap": torch.cat(heightmap_chunks, dim=0),
            "bar_delta_b": torch.cat(label_chunks, dim=0),
            "level": levels,
            "heightmap_normalized": True,
            "heightmap_channels": ("height", "valid_mask"),
            "fixed_level": args_cli.fixed_level,
            "levels": requested_levels,
            "checkpoint": resume_path,
        },
        output,
    )
    print(f"saved {collected} samples to {output}")
    level_counts = torch.bincount(levels, minlength=int(env_cfg.curriculum_max_level) + 1)
    for level, count in enumerate(level_counts.tolist()):
        if count > 0:
            print(f"level {level}: {count} samples")


if __name__ == "__main__":
    main()
    simulation_app.close()
