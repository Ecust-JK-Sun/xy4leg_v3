from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args_v3 as cli_args  # isort: skip


DEFAULT_TASK = "Isaac-XY4Leg-Gripper-Truss-V3-Student-DepthBar-Direct-v0"
DEFAULT_OUTPUT_DIR = "logs/rsl_rl/xy4leg_truss_v3_depth_bar/visualization"
DEFAULT_LOG_ROOT = str(Path(__file__).resolve().parents[5] / "logs" / "rsl_rl")

parser = argparse.ArgumentParser(description="Visualize Truss V3 front/rear depth cameras and fused heightmap.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_index", type=int, default=0)
parser.add_argument("--num_frames", type=int, default=8)
parser.add_argument("--warmup_steps", type=int, default=20)
parser.add_argument("--frame_interval", type=int, default=20)
parser.add_argument("--fixed_level", type=int, default=0, help="Set 0--4 for fixed curriculum level, -1 to keep cfg.")
parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
parser.add_argument("--task", type=str, default=DEFAULT_TASK)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=0)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata

import cv2
import gymnasium as gym
import numpy as np
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


def _to_numpy_2d(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32)


def _depth_to_color(depth_m: torch.Tensor, valid: torch.Tensor, max_depth: float) -> np.ndarray:
    depth = _to_numpy_2d(depth_m)
    valid_np = _to_numpy_2d(valid.float()) > 0.5
    normalized = 1.0 - np.clip(depth / max(max_depth, 1.0e-6), 0.0, 1.0)
    image_u8 = (normalized * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(image_u8, cv2.COLORMAP_TURBO)
    color[~valid_np] = (0, 0, 0)
    return color


def _heightmap_to_color(heightmap: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    height = _to_numpy_2d(heightmap[0])
    valid = _to_numpy_2d(heightmap[1]) > 0.5
    image_u8 = (np.clip(height, 0.0, 1.0) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(image_u8, cv2.COLORMAP_TURBO)
    color[~valid] = (0, 0, 0)
    valid_image = np.repeat((valid.astype(np.uint8) * 255)[:, :, None], 3, axis=2)
    return color, valid_image


def _add_label(image: np.ndarray, label: str) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    labeled = cv2.copyMakeBorder(image, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(labeled, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return labeled


def _resize(image: np.ndarray, scale: int) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)


def _draw_heightmap_axes_and_bar(image: np.ndarray, bar_delta_b: torch.Tensor, cfg) -> np.ndarray:
    scale = max(image.shape[1] // int(cfg.depth_bar_heightmap_size), 1)
    x_min, x_max = cfg.depth_bar_heightmap_x_range
    y_min, y_max = cfg.depth_bar_heightmap_y_range
    dx = float(bar_delta_b[0].detach().cpu())
    dy = float(bar_delta_b[1].detach().cpu())
    col = int((dx - x_min) / max(x_max - x_min, 1.0e-6) * (int(cfg.depth_bar_heightmap_size) - 1) * scale)
    row = int((dy - y_min) / max(y_max - y_min, 1.0e-6) * (int(cfg.depth_bar_heightmap_size) - 1) * scale)
    if 0 <= row < image.shape[0] and 0 <= col < image.shape[1]:
        cv2.drawMarker(image, (col, row), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
        cv2.circle(image, (col, row), 8, (0, 0, 255), 1)

    cv2.arrowedLine(image, (12, image.shape[0] - 14), (64, image.shape[0] - 14), (255, 255, 255), 1)
    cv2.putText(image, "+x", (68, image.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.arrowedLine(image, (12, image.shape[0] - 14), (12, image.shape[0] - 52), (255, 255, 255), 1)
    cv2.putText(image, "-y", (16, image.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return image


def _make_composite(front: np.ndarray, rear: np.ndarray, heightmap: np.ndarray, valid: np.ndarray) -> np.ndarray:
    top = np.concatenate((_add_label(front, "front depth"), _add_label(rear, "rear depth")), axis=1)
    bottom = np.concatenate((_add_label(heightmap, "fused base heightmap"), _add_label(valid, "heightmap valid mask")), axis=1)
    return np.concatenate((top, bottom), axis=0)


def _heightmap_valid_bounds(heightmap: torch.Tensor, cfg) -> tuple[float, float, float, float] | None:
    valid_indices = torch.nonzero(heightmap[1] > 0.5, as_tuple=False)
    if valid_indices.numel() == 0:
        return None
    rows = valid_indices[:, 0].float()
    cols = valid_indices[:, 1].float()
    grid = max(int(cfg.depth_bar_heightmap_size) - 1, 1)
    x_min, x_max = cfg.depth_bar_heightmap_x_range
    y_min, y_max = cfg.depth_bar_heightmap_y_range
    x_values = float(x_min) + cols / float(grid) * float(x_max - x_min)
    y_values = float(y_min) + rows / float(grid) * float(y_max - y_min)
    return (
        float(torch.min(x_values).detach().cpu()),
        float(torch.max(x_values).detach().cpu()),
        float(torch.min(y_values).detach().cpu()),
        float(torch.max(y_values).detach().cpu()),
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    if args_cli.fixed_level >= 0:
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

    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_index = max(0, min(int(args_cli.env_index), env.unwrapped.num_envs - 1))

    obs = env.get_observations()
    saved = 0
    total_steps = args_cli.warmup_steps + args_cli.num_frames * args_cli.frame_interval
    for step in range(total_steps + 1):
        if step >= args_cli.warmup_steps and (step - args_cli.warmup_steps) % args_cli.frame_interval == 0:
            base_env = env.unwrapped
            front_depth, front_valid = base_env._camera_depth_m(base_env.front_tiled_camera, base_env.cfg.tiled_camera)
            rear_depth, rear_valid = base_env._camera_depth_m(base_env.rear_tiled_camera, base_env.cfg.rear_tiled_camera)
            heightmap = obs["heightmap"][env_index]
            label = obs["bar_estimate_label"][env_index]

            scale = 4
            max_depth = float(base_env.cfg.depth_bar_max_depth)
            front_color = _resize(_depth_to_color(front_depth[env_index], front_valid[env_index], max_depth), scale)
            rear_color = _resize(_depth_to_color(rear_depth[env_index], rear_valid[env_index], max_depth), scale)
            height_color, valid_image = _heightmap_to_color(heightmap)
            height_color = _draw_heightmap_axes_and_bar(_resize(height_color, scale), label, base_env.cfg)
            valid_image = _resize(valid_image, scale)

            step_dir = output_dir / f"step_{step:04d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(step_dir / "front_depth.png"), front_color)
            cv2.imwrite(str(step_dir / "rear_depth.png"), rear_color)
            cv2.imwrite(str(step_dir / "heightmap.png"), height_color)
            cv2.imwrite(str(step_dir / "valid_mask.png"), valid_image)
            cv2.imwrite(str(step_dir / "composite.png"), _make_composite(front_color, rear_color, height_color, valid_image))

            valid_rate = float(heightmap[1].float().mean().detach().cpu())
            valid_bounds = _heightmap_valid_bounds(heightmap, base_env.cfg)
            with open(step_dir / "meta.txt", "w", encoding="utf-8") as f:
                f.write(f"step: {step}\n")
                f.write(f"frame_index: {saved}\n")
                f.write(f"env_index: {env_index}\n")
                f.write(f"fixed_level: {args_cli.fixed_level}\n")
                f.write(f"bar_delta_b_m: {label.detach().cpu().tolist()}\n")
                f.write(f"heightmap_valid_rate: {valid_rate:.6f}\n")
                if valid_bounds is not None:
                    f.write(
                        "heightmap_valid_bounds_m: "
                        f"x=[{valid_bounds[0]:.3f}, {valid_bounds[1]:.3f}], "
                        f"y=[{valid_bounds[2]:.3f}, {valid_bounds[3]:.3f}]\n"
                    )
                else:
                    f.write("heightmap_valid_bounds_m: none\n")
                f.write(f"checkpoint: {resume_path}\n")
                f.write("heightmap display axes: +x is right, -y is up\n")
                f.write("white/red cross marks oracle bar_delta_b projected to the heightmap grid\n")

            saved += 1
            print(f"saved {saved}/{args_cli.num_frames}: {step_dir / 'composite.png'}")
            if saved >= args_cli.num_frames:
                break

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if reset_policy is not None:
                reset_policy(dones)

    env.close()
    print(f"saved visualization frames to {output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
