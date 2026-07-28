# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the Truss V3 teacher with a project-local copy of IsaacLab's RSL-RL launcher."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import os
import struct
import sys
import time as time_module
import types
from pathlib import Path

# Avoid cuBLASLt/TF32 addmm paths that can fail on some CUDA driver/GPU
# combinations during small MLP Linear calls.
os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")

from isaaclab.app import AppLauncher

# Local copy: edit cli_args_v3.py when V3 needs task-specific command-line options.
import cli_args_v3 as cli_args  # isort: skip

# DEFAULT_TASK = "Isaac-XY4Leg-Gripper-Truss-V3-Walk-Direct-v0"
# DEFAULT_TASK = "Isaac-XY4Leg-Gripper-Truss-V3-Teacher-Direct-v0"
# DEFAULT_TASK = "Isaac-XY4Leg-Gripper-Truss-V3-Student-Direct-v0"
DEFAULT_TASK = "Isaac-XY4Leg-Gripper-Truss-V3-Student-OracleBar-Direct-v0"


DEFAULT_NUM_ENVS = 128
# None keeps each task's own default: Walk=20000, Teacher=50000, DR=2500.
DEFAULT_MAX_ITERATIONS: int | None = None
DEFAULT_HEADLESS = True
DEFAULT_LOG_ROOT = str(Path(__file__).resolve().parents[5] / "logs" / "rsl_rl")
LONG_RUN_SAVE_FRACTION = 0.02
LONG_RUN_MIN_ITERATIONS = 30_000
SHORT_RUN_SAVE_INTERVAL = 500

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Record periodic training videos. Use --no-video to disable.",
)
parser.add_argument("--video_length", type=int, default=500, help="Length of each video (in environment steps).")
parser.add_argument(
    "--video_interval",
    type=int,
    default=None,
    help="PPO learning iterations between training videos. Defaults to the checkpoint save interval.",
)
parser.add_argument(
    "--num_envs", type=int, default=DEFAULT_NUM_ENVS, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--pretrained_actor",
    type=str,
    default=None,
    help="Load only actor weights from a checkpoint, e.g. Walk -> Crossing.",
)
parser.add_argument(
    "--teacher_checkpoint",
    type=str,
    default=None,
    help="Teacher PPO checkpoint for DistillationRunner. Defaults to the student agent config.",
)
parser.add_argument(
    "--depth_bar_estimator_checkpoint",
    type=str,
    default=None,
    help="TorchScript heightmap estimator checkpoint for DepthBar tasks.",
)
parser.add_argument(
    "--max_iterations",
    type=int,
    default=DEFAULT_MAX_ITERATIONS,
    help="RL Policy training iterations.",
)
parser.add_argument(
    "--save_interval",
    type=int,
    default=None,
    help="PPO learning iterations between model checkpoints. Defaults to 2% of long runs or 500 for short runs.",
)
parser.add_argument(
    "--scalar_log_interval",
    type=int,
    default=10,
    help="PPO learning iterations between TensorBoard scalar writes.",
)
parser.add_argument(
    "--csv_log_interval",
    type=int,
    default=100,
    help="PPO learning iterations between sampled CSV scalar writes. Use 0 to disable sampled CSV.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=DEFAULT_HEADLESS)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video or "DepthBar" in args_cli.task:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("highest")


from tensorboard.compat.proto import event_pb2, summary_pb2


def _make_crc32c_table() -> tuple[int, ...]:
    table = []
    polynomial = 0x82F63B78
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = (crc >> 1) ^ polynomial if crc & 1 else crc >> 1
        table.append(crc & 0xFFFFFFFF)
    return tuple(table)


_CRC32C_TABLE = _make_crc32c_table()


def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return (~crc) & 0xFFFFFFFF


def _masked_crc32c(data: bytes) -> int:
    crc = _crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


class LightweightSummaryWriter:
    """TensorBoard scalar writer without TensorBoard's numpy/gfile dependencies."""

    def __init__(self, log_dir: str | os.PathLike | None = None, flush_secs: int = 10, **_kwargs):
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.flush_secs = flush_secs
        self.scalar_log_interval = max(int(args_cli.scalar_log_interval), 1)
        self.csv_log_interval = max(int(args_cli.csv_log_interval), 0)
        self._last_flush = time_module.time()
        self._event_file = None
        self._csv_file = None
        self._csv_writer = None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            started_at = int(time_module.time())
            host = os.uname().nodename if hasattr(os, "uname") else "localhost"
            event_path = self.log_dir / f"events.out.tfevents.{started_at}.{host}.{os.getpid()}"
            self._event_file = event_path.open("ab")
            self._write_event(
                event_pb2.Event(
                    wall_time=time_module.time(),
                    file_version="brain.Event:2",
                )
            )
            if self.csv_log_interval > 0:
                csv_path = self.log_dir / "scalars_sampled.csv"
                needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
                self._csv_file = csv_path.open("a", newline="")
                self._csv_writer = csv.writer(self._csv_file)
                if needs_header:
                    self._csv_writer.writerow(("wall_time", "step", "tag", "value"))
                    self._csv_file.flush()

    def _write_event(self, event: event_pb2.Event) -> None:
        if self._event_file is None:
            return
        data = event.SerializeToString()
        header = struct.pack("<Q", len(data))
        self._event_file.write(header)
        self._event_file.write(struct.pack("<I", _masked_crc32c(header)))
        self._event_file.write(data)
        self._event_file.write(struct.pack("<I", _masked_crc32c(data)))

    @staticmethod
    def _as_float(scalar_value) -> float:
        if hasattr(scalar_value, "detach"):
            scalar_value = scalar_value.detach()
            if scalar_value.numel() > 1:
                scalar_value = scalar_value.mean()
            scalar_value = scalar_value.cpu().item()
        elif hasattr(scalar_value, "item"):
            scalar_value = scalar_value.item()
        return float(scalar_value)

    @staticmethod
    def _as_step(global_step) -> int:
        return 0 if global_step is None else int(global_step)

    @staticmethod
    def _should_write(step: int, interval: int) -> bool:
        return interval > 0 and step % interval == 0

    def add_scalar(self, tag: str, scalar_value, global_step=None, walltime=None):
        step = self._as_step(global_step)
        now = time_module.time() if walltime is None else walltime
        write_event = self._should_write(step, self.scalar_log_interval)
        write_csv = self._csv_writer is not None and self._should_write(step, self.csv_log_interval)
        if write_event or write_csv:
            value = self._as_float(scalar_value)
        if write_event:
            summary = summary_pb2.Summary(
                value=[
                    summary_pb2.Summary.Value(
                        tag=tag,
                        simple_value=value,
                    )
                ]
            )
            self._write_event(
                event_pb2.Event(wall_time=now, step=step, summary=summary)
            )
        if write_csv:
            self._csv_writer.writerow((now, step, tag, value))
        if now - self._last_flush >= self.flush_secs:
            self.flush()

    def flush(self):
        if self._event_file is not None:
            self._event_file.flush()
        if self._csv_file is not None:
            self._csv_file.flush()
        self._last_flush = time_module.time()

    def close(self):
        self.flush()
        if self._event_file is not None:
            self._event_file.close()
            self._event_file = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None


tensorboard_module = types.ModuleType("torch.utils.tensorboard")
tensorboard_module.SummaryWriter = LightweightSummaryWriter
sys.modules["torch.utils.tensorboard"] = tensorboard_module


class IterationRecordVideo(gym.wrappers.RecordVideo):
    """Name step-triggered videos using PPO learning iterations."""

    def __init__(self, *args, steps_per_iteration: int, **kwargs):
        self.steps_per_iteration = max(int(steps_per_iteration), 1)
        super().__init__(*args, **kwargs)

    def start_recording(self, video_name: str):
        if "-step-" in video_name:
            iteration = max(self.step_id, 0) // self.steps_per_iteration
            video_name = f"{self.name_prefix}-{iteration}"
        super().start_recording(video_name)


class TrussV3OnPolicyRunner(OnPolicyRunner):
    """RSL-RL runner that includes V3 curriculum state in checkpoints."""

    def save(self, path: str, infos: dict | None = None) -> None:
        infos = dict(infos or {})
        infos["truss_v3_curriculum"] = self.env.unwrapped.get_curriculum_state()
        super().save(path, infos)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict | None:
        infos = super().load(path, load_cfg, strict, map_location)
        if infos and "truss_v3_curriculum" in infos:
            self.env.unwrapped.set_curriculum_state(infos["truss_v3_curriculum"])
            self.env.reset()
        return infos


def _as_float(value) -> float:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().reshape(-1).sum().cpu().item())
    return float(value)


def _patch_weighted_curriculum_logger(runner) -> None:
    """Make train-time per-level curriculum success scalars episode-weighted."""
    logger = getattr(runner, "logger", None)
    if logger is None or getattr(logger, "_truss_v3_weighted_curriculum_patch", False):
        return
    original_log = logger.log

    def _order_curriculum_level_metrics(ep_info: dict, max_level: int) -> None:
        level_keys = {
            f"Curriculum/level_{level}_{suffix}"
            for level in range(max_level + 1)
            for suffix in ("num_envs", "episodes", "success_count", "success_rate")
        }
        ordered = {key: value for key, value in ep_info.items() if key not in level_keys}
        for level in range(max_level + 1):
            for suffix in ("num_envs", "episodes", "success_count", "success_rate"):
                key = f"Curriculum/level_{level}_{suffix}"
                if key in ep_info:
                    ordered[key] = ep_info[key]
        ep_info.clear()
        ep_info.update(ordered)

    def log_with_weighted_curriculum(*args, **kwargs):
        ep_extras = getattr(logger, "ep_extras", None)
        env = getattr(runner, "env", None)
        unwrapped = getattr(env, "unwrapped", None)
        cfg = getattr(unwrapped, "cfg", None)
        max_level = int(getattr(cfg, "curriculum_max_level", -1))
        if ep_extras and max_level >= 0:
            level_num_envs = None
            curriculum_level = getattr(unwrapped, "_curriculum_level", None)
            if torch.is_tensor(curriculum_level):
                level_num_envs = torch.bincount(
                    curriculum_level.detach().to(device="cpu", dtype=torch.long).clamp(
                        min=0, max=max_level
                    ),
                    minlength=max_level + 1,
                ).to(dtype=torch.float32)
            for level in range(max_level + 1):
                num_envs_key = f"Curriculum/level_{level}_num_envs"
                episode_key = f"Curriculum/level_{level}_episodes"
                success_count_key = f"Curriculum/level_{level}_success_count"
                success_rate_key = f"Curriculum/level_{level}_success_rate"
                total_episodes = 0.0
                total_success = 0.0
                for ep_info in ep_extras:
                    count = _as_float(ep_info.get(episode_key, 0.0))
                    total_episodes += count
                    if success_count_key in ep_info:
                        total_success += _as_float(ep_info[success_count_key])
                    elif success_rate_key in ep_info:
                        total_success += _as_float(ep_info[success_rate_key]) * count
                for ep_info in ep_extras:
                    if level_num_envs is not None:
                        ep_info[num_envs_key] = float(level_num_envs[level].item())
                    ep_info[episode_key] = total_episodes
                    ep_info[success_count_key] = total_success
                    if total_episodes > 0.0:
                        ep_info[success_rate_key] = total_success / total_episodes
                    else:
                        ep_info.pop(success_rate_key, None)
            for ep_info in ep_extras:
                _order_curriculum_level_metrics(ep_info, max_level)
        return original_log(*args, **kwargs)

    logger.log = log_with_weighted_curriculum
    logger._truss_v3_weighted_curriculum_patch = True


def initialize_actor_standing_bias(runner: OnPolicyRunner) -> None:
    """Center a fresh policy on the robot's default standing joint targets."""
    actor = runner.alg.actor
    output_layer = next(
        module
        for module in reversed(actor.mlp)
        if isinstance(module, torch.nn.Linear)
    )
    nominal_action = runner.env.unwrapped.nominal_action.to(
        device=output_layer.bias.device,
        dtype=output_layer.bias.dtype,
    )
    if output_layer.bias.numel() != nominal_action.numel():
        raise RuntimeError(
            f"Actor output size {output_layer.bias.numel()} does not match "
            f"nominal action size {nominal_action.numel()}"
        )
    with torch.no_grad():
        output_layer.bias.copy_(nominal_action)


def load_pretrained_actor(runner: OnPolicyRunner, checkpoint_path: str) -> None:
    """Load policy and observation-normalizer weights without optimizer/curriculum state."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained actor checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=runner.device, weights_only=False)
    actor_state = checkpoint.get("actor_state_dict")
    if actor_state is None:
        raise KeyError(f"Checkpoint has no actor_state_dict: {path}")
    try:
        runner.alg.actor.load_state_dict(actor_state, strict=True)
        adapted_keys: list[str] = []
        inserted_gripper_obs_adapter = False
        reset_privileged_normalizer = False
    except RuntimeError as exc:
        target_state = runner.alg.actor.state_dict()
        env_cfg = runner.env.unwrapped.cfg
        deployable_dim = int(env_cfg.deployable_observation_dim)
        copy_obs_dim = int(getattr(env_cfg, "pretrained_actor_copy_obs_dim", None) or deployable_dim)
        inserted_gripper_obs_dim = 4 if bool(getattr(env_cfg, "include_gripper_position_obs", False)) else 0
        source_deployable_without_gripper_dim = deployable_dim - inserted_gripper_obs_dim
        adapted_state = {}
        adapted_keys = []
        skipped_keys = []
        inserted_gripper_obs_keys = []
        generic_input_adapter_used = False

        def _make_default_like(tensor: torch.Tensor, key: str) -> torch.Tensor:
            if "_var" in key or "_std" in key:
                return torch.ones_like(tensor)
            return torch.zeros_like(tensor)

        def _try_insert_gripper_obs(source: torch.Tensor, target: torch.Tensor, key: str):
            if inserted_gripper_obs_dim <= 0:
                return None
            if source.ndim == 2 and target.ndim == 2 and source.shape[0] == target.shape[0]:
                source_obs_dim = int(source.shape[1])
                target_obs_dim = int(target.shape[1])
                source_privileged_dim = source_obs_dim - source_deployable_without_gripper_dim
                target_privileged_dim = target_obs_dim - deployable_dim
                if (
                    source_deployable_without_gripper_dim > 0
                    and source_privileged_dim > 0
                    and source_privileged_dim == target_privileged_dim
                    and target_obs_dim == source_obs_dim + inserted_gripper_obs_dim
                ):
                    adapted = _make_default_like(target, key)
                    adapted[:, :source_deployable_without_gripper_dim] = source[
                        :, :source_deployable_without_gripper_dim
                    ]
                    adapted[:, deployable_dim:] = source[
                        :, source_deployable_without_gripper_dim:
                    ]
                    return adapted
            if source.ndim == 1 and target.ndim == 1:
                source_obs_dim = int(source.shape[0])
                target_obs_dim = int(target.shape[0])
                source_privileged_dim = source_obs_dim - source_deployable_without_gripper_dim
                target_privileged_dim = target_obs_dim - deployable_dim
                if (
                    source_deployable_without_gripper_dim > 0
                    and source_privileged_dim > 0
                    and source_privileged_dim == target_privileged_dim
                    and target_obs_dim == source_obs_dim + inserted_gripper_obs_dim
                ):
                    adapted = _make_default_like(target, key)
                    adapted[:source_deployable_without_gripper_dim] = source[
                        :source_deployable_without_gripper_dim
                    ]
                    adapted[deployable_dim:] = source[source_deployable_without_gripper_dim:]
                    return adapted
            return None

        for key, source in actor_state.items():
            target = target_state.get(key)
            if target is None:
                skipped_keys.append(key)
                continue
            if source.shape == target.shape:
                adapted_state[key] = source
                continue
            inserted_adapted = _try_insert_gripper_obs(source, target, key)
            if inserted_adapted is not None:
                adapted_state[key] = inserted_adapted
                adapted_keys.append(key)
                inserted_gripper_obs_keys.append(key)
                continue
            if (
                source.ndim == 2
                and target.ndim == 2
                and source.shape[0] == target.shape[0]
                and min(source.shape[1], target.shape[1]) >= copy_obs_dim
            ):
                adapted = torch.zeros_like(target)
                adapted[:, :copy_obs_dim] = source[:, :copy_obs_dim]
                adapted_state[key] = adapted
                adapted_keys.append(key)
                generic_input_adapter_used = True
                continue
            if (
                source.ndim == 1
                and target.ndim == 1
                and min(source.shape[0], target.shape[0]) >= copy_obs_dim
            ):
                adapted = _make_default_like(target, key)
                adapted[:copy_obs_dim] = source[:copy_obs_dim]
                adapted_state[key] = adapted
                adapted_keys.append(key)
                generic_input_adapter_used = True
                continue
            skipped_keys.append(key)
        if not adapted_keys:
            raise RuntimeError(
                f"Could not adapt actor checkpoint {path} to current actor input shape"
            ) from exc
        result = runner.alg.actor.load_state_dict(adapted_state, strict=False)
        inserted_gripper_obs_adapter = bool(inserted_gripper_obs_keys)
        reset_privileged_normalizer = generic_input_adapter_used and not inserted_gripper_obs_adapter
        if inserted_gripper_obs_adapter:
            print(
                "[INFO]: Adapted pretrained actor by inserting gripper-angle obs: "
                f"{inserted_gripper_obs_keys}; mapped source 0:{source_deployable_without_gripper_dim} "
                f"-> target 0:{source_deployable_without_gripper_dim}, "
                f"source {source_deployable_without_gripper_dim}: -> target {deployable_dim}:"
            )
        if generic_input_adapter_used:
            print(
                "[INFO]: Adapted pretrained actor input weights for deployable interface: "
                f"{adapted_keys}; copied first {copy_obs_dim} obs dims"
            )
        if result.missing_keys:
            print(f"[INFO]: Missing actor keys kept at initialization: {result.missing_keys}")
        if skipped_keys:
            print(f"[INFO]: Skipped incompatible actor keys: {skipped_keys}")
    normalizer = runner.alg.actor.obs_normalizer
    deployable_dim = runner.env.unwrapped.cfg.deployable_observation_dim
    if (
        "reset_privileged_normalizer" in locals()
        and reset_privileged_normalizer
        and hasattr(normalizer, "_mean")
        and normalizer._mean.shape[-1] > deployable_dim
    ):
        with torch.no_grad():
            normalizer._mean[..., deployable_dim:] = 0.0
            normalizer._var[..., deployable_dim:] = 1.0
            normalizer._std[..., deployable_dim:] = 1.0
            # Keep a small effective count so crossing-specific privileged
            # statistics adapt quickly without discarding proprio statistics.
            normalizer.count.fill_(runner.env.num_envs)
    elif (
        "inserted_gripper_obs_adapter" in locals()
        and inserted_gripper_obs_adapter
        and hasattr(normalizer, "count")
    ):
        with torch.no_grad():
            # The inserted 4D gripper-angle statistics are new, while the copied
            # privileged statistics remain semantically aligned. Keep adaptation
            # fast without zeroing the teacher's privileged normalizer.
            normalizer.count.fill_(runner.env.num_envs)
    print(f"[INFO]: Loaded pretrained actor only from: {path}")


def compute_auto_save_interval(max_iterations: int) -> int:
    """Return the shared checkpoint/video interval in PPO learning iterations."""
    if max_iterations < LONG_RUN_MIN_ITERATIONS:
        return SHORT_RUN_SAVE_INTERVAL
    return max(int(round(max_iterations * LONG_RUN_SAVE_FRACTION)), 1)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    if agent_cfg.max_iterations <= 0:
        raise ValueError(f"--max_iterations must be positive, got {agent_cfg.max_iterations}")
    auto_save_interval = compute_auto_save_interval(agent_cfg.max_iterations)
    save_interval = args_cli.save_interval if args_cli.save_interval is not None else auto_save_interval
    if save_interval <= 0:
        raise ValueError(f"--save_interval must be positive, got {save_interval}")
    video_iteration_interval = (
        args_cli.video_interval if args_cli.video_interval is not None else save_interval
    )
    if video_iteration_interval <= 0:
        raise ValueError(f"--video_interval must be positive, got {video_iteration_interval}")
    if args_cli.scalar_log_interval <= 0:
        raise ValueError(f"--scalar_log_interval must be positive, got {args_cli.scalar_log_interval}")
    if args_cli.csv_log_interval < 0:
        raise ValueError(f"--csv_log_interval must be non-negative, got {args_cli.csv_log_interval}")
    agent_cfg.save_interval = save_interval
    print(
        "[INFO] Saving checkpoints/videos every "
        f"{save_interval}/{video_iteration_interval} PPO iterations "
        f"(max_iterations={agent_cfg.max_iterations}, auto_interval={auto_save_interval})."
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    if args_cli.depth_bar_estimator_checkpoint is not None:
        if not hasattr(env_cfg, "depth_bar_estimator_checkpoint"):
            raise ValueError(
                "--depth_bar_estimator_checkpoint is only valid for environments with depth bar support"
            )
        env_cfg.depth_bar_estimator_checkpoint = args_cli.depth_bar_estimator_checkpoint
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join(DEFAULT_LOG_ROOT, agent_cfg.experiment_name)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        teacher_checkpoint_path = args_cli.teacher_checkpoint or getattr(
            agent_cfg, "teacher_checkpoint_path", None
        )
        if (
            agent_cfg.algorithm.class_name == "Distillation"
            and agent_cfg.resume
            and args_cli.teacher_checkpoint
        ):
            raise ValueError("--resume and --teacher_checkpoint cannot be used together")
        if (
            agent_cfg.algorithm.class_name == "Distillation"
            and not agent_cfg.resume
            and teacher_checkpoint_path
        ):
            resume_path = str(Path(teacher_checkpoint_path).expanduser().resolve())
            if not Path(resume_path).is_file():
                raise FileNotFoundError(f"Teacher checkpoint does not exist: {resume_path}")
        elif agent_cfg.resume and args_cli.checkpoint:
            checkpoint_path = Path(args_cli.checkpoint).expanduser()
            if checkpoint_path.is_file():
                resume_path = str(checkpoint_path.resolve())
            else:
                resume_path = get_checkpoint_path(
                    log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
                )
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_step_interval = video_iteration_interval * agent_cfg.num_steps_per_env
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % video_step_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = IterationRecordVideo(
            env,
            steps_per_iteration=agent_cfg.num_steps_per_env,
            **video_kwargs,
        )

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = TrussV3OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
        )
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    _patch_weighted_curriculum_logger(runner)
    if not agent_cfg.resume and agent_cfg.class_name == "OnPolicyRunner":
        initialize_actor_standing_bias(runner)
        if args_cli.pretrained_actor:
            load_pretrained_actor(runner, args_cli.pretrained_actor)
    elif agent_cfg.resume and args_cli.pretrained_actor:
        raise ValueError("--resume and --pretrained_actor cannot be used together")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
