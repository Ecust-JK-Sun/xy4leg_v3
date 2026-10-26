# XY4Leg Truss V3

This repository is an overlay for the IsaacLab workspace. It contains the
`xy4leg_truss_v3` direct task, sim2real helper scripts, selected training
checkpoints, and the gripper actuator helper required by the V3 config.

## Layout

```text
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/
source/isaaclab_tasks/test/test_xy4leg_truss_v3_logic.py
logs/rsl_rl/... selected checkpoints
```

Copy or clone this repository at the IsaacLab root, then install/run IsaacLab
as usual.

## 本轮 Truss V3 更新（2026-10-26 整理）

本轮聚焦 crossing 的 Sim2Real 通信、推理频率和部署接口。修复代码已在
[`a5bfcd0`](https://github.com/Ecust-JK-Sun/xy4leg_v3/commit/a5bfcd078c9d6c7e7baf1707b08dc582e308866d)
发布；本次补齐完整代码报告、原始补丁和文档入口。

| 内容 | 入口 |
| --- | --- |
| 逐文件完整代码 diff：11 个运行代码文件、测试及两份测试参数 | [代码改动报告](source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/SIM2REAL_CODE_CHANGE_REPORT_2026-10-22.md) |
| 与 `e8a0a32 → a5bfcd0` 对应的代码补丁 | [原始 `.patch`](source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/SIM2REAL_CODE_CHANGES_2026-10-22.patch) |
| 通信问题、实际频率、延迟指标及测量命令 | [通信与频率检查](source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/COMMUNICATION_TIMING_REVIEW_2026-10-22.md) |
| 部署参数恢复、反馈检查和异常处理 | [修复记录](source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/SIM2REAL_FIXES_2026-10-22.md) |
| Crossing 各方向模型、训练开始时间及指标口径 | [项目进展表](source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/PROGRESS_AND_SIM2REAL_REVIEW_2026-10-22.md) |
| 当前部署入口与参数说明 | [Sim2Real README](source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/README.md) |

主要改动包括非阻塞串口反馈、逐电机有效性/年龄检查、独立相机采集、绝对周期调度，
以及实际 Hz、队列和 p95 延迟统计。当前默认 `68500` 的策略周期为 **40 ms / 25 Hz**；
反馈默认 **200 Hz 轮询**。默认保留周期发送，可选 `--tx_mode on_update`。

发布时的独立检出通过 49 项 CPU/mock 检查；本机原工作区通过 53 项。
实机链路尚未验收，本轮没有增加安全停机动作。

默认 RealCam DepthBar `model_68500.pt`、其完整 `params/` 和
`depth_crop_estimator_realcam_xmax060.pt` 需要另外准备，具体路径见部署 README。
测试 fixture 不能替代实机模型配套参数。本次整理未上传新权重或完整训练日志。

## External Asset

The robot USD is not included in git because the current file is larger than
GitHub's normal 100 MB file limit:

```text
source/isaaclab_tasks/isaaclab_tasks/direct/xy4legv2_amp/usd/xy4legv5_gripper_truss_v2.usd
```

Place that USD at the path above before running training or play commands.
If the asset should be versioned later, use Git LFS.

## Key Checkpoints

See [CHECKPOINTS.md](CHECKPOINTS.md) for the historical checkpoints already in this
repository and the current crossing model references.

## Main Documents

```text
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/TRUSS_V3_DESIGN.md
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/TRUSS_V3_HISTORY.md
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/README.md
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/HANDOVER.md
```
