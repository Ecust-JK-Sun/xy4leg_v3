# Checkpoints and Current Crossing References

Only a small set of useful checkpoints is included. Full training logs,
videos, datasets, and intermediate checkpoints are intentionally excluded.

## Current crossing references（2026-10-22 审查，2026-10-26 整理）

当前模型选择只比较 crossing。下表是审查中建议保留或复评的模型引用；
训练开始时间取 run 目录名，均为北京时间，不代表 checkpoint 的精确保存时间，
也不包含前序训练耗时。完整路径、指标口径及限制见
[项目进展表](source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/PROGRESS_AND_SIM2REAL_REVIEW_2026-10-22.md)。

| 方向 | 训练 run 开始时间（北京时间） | 模型 | 用途 |
| --- | --- | --- | --- |
| 历史 heightmap DepthBar | 2026-07-22 15:49:26 | `model_30000.pt` | 前后相机 heightmap 历史基线，接口与当前 RealCam 模型不同 |
| 当前默认 RealCam DepthBar | 2026-08-05 04:55:55 | `model_68500.pt` | 当前部署基线，80D actor、24D action、25 Hz 策略周期 |
| Noisy OracleBar PPO | 2026-08-28 03:08:45 | `model_27000.pt` / `model_27600.pt` | 带噪横杆真值输入的仿真控制器参考 |
| 低频安全 Teacher | 2026-09-08 02:11:53 | `model_96000.pt` / `model_96800.pt` | 92D 特权 actor 候选，仍需复评和蒸馏 |

本次更新仅发布修复代码、测试和文档，没有新增上述 checkpoint 权重。
部署 `68500` 还需要模型配套的完整 `params/` 和 RealCam 深度 estimator；
不能使用下列旧 heightmap estimator 替代，也不能使用测试 fixture 替代真实训练参数。

## Historical checkpoints included in this repository

以下是七月已保存的历史产物，保留用于历史复现和对照。

### Walk Sim2Real（历史记录）

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_walk_sim2real/2026-07-24_18-58-21_deployable_interface/model_28800.pt
```

Historical deployable-interface walk policy. This was used for real walk
validation and as the sim2real interface reference.

### Gripper Crossing Teacher

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_teacher/2026-07-21_22-36-26_learn_gripper/model_36000.pt
```

Learned-gripper privileged crossing teacher selected from play/eval because it
was more conservative than later checkpoints.

### Crossing Teacher Sim2Real

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_teacher_sim2real/2026-07-26_21-09-53_sim2real_ft_from_teacher36000/model_20000.pt
```

Privileged teacher sim2real fine-tuned from the learned-gripper teacher. This
is the selected teacher for oracle-bar sim2real distillation.

### OracleBar Student Sim2Real

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_student_oracle_bar_sim2real/2026-07-27_14-17-41_from_teacher_sim2real20000/model_13500.pt
```

Selected oracle-bar sim2real student. This is the preferred resume checkpoint
for DepthBar sim2real training.

### DepthBar Student Sim2Real

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar_sim2real/2026-07-27_19-37-56_depth_from_oracle_sim2real13500/model_24000.pt
```

Historical DepthBar sim2real checkpoint, retained for continuation and comparison.
The current deployment reference is the RealCam `model_68500.pt` listed above.

### Depth Bar Estimator

```text
logs/rsl_rl/xy4leg_truss_v3_depth_bar/depth_bar_estimator_l2_055-070_l3_060-080_l4_065-090.pt
```

TorchScript heightmap-to-bar estimator used by DepthBar tasks.
