# Included Checkpoints

完整训练和实机结论见 [项目总结](source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/README.md)。本文件仅维护仓库内模型清单；完整训练日志、数据集和中间 checkpoint 不纳入版本控制。

## 2026-11-09 默认部署模型

目录：`logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar_sim2real/`。
策略为 RealCam DepthBar `68500`，80D actor、24D action、25 Hz；模型和 params 必须配套使用，旧 heightmap estimator 或测试 fixture 均不能替代。

| 文件 | bytes | SHA256 |
| --- | ---: | --- |
| [model_68500.pt](logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar_sim2real/model_68500.pt) | 3383661 | `6a2406a3a6770dd90a326fc15aac5b50ff1a6bb56c6fa2cf81b50b02c10acfc6` |
| [depth_crop_estimator_realcam_xmax060.pt](logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar_sim2real/depth_crop_estimator_realcam_xmax060.pt) | 289926 | `3d8580f29511139418373fae63e5911edd29de70add9f63af1f063c9a282aadb` |
| [params/env.yaml](logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar_sim2real/params/env.yaml) | 22416 | `c4c42dc87d0fa865d34f6d686f951324175f1684172930e3149ca10f0553b0a4` |
| [params/agent.yaml](logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar_sim2real/params/agent.yaml) | 1251 | `efd9f47b7285bae0c94f8e6d5fca25b656332794a85e3279afb70f13989eeaa2` |

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
