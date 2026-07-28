# Selected Checkpoints

Only a small set of useful checkpoints is included. Full training logs,
videos, datasets, and intermediate checkpoints are intentionally excluded.

## Walk Sim2Real

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_walk_sim2real/2026-07-24_18-58-21_deployable_interface/model_28800.pt
```

Recommended deployable-interface walk policy. This was used for real walk
validation and as the sim2real interface reference.

## Gripper Crossing Teacher

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_teacher/2026-07-21_22-36-26_learn_gripper/model_36000.pt
```

Learned-gripper privileged crossing teacher selected from play/eval because it
was more conservative than later checkpoints.

## Crossing Teacher Sim2Real

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_teacher_sim2real/2026-07-26_21-09-53_sim2real_ft_from_teacher36000/model_20000.pt
```

Privileged teacher sim2real fine-tuned from the learned-gripper teacher. This
is the selected teacher for oracle-bar sim2real distillation.

## OracleBar Student Sim2Real

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_student_oracle_bar_sim2real/2026-07-27_14-17-41_from_teacher_sim2real20000/model_13500.pt
```

Selected oracle-bar sim2real student. This is the preferred resume checkpoint
for DepthBar sim2real training.

## DepthBar Student Sim2Real

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar_sim2real/2026-07-27_19-37-56_depth_from_oracle_sim2real13500/model_24000.pt
```

Current in-progress DepthBar sim2real checkpoint. It is included mainly for
continuation and comparison.

## Depth Bar Estimator

```text
logs/rsl_rl/xy4leg_truss_v3_depth_bar/depth_bar_estimator_l2_055-070_l3_060-080_l4_065-090.pt
```

TorchScript heightmap-to-bar estimator used by DepthBar tasks.
