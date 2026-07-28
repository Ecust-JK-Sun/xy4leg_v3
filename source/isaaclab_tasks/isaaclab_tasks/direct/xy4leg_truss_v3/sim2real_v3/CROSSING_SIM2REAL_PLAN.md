# Crossing Sim2Real Plan

本文档记录 V3 learned-gripper crossing 的 sim2real 路线。当前 walk sim2real
已经验证可用，后续 crossing 不直接沿用旧的 76D DepthBar student，而是统一到 walk
sim2real 已验证过的 deployable interface。

## Why Not Directly Deploy Old DepthBar

旧 gripper DepthBar crossing student 的 actor 输入是：

```text
72D deployable + 4D bar_estimate = 76D
```

它没有 `gripper_angle_rad`，也没有用 walk sim2real 中验证过的 gripper latch、慢步频和
sim2real DR 接口训练。当前 walk sim2real 的 actor 输入是：

```text
76D deployable + 4D bar_estimate = 80D
```

其中 deployable 最后 4D 是：

```text
gripper_angle_rad, policy leg order RF, LF, RB, LB
```

为了降低实机风险，crossing sim2real 后续全部使用 80D actor 输入。

## New Task Route

### 1. Crossing Teacher Sim2Real

任务：

```text
Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Teacher-Sim2Real-Direct-v0
```

日志目录：

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_teacher_sim2real
```

作用：

```text
PPO crossing policy
actor obs = policy = 76D deployable + 16D privileged = 92D
critic obs = critic
action = 24D
```

初始化 checkpoint：

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_teacher/2026-07-21_22-36-26_learn_gripper/model_36000.pt
```

这一步的目标不是追最高速度，而是得到保守、低冲击、夹爪时机稳定的 crossing policy。
这一步不是最终可部署 actor；它仍然使用仿真 privileged 信息来降低 crossing PPO 探索难度。
最终部署接口在 Stage 2/3 的 student 中固定为 80D。

曾尝试从 walk-sim2real `model_28800.pt` 初始化 privileged teacher，但该路线到 33.5k
仍卡在 level0，`hind_complete_rate=0` 且 `no_progress` 很高。当前推荐从已经学会跨越的
learned-gripper teacher `model_36000.pt` 做 sim2real finetune，再加入 walk-sim2real 中
验证过的 gripper angle obs、慢步频、DR、action delay 和 gripper latch。

### 2. OracleBar Student Sim2Real

任务：

```text
Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Student-OracleBar-Sim2Real-Direct-v0
```

日志目录：

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_student_oracle_bar_sim2real
```

作用：

```text
student obs = deployable + oracle bar_estimate = 80D
teacher obs = policy = 92D privileged
action = 24D
```

训练时通过 `--teacher_checkpoint` 指向阶段 1 的 best teacher checkpoint。

### 3. DepthBar Student Sim2Real

任务：

```text
Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Student-DepthBar-Sim2Real-Direct-v0
```

日志目录：

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar_sim2real
```

作用：

```text
front/rear depth cameras
-> base-frame fused heightmap
-> depth bar estimator
-> 4D bar_estimate
-> 80D deployable actor
```

这一步从阶段 2 的 oracle-bar student checkpoint `--resume`，保持同一个
80D deployable+bar actor 接口，只把环境中的 `bar_estimate` 来源从 oracle
切换为 depth-derived estimator。checkpoint 中同时包含 student 和 frozen teacher
state，因此不再额外传 `--teacher_checkpoint`。

## Shared Sim2Real Parameters

crossing sim2real 继承 walk sim2real 的硬件相关设置：

```text
include_gripper_position_obs = true
gripper observation = gripper_angle_rad
gripper latch min hold = 0.8 s
gripper latch deadband = 0.10
gripper action step = 0.4 deg / policy step
gait_frequency = 0.35 Hz
action delay = 0--3 policy steps
joint pos bias/noise = +/-0.02 rad / 0.015 rad
joint velocity noise = 0.08 rad/s
friction/mass/actuator DR enabled
```

crossing level ranges use the latest near/balanced ranges:

```text
level0: 0.00--0.15
level1: 0.15--0.30
level2: 0.55--0.65
level3: 0.60--0.70
level4: 0.65--0.75
```

Teacher Sim2Real uses normal curriculum promotion/demotion. OracleBar and DepthBar Sim2Real
students use balanced sampling:

```text
level weights: 0.05, 0.05, 0.30, 0.30, 0.30
```

## Training Commands

### Stage 1: Teacher Sim2Real

```bash
cd /home/lab202/IsaacLab

python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/train_v3.py \
  --task Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Teacher-Sim2Real-Direct-v0 \
  --pretrained_actor logs/rsl_rl/xy4leg_gripper_truss_v3_teacher/2026-07-21_22-36-26_learn_gripper/model_36000.pt \
  --run_name sim2real_ft_from_teacher36000 \
  --max_iterations 50000 \
  --num_envs 128 \
  --headless
```

Recommended checkpoints to inspect:

```text
model_5000.pt
model_10000.pt
model_20000.pt
model_49999.pt
```

### Stage 2: OracleBar Student Sim2Real

Replace `<teacher_model.pt>` with the selected Stage 1 checkpoint.

```bash
cd /home/lab202/IsaacLab

python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/train_v3.py \
  --task Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Student-OracleBar-Sim2Real-Direct-v0 \
  --teacher_checkpoint <teacher_model.pt> \
  --run_name from_teacher_sim2real \
  --max_iterations 20000 \
  --num_envs 128 \
  --headless
```

### Stage 3: DepthBar Student Sim2Real

Replace `<oracle_student_model.pt>` with the selected Stage 2 OracleBar student checkpoint.
Replace `<depth_estimator.pt>` with the estimator selected for the current camera/heightmap setup.

```bash
cd /home/lab202/IsaacLab

python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/train_v3.py \
  --task Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Student-DepthBar-Sim2Real-Direct-v0 \
  --resume \
  --checkpoint <oracle_student_model.pt> \
  --depth_bar_estimator_checkpoint <depth_estimator.pt> \
  --run_name depth_from_oracle_sim2real \
  --max_iterations 50000 \
  --num_envs 128 \
  --headless
```

This resumes from the OracleBar distillation checkpoint. The student keeps the trained
deployable actor weights, while the environment feeds depth-derived `bar_estimate` through the
same 4D bar input slot.

## Real-Side Crossing Play

Do not overload the walk-only `play_real.py` with perception code. The recommended real
deployment entry should be a separate:

```text
play_cross_real.py
```

It should reuse the walk sim2real hardware interface:

```text
motor mapping
radar/projected gravity
gripper calibration
gripper latch
previous raw action handling
```

and add one bar source:

```text
--bar_source dummy/manual/udp/depth
```

Initial real validation order:

```text
1. dummy bar, no motors: falls back to walk-like behavior
2. manual bar, no motors: verify bar_estimate scaling and action response
3. suspended robot with manual bar
4. ground, no physical bar, dummy far bar
5. near physical bar, level0/1 distance
6. level2/3/4 distances after stable near-bar behavior
```
