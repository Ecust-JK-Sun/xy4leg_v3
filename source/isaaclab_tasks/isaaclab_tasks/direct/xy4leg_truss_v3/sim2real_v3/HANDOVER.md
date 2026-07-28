# Sim2Real V3 Handover

本文档面向另一台实机部署电脑。当前仓库侧主要负责仿真训练；实机侧需要按这里的接口构造
policy observation，并对夹爪 action 做低层后处理。

## 当前训练任务

任务名：

```text
Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Walk-Sim2Real-Direct-v0
```

训练日志目录：

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_walk_sim2real
```

本次要拷贝到实机电脑的训练 run：

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_walk_sim2real/2026-07-24_18-58-21_deployable_interface
```

推荐先部署验证的 checkpoint：

```text
model_28800.pt
```

备选 checkpoint：

```text
model_29400.pt
model_29999.pt
```

不建议优先用 `model_24600.pt`，该保存点附近日志里出现过一次明显 reset 异常。

初始化 checkpoint：

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_walk/2026-07-20_11-59-48_learn_gripper/model_49999.pt
```

这版 actor 不再使用旧 88D `policy` privileged observation，而是使用：

```text
actor obs = deployable + bar_estimate = 76D + 4D = 80D
action    = 24D
```

实机电脑不仅需要拷贝日志目录，还需要使用包含当前 V3 task 注册和 `sim2real_v3`
脚本的代码版本。最小需要确认这些文件存在并与训练侧接口一致：

```text
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/__init__.py
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/truss_v3_env.py
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/truss_v3_env_cfg.py
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/agents/rsl_rl_ppo_cfg.py
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/play_real.py
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/gripper_calibration.yaml
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/damiao.yaml
```

如果实机电脑只拷贝 log、不更新代码，`gym.make` 可能找不到
`Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Walk-Sim2Real-Direct-v0`，或者 obs 维度仍停留在旧版本。

## Actor Observation

`deployable` 顺序必须严格一致：

```text
0:20    joint_pos_norm, 20D, policy leg order RF, LF, RB, LB
20:40   joint_vel_norm, 20D, policy leg order RF, LF, RB, LB
40:43   projected_gravity_b, 3D
43:46   root_ang_vel_b / 5, 3D
46:48   gait_clock = [sin(2*pi*RF_phase), cos(2*pi*RF_phase)], 2D
48:72   previous raw action, 24D
72:76   gripper_angle_rad, 4D, policy leg order RF, LF, RB, LB
```

`bar_estimate` 在 walk-only 实机测试中使用 dummy value：

```text
[2.0, 0.0, -1.0, 1.0]
```

这对应“横杆很远且可见”。后续进入 camera/depth crossing 时，该 4D 通道替换为
depth-derived bar estimate：

```text
[bar_dx_b / 1.2, bar_dy_b / 0.3, bar_dz_b / 0.5, visible]
```

当前不加入夹爪速度，因为实机夹爪速度反馈噪声较大。

## Gripper Position Observation

训练侧新增的夹爪观测是 `gripper_angle_rad`，不是归一化开闭比例：

```text
closed_angle = -30 deg = -0.5236 rad
open_angle   = -14 deg = -0.2443 rad
```

仿真中由同一条腿两个 USD 夹爪关节的平均角度得到：

```text
gripper_angle_rad = mean(two USD gripper joint positions)
gripper_angle_rad = clamp(gripper_angle_rad, closed_angle, open_angle)
```

实机侧需要用每个夹爪电机自己的标定值映射到同样的仿真角度范围。当前 V2 遗留标定值为：

```text
hardware order: LB, LF, RB, RF
travel/open position values: [-6.96, -7.38, -10.15, -7.0]
```

如果沿用 V2 公式，可先按：

```text
opening_alpha = clip(real_gripper_pos / travel_pos, 0, 1)
gripper_angle_rad = closed_angle + opening_alpha * (open_angle - closed_angle)
```

再重排到 policy leg order：

```text
RF, LF, RB, LB
```

上线前建议重新标定四个夹爪：

```text
closed_pos: 夹爪完全闭合时的电机反馈
open_pos:   夹爪完全张开时的电机反馈
opening_alpha = clip((pos - closed_pos) / (open_pos - closed_pos), 0, 1)
gripper_angle_rad = closed_angle + opening_alpha * (open_angle - closed_angle)
```

## Action Interface

policy 输出仍保持 24D 连续 action：

```text
0:20    leg joint normalized position action
20:24   gripper action, policy leg order RF, LF, RB, LB
```

训练侧没有把 policy 输出改成离散动作。夹爪执行前只做后处理：

```text
continuous action -> deadband -> sign {-1, +1} -> latch/min-hold -> torque command
```

仿真训练当前参数：

```text
gripper_action_latch_deadband:    0.10
gripper_action_latch_min_hold_s:  0.8 s
gripper_action_step:              0.4 deg / policy step
```

实机侧建议保持同样语义：

```text
if action > +0.10: desired = +1
if action < -0.10: desired = -1
else:              keep previous desired

only allow desired to change when previous command has been held >= 0.8 s
```

当前仓库 `play_real.py` 已提供参考参数：

```text
--gripper_binary_latch / --no-gripper_binary_latch
--gripper_latch_deadband 0.10
--gripper_latch_min_hold_s 0.8
--gripper_latch_initial_action -1.0
```

V3 仿真语义：

```text
+1 = open direction
-1 = close direction
```

但 DaMiao 电机 torque sign 取决于实机接线和夹爪结构。当前 `gripper_calibration.yaml`
默认：

```text
torque_sign = -1.0 for LB/LF/RB/RF
```

实机侧如果发现某条腿开合方向反了，只改对应腿的 `torque_sign`，不要改 policy。

## Timing

仿真 policy step：

```text
sim.dt = 0.02
decimation = 2
policy_dt = 0.04 s
policy_hz = 25 Hz
```

Walk-Sim2Real 训练已下调：

```text
gait_frequency:       0.35 Hz
forward velocity:     0.025--0.06 m/s
forward speed limit:  0.10 m/s
```

目的是降低步态频率和夹爪开合频率，匹配真实夹爪 1--2 s 的完整开合时间。

## Required Real-Side Checks

1. 确认 actor obs 实际拼接为 80D。
2. 确认 `gripper_angle_rad` 顺序是 `RF, LF, RB, LB`。
3. 确认不把夹爪速度放进 observation。
4. 确认后 4 维 action 不直接连续映射为高频变化力矩，而是经过 sign+latch。
5. 确认每条腿夹爪 `torque_sign` 单独可调。
6. dry run 时打印：

```text
obs_dim
gripper_angle_rad
raw_gripper_action
latched_gripper_action
gripper_torque_command
```

## Minimal Real-Play Command

推荐先在实机电脑 dry run，不开电机：

```bash
cd /home/lab202/IsaacLab

python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/play_real.py \
  --task Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Walk-Sim2Real-Direct-v0 \
  --checkpoint logs/rsl_rl/xy4leg_gripper_truss_v3_walk_sim2real/2026-07-24_18-58-21_deployable_interface/model_28800.pt \
  --headless \
  --max_steps 200 \
  --print_interval 25
```

期望启动时打印：

```text
Actor obs groups: ['deployable', 'bar_estimate'] -> real obs mode: deployable_bar
deployable_observation_dim: 76
expected actor obs dim: 80
```

循环打印中应能看到：

```text
grip_angle=[...]
grip_raw=[...]
grip_latch=[...]
grip_cmd=[...]
```

dry run 不加 `--enable_motors` 时不会读取真实电机反馈，`grip_angle` 会使用默认闭合角
`[-0.5236, -0.5236, -0.5236, -0.5236]`。它只用于检查 checkpoint、task 注册、
obs/action 维度和策略前向是否能跑通。

带雷达但不开电机的 dry run：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/play_real.py \
  --task Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Walk-Sim2Real-Direct-v0 \
  --checkpoint logs/rsl_rl/xy4leg_gripper_truss_v3_walk_sim2real/2026-07-24_18-58-21_deployable_interface/model_28800.pt \
  --headless \
  --enable_radar \
  --max_steps 200 \
  --print_interval 25
```

确认 dry run 正常后，再进行悬空或低负载电机测试：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/play_real.py \
  --task Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Walk-Sim2Real-Direct-v0 \
  --checkpoint logs/rsl_rl/xy4leg_gripper_truss_v3_walk_sim2real/2026-07-24_18-58-21_deployable_interface/model_28800.pt \
  --headless \
  --enable_radar \
  --enable_motors \
  --max_steps 300 \
  --gripper_torque_scale 0.5 \
  --print_interval 25
```

正式地面短测再去掉或放大 `--max_steps`，并逐步把 `--gripper_torque_scale`
从 `0.5` 调回 `1.0`。

## If Modifying Another play_real.py

如果实机电脑不用本目录的 `play_real.py`，而是在已有脚本上改，必须实现以下最小接口：

1. 用 RSL-RL `OnPolicyRunner` 加载 checkpoint，并按 agent yaml 的 actor obs groups 判断：

```text
actor obs groups must be ['deployable', 'bar_estimate']
```

2. 给 policy 的 observation 必须是 dict：

```python
obs = {
    "deployable": deployable_tensor_1x76,
    "bar_estimate": bar_estimate_tensor_1x4,
}
```

不能把 80D 拼成旧的 `"policy"` key，也不能再追加 16D privileged 常量。

3. `deployable` 必须严格按本文档的 76D 顺序拼接；最后 4D 是 `gripper_angle_rad`，
不是 `0..1` 开闭比例，也不是夹爪速度。

4. walk-only 阶段 `bar_estimate` 固定为：

```text
[2.0, 0.0, -1.0, 1.0]
```

5. policy 输出后 4D 仍是连续值，只在发送给夹爪电机前做：

```text
deadband 0.10 -> sign {-1, +1} -> min_hold 0.8 s -> torque_sign/scale
```

6. `previous raw action` 必须使用 policy 原始输出 `action`，不要用 smoothing 或 latch 后的
`command_action` 回填。当前训练观测里历史动作对应的是策略输出，不是低层后处理后的电机命令。

## Validation Order

建议实机侧按以下顺序验证：

```text
1. no motors dry run: 检查 obs/action 维度和数值范围
2. motors enabled, robot unloaded/suspended: 检查夹爪 sign、latch 和腿部关节方向
3. low torque gripper test: gripper_torque_scale 0.3--0.5
4. ground walk short run: max_steps 200--500
5. longer walk run: 20 s，确认 base 晃动、足端冲击、夹爪时机
```

当前阶段只验证 walk sim2real。camera/depth crossing sim2real 需要等
`deployable + gripper position + depth bar_estimate` 的 crossing student 重新训练完成后再做。
