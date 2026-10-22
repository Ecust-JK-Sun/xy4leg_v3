# Truss V3 Sim2Real Crossing

这个目录用于 V3 learned-gripper 策略的实机播放，当前默认是 RealCam DepthBar crossing。
2026-10-22 的部署参数恢复、反馈有效性和异常处理更新见 [修复记录](SIM2REAL_FIXES_2026-10-22.md)。
本轮重点检查的通信、25 Hz 推理调度、延迟测量和启动参数见 [通信与频率检查](COMMUNICATION_TIMING_REVIEW_2026-10-22.md)。
跨越模型及其完整训练时间见 [项目进展表](../PROGRESS_AND_SIM2REAL_REVIEW_2026-10-22.md)。

默认不发送电机命令。只有显式传入 `--enable_motors` 时才会打开串口、使能电机并发送目标。
带电 crossing 还要求 `--enable_camera --enable_radar`，并会加载模型配套 `params/`，不能只拷贝 `.pt`。
本轮未新增安全停机动作。下文 Walk 训练与命令段落保留为历史资料，当前部署以修复记录为准。

## 文件说明

```text
damiao.yaml                 DaMiao MIT 控制增益
gripper_calibration.yaml    四个真实夹爪的行程/力矩标定
motor_control.py            DaMiao 串口协议
motor_session_gripper.py    四路串口 + 24 维硬件命令封装
serial_threads.py           电机反馈读取线程和动作发送线程
threads_shared.py           线程共享最新 obs/action
runtime_timing.py           绝对周期调度、异步相机缓存和 Hz/延迟统计
radar_receiver.py           UDP 雷达/动捕数据接收
play_real.py                V3 crossing real-play 入口
HANDOVER.md                 实机电脑需要实现/核对的 sim2real 接口说明
CROSSING_SIM2REAL_PLAN.md   crossing sim2real 训练/部署路线
```

未复制旧的 `play_gripper_truss_real_thread.py`，因为它属于 V2 skrl/AMP 体系，不适用于 V3
RSL-RL checkpoint。

## 默认策略

当前默认 checkpoint：

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar_sim2real/2026-08-05_04-55-55_realcam_crop_noise_l2far_render12hz_from42000/model_68500.pt
```

对应任务：

```text
Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Student-DepthBar-Sim2Real-Direct-v0
```

该策略是 24 维 action：

```text
20D leg joint normalized position action + 4D learned gripper action
```

V3 仿真中腿部 action 解码为：

```text
target = action_offset + action_scale * action
action_scale = 0.5 * (soft_upper - soft_lower)
```

注意这里的 `action_scale` 是半宽，不是 V2 旧 play 脚本里的全宽。

## 硬件映射

V3 policy leg order：

```text
RF, LF, RB, LB
```

硬件 buffer order：

```text
LB, LF, RB, RF
```

硬件 24 维命令是 joint-major interleaved：

```text
[LB1, LF1, RB1, RF1, LB2, LF2, RB2, RF2, ..., LB6, LF6, RB6, RF6]
```

其中 Joint1-5 使用位置 MIT 控制，Joint6 是每条腿的夹爪电机，使用力矩控制。

夹爪 action 第一版按符号转力矩，并从 `gripper_calibration.yaml` 读取每个夹爪的
`torque_sign / torque_scale / deadband / max_torque`：

```text
torque = policy_gripper_action * torque_sign * torque_scale * gripper_torque_scale
```

默认 `torque_sign=-1.0`，`gripper_torque_scale=1.0`。如果某条腿夹爪开合方向反了，优先调整
`gripper_calibration.yaml` 中对应腿的 `torque_sign`，不要改 policy checkpoint。

`close_pos` 保留了 V2 的四个实机夹爪行程标定值：

```text
LB=-6.96, LF=-7.38, RB=-10.15, RF=-7.0
```

旧 88D walk play 不使用夹爪位置。新的 Walk-Sim2Real 80D actor 会使用每条腿一个
`gripper_angle_rad`，当前 `play_real.py` 已按 `close_pos` 近似映射到仿真角度范围；
实机上线前建议按 `HANDOVER.md` 重新标定 `closed_pos/open_pos`。

## Observation

当前推荐的 Walk-Sim2Real checkpoint 使用 80 维 actor observation：

```text
20 joint_pos_norm
20 joint_vel_norm
3  projected_gravity
3  root_ang_vel_b / 5
2  gait_clock
24 previous raw action
4  gripper_angle_rad
4  dummy bar_estimate
```

对应 RSL-RL actor obs groups：

```text
['deployable', 'bar_estimate']
```

`play_real.py` 会根据 agent yaml 自动进入 `deployable_bar` 模式，给 policy 传入：

```python
{"deployable": tensor_1x76, "bar_estimate": tensor_1x4}
```

旧 88D `policy` observation 仅作为兼容旧 checkpoint 的 fallback，不作为当前部署推荐路线。

## Recommended Route

最终目标不是单独让 walk 任务部署，而是让相机/depth bar student 能部署。因此推荐路线是：

```text
1. Deployable-interface Walk-Sim2Real
   proprioception + previous action + gait clock + gripper position + dummy bar_estimate

2. Privileged Crossing Teacher
   用更稳的 walk actor 初始化，actor 使用 privileged policy obs 学习跨越

3. Oracle-Bar Student
   student 输入固定为 deployable + oracle bar_estimate，通过 distillation 验证可部署接口能闭环跨越

4. Depth-Bar Student
   从 oracle-bar student checkpoint resume，把 oracle bar_estimate 换成 depth/heightmap estimator 输出

5. Sim2Real Robustness
   逐步加入 bar estimate 噪声、延迟、漏检、相机 DR 和更真实的电机/传感器误差
```

关键原则：

```text
walk 只作为实机稳定步态先验和硬件链路验证；
最终 policy 的 actor 输入应尽早固定为 deployable + bar_estimate；
不要长期依赖 88D policy obs 中的 privileged 常量。
```

crossing sim2real 的具体训练阶段、任务名和命令记录在：

```text
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/CROSSING_SIM2REAL_PLAN.md
```

## Deployable-Interface Walk-Sim2Real

新增任务：

```text
Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Walk-Sim2Real-Direct-v0
```

日志目录：

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_walk_sim2real
```

actor observation：

```text
76D deployable proprioception/action history/gripper position
4D dummy bar_estimate = [far_dx, 0, nominal_dz, visible]
```

76D deployable 的顺序为：

```text
20 joint_pos_norm
20 joint_vel_norm
3  projected_gravity
3  root_ang_vel_b / 5
2  gait_clock
24 previous raw action
4  gripper_angle_rad
```

critic observation 仍使用内部 `critic`，用于保持训练稳定；部署时只导出 actor。

该任务使用更偏实机的 DR 和稳定性 reward：

```text
friction static/dynamic:      0.6--1.4 / 0.5--1.3
base mass add:               -0.8--0.8 kg
actuator stiffness/damping:   0.75--1.25 / 0.7--1.3
action delay:                0--3 policy steps
joint pos bias/noise:         +/-0.02 rad / 0.015 rad
joint velocity noise:         0.08 rad/s
gait_frequency:               0.35 Hz
target forward speed:         0.025--0.06 m/s
gripper target speed:         0.4 deg/policy step
gripper action latch:         0.8 s min hold, 0.10 deadband
```

从旧 learned-gripper walk checkpoint 初始化时，`train_v3.py --pretrained_actor` 会做
partial actor loading：

```text
旧 88D actor: 72D deployable + 16D privileged
新 80D actor: 76D deployable + 4D bar_estimate
```

加载时保留旧 checkpoint 的前 72D deployable 权重，丢弃旧 privileged 输入权重；新增
4D `gripper_angle_rad` 和 4D `bar_estimate` 输入权重从 0 初始化。

启动命令：

```bash
cd /home/lab202/IsaacLab

python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/train_v3.py \
  --task Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Walk-Sim2Real-Direct-v0 \
  --pretrained_actor logs/rsl_rl/xy4leg_gripper_truss_v3_walk/2026-07-20_11-59-48_learn_gripper/model_49999.pt \
  --num_envs 128 \
  --headless
```

本次 30k 训练结果建议优先验证：

```text
首选: model_28800.pt
备选: model_29400.pt
最终: model_29999.pt
```

选择标准：

```text
fall_or_tilt 接近 0
mean_episode_length 接近 20s 满长
action_saturation < 0.02
action_abs_gt_075 < 0.10
touchdown_impact / base_motion / slip 低于旧 walk
视频中机体晃动和落足冲击明显更小
```

新任务 checkpoint 可直接用 `play_real.py` 加载；脚本会根据 agent `obs_groups` 自动选择
旧 88D `policy` 模式或新 80D `deployable + bar_estimate` 模式。

夹爪 action 后处理默认启用：

```text
--gripper_binary_latch
--gripper_latch_open_deadband 0.06
--gripper_latch_close_deadband 0.10
--gripper_latch_min_hold_s 0.8
```

实机后腿夹爪如果出现“夹住后不张开”，优先检查打印中的
`grip_raw/grip_latch/grip_cmd/grip_angle`。当前
`gripper_calibration.yaml` 中对应腿的 `open_torque_scale/max_torque` 应以实机标定为准。
如果 `grip_latch=+1` 但仍打不开，需要结合实际 `grip_cmd`、接线和机械状态检查，
再决定是否调整标定。本轮部署修复没有修改夹爪标定文件。

## 雷达接口

`radar_receiver.py` 默认监听 UDP `8888`，期望 JSON 字段：

```json
{
  "t": 0.0,
  "p": [0.0, 0.0, 0.0],
  "q": [0.0, 0.0, 0.0, 1.0],
  "v": [0.0, 0.0, 0.0],
  "w": [0.0, 0.0, 0.0]
}
```

`q` 是 xyzw。脚本用 `q` 计算 body-frame projected gravity。

默认假设 `v/w` 已经是 body frame。如果雷达输出 world frame，启动时加：

```bash
--radar_velocity_world
```

不加 `--enable_radar` 时，脚本使用：

```text
projected_gravity = [0, 0, -1]
root_lin_vel_b    = [0.05, 0, 0]
root_ang_vel_b    = [0, 0, 0]
```

## Dry Run

先只验证 checkpoint、obs/action 维度和映射，不开电机：

```bash
cd /home/lab202/IsaacLab

python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/play_real.py \
  --task Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Walk-Sim2Real-Direct-v0 \
  --checkpoint logs/rsl_rl/xy4leg_gripper_truss_v3_walk_sim2real/2026-07-24_18-58-21_deployable_interface/model_28800.pt \
  --headless \
  --max_steps 200
```

带雷达 dry run：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/play_real.py \
  --task Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Walk-Sim2Real-Direct-v0 \
  --checkpoint logs/rsl_rl/xy4leg_gripper_truss_v3_walk_sim2real/2026-07-24_18-58-21_deployable_interface/model_28800.pt \
  --headless \
  --enable_radar \
  --max_steps 200
```

## Enable Motors

确认 dry run 输出正常后，再显式使能电机：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/play_real.py \
  --task Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Walk-Sim2Real-Direct-v0 \
  --checkpoint logs/rsl_rl/xy4leg_gripper_truss_v3_walk_sim2real/2026-07-24_18-58-21_deployable_interface/model_28800.pt \
  --headless \
  --enable_radar \
  --enable_motors
```

可选保守参数：

```bash
--action_smoothing 0.5
--gripper_torque_scale 0.5
--startup_hold_s 5.0
```

## DepthBar Crossing

当前 `play_real.py` 也支持 depthbar sim2real crossing。默认任务和 checkpoint 已切换为：

```text
Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Student-DepthBar-Sim2Real-Direct-v0
logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar_sim2real/2026-08-05_04-55-55_realcam_crop_noise_l2far_render12hz_from42000/model_68500.pt
```

默认 depth estimator：

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_depth_crop_realcam/depth_crop_estimator_realcam_xmax060.pt
```

先做不开电机的相机 dry run：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/play_real.py \
  --enable_camera \
  --enable_radar \
  --max_steps 300 \
  --print_interval 25 \
  --bar_estimate_lowpass_alpha 0.25 \
  --headless
```

正常输出中应看到：

```text
bar_obs=[dx/dx_scale, dy/dy_scale, dz/dz_scale, visible]
bar_m=[dx, dy, dz]
bar_vis=raw_visible/memory_used
```

确认相机估计稳定后，再加 `--enable_motors`。如果相机安装不是倒置，启动时加
`--no-camera_flip`。

`play_real.py` 现在从模型配套的训练快照恢复 `dx_scale/dy_scale/dz_scale`。
默认 `68500` 使用 `1.2 / 0.3 / 0.5`，不能直接使用后来训练分支的 `1.2 / 0.6 / 1.0`。
低通 alpha 也优先恢复训练值；`68500` 为 `1.0`。如果实机需要额外去抖，显式传入
`--bar_estimate_lowpass_alpha 0.30` 等覆盖值，并记录它对感知延迟的影响。

## Safety Checklist

1. 先运行 dry run，确认当前 DepthBar actor 输入是 80 维，action 是 24 维。
2. 确认打印的 `Policy leg order` 和 `Hardware leg order` 符合当前接线。
3. 确认 Joint5 的 sign 是否仍为负号映射。
4. 带电 crossing 参数需要同时包含 `--enable_motors --enable_camera --enable_radar`。
5. 确认夹爪开合方向；如果方向反，先改 gripper sign。
6. 检查雷达 projected gravity 在水平站立时接近 `[0, 0, -1]`；未收到有效雷达数据时带电入口不会继续启动。

## Known Limitations

- depthbar crossing 现在使用单前向 RealSense depth crop + estimator + nominal-velocity memory。
  实机侧没有准确 body speed 时，memory 使用训练快照的 nominal velocity，`68500` 为 `0.035 m/s`。
- 第一版没有做 IMU/雷达延迟补偿。
- 第一版没有按实机电机温度/力矩做保护停机。
- 电机串口默认仍沿用 `/dev/ttyUSB0..3`，可用 `--motor_ports` 按 LB LF RB RF 顺序指定
  稳定设备路径；真实设备身份和接线仍需匹配。
