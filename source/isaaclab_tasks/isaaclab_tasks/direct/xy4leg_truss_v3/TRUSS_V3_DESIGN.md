# XY4Leg Truss V3 训练架构

本文档只说明 `xy4leg_truss_v3` 的训练架构、任务分层和设计思路。训练日志、
效果复盘和参数演变记录见：

```text
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/TRUSS_V3_HISTORY.md
```

## 目标

V3 的第一目标是训练一个 privileged teacher，让机器人在两根纵梁上稳定通过单根横杆。
Teacher 阶段先不引入 AMP、camera、foothold estimator、PAS、GRU 或 VAE；teacher
稳定后再进入 student 和感知闭环。

V3 与 V2 的关系：

- 复用 V2 的机器人 USD、关节定义和部分正常行走统计。
- 不复用 V2 的 AMP 训练框架和外部 `skrl/amp.py` patch。
- `sim_output_truss_0520_polynomial.npz` 只用于 level 0 的 reset pose，原因是该数据
  只包含前脚通过，base 和后脚未完成完整跨越。

## 任务分层

V3 当前分为六类 task：

```text
Walk                 -> 纵梁行走预训练，夹爪按 gait phase 固定开合
Gripper Walk         -> 纵梁行走预训练，策略学习夹爪开合
Teacher nominal      -> 无 domain randomization 的 crossing teacher
Gripper Teacher      -> 无 domain randomization 的 crossing teacher，策略学习夹爪开合
Teacher DR           -> 固定 level 4，从 nominal checkpoint 续训鲁棒性
Student OracleBar    -> deployable proprioception + oracle 低维横杆估计
Student DepthBar     -> front/rear depth camera -> heightmap estimator -> 低维横杆估计
```

默认 Walk 和 Teacher 使用兼容的 20 维 action 接口，因此可以通过 `--pretrained_actor`
只迁移 Walk actor 和 observation normalizer 到 Teacher。Gripper Walk 是独立的 24 维
action 任务，它的 checkpoint 只用于 Gripper Teacher，不能直接加载到默认 20 维 Teacher。

## 场景与物理

场景包含：

- XY4Leg 5-DOF gripper robot。
- 两根纵梁。
- 单根横杆。
- IsaacLab 默认网格地面，作为背景和跌落承接面。

主要几何和物理设定：

```text
rail_top_height:        0.50 m
rail_center_distance:   0.296 m
crossbar_width:         0.04 m
crossbar_length:        0.54 m
gravity:                (0, 0, -1.6)
policy dt:              0.04 s
sim dt:                 0.02 s
```

足端正常接触、clearance 和 reward 以纵梁/横杆高度为准；地面主要用于可视化和失稳后承接。

## Observation 与 Action

Actor observation 为 84 维：

```text
68 维 deployable proprioception
16 维 teacher privilege
```

68 维可部署状态包含：

- 20 维主关节位置。
- 20 维主关节速度。
- projected gravity。
- body angular velocity。
- gait clock。
- 20 维上一动作。

16 维 teacher privilege 包含：

- body velocity。
- 横杆相对状态。
- 四脚 signed distance。
- 4 维 crossing stage one-hot。

Critic observation 为 106 维，在 actor observation 后追加：

- 四脚 body-frame 位置。
- 四脚接触状态。
- 四脚 clearance。
- base clearance。
- 非足端横杆碰撞力。

Action 为 20 维，对应四条腿各 5 个主关节。动作映射为：

```text
joint_target = midpoint + 0.5 * (upper - lower) * clamp(action, -1, 1)
```

策略 wrapper 保持 `clip_actions=1.0`，实际输出饱和主要通过 soft limit reward 约束，
避免窄 action clip 掩盖策略本身的动作幅度问题。

Gripper Walk 的 action 为 24 维：

```text
20 维主关节 action + 4 维 gripper action
```

真实夹爪是一个电机同时控制同一个夹爪的两个开合关节，因此策略只输出每条腿一个
gripper action。环境将 RF/LF/RB/LB 的 4 个 gripper action 分别转成开/合增量，并把
同一个目标角同步写到对应夹爪的两个 USD 关节。

## Student 感知接口

最终目标不是纯本体感知跨越，而是带 camera/depth 感知的跨越。V3 将感知和控制拆成两层：

```text
camera/depth -> bar estimator -> low-dimensional bar_estimate -> student policy
```

这样做的原因是控制策略输入保持低维、可解释，感知模块可以单独验证误差、延迟和漏检。
不直接端到端把图像送进 policy，主要是为了降低训练和实机调试复杂度。

### Oracle bar_estimate

第一阶段使用 oracle `bar_estimate`，即从仿真真值直接构造“相机最终应该估出来的横杆状态”。
当前接口为 4 维：

```text
bar_dx_b / 1.2
bar_dy_b / 0.3
bar_dz_b / 0.5
visible
```

其中 `bar_delta_b = (bar_dx_b, bar_dy_b, bar_dz_b)` 是横杆中心相对 base 的 body-frame
位移：

```python
bar_delta_w = bar_pos - root_pos
bar_delta_b = quat_apply_inverse(root_quat, bar_delta_w)
```

`visible` 在 oracle 阶段固定为 1。student observation 因此为：

```text
68 维 deployable proprioception + 4 维 bar_estimate
```

Oracle `bar_estimate` 不解决视觉问题，它只回答一个控制问题：

```text
如果 student 知道横杆相对位置，它能否稳定学会接近、跨越和落轨？
```

### Noisy bar_estimate

Oracle 阶段通过后，下一阶段在 `bar_estimate` 上加入感知误差，模拟 camera/depth estimator
输出：

```text
per-step gaussian noise
per-episode calibration bias
0--2 policy step latency
low-pass smoothing
dropout / visible false
field-of-view gating
```

当前 noisy 验证 task 为：

```text
Isaac-XY4Leg-Gripper-Truss-V3-Student-NoisyBar-Direct-v0
```

默认参数为：

```text
gaussian noise std:       dx=0.03 m, dy=0.015 m, dz=0.015 m
episode bias range:       dx=+-0.02 m, dy=+-0.01 m, dz=+-0.01 m
random delay:             0--2 policy steps
lowpass alpha:            0.7
dropout probability:      0.02 per policy step
FOV x range:              -0.15--1.45 m
FOV y range:              -0.45--0.45 m
FOV z range:              -0.85--0.25 m
```

噪声只加在 student 的 `bar_estimate` observation group 上，不改变仿真物理、
crossing state、reward 或 teacher privileged observation。`visible=0` 时保持上一帧
filtered estimate，只把可见性标志置 0，模拟 estimator 暂时丢失检测但下游仍保留最近估计。
这一步用于验证 policy 对非完美估计的敏感度，避免直接进入 camera/depth 后无法区分
“感知不准”和“控制 policy 对误差不鲁棒”。

如果 full noisy 直接失败，先跑两个 ablation task：

```text
Student-NoisyBar-NoiseOnly:  只加 dx/dy/dz noise 和 episode bias，visible 恒为 1，无延迟
Student-NoisyBar-Delayed:    加 noise/bias/0--2 step delay/low-pass，visible 仍恒为 1
```

对应 task id：

```text
Isaac-XY4Leg-Gripper-Truss-V3-Student-NoisyBar-NoiseOnly-Direct-v0
Isaac-XY4Leg-Gripper-Truss-V3-Student-NoisyBar-Delayed-Direct-v0
```

这两个 ablation 用来判断失败来源：如果 `NoiseOnly` 已失败，是 policy 对横杆位置误差
本身过敏；如果 `NoiseOnly/Delayed` 通过但 full noisy 失败，主因就是 `visible=0`、
FOV gating 或检测丢失引入的 out-of-distribution 输入。

### Camera/depth estimator 差距

Camera/depth estimator 相比 oracle `bar_estimate` 需要额外解决：

- 相机内外参误差。
- depth 噪声和空洞。
- 横杆遮挡、漏检和误检。
- 运动模糊和光照/材质 domain gap。
- 视野外横杆。
- 图像处理与控制之间的延迟。

因此 V3 的顺序是：

```text
oracle bar_estimate student
-> noisy/delayed bar_estimate student
-> depth-derived bar_estimate student
-> 实机 camera/depth estimator + student policy
```

### Depth-derived bar_estimate

Depth-derived 阶段先使用固定 oracle-bar student policy 验证 perception 接口，再从
oracle student checkpoint 继续 fine-tune，使 policy 适配 depth-derived `bar_estimate`
的闭环误差：

```text
front/rear depth cameras
-> camera points -> base_link local heightmap
-> DepthBarEstimator -> metric bar_delta_b(dx, dy, dz)
-> normalize -> policy bar_estimate(dx/1.2, dy/0.3, dz/0.5, visible=1)
```

当前 task：

```text
Isaac-XY4Leg-Gripper-Truss-V3-Student-DepthBar-Direct-v0
```

该 task 会挂载前后两个 64x64 depth camera，并额外返回：

```text
heightmap           # [num_envs, 2, 64, 64]
                    # channel 0: fused base-frame height, normalized to 0--1
                    # channel 1: valid mask
bar_estimate_label  # [num_envs, 3], metric body-frame bar_delta_b
```

heightmap 的局部机体系范围默认为：

```text
x: [-0.85, 1.45] m
y: [-0.45, 0.45] m
z: [-0.85, 0.25] m, normalized to 0--1
```

当前相机安装在 `base_link` 下方：

```text
front camera: pos=( 0.00, 0.0, -0.16), pitch approximately -30 deg, 64x64 depth
rear camera:  pos=(-0.20, 0.0, -0.16), pitch approximately +30 deg, 64x64 depth
focal_length: 14.0
```

如果 `depth_bar_estimator_checkpoint` 为空，policy 的 `bar_estimate` 暂时使用 oracle
fallback；这用于先验证相机渲染、数据形状、label 和 policy 接口。如果提供 TorchScript
estimator checkpoint，则 env 会用 estimator 输出替代 oracle。给 policy 的 `visible`
固定为 1，丢检/last-valid 逻辑先留在 estimator 层内部处理。

## Gait 与 Gripper

V3 当前使用固定相位 gait clock：

```text
gait_frequency:     0.5 Hz
gait_duty_factor:   0.82
phase_offsets:      RF=0.0, LF=0.5, RB=0.75, LB=0.25
```

默认 Walk、Teacher 和 Student 中，四组 gripper 使用 gait-phase FSM：

- swing 相打开。
- stance 相闭合。
- 使用 `gripper_phase_hysteresis` 防止频繁开合。

Gripper Walk 中不使用 gait-phase gripper FSM，而是让 policy 学习四个 gripper
action。每个 action 只决定该夹爪下一步向打开还是闭合方向移动，步长由
`gripper_action_step` 决定，并限制在 `gripper_closed_angle` 到 `gripper_open_angle`
之间。

接触判定聚合每条腿的 `Link5` 和两个 gripper 指端，用同一套接触状态驱动支撑、
落地、打滑、悬空时间、rail-valid 和 landing condition。

## Walk 预训练

Walk task 将横杆移到远处，只训练纵梁前进和动作稳定性。它的目标不是最大速度，
而是为 crossing teacher 提供一个低冲击、少滑动、动作不过饱和的初始 actor。

Walk reward 主要包含：

- 前进速度区间奖励。
- upright 和 heading。
- stance 接触与 swing 离地。
- 纵梁横向居中和期望 stance 接触。
- swing clearance 误差。
- base height 和 base motion。
- touchdown impact、slip、joint velocity、energy。
- action magnitude、action rate、action acceleration、soft action limit。

Walk 默认开启轻量 domain randomization：

```text
friction / base mass / actuator gains
0--2 step action delay
joint position bias
joint position and velocity noise
```

## Crossing State

Crossing 使用逐脚状态机，不直接用单步 signed distance 判断成功。每只脚依次经过：

```text
0: before bar
1: cleared bar near crossbar
2: passed bar
```

脚从 0 到 1 需要同时满足：

- 位于横杆附近的 x window。
- 足端 clearance 达到阈值。
- 处于对应纵梁 y 范围内。

脚从 1 到 2 需要：

- signed distance 超过 pass margin。
- rail-valid 仍成立。

该状态机避免“从横杆旁边绕过”或“初始位置直接算通过”的奖励投机。

## Crossing Curriculum

Teacher nominal 使用五级课程：

```text
level 0: hind completion
level 1: front crossing
level 2: medium approach
level 3: full approach
level 4: randomized full
```

每个环境维护独立 curriculum state：

```text
level
success_ema
episode_count
failure_streak
```

课程推进规则：

- 至少累计 `curriculum_min_episodes` 后才允许升降级。
- `success_ema >= curriculum_promote_threshold` 时升级。
- `success_ema <= curriculum_demote_threshold` 或连续失败过多时降级。
- checkpoint 会保存并恢复 curriculum state。

level 4 成功条件：

- 四脚均 passed bar。
- base 超过 `root_pass_margin`。
- 至少三脚有效支撑保持 `stable_landing_s`。
- 四脚位于纵梁范围内。
- 姿态、yaw、横向偏移、角速度、高度均在阈值内。
- 没有非足端横杆碰撞。

level 0 到 level 3 使用逐步放宽的 landing condition，让策略先学会后腿和 base
完成，再扩展到完整 approach。

Student distillation 复用同一套 crossing state、success 判定和 reward，但课程距离更偏向
控制可学性。Teacher 的 level 0 是“后腿完成”bootstrap，横杆几乎在机体下方；student
使用 oracle `bar_estimate` 后，目标是从近距离完整 crossing 逐步推到远距离 approach：

```text
teacher level distance ranges:
  level 0: -0.02--0.02 m
  level 1:  0.15--0.35 m
  level 2:  0.35--0.50 m
  level 3:  0.50--0.80 m
  level 4:  0.80--1.20 m

student/depth level distance ranges:
  level 0:  0.00--0.15 m
  level 1:  0.15--0.30 m
  level 2:  0.55--0.70 m
  level 3:  0.60--0.80 m
  level 4:  0.65--0.90 m
```

Student 的升阶阈值略低于 teacher：

```text
teacher promote/demote: 0.70 / 0.30
student promote/demote: 0.65 / 0.25
```

这样做的目的是避免 student 在 level 0/1 反复退回，先确认“deployable proprioception +
bar_estimate”能稳定闭环，再逐步恢复到 level 4 全阶段跨越。Teacher 课程保持不变。

## Reward 设计

V3 reward 的主线是：

```text
能完成 crossing
但不要用跌倒、碰撞、大冲击、动作饱和或长期悬空换成功
```

主要 reward 组：

- Walking base terms：forward velocity、upright、heading、support、gait contact。
- Rail walking terms：rail center、rail contact、swing clearance、walk foothold。
- Crossing progress terms：front/hind/root monotonic progress。
- Crossing event terms：front/hind/root event、success event。
- Landing terms：landing rail center、landing contact、landing stability、stable contact progress。
- Stability penalties：base motion、touchdown impact、slip、joint velocity、energy。
- Action penalties：action magnitude、soft action limit、action rate、action acceleration。
- Failure penalties：forbidden contact、long air without progress、termination。

Progress reward 只奖励历史最大进度的新增部分，reset 初始状态不产生 progress 或 event
奖励。失败步不再获得 crossing 正奖励。

## Reset 与专家数据

reset 分两类：

- 普通 reset：机器人回到 `robot.init_state.joint_pos` 对应的默认站立姿态，并根据课程等级采样横杆距离。
- 可选专家 reset：level 0 可以按概率使用 `sim_output_truss_0520_polynomial.npz` 中的后腿完成片段，
  只作为早期 bootstrap 姿态来源。

reset 后会根据实际关节姿态反算：

```text
raw_actions
previous_actions
previous_previous_actions
applied_actions
action_history
```

这样 fresh policy 和 resume policy 都从一致的 action history 开始。

## Domain Randomization

Nominal Teacher 默认关闭 DR，用于先验证课程和 reward 是否可学。

Teacher DR task 固定在 level 4，并开启：

- rigid body material randomization。
- base mass randomization。
- actuator stiffness/damping randomization。
- 0--2 step action delay。
- joint position bias。
- joint position / velocity observation noise。

DR 预期在 nominal teacher 达到稳定成功率后，从 nominal checkpoint 续训。

## 训练入口

V3 使用独立入口：

```text
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/train_v3.py
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/play_v3.py
```

`train_v3.py` 默认 task 为 nominal teacher。常用命令：

Walk 预训练：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/train_v3.py \
  --task Isaac-XY4Leg-Gripper-Truss-V3-Walk-Direct-v0 --headless
```

从 Walk checkpoint 启动 Teacher：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/train_v3.py \
  --task Isaac-XY4Leg-Gripper-Truss-V3-Teacher-Direct-v0 \
  --pretrained_actor /path/to/xy4leg_truss_v3_walk/model_x.pt --headless
```

直接训练 nominal Teacher：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/train_v3.py --headless
```

训练 oracle bar-estimate Student：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/train_v3.py \
  --task Isaac-XY4Leg-Gripper-Truss-V3-Student-OracleBar-Direct-v0 --headless
```

播放 noisy bar-estimate Student：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/play_v3.py \
  --task Isaac-XY4Leg-Gripper-Truss-V3-Student-NoisyBar-Direct-v0 \
  --checkpoint /path/to/oracle_bar_student/model_x.pt \
  --video --video_length 500 --headless
```

定量评估 oracle/noisy Student：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/eval_v3.py \
  --task Isaac-XY4Leg-Gripper-Truss-V3-Student-NoisyBar-Direct-v0 \
  --checkpoint /path/to/oracle_bar_student/model_x.pt \
  --num_envs 128 --num_episodes 1000 --headless
```

采集 heightmap bar-estimator 数据：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/collect_depth_bar_dataset_v3.py \
  --checkpoint logs/rsl_rl/xy4leg_truss_v3_student_oracle_bar/2026-07-17_14-09-48_nominal/model_35000.pt \
  --output logs/rsl_rl/xy4leg_truss_v3_depth_bar/dataset_levels0-4_l2_055-070_l3_060-080_l4_065-090.pt \
  --levels 0,1,2,3,4 --num_samples 100000 --num_envs 128 --headless
```

训练并导出 heightmap bar-estimator：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/train_depth_bar_estimator_v3.py \
  --dataset logs/rsl_rl/xy4leg_truss_v3_depth_bar/dataset_levels0-4_l2_055-070_l3_060-080_l4_065-090.pt \
  --output logs/rsl_rl/xy4leg_truss_v3_depth_bar/depth_bar_estimator_l2_055-070_l3_060-080_l4_065-090.pt
```

评估 depth-derived bar_estimate：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/eval_v3.py \
  --task Isaac-XY4Leg-Gripper-Truss-V3-Student-DepthBar-Direct-v0 \
  --checkpoint logs/rsl_rl/xy4leg_truss_v3_student_oracle_bar/2026-07-17_14-09-48_nominal/model_35000.pt \
  --depth_bar_estimator_checkpoint logs/rsl_rl/xy4leg_truss_v3_depth_bar/depth_bar_estimator_l2_055-070_l3_060-080_l4_065-090.pt \
  --fixed_level 4 --num_envs 128 --num_episodes 1000 --headless
```

从 oracle-bar student checkpoint 继续 fine-tune DepthBar student：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/train_v3.py \
  --task Isaac-XY4Leg-Gripper-Truss-V3-Student-DepthBar-Direct-v0 \
  --resume \
  --checkpoint logs/rsl_rl/xy4leg_truss_v3_student_oracle_bar/2026-07-17_14-09-48_nominal/model_35000.pt \
  --depth_bar_estimator_checkpoint logs/rsl_rl/xy4leg_truss_v3_depth_bar/depth_bar_estimator_l2_055-070_l3_060-080_l4_065-090.pt \
  --run_name ft_depth_from_oracle35000_l2_055-070_l3_060-080_l4_065-090 \
  --num_envs 128 --headless
```

从 checkpoint resume：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/train_v3.py \
  --resume --load_run RUN_DIR --checkpoint model_x.pt --headless
```

DR 续训：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/train_v3.py \
  --task Isaac-XY4Leg-Gripper-Truss-V3-Teacher-DR-Direct-v0 \
  --resume --headless
```

播放 checkpoint：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/play_v3.py \
  --checkpoint /path/to/model.pt
```

## 日志与模型保存

训练日志固定写入仓库内：

```text
logs/rsl_rl/<experiment_name>/<timestamp>
```

模型和视频保存间隔按总训练轮次自动设置：

```text
max_iterations < 30000: every 500 PPO iterations
max_iterations >= 30000: every 2% total PPO iterations
```

标量日志：

```text
events.out.tfevents...       # TensorBoard 读取
scalars_sampled.csv          # 低频 CSV 采样
```

默认写入频率：

```text
--scalar_log_interval 10
--csv_log_interval 100
```

RSL-RL 默认保存 `model_<iteration>.pt` 和最终模型；当前没有单独的 `best_model.pt`。
模型选择需要结合 TensorBoard 指标和训练视频人工判断。

## 验收标准与后续

Nominal teacher 验收目标：

- 3 个 seed。
- 至少 1000 个确定性 evaluation episode。
- 成功率不低于 90%。
- fall、yaw、lateral、angular velocity、forbidden contact 均保持低水平。
- 视频中不应依赖大跨步、强冲击或明显动作饱和。

DR teacher 验收目标：

- 相同 evaluation 协议。
- 成功率不低于 80%。
- 动作形态仍可部署，不因 DR 产生明显高频抖动。

达标后再进入 student 阶段：

- 先做 oracle `bar_estimate` student，验证低维横杆状态足够支持闭环 crossing。
- 再做 noisy/delayed `bar_estimate` student，加入估计噪声、延迟、低通和 dropout。
- 然后训练 depth-derived bar estimator，把深度图转成同一 4 维 `bar_estimate` 接口。
- 最后接入实机 camera/depth estimator，并视效果决定是否需要 PPO fine-tune。
