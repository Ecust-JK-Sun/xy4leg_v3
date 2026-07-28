# Truss V3 修改历史

本文档记录 `xy4leg_truss_v3` 的训练日志、效果判断，以及每轮训练后推动的
主要代码和 reward 修改。日志根目录为：

```text
logs/rsl_rl/xy4leg_truss_v3_walk
logs/rsl_rl/xy4leg_gripper_truss_v3_walk
logs/rsl_rl/xy4leg_truss_v3_teacher
logs/rsl_rl/xy4leg_gripper_truss_v3_teacher
logs/rsl_rl/xy4leg_truss_v3_student
logs/rsl_rl/xy4leg_truss_v3_student_oracle_bar
logs/rsl_rl/xy4leg_truss_v3_depth_bar
logs/rsl_rl/xy4leg_gripper_truss_v3_student_oracle_bar
logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar
logs/rsl_rl/xy4leg_gripper_truss_v3_walk_sim2real
logs/rsl_rl/xy4leg_gripper_truss_v3_teacher_sim2real
logs/rsl_rl/xy4leg_gripper_truss_v3_student_oracle_bar_sim2real
logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar_sim2real
```

训练架构、任务分层、observation/action、课程和验收思路见：

```text
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/TRUSS_V3_DESIGN.md
```

## 阅读约定

- **训练效果**：优先看后段趋势，少数 batch 中的 `success_rate=1.0` 不单独视为稳定收敛。
- **触发修改**：该训练暴露的问题直接推动后续修改。
- **当前状态**：以当前源码为准，不代表已经有完整训练验证。
- **历史分析记录**：曾经分析过但当前日志目录中不一定还保留的 run。

## 训练日志索引

| 类型 | 训练目录 | 主要配置/现象 | 训练效果简述 | 训练后采取的修改 |
|---|---|---|---|---|
| Teacher | `2026-07-05_19-27-03` | 早期 V3 teacher，`gait_frequency=0.8`，`duty=0.75`，无 DR | 后段成功率接近 0，`fall_or_tilt` 约 0.57；后腿通过率已有信号但不能形成稳定 success | 修 reset/action history、逐脚 crossing state、接触聚合、reward/reset 日志和 curriculum checkpoint |
| Teacher | `2026-07-06_03-52-48` | 早期 landing curriculum，`landing_velocity_sigma=0.08` | 后段 success 约 0.48，level 约 1.17；能部分跨越，但跌倒仍约 0.52 | 增强落地课程、reset 分类、每腿 rail/contact 诊断；开始准备 Walk 预训练和五级课程 |
| Walk | `2026-07-06_16-38-39` | Walk-only，15k，`gait_frequency=0.8`，DR 关闭 | 初版 walking；无 crossing success 指标，仍有跌倒和 gait/contact 投机风险 | 修正 `support/gait_contact`，只奖励期望 stance 接触；加入 base motion、swing clearance、foothold touchdown |
| Walk | `2026-07-06_22-25-14` | Walk-only，20k，base motion 已启用 | walking 继续稳定，但仍偏激进，动作和落足质量不足以直接上实机 | 第二轮 Walk 稳定化：速度区间奖励、penalty clamp、reset 起点后移和更多 gait 诊断 |
| Walk | `2026-07-07_02-39-38` | Walk-only，50k，DR 关闭，速度约 0.10 m/s | nominal 行走基本稳定，但步态偏低/高抬腿，缺少 sim-to-real 鲁棒性；只有 `model_*.pt`，无 `best_model` | 直接在 Walk 中加入 DR、动作延迟、观测噪声；步频改 0.5，duty 0.82；奖励转向稳、低冲击、少滑动 |
| Walk | `2026-07-07_20-47-04` | Walk-only，50k，DR 开启，`max_action_delay_steps=2`，稳定性 reward 较强 | 后段平均速度约 0.08 m/s，动作更保守；作为后续 teacher 的默认 `pretrained_actor` 来源 | 用该 Walk checkpoint 初始化 Teacher；Teacher 保持 nominal，无 DR |
| Teacher | `2026-07-08_16-30-24` | nominal Teacher，gait 0.5/0.82，gravity 已调至 `-1.6`，DR 关闭 | 后段 success 约 0.50，level 约 1.34；`model_49000.pt` 视频较好但仍低趴、晃动、实机风险高 | 尝试加强稳定性 reward：base motion、base height、touchdown impact、action accel、landing angular stability；同时修 quantile 空 tensor、CUDA/cuBLASLt、TensorBoard writer 报错 |
| Teacher | `2026-07-09_14-46-07` | 历史分析记录：稳定性约束过强，150k 目标，实际约 94k | 退化：success 约 0，level 卡 0；front/root 完成但 hind 不过，`no_progress` 约 0.70 | 回滚过强稳定性：恢复 `stable_landing_s=0.40`，提高 hind/success 奖励，减轻 base/action/impact 惩罚 |
| Teacher | `2026-07-10_22-30-48` | 平衡版 reward，50k，event/csv 抽样前仍有大 CSV | 明显最好：20k 左右进入 level 4，后段 success 约 0.84，hind complete 约 0.92，fall 约 0.14；但前脚常直接大跨步越过横杆 | 日志改为 TensorBoard event + 抽样 CSV；加入前脚 approach foothold 和 overshoot shaping，尝试修正大跨步 |
| Teacher | `2026-07-11_13-26-02` | 前脚 shaping 初版：`rew_front_approach=1.5`、`rew_front_step_overshoot=-0.8`，对 level 0 也生效 | 退化：到 23.3k 仍 level 0，success 0；front complete 从一开始就是 1，但 hind 卡住，`no_progress` 约 0.67 | 将前脚 shaping gate 到 `curriculum_level > 0`，并大幅减轻 overshoot 惩罚 |
| Teacher | `2026-07-11_21-03-06` | 轻量前脚 shaping 完整 50k 验证，参数仍出现在 env config 中 | 仍退化：49.9k `level_mean=0`，尾段 success 0；front complete 维持 1，但 hind tail avg 约 0.01，`no_progress` tail avg 约 0.59 | 回到 `2026-07-10_22-30-48` 的有效 reward 面：移除前脚 shaping 配置与 reward 计算 |
| Teacher | `2026-07-12_13-09-54` | 移除前脚 shaping 后的 50k 验证，配置理论上接近 `2026-07-10_22-30-48` | 恢复到 level 4，约 15.1k 到 level 4；尾段 success 约 0.71，低于 `2026-07-10` 的约 0.83；tail `fall_or_tilt` 约 0.21、`no_progress` 约 0.08 | 去掉 `_RESET_MOTION` / motion reset，后续 reset 直接使用 `robot.init_state.joint_pos` 对应默认关节姿态 |
| Current | 已回退到 `2026-07-12_13-09-54` reset 配置 | 撤销高课程前脚 reward gate，并恢复 `_RESET_MOTION`、`hind_reset_pose_probability=0.5`、`joint_reset_noise=0.02` | 用于隔离变量，确认 level 0 能否恢复到 `2026-07-12_13-09-54` 的升阶表现 | 暂不改 level 设置；新增 `Crossing/early_front_event_rate` 只读诊断，不参与 reward |
| Teacher | `2026-07-14_02-51-07` | 恢复 reset motion 后的 nominal teacher，新增 `early_front_event_rate` 诊断 | 50k 完整训练，约 15.8k 到 level 4；末尾 success 约 0.92，尾段 success 约 0.81；尾段 `fall_or_tilt` 约 0.14，`early_front_event_rate=0` | 作为 nominal student 第一版 teacher checkpoint：`model_49999.pt` |
| Student | `2026-07-14_13-50-46_nominal` | nominal distillation，student 只使用 deployable obs，teacher 使用完整 policy obs；不加 DR | 20k 后仍卡 level 0，尾段 success 约 0.002，`fall_or_tilt` 约 0.998；behavior loss 降到约 0.008，但闭环 rollout 不稳定 | 新增 oracle `bar_estimate` 输入，验证低维横杆感知是否能解决纯本体感知信息不足 |
| Student OracleBar | `2026-07-15_13-09-10_nominal` | nominal distillation，student 使用 `deployable + oracle bar_estimate`，teacher 使用完整 policy obs；`init_std=0.15`、`learning_rate=1e-3` | 50k 后仍卡 level 0，尾段 success 约 0.008，`fall_or_tilt` 约 0.99；behavior loss 降到约 0.0026，但 `action_accel/base_motion/touchdown_impact` 明显高于 teacher，视频中机体抖动和落足不稳 | 将 student `init_std` 从 0.15 降到 0.03，distillation `learning_rate` 从 1e-3 降到 3e-4，先隔离随机 rollout 抖动问题 |
| Student OracleBar | `2026-07-16_00-24-16_nominal` | nominal distillation，`deployable + oracle bar_estimate`，`init_std=0.03`、`learning_rate=3e-4` | 50k 后已明显脱离 level 0，尾段 success 约 0.45，最好 batch success 约 0.92；但 `level_mean` 尾段约 1.4，level 2+ 成功率弱，base motion / touchdown impact 仍高于 teacher | Student-only 重新切分课程距离，并稍降低升降级阈值；Teacher 课程不动 |
| Student OracleBar | `2026-07-17_14-09-48_nominal` | nominal distillation，student 课程改为 `0.00--0.15 / 0.15--0.30 / 0.30--0.50 / 0.50--0.80 / 0.80--1.20`，promote/demote `0.65/0.25` | 明显改善：约 16k 到 level 4，尾段 `level_mean` 约 3.99，tail success 约 0.67，`fall_or_tilt` 约 0.23；仍低于 teacher，`no_progress` 约 0.11 | 新增 `Student-NoisyBar` 验证入口，只扰动 4 维 `bar_estimate`，用于评估 policy 对感知误差/延迟/漏检的敏感度 |
| Student NoisyBar | `model_35000.pt` eval | NoiseOnly/Delayed ablation，visible 恒为 1；full noisy 会触发 visible/FOV/dropout OOD | NoiseOnly success 约 0.747，Delayed success 约 0.753；full noisy success 约 0.008 | 不先做 full noisy fine-tune，改为推进 depth-derived bar estimator，policy visible 固定为 1 |
| Student DepthBar | `eval_depth_l2_055-070_l3_060-080_l4_065-090` | front/rear depth -> base-frame heightmap -> TorchScript estimator；固定 oracle student `model_35000.pt` | fixed-level success：L0 0.713、L1 0.634、L2 0.553、L3 0.711、L4 0.754；estimator 误差约 2cm 级 | 从 oracle student checkpoint 继续 DepthBar fine-tune，适配 depth-derived estimate 的闭环误差 |
| Student DepthBar | `2026-07-19_18-06-50_ft_depth_from_oracle35000_l2_055-070_l3_060-080_l4_065-090` | 从 oracle student `model_35000.pt` resume，使用 `depth_bar_estimator_l2_055-070_l3_060-080_l4_065-090.pt` | 训练内稳定到 level 4；后 20% sampled success 约 0.75，最后 sampled success 约 0.917；fall/no_progress tail 约 0.156/0.094 | DepthBar 闭环流程已打通；下一步做 `model_51600.pt` fixed-level eval，再进入感知噪声/延迟和轻量 DR |
| Gripper Walk | `2026-07-20_11-59-48_learn_gripper` | 独立任务 `Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Walk-Direct-v0`，日志目录 `xy4leg_gripper_truss_v3_walk` | 50k 训练完成，末段走满率高，`fall_or_tilt` 约 0.0025，动作饱和接近 0；可作为 learned-gripper teacher 初始化 | 新增 Gripper Teacher 注册，使用 24 维 action checkpoint 初始化 crossing teacher |
| Gripper Teacher | `2026-07-21_01-22-13_learn_gripper` | 初版使用 nominal teacher 旧课程，训练到约 32k 中断；已进入 level 4，tail success 约 0.82 | 24 维 action：20 维主关节 + 4 维夹爪；旧 level 4 为 `[0.80, 1.20]` | 后续改用 student/depth-bar 修正课程：L0 `[0.00,0.15]`、L1 `[0.15,0.30]`、L2 `[0.55,0.70]`、L3 `[0.60,0.80]`、L4 `[0.65,0.90]`，promote/demote `0.65/0.25` |
| Gripper Teacher | `2026-07-21_13-40-58_learn_gripper` | 从 `model_32000.pt` 续训到约 41.9k；`model_33440.pt` play 效果较好，但各 level 仍一般，夹爪开合时机偶有偏差；继续训练后速度升高、冲击和 slip 增大，动作偏激进 | `model_33440.pt` 是当前较均衡 checkpoint；后续应从该点短续训 | 新增 gripper phase shaping，所有 learned-gripper 训练启用；Gripper Teacher 额外加入 `forward_speed_limit=0.25`，并轻微加强 `touchdown_impact/base_motion/action_rate/action_accel/slip`，略降 `hind_progress/success_event`，限制明显快速跳跨 |
| Gripper Teacher | `2026-07-21_16-34-08_learn_gripper` | 从 `model_33440.pt` 附近续训，启用夹爪相位 shaping 和 forward speed limit；保存到 `model_43439.pt` | tail success 约 0.86，`fall_or_tilt` 约 0.14，速度约 0.27；比上一版慢一些，但仍存在足端擦碰横杆 | 为减少足端碰杆，新增 `foot_bar_contact` 诊断和直接惩罚试验 |
| Gripper Teacher | `2026-07-21_18-59-55_learn_gripper` | 从 `model_34000.pt` 续训，加入 `rew_foot_bar_contact=-2.0` 直接惩罚 | tail success 约 0.85，速度约 0.23；play 中 `model_36000.pt` 观感较好，`model_38000.pt` / level4 不理想；直接 foot-bar 接触惩罚容易受 rail 接触诊断噪声影响 | 将 `foot_bar_contact` 降为诊断项，改用更保守的 `low_bar_clearance` 惩罚约束足端低过杆 |
| Gripper Teacher | `2026-07-21_22-36-26_learn_gripper` | 从 `model_34000.pt` 续训，`rew_low_bar_clearance=-1.5`，`rew_foot_bar_contact=0` | tail success 约 0.91，`fall_or_tilt` 约 0.09，速度约 0.24；`model_36000.pt` 相比 `model_38000.pt` 更保守，人工 play 后选为 student teacher | 固定 teacher checkpoint 为 `model_36000.pt`，开始 learned-gripper oracle-bar student |
| Gripper Student OracleBar | `2026-07-22_12-50-45_teacher36000` | 24D learned-gripper student，输入 `deployable + oracle bar_estimate`，teacher 为 gripper teacher `model_36000.pt`；训练到约 11.8k 后中断 | behavior loss 很快降到 `1e-3` 以下；tail success 约 0.81；`model_4000.pt` 与 `model_10000.pt` 都有约 0.94 batch success，后续选 `model_10000.pt` 推进 depth-bar | 注册 learned-gripper DepthBar student，从 oracle student `model_10000.pt` resume |
| Gripper Student DepthBar | `2026-07-22_15-49-26_from_oracle10000_teacher36000` | 从 oracle student `model_10000.pt` resume，使用 depth/heightmap estimator `depth_bar_estimator_l2_055-070_l3_060-080_l4_065-090.pt` | 训练从开始到 5w 基本平台期；区间 success 约 0.83-0.85；综合排序首选 `model_30000.pt`，最终 `model_59999.pt` 不作为 best | 将 gripper student/oracle/depth level 改近到 L2 `[0.55,0.65]`、L3 `[0.60,0.70]`、L4 `[0.65,0.75]`，并做 level2 split eval |
| Gripper Student DepthBar Eval | `eval/level.txt` 与 `eval/split_l2_l4_model30000/*` | 使用 `model_30000.pt` 在近距离 level 上 fixed-level 评估；重点看 level2/4 | level4 success 0.846；level2 success 0.773。level2 split 显示 x 越大越稳定：`0.55--0.60` 0.767，`0.70--0.75` 0.830 | 判断训练长期停在 level4，level2/3 样本不足；新增 balanced curriculum |
| Gripper Student DepthBar | `2026-07-23_14-31-20_balanced_levels_from_model30000` | 从 depth-bar `model_30000.pt` resume，启用 balanced level weights `0.05/0.05/0.30/0.30/0.30`，完整训到 `model_49999.pt` | 后 20% sampled success 约 0.830，fall 约 0.160；level2/3/4 加权 success 分别约 0.786/0.805/0.821，覆盖明显比上一版均衡 | 候选 best model：`model_42000.pt`、`model_45000.pt`、`model_48500.pt`、`model_49999.pt`；后续需 fixed-level eval 再定 |
| Gripper Walk Sim2Real | `2026-07-24_02-38-20_deployable_interface` | 新任务 `Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Walk-Sim2Real-Direct-v0`；actor obs 改为 `deployable(76D)+bar_estimate(4D)=80D`，加入夹爪角度、慢步频、action delay、观测噪声、Sim2Real DR、夹爪 latch | 后 5k 基本走满 20s：`time_out` 约 0.996，`fall_or_tilt` 约 0.004，action saturation 约 0 | 确认 deployable interface 可训练；后续再跑一版并选更适合实机的 checkpoint |
| Gripper Walk Sim2Real | `2026-07-24_18-58-21_deployable_interface` | 同一套 deployable-interface walk sim2real，30k 完整训练，并导出 `policy.pt/onnx` | 后 5k `time_out=1.0`、`fall_or_tilt=0`、`action_saturation=0`、平均前进距离约 1.27m；实机 walk 初测反馈可用 | 选 `model_28800.pt` 作为 walk sim2real 推荐 checkpoint 和 crossing sim2real 初始化来源 |
| Crossing Teacher Sim2Real | `2026-07-25_15-55-43_deployable_from_walksim_model28800` | 错误路线：将 Teacher PPO actor 直接设成 `deployable+bar_estimate`，本质接近 oracle-bar deployable policy，不是 privileged teacher | 后段仍 level0，success 0；`no_progress` 平均约 0.64、最后约 0.875；不应续训 | 废弃该路线；改回三阶段：privileged teacher -> oracle-bar student -> depth-bar student，只把 walk-sim2real 的硬件接口/DR/慢步频/latch 加到对应阶段 |
| Crossing Teacher Sim2Real | `2026-07-26_13-04-34_privileged_from_walksim_model28800` | 已修正为 privileged teacher，actor obs 为 `policy=92D`，但仍从 walk-sim2real `model_28800.pt` 初始化 | 到 33.5k 仍 level0；front/root complete 接近 1，但 `hind_complete_rate=0`，`no_progress` 约 0.7--0.9，说明没有学出后腿跨越 | 放弃从 walk-only prior 重新探索 crossing；新增 88D teacher -> 92D sim2real teacher adapter，从 gripper teacher `model_36000.pt` 做 sim2real finetune |

## 1. V3 Teacher 初始稳定性修复

**训练关联**

- 触发训练：`2026-07-05_19-27-03`
- 首次改善验证：`2026-07-06_03-52-48`

初始 V3 目标是脱离 V2 AMP/skrl 体系，只做单根横杆的 privileged teacher。
`2026-07-05_19-27-03` 暴露的主要问题是 reset 后动作历史不一致、接触定义不稳、
crossing 状态容易投机、失败步仍能拿到 crossing 正奖励。

随后加入：

- reset 后由实际关节姿态反算 action/history。
- fresh actor 的最后一层 bias 初始化为默认站立动作。
- 每条腿聚合 `Link5` 与两个 gripper 指端接触，统一用于支撑、落地、滑动和悬空。
- crossing 改为逐脚状态机：先在横杆附近满足 clearance，再允许通过。
- crossing progress 只累计历史最大增量，且必须姿态、横向位置和碰撞状态有效。
- curriculum 状态随 RSL-RL checkpoint 保存和恢复。
- `RewardRate/*`、`ResetRate/*` 改为按实际 episode 时长和 reset 类型输出。

## 2. 落地课程与五级 Crossing Curriculum

**训练关联**

- 触发训练：`2026-07-06_03-52-48`
- 后续 Teacher 验证：`2026-07-08_16-30-24`

`2026-07-06_03-52-48` 已能获得约 0.48 的后段 success，但跌倒仍高，课程推进有限。
之后将课程明确拆为五级：

```text
level 0: hind completion
level 1: front crossing
level 2: medium approach
level 3: full approach
level 4: randomized full
```

success 随课程收紧：level 0 聚焦后腿和 base 完成，中间等级要求四脚通过并至少
三脚有效支撑，level 4 再要求四脚都位于纵梁范围。

同时新增：

- 每条腿 episode contact / rail-valid / terminal 状态日志。
- reset 原因分类：height、tilt、yaw、lateral、angular velocity、forbidden contact、no progress。
- level 0/1 的 no-progress 检查能在 episode 内实际触发。

## 3. Walk 预训练

**训练关联**

- 初版 Walk：`2026-07-06_16-38-39`
- 中间修正：`2026-07-06_22-25-14`
- nominal Walk 基线：`2026-07-07_02-39-38`
- DR Walk 基线：`2026-07-07_20-47-04`

新增 `Isaac-XY4Leg-Gripper-Truss-V3-Walk-Direct-v0`，横杆放到 100 m 外，只训练：

- 纵梁上前进。
- stance 接触和 swing 离地。
- 纵梁横向居中。
- base height 和 base motion。
- action magnitude、action rate、action acceleration。
- touchdown impact、slip、joint velocity、energy。

`2026-07-07_02-39-38` nominal Walk 基本可行，但动作低趴、步态不够实机友好。
之后将 Walk 改为更保守：

```text
gait_frequency: 0.5
gait_duty_factor: 0.82
forward_velocity_min/max: 0.03 / 0.09
domain_randomization_enabled: true
max_action_delay_steps: 2
joint position/velocity noise: enabled
```

`2026-07-07_20-47-04/model_49999.pt` 后续作为 Teacher 默认 `pretrained_actor`。

## 4. 低重力与硬件部署稳定性尝试

**训练关联**

- 修改前较好 Teacher：`2026-07-08_16-30-24`
- 过强稳定性回归：`2026-07-09_14-46-07`
- 平衡后验证：`2026-07-10_22-30-48`

训练过程中明确将 V3 gravity 从早期值调整为：

```text
gravity = (0, 0, -1.6)
```

`2026-07-08_16-30-24` 的后段 success 约 0.50，level 约 1.34，视频中动作已经能跨越，
但仍低趴、晃动、落地冲击较明显。随后曾尝试大幅加强稳定性：

- `rew_base_motion` 从 0 加到较强负值。
- `rew_walk_base_height` 启用。
- `rew_touchdown_impact`、`rew_action_accel`、`rew_slip`、`rew_joint_velocity` 加强。
- `stable_landing_s` 曾提高到 0.60。
- landing stability 同时考虑 root angular speed。

`2026-07-09_14-46-07` 证明这组约束过强：策略学成“前脚和身体过去，后脚不动”，
level 卡 0，no-progress 成主要 reset。

随后改为平衡版：

```text
stable_landing_s: 0.40
rew_forward_velocity: 1.8
rew_hind_progress: 12.0
rew_success_event: 14.0
rew_base_motion: -0.15
rew_touchdown_impact: -0.04
rew_action_accel: -0.03
rew_termination: -8.0
```

该版本在 `2026-07-10_22-30-48` 得到当前最好的完整训练结果。

## 5. 训练入口、checkpoint 和日志

**训练关联**

- 修改前：`2026-07-08_16-30-24` 仍固定保存间隔 500，TensorBoard 后来出现写线程错误。
- 首次验证保存间隔：`2026-07-10_22-30-48` 使用 `save_interval=1000`。
- 首次验证 event writer：`2026-07-11_13-26-02`。

改动：

- `train_v3.py` 中加入 cuBLASLt/TF32 workaround，避免 `CUBLAS_STATUS_NOT_SUPPORTED`。
- `_log_resets()` 对空 `terminal_swing_clearance` 做保护，避免 `torch.quantile()` 空 tensor 崩溃。
- checkpoint/video 保存间隔随 `max_iterations` 自动计算：

```text
max_iterations < 30000: every 500 iterations
max_iterations >= 30000: every 2% total iterations
```

- 原先为了绕过 TensorBoard gfile 线程错误，临时将标量全部写入 `scalars.csv`；这导致
  `2026-07-10_22-30-48/scalars.csv` 达到约 356 MB。
- 当前改为轻量 TensorBoard event writer：

```text
events.out.tfevents...       # 默认每 10 PPO iteration 写一次 scalar
scalars_sampled.csv          # 默认每 100 PPO iteration 抽样
```

可通过命令行调整：

```bash
--scalar_log_interval 10
--csv_log_interval 100
```

## 6. 2026-07-10 最佳 Teacher 与“大跨步”问题

**训练关联**

- 触发训练：`2026-07-10_22-30-48`
- 失败验证：`2026-07-11_13-26-02`
- 再次失败验证：`2026-07-11_21-03-06`
- 当前待验证：关闭前脚 shaping 后的下一次完整 run

`2026-07-10_22-30-48` 是当前最好的完整训练：

```text
20k 左右进入 level 4
后段 success_rate 约 0.84
后段 hind_complete_rate 约 0.92
后段 fall_or_tilt 约 0.14
```

问题是视频中前脚经常直接大步跨过前方横杆，而不是先接近横杆再跨越。为此加入：

```text
front_approach_foothold: 奖励前脚先落在横杆前约 -0.12 m
front_step_overshoot:    对前脚过杆后落得过远做轻惩罚
```

初版在 `2026-07-11_13-26-02` 失败，原因是 level 0 reset 中前脚本来就经常已完成，
导致 approach 正奖励永远为 0，而 overshoot 惩罚一直生效。随后将 shaping 改为
level-gated 并减小权重：

```text
front shaping 只在 curriculum_level > 0 时启用
front_pass_landing_max: 0.38
front_pass_landing_overshoot_sigma: 0.12
rew_front_approach_foothold: 0.8
rew_front_step_overshoot: -0.2
```

`2026-07-11_21-03-06` 仍然在 50k 训练中卡在 level 0。虽然日志中
`front_approach_foothold` 和 `front_step_overshoot` 为 0，但该系列修改没有恢复到
`2026-07-10_22-30-48` 的学习速度。因此当前处置是从训练代码中移除这两个项，
恢复到 `2026-07-10_22-30-48` 的有效 reward 面。

下一次完整训练需要重点观察：

```text
Curriculum/level_mean
Crossing/success_rate
Crossing/hind_complete_rate
ResetRate/no_progress
RewardRate/hind_progress
RewardRate/termination
```

预期是：课程重新接近 `2026-07-10_22-30-48`，在 12k--20k 左右升到 level 4。`2026-07-12_13-09-54`
已验证移除前脚 shaping 后课程可以恢复，但视频中仍存在横杆离机体较远时直接大跨步的问题。

曾短暂尝试高课程前脚 reward gate，但 `2026-07-13_01-08-48` 卡在 level 0。虽然分析认为
主因更可能是同时去掉 `_RESET_MOTION` 后 level 0 bootstrap 变难，但为隔离变量，当前代码已撤销
该 gate，并恢复到 `2026-07-12_13-09-54` 的 reset 配置：

```text
reset_motion_file: sim_output_truss_0520_polynomial.npz
hind_reset_pose_probability: 0.5
joint_reset_noise: 0.02
```

同时新增只读诊断：

```text
Crossing/early_front_event_rate
early_front_event_root_distance: 0.45
```

定义：一个 episode 内只要发生过 `front_event=True` 且当时
`root_signed_distance < -0.45`，就记为 early front event。该指标只用于判断
“前脚是否在机体还没接近横杆时就过杆”，不参与 reward、success 或 curriculum。

## 7. Student Distillation 与 Bar Estimate

**训练关联**

- Teacher 来源：`2026-07-14_02-51-07/model_49999.pt`
- 纯本体 student：`2026-07-14_13-50-46_nominal`
- Oracle bar student 高随机性：`2026-07-15_13-09-10_nominal`
- Oracle bar student 低随机性：`2026-07-16_00-24-16_nominal`
- Student curriculum 修正后：`2026-07-17_14-09-48_nominal`

纯 `deployable` student 失败说明只用本体状态无法可靠判断横杆位置。加入 oracle
`bar_estimate` 后，高 `init_std=0.15` 版本仍失败，主要表现为闭环 rollout 抖动、
高 `fall_or_tilt` 和较大的 `action_accel/base_motion/touchdown_impact`。将 student
动作分布初始标准差降到 0.03，并把 distillation learning rate 降到 `3e-4` 后，
`2026-07-16_00-24-16_nominal` 已经能脱离 level 0：

```text
Crossing/success_rate tail: roughly 0.45
best batch success: roughly 0.92
Curriculum/level_mean tail: roughly 1.4
```

但该版本主要学会了近距离跨越，level 2 之后成功率明显不足。当前判断是：

- 低维横杆相对位置对 student 有帮助，oracle `bar_estimate` 路线成立。
- 失败点不再是“完全没有横杆信息”，而是 student curriculum 对 level 1/2 的过渡仍偏陡。
- 现在不应先加 DR，否则会把感知、课程和鲁棒性问题混在一起。

当时版本只对 Student env 重新切分课程距离，Teacher env 不变：

```text
student level 0: 0.00--0.15 m
student level 1: 0.15--0.30 m
student level 2: 0.30--0.50 m
student level 3: 0.50--0.80 m
student level 4: 0.80--1.20 m
student promote/demote: 0.65 / 0.25
```

该修改的目的不是直接解决最终实机感知，而是先验证 deployable proprioception 加低维
横杆估计能否完成 level 4 全阶段 crossing。通过后再把 oracle `bar_estimate` 替换为
带噪声、延迟、低通和 dropout 的估计，最后接 camera/depth estimator。

`2026-07-17_14-09-48_nominal` 验证 student 课程修正有效：

```text
level_mean tail: roughly 3.99
success_rate tail: roughly 0.67
fall_or_tilt tail: roughly 0.23
no_progress tail: roughly 0.11
```

当前新增 noisy bar-estimate 验证 task：

```text
Isaac-XY4Leg-Gripper-Truss-V3-Student-NoisyBar-Direct-v0
```

该 task 不改物理和 reward，只在 student `bar_estimate` 上加入：

```text
gaussian noise std: dx=0.03 m, dy=0.015 m, dz=0.015 m
episode bias:       dx=+-0.02 m, dy=+-0.01 m, dz=+-0.01 m
random delay:       0--2 policy steps
lowpass alpha:      0.7
dropout:            0.02 per policy step, visible=0 and hold last estimate
FOV gate:           x=-0.15--1.45, y=-0.45--0.45, z=-0.85--0.25 m
```

验证标准不是绝对成功率，而是和 oracle baseline 的差值。如果 noisy 后 success 只小幅下降，
说明 policy 对 estimator 误差有余量，可以进入 depth-derived bar estimator；如果明显掉级或
`no_progress/fall_or_tilt` 大幅上升，则先做 noisy bar_estimate fine-tune。

`model_35000.pt` 的首轮 noisy/oracle eval 结果：

```text
oracle: success 0.723, fall_or_tilt 0.223, action_saturation 0.041
full noisy: success 0.008, fall_or_tilt 0.992, action_saturation 0.388
```

这说明 full noisy 输入已经严重 OOD，不能直接进入 depth/camera estimator。当前已新增
两个 ablation task：

```text
Isaac-XY4Leg-Gripper-Truss-V3-Student-NoisyBar-NoiseOnly-Direct-v0
Isaac-XY4Leg-Gripper-Truss-V3-Student-NoisyBar-Delayed-Direct-v0
```

下一步先判断是横杆数值误差本身导致失败，还是 `visible=0` / FOV / dropout 导致失败。

ablation 结果显示 `NoiseOnly` 和 `Delayed` 均可接受：

```text
NoiseOnly: success 0.747, fall_or_tilt 0.185, visible_rate 1.0
Delayed:   success 0.753, fall_or_tilt 0.203, delay_steps roughly 1.0
```

因此当前决策是：先推进 depth-derived bar estimator，但给 policy 的 `visible` 固定为 1。
已新增：

```text
Isaac-XY4Leg-Gripper-Truss-V3-Student-DepthBar-Direct-v0
depth_bar_estimator.py
collect_depth_bar_dataset_v3.py
train_depth_bar_estimator_v3.py
```

DepthBar task 已改为挂载前后两个 64x64 depth camera，先把两路 depth 反投影到
`base_link` 坐标系，再融合成 `[height, valid_mask]` 两通道 local heightmap。
输出字段为 `heightmap` 和 metric `bar_estimate_label`。
没有 estimator checkpoint 时使用 oracle fallback，用于验证 camera/data/policy 接口；提供
TorchScript estimator 后，env 使用 depth-derived `dx/dy/dz`，并把 `visible=1` 送入 policy。

## 2026-07-19 进入 heightmap estimator 数据阶段

已完成前后 depth camera 和 base-frame fused heightmap 的可视化验证。可视化脚本输出已改为
按 step 分目录：

```text
vis_level0/step_0020/composite.png
```

并在 `meta.txt` 里记录 `heightmap_valid_bounds_m`，用于检查实际覆盖范围。

根据可视化结果，前后相机 FOV 已扩大，heightmap x 范围扩为 `[-0.85, 1.45]`，用于补
机体下方和前方横杆区域。下一步不是继续调 reward，而是采集 balanced level0--4
heightmap 数据，训练第一个 `DepthBarEstimator`：

```text
collect_depth_bar_dataset_v3.py --levels 0,1,2,3,4
train_depth_bar_estimator_v3.py --dataset dataset_levels0-4.pt
eval_v3.py --fixed_level LEVEL --depth_bar_estimator_checkpoint ...
```

采集脚本现在会保存每个样本的 `level`，训练脚本会打印 per-level 样本数。

离线分析 `dataset_levels0-4.pt` 和 `depth_bar_estimator.pt` 后发现，level4 中
`dx=0.8--1.2m` 的横杆投影有效率只有约 25%，estimator 在这段存在少量灾难性
`dx` 反号错误。为避免把不可观测状态交给 estimator，student/DepthBar 的 level4
距离范围从 `(0.80, 1.20)` 缩小为 `(0.65, 0.90)`；teacher 原始课程设置暂不变。

逐级评估 `depth_bar_estimator_l4_065-090.pt` 后，level2 明显低于相邻阶段：

```text
DepthBar level2: success roughly 0.19, fall_or_tilt roughly 0.80
Oracle fallback level2: success roughly 0.31, fall_or_tilt roughly 0.69
```

由于 oracle fallback 同样失败，问题主要不是 estimator，而是 level2 横杆距离
`(0.30, 0.50)` 与 nominal 前脚位置区域重叠。前脚初始落点大致在
`RF x=0.19--0.29m`、`LF x=0.22--0.36m`，level2 容易让横杆贴近或压到前脚附近。
为先隔离这个 reset 几何问题，student/DepthBar 的 level2 做保守调整：

```text
level2: (0.30, 0.50) -> (0.45, 0.60)
```

其他 level、reward、estimator 结构暂不改。

随后对 level2 做 oracle bar-estimate 距离扫描，每段 500 episodes：

```text
0.35--0.45: success 0.096, fall_or_tilt 0.904
0.40--0.50: success 0.138, fall_or_tilt 0.862
0.45--0.60: success 0.404, fall_or_tilt 0.596
0.50--0.65: success 0.547, fall_or_tilt 0.453
0.55--0.70: success 0.637, fall_or_tilt 0.363
0.60--0.75: success 0.719, fall_or_tilt 0.281
```

结果确认 `0.35--0.50m` 是明显的初始几何/前脚冲突区。为避免 level2 成为非单调课程
低谷，同时保持和 level3/4 的过渡，student/DepthBar 课程进一步调整为：

```text
level0: 0.00--0.15
level1: 0.15--0.30
level2: 0.55--0.70
level3: 0.60--0.80
level4: 0.65--0.90
```

新课程 oracle fixed-level 1000 episode 验证：

```text
level0: success 0.723
level1: success 0.723
level2: success 0.711
level3: success 0.806
level4: success 0.882
```

重新采集 balanced dataset 并训练 `depth_bar_estimator_l2_055-070_l3_060-080_l4_065-090.pt`
后，depth-derived fixed-level 1000 episode 验证为：

```text
level0: success 0.713, oracle gap -0.010
level1: success 0.634, oracle gap -0.089
level2: success 0.553, oracle gap -0.158
level3: success 0.711, oracle gap -0.095
level4: success 0.754, oracle gap -0.128
```

离线 estimator L1 误差约为 `dx=2.0--2.2cm`、`dy=0.4cm`、`dz=1.6cm`，
没有单个 level 的明显拟合崩溃。当前主要差距来自 oracle student 对 depth-derived
bar estimate 的闭环敏感性，下一步应从 oracle student checkpoint 继续做 DepthBar
distillation/fine-tune，而不是继续单独扩大 estimator 数据。

`2026-07-19_18-06-50_ft_depth_from_oracle35000_l2_055-070_l3_060-080_l4_065-090`
从 oracle student `model_35000.pt` resume，在 DepthBar env 中使用同一个
TorchScript estimator 做闭环 distillation/fine-tune。训练保存到 `model_51600.pt`。

训练内 sampled scalar 后段表现：

```text
Curriculum/level_mean tail:        4.000
Crossing/success_rate tail:        0.750
Crossing/success_rate last:        0.917
Crossing/hind_complete_rate tail:  0.921
Crossing/root_complete_rate tail:  0.945
ResetRate/fall_or_tilt tail:       0.156
ResetRate/no_progress tail:        0.094
Landing/max_stable_fraction tail:  0.878
Diagnostics/action_saturation tail: 0.036
Loss/behavior tail:                0.00042
```

这说明相机/heightmap estimator/student policy 的基本闭环流程已经走通。训练内
level4 表现接近 oracle/teacher 的动作模式，但 `model_51600.pt` 还没有单独跑
fixed-level 1000 episode eval，因此当前结论仍以训练内统计和视频观感为主。

## 当前建议

- 当前最可靠的 teacher checkpoint 来自：

```text
logs/rsl_rl/xy4leg_truss_v3_teacher/2026-07-14_02-51-07/model_49999.pt
```

- 当前 DepthBar student fine-tune checkpoint：

```text
logs/rsl_rl/xy4leg_truss_v3_student_depth_bar/2026-07-19_18-06-50_ft_depth_from_oracle35000_l2_055-070_l3_060-080_l4_065-090/model_51600.pt
```

- fine-tune 起点仍是 oracle student `model_35000.pt`，可作为回退基线。
- 当前 depth estimator 使用：

```text
logs/rsl_rl/xy4leg_truss_v3_depth_bar/depth_bar_estimator_l2_055-070_l3_060-080_l4_065-090.pt
```

- 下一步先对 `model_51600.pt` 做 fixed-level 1000 episode eval；通过后再加入
  depth estimate 的轻量噪声/延迟和小范围 DR。

## 2026-07-21 Learned Gripper Teacher

新增 learned-gripper walk/teacher 任务后，gripper action interface 与无夹爪 teacher
分离：

```text
Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Walk-Direct-v0
Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Teacher-Direct-v0
```

learned gripper 版本使用 24D action：

```text
20D leg joints + 4D gripper actions
```

每个 gripper action 控制同一夹爪两个 USD 关节的共同开合，匹配真实结构中
“一个电机带动一个夹爪两个关节”的控制方式。非 gripper 任务仍保持原 20D action，
不受影响。

`xy4leg_gripper_truss_v3_walk/2026-07-20_11-59-48_learn_gripper/model_49999.pt`
完成 learned-gripper walk 预训练，后段 walk 稳定，`mean_episode_length` 接近满长，
`fall_or_tilt` 约 `0.0025`，作为 gripper teacher 初始化来源。

`xy4leg_gripper_truss_v3_teacher/2026-07-21_01-22-13_learn_gripper` 从 learned-gripper
walk actor 初始化训练 gripper teacher。训练到 `model_32000.pt` 中断时已经进入 level4，
后段统计约为：

```text
success_rate tail:       ~0.87
fall_or_tilt tail:       ~0.13
mean_forward_speed tail: ~0.19 m/s
```

续训 `2026-07-21_13-40-58_learn_gripper` 使用 student/depth-bar 修正后的课程：

```text
level0: 0.00--0.15
level1: 0.15--0.30
level2: 0.55--0.70
level3: 0.60--0.80
level4: 0.65--0.90
```

`model_33440.pt` 的逐 level play 观感相对稳定，但继续训练到后段后，策略逐渐偏向
更快、更激进的跨越。`2026-07-21_13-40-58` 后段约为：

```text
success_rate tail:       ~0.863
fall_or_tilt tail:       ~0.135
mean_forward_speed tail: ~0.310 m/s
episode_duration tail:   ~3.79 s
```

为了限制继续训练后向“快速跳跃式跨越”漂移，随后增加轻量稳定化项：

- `forward_speed_limit`：限制过高前向速度，避免通过速度奖励换取跳跃式跨越。
- `gripper_phase`：鼓励夹爪开合与步态 swing/stance 相位一致。
- 加强 `touchdown_impact`、`action_accel`、`slip` 等稳定性惩罚。

`2026-07-21_16-34-08_learn_gripper` 从 `model_33440.pt` 续训。该版本确认速度惩罚
有效，但继续训练仍会逐渐变快。后段统计为：

```text
success_rate tail:       ~0.860
fall_or_tilt tail:       ~0.140
mean_forward_speed tail: ~0.273 m/s
episode_duration tail:   ~4.63 s
```

该 run 中较好的 checkpoint：

```text
model_34000.pt: success=1.000, fall=0.000, speed=0.185, duration=5.87
model_42000.pt: success=0.986, fall=0.014, speed=0.251, duration=4.91
```

其中 `model_34000.pt` 更保守，作为后续“减少足端碰横杆”实验的回退起点。由于视频中
仍可能出现足端擦碰横杆，新增 `foot_bar_contact` 诊断和软惩罚：

- 使用足端/夹爪末端接触传感器，在横杆附近空间窗口内检测接触。
- 记录 `Diagnostics/foot_bar_contact_rate`、
  `Diagnostics/foot_bar_contact_terminal_rate` 和
  `Diagnostics/foot_bar_contact_force`。
- 仅在 `XY4LegGripperTrussV3TeacherEnvCfg` 中启用
  `rew_foot_bar_contact=-2.0`，不改变成功判据和终止条件。

`2026-07-21_18-59-55_learn_gripper` 从 `model_34000.pt` 继续训练 5000
iterations。level4 没有退化，速度保持较低，但 `model_37500.pt` 的 level4 play
观感一般，足端/夹爪开合仍有贴杆风险。后段统计为：

```text
success_rate tail:                  ~0.855
fall_or_tilt tail:                  ~0.140
mean_forward_speed tail:            ~0.232 m/s
foot_bar_contact_rate tail:         ~0.217
foot_bar_contact_terminal tail:     ~0.435
foot_bar_contact_force tail:        ~10.35
```

由于 `foot_bar_contact` 指标基于 net contact force + 空间窗口，可能把横杆附近正常踩
rail 误报为碰 crossbar。随后将 `foot_bar_contact` 降级为诊断项，把训练主约束改为
几何式 `low_bar_clearance`：

- 只在足端接近横杆 x 窗口、处于期望 swing、且向前跨越时生效。
- 当足端相对 rail top 的 clearance 低于 `0.08m` 时按缺口平方扣分。
- 新增 `RewardRate/low_bar_clearance`、
  `Diagnostics/low_bar_clearance_rate` 和
  `Diagnostics/low_bar_clearance_deficit`。
- `XY4LegGripperTrussV3TeacherEnvCfg` 使用 `rew_low_bar_clearance=-1.5`，
  `rew_foot_bar_contact=0.0`。

`2026-07-21_22-36-26_learn_gripper` 再次从 `model_34000.pt` 续训，启用
`low_bar_clearance`。该版本相比 `18-59-55` 有明显改善：

```text
success_rate tail:                  0.910
fall_or_tilt tail:                  0.090
mean_forward_speed tail:            0.242 m/s
landing_stable_fraction tail:       0.913
action_saturation tail:             0.027
low_bar_clearance_rate tail:        0.0106
low_bar_clearance_deficit tail:     0.0050 m
foot_bar_contact_rate tail:         0.172
foot_bar_contact_force tail:        2.77
```

checkpoint 对比：

```text
model_36000.pt:
  success_rate        0.942
  fall_or_tilt        0.058
  mean_forward_speed  0.222
  low_clear_rate      0.010
  touchdown_impact   -0.075

model_38000.pt:
  success_rate        0.958
  fall_or_tilt        0.042
  mean_forward_speed  0.239
  low_clear_rate      0.020
  touchdown_impact   -0.088
```

人工 play 对比后，`model_36000.pt` 比 `model_38000.pt` 更保守，速度更低、
touchdown impact 更小，作为 learned-gripper student 的 teacher checkpoint。

新增独立 oracle-bar student 任务：

```text
Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Student-OracleBar-Direct-v0
```

该任务保持 gripper teacher 的 24D action interface，student 输入为
`deployable + oracle bar_estimate`，teacher 输入为完整 `policy` observation。
默认 teacher checkpoint：

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_teacher/2026-07-21_22-36-26_learn_gripper/model_36000.pt
```

`xy4leg_gripper_truss_v3_student_oracle_bar/2026-07-22_12-50-45_teacher36000`
训练到 `model_11000.pt` 后中断。behavior loss 在 1000 iterations 后已降到
`1e-3` 以下，说明 oracle-bar distillation 很快收敛；实际 crossing 指标存在采样波动，
不建议只用最后 checkpoint。较好的候选为：

```text
model_4000.pt:
  success_rate        0.938
  fall_or_tilt        0.063
  mean_forward_speed  0.200
  low_clear_rate      0.006

model_10000.pt:
  success_rate        0.938
  fall_or_tilt        0.063
  mean_forward_speed  0.207
  low_clear_rate      0.018
```

训练视频中 `model_4000/10000/11000` 动作均无明显异常。为推进相机闭环，新增 learned-gripper
DepthBar student 任务：

```text
Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Student-DepthBar-Direct-v0
```

该任务继承 learned-gripper oracle student 的 24D action interface 和
`deployable + bar_estimate` student 输入结构，只是 `bar_estimate` 由 depth/heightmap
estimator 产生。第一版先从 oracle student `model_10000.pt` resume，复用已有
estimator：

```text
logs/rsl_rl/xy4leg_truss_v3_depth_bar/depth_bar_estimator_l2_055-070_l3_060-080_l4_065-090.pt
```

## 2026-07-23 Gripper DepthBar Plateau And Near-Level Adjustment

`xy4leg_gripper_truss_v3_student_depth_bar/2026-07-22_15-49-26_from_oracle10000_teacher36000`
从 oracle student `model_10000.pt` resume，训练到 `model_59999.pt`。整体表现从早期到
5w 轮基本处在同一平台，继续增加迭代没有稳定提升：

```text
10000--20000: success 0.825, fall_or_tilt 0.175, behavior_loss 0.000856
20000--30000: success 0.836, fall_or_tilt 0.162, behavior_loss 0.000730
30000--40000: success 0.850, fall_or_tilt 0.149, behavior_loss 0.000690
40000--50000: success 0.833, fall_or_tilt 0.164, behavior_loss 0.000675
50000--60000: success 0.833, fall_or_tilt 0.164, behavior_loss 0.000666
```

按 success、fall、action saturation、low-bar-clearance 和 behavior loss 综合排序，
当前首选 checkpoint 为：

```text
model_30000.pt:
  success_rate        0.958
  fall_or_tilt        0.042
  mean_forward_speed  0.206
  action_saturation   0.023
  low_clear_rate      0.0047
```

`model_37000.pt` 和 `model_47000.pt` 也达到过 `0.958` success，但 speed/low-clearance
或视觉 rollout 表现略差；最终 `model_59999.pt` 不作为最优模型。

为改善 depth-bar 下 level4 横杆过远导致的相机可观测性和大跨步倾向，同时避开历史扫描中
确认的前脚初始化冲突区 `0.35--0.50m`，只调整 learned-gripper student/oracle/depth-bar
课程，暂不改 gripper teacher 原始课程：

```text
level0: 0.00--0.15
level1: 0.15--0.30
level2: 0.55--0.65
level3: 0.60--0.70
level4: 0.65--0.75
```

随后对 level2 进一步按 0.05 m 区间评估：

```text
0.55--0.60: success 0.767, fall_or_tilt 0.228
0.60--0.65: success 0.785, fall_or_tilt 0.213
0.65--0.70: success 0.789, fall_or_tilt 0.208
0.70--0.75: success 0.830, fall_or_tilt 0.168
```

结果显示 x 越大越稳定，但 `Heightmap/valid_rate` 没有随 x 增大而提升，因此主要问题不是
远处相机可见性，而是训练分布。gripper oracle/depth student 从很早开始几乎全部处于
level4，level2/3 样本不足；而 `0.55--0.65m` 又更接近前脚/机体初始化区域，动作相位更难。

为让各阶段距离持续得到训练，新增 balanced curriculum 机制。默认关闭；gripper oracle/depth
student 启用：

```text
level weights: 0.05, 0.05, 0.30, 0.30, 0.30
level0: 0.00--0.15
level1: 0.15--0.30
level2: 0.55--0.65
level3: 0.60--0.70
level4: 0.65--0.75
```

balanced curriculum 训练时跳过自动 promote/demote，每个 reset 按权重采样下一 episode 的
逻辑 level 和横杆距离；fixed-level eval 会通过关闭 `curriculum_enabled` 保持指定 level。

`xy4leg_gripper_truss_v3_student_depth_bar/2026-07-23_14-31-20_balanced_levels_from_model30000`
从 `model_30000.pt` 开始继续训练，并完整跑到 `model_49999.pt`。后 20% sampled
统计显示总体 success 与上一版接近，但 level 分布明显更均衡：

```text
success_rate tail avg:              0.830
fall_or_tilt tail avg:              0.160
no_progress tail avg:               0.006
level_mean tail avg:                2.73
landing_stable_fraction tail avg:   0.835
behavior_loss tail avg:             0.00070
```

后 20% 按 episode 加权的 per-level 结果：

```text
level0: episodes 365,  success 0.953
level1: episodes 340,  success 0.818
level2: episodes 2164, success 0.786
level3: episodes 2364, success 0.805
level4: episodes 2121, success 0.821
```

这说明 balanced sampler 达到了目的：不再长期集中在 level4，level2/3/4 都有持续样本。
整体 success 没有显著超过上一版，但训练覆盖更接近后续 fixed-level eval 和实机部署需要。

按 sampled success、fall/no-progress、level_mean 和 reward 粗排，候选 checkpoint 为：

```text
model_42000.pt:
  success 0.944, fall 0.056, level_mean 2.80
  L2/L3/L4 sampled success: 0.875 / 1.000 / 0.900

model_45000.pt:
  success 0.896, fall 0.104, level_mean 2.81
  L2/L3/L4 sampled success: 0.667 / 1.000 / 0.788

model_48500.pt:
  success 0.906, fall 0.094, level_mean 2.96
  L2/L3/L4 sampled success: 0.744 / 0.684 / 0.758

model_49999.pt:
  final checkpoint; sampled success 0.771, fall 0.229
  保留为完整训练终点对照，不直接视为 best
```

`model_42000.pt` 单点指标最好，但该 sampled batch 中 level4 episode 数较少；实际 best
仍需对 `42000/45000/48500/49999` 做 fixed-level eval 后决定。

train-time `Curriculum/level_n_success_rate` 日志修正为当前 PPO iteration 内的加权统计：
`sum(level_n_success_count) / sum(level_n_episodes)`。同时保留
`Curriculum/level_n_episodes` 和 `Curriculum/level_n_success_count`，避免 RSL-RL logger
对多个 reset batch 的 success rate 做非加权平均。

## 2026-07-24/25 Walk-Sim2Real Interface And Crossing Route Correction

### Sim2Real 前的 Gripper DepthBar 是怎么训练的

在进入 sim2real 前，`Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Student-DepthBar-Direct-v0`
不是直接从 teacher PPO 开始训练，而是沿用了已经验证过的三段流程：

```text
1. learned-gripper privileged teacher
2. learned-gripper oracle-bar student
3. learned-gripper depth-bar student
```

具体来源为：

```text
Teacher checkpoint:
logs/rsl_rl/xy4leg_gripper_truss_v3_teacher/2026-07-21_22-36-26_learn_gripper/model_36000.pt

OracleBar student run:
logs/rsl_rl/xy4leg_gripper_truss_v3_student_oracle_bar/2026-07-22_12-50-45_teacher36000

OracleBar student checkpoint used for DepthBar resume:
model_10000.pt

Depth estimator:
logs/rsl_rl/xy4leg_truss_v3_depth_bar/depth_bar_estimator_l2_055-070_l3_060-080_l4_065-090.pt
```

当时的 learned-gripper DepthBar student 输入仍是旧 deployable interface：

```text
deployable = 72D
bar_estimate = 4D
student actor obs = 76D
teacher obs = policy privileged obs
action = 24D = 20D leg + 4D gripper
```

关键点：

- `deployable` 中还没有夹爪关节角度。
- 没有 walk-sim2real 后来加入的 `gripper_angle_rad` 观测。
- 没有 sim2real 专用的更强 DR、夹爪 latch、慢步频和更低速度限制。
- `bar_estimate` 由 depth/heightmap estimator 输出，`visible` 固定为 1。
- 训练方式是从 oracle-bar student checkpoint resume，而不是重新随机初始化：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/train_v3.py \
  --task Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Student-DepthBar-Direct-v0 \
  --resume \
  --checkpoint logs/rsl_rl/xy4leg_gripper_truss_v3_student_oracle_bar/2026-07-22_12-50-45_teacher36000/model_10000.pt \
  --depth_bar_estimator_checkpoint logs/rsl_rl/xy4leg_truss_v3_depth_bar/depth_bar_estimator_l2_055-070_l3_060-080_l4_065-090.pt \
  --run_name from_oracle10000_teacher36000 \
  --max_iterations 50000 \
  --num_envs 128 \
  --headless
```

由于 resume 起点是 `model_10000.pt`，完整训练保存到了 `model_59999.pt`。后续
`2026-07-23_14-31-20_balanced_levels_from_model30000` 又从该 DepthBar run 的
`model_30000.pt` 继续，启用 balanced level sampling。

### Walk-Sim2Real 做了哪些改动

新增任务：

```text
Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Walk-Sim2Real-Direct-v0
```

对应 runner：

```text
experiment_name = xy4leg_gripper_truss_v3_walk_sim2real
obs_groups = {"actor": ["deployable", "bar_estimate"], "critic": ["critic"]}
```

这里的“80D”应严格理解为 actor 输入 80D，不是 `deployable` 单独 80D：

```text
deployable = 76D
bar_estimate = 4D
actor obs = 80D
```

`deployable` 的 76D 顺序为：

```text
0:20    joint_pos_norm, 20D
20:40   joint_vel_norm, 20D
40:43   projected_gravity_b, 3D
43:46   root_ang_vel_b / 5, 3D
46:48   gait_clock, 2D
48:72   previous raw action, 24D
72:76   gripper_angle_rad, 4D
```

`gripper_angle_rad` 是同一夹爪两个 USD 关节位置的平均角度，单位 rad，直接 clamp 到
仿真开合角度范围；不是 `0..1` 开闭状态，也不包含夹爪速度。夹爪速度没有加入，
原因是实机反馈噪声较大。

walk-only 阶段没有真实横杆，`bar_estimate` 使用 dummy value：

```text
[2.0, 0.0, -1.0, 1.0]
```

它的意义是“横杆很远且可见”。后续 crossing/depth-bar sim2real 会把这个 4D 通道换成：

```text
[bar_dx_b / 1.2, bar_dy_b / 0.3, bar_dz_b / 0.5, visible]
```

walk-sim2real 还加入了以下硬件相关改动：

```text
include_gripper_position_obs = true
pretrained_actor_copy_obs_dim = 72
domain_randomization_enabled = true
max_action_delay_steps = 3
joint_position_bias_range = 0.02
joint_position_noise_std = 0.015
joint_velocity_noise_std = 0.08
gait_frequency = 0.35
forward_velocity_min/max = 0.025 / 0.06
forward_speed_limit = 0.10
action_soft_limit = 0.55
gripper_action_step = 0.4 deg / policy step
gripper_action_latch_min_hold_s = 0.8
gripper_action_latch_deadband = 0.10
```

Sim2Real DR 相比普通 DR 更宽：

```text
static friction: 0.6--1.4
dynamic friction: 0.5--1.3
base mass add: -0.8--0.8 kg
leg actuator stiffness scale: 0.75--1.25
leg actuator damping scale: 0.7--1.3
```

reward 也向实机稳定性偏移：

```text
rew_forward_velocity = 1.2
rew_forward_speed_limit = -0.15
rew_touchdown_impact = -0.08
rew_base_motion = -1.0
rew_action_magnitude = -0.10
rew_action_soft_limit = -2.5
rew_action_rate = -0.16
rew_action_accel = -0.12
rew_joint_velocity = -0.008
rew_energy = -0.0008
rew_slip = -1.0
rew_gripper_phase = -0.15
```

夹爪 policy 输出没有改成离散动作，仍然是连续 `[-1, 1]`。训练和实机 play 的约定是：

```text
policy raw gripper action
-> deadband
-> sign {-1, +1}
-> latch/min_hold
-> torque sign/scale
```

同时，obs 里的 previous action 必须回填 policy 原始输出，不能回填 latch 后的低层命令。

`xy4leg_gripper_truss_v3_walk_sim2real/2026-07-24_18-58-21_deployable_interface`
是当前 walk-sim2real 推荐 run。后 5k 训练统计：

```text
ResetRate/time_out avg:          1.000
ResetRate/fall_or_tilt avg:      0.000
Diagnostics/action_saturation:   0.000
Walk/episode_duration avg:       19.96 s
Walk/forward_distance avg:       1.27 m
```

当前推荐 checkpoint：

```text
logs/rsl_rl/xy4leg_gripper_truss_v3_walk_sim2real/2026-07-24_18-58-21_deployable_interface/model_28800.pt
```

### Crossing Sim2Real 路线修正

`2026-07-25_15-55-43_deployable_from_walksim_model28800` 是错误路线验证：

```text
Teacher PPO actor obs = deployable + bar_estimate
```

这会把 teacher 变成一个直接用 oracle bar 的 deployable PPO policy，和 oracle-bar
student 的最终接口过于接近，失去了 privileged teacher 的意义。该 run 后段仍卡在 level0：

```text
Crossing/success_rate avg:   0.000
Curriculum/level_mean avg:   0.000
ResetRate/no_progress avg:   0.635
ResetRate/no_progress last:  0.875
```

因此不应从该 checkpoint 续训。当前修正后的 crossing sim2real 路线是：

```text
1. Privileged Teacher Sim2Real
   actor obs = policy = 76D deployable + 16D privileged = 92D
   critic obs = critic
   initialize from gripper teacher model_36000.pt
   map old 88D policy -> new 92D policy by inserting 4D gripper_angle_rad obs

2. Oracle-Bar Student Sim2Real
   student obs = deployable + oracle bar_estimate = 80D
   teacher obs = policy = 92D privileged

3. Depth-Bar Student Sim2Real
   student obs = deployable + depth-derived bar_estimate = 80D
   teacher still uses Stage 1 privileged teacher checkpoint
```

crossing sim2real 继承 walk-sim2real 的硬件接口和 DR，但 crossing teacher 的速度阈值略放宽：

```text
gait_frequency = 0.35
forward_velocity_min/max = 0.025 / 0.07
forward_speed_limit = 0.14
gripper_action_latch_min_hold_s = 0.8
include_gripper_position_obs = true
deployable_observation_dim = 76
actor obs for final students = 80D
```

`2026-07-26_13-04-34_privileged_from_walksim_model28800` 验证了“虽然 teacher 已经是
privileged，但从 walk-sim2real 初始化仍不够”。到 33.5k 时：

```text
Crossing/front_complete_rate:  ~1.0
Crossing/root_complete_rate:   ~0.98
Crossing/hind_complete_rate:   0.0
Crossing/success_rate:         0.0
ResetRate/no_progress:         ~0.7--0.9
Walk/mean_forward_speed:       ~0.01 m/s
```

因此后续改为从已经学会 crossing 的 learned-gripper teacher checkpoint 做 sim2real
finetune，而不是让 walk-only prior 在带 DR/latch/慢步频的环境里重新探索 hind crossing。
`train_v3.py` 已新增 88D -> 92D adapter：

```text
source 0:72  deployable without gripper angle -> target 0:72
target 72:76 new gripper_angle_rad             -> initialized neutral
source 72:88 privileged                        -> target 76:92
```

完整命令和部署路线另见：

```text
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/CROSSING_SIM2REAL_PLAN.md
```
