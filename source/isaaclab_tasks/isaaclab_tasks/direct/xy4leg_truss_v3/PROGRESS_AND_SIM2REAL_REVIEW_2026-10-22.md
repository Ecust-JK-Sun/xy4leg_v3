# XY4Leg Truss V3 进展与 Sim2Real 审查（2026-10-22）

**结论**

本次模型选择只比较 crossing，排除 Walk。V3 已打通 privileged Teacher → OracleBar Student → DepthBar Student 的训练和早期实机尝试。当前应分别保留“实机初测基线”“低频安全 Teacher 候选”“OracleBar 控制器上限”，不能用一个最新 checkpoint 代表全部最优进展。

低频安全 Teacher 优先复评 **2026-09-08 02:11:53 安全分支的 `model_96000.pt` / `model_96800.pt`**。更新的 SmoothSafe 分支虽然普通成功率较高，安全成功率明显退化。部署入口仍默认加载 2026-08-05 04:55:55 的 DepthBar `model_68500.pt`。

下文“问题 1–12”和原 JSON 是修复前的审查快照。本轮已修复配置漂移、异常动作、无效反馈、启动竞争和感知 NaN 等问题，具体见 [修复记录与当前使用方式](sim2real_v3/SIM2REAL_FIXES_2026-10-22.md)。按用户要求，没有添加安全停机动作。全程未连接或使能真实硬件。

后续重点检查通信和推理频率，已完成非阻塞反馈、独立相机采集及实际 Hz/延迟统计；结果和测量边界见 [通信与频率检查](sim2real_v3/COMMUNICATION_TIMING_REVIEW_2026-10-22.md)。当前 `68500` 的策略目标是 25 Hz，仍需在实机记录验收。

**证据和指标口径**

- 复核本地 34 个相关 run 的 `scalars_sampled.csv`、保存的 `params/env.yaml` / `agent.yaml`、关键 checkpoint，以及当前训练与部署源码。此表不是对全部历史实验的穷举。
- 下表“末段”为 `[最后记录 iteration - 5000, 最后记录 iteration]` 内抽样标量的算术均值，**不是独立评估，也不是按 episode 数加权的成功率**。不足 5000 轮的短 run 使用已有窗口。完整窗口和样本数见 [JSON 证据](sim2real_v3/REVIEW_EVIDENCE_2026-10-22.json)。
- checkpoint 候选窗口使用 `[checkpoint iteration - 3000, checkpoint iteration]`；模型保存点及其邻近训练窗口的成绩不能当作该模型确定性推理的评测成绩。
- 不同 run 的课程距离、DR、感知输入、初始姿态不同，不应直接跨组排序。
- 当前 `safe_success` 要求完成 crossing，且整个 episode 未发生代码判定的足端碰杆或几何包络违规。`foot_bar_contact_rate` / `foot_bar_envelope_rate` 是 episode 内违规步数比例的聚合，不能读成“多少比例的 episode 碰过杆”。碰杆判定使用邻近横杆的接触组总力，仍是代理指标，可能混入轨道接触，见 [状态计算](truss_v3_env.py#L1005) 和 [安全成功定义](truss_v3_env.py#L1251)。

**目前值得保留的成果**

训练开始时间取 run 目录名，均为北京时间（Asia/Shanghai）；日志末次时间取 CSV 的 `wall_time`，不等于精确训练结束或该 checkpoint 的保存时间。续训 run 的时间仅表示本轮，不包含前序训练耗时。

| 方向 | 训练 run 开始时间 | 日志最后记录时间 | 推荐保留／复评的模型 | 已有证据与用途 |
| --- | --- | --- | --- | --- |
| 历史 learned-gripper DepthBar | **2026-07-22 15:49:26** | 2026-07-23 12:29:42 | `model_30000.pt` | 固定 L2 / L4 各 1000 episodes，成功率 **77.3% / 84.6%**；76D、前后相机 heightmap 历史基线，与当前 80D 单前相机接口不同 |
| 当前默认 RealCam DepthBar | **2026-08-05 04:55:55** | 2026-08-05 13:21:19 | `model_68500.pt` | run 末段成功率 **79.03%**、fall/tilt **16.21%**；有实机动作可用及夹爪、感知问题的反馈，属于当前部署基线 |
| Noisy OracleBar PPO | **2026-08-28 03:08:45** | 2026-08-28 10:54:05 | `model_27000.pt` / `model_27600.pt` | run 末段普通／安全成功 **91.62% / 79.28%**，速度 **0.2292 m/s**；使用带噪横杆真值、课程距离 0.50–0.55 m，是 80D 控制器的仿真参考 |
| 低频安全 Teacher | **2026-09-08 02:11:53** | 2026-09-08 10:51:09 | **`model_96000.pt` / `model_96800.pt`** | run 末段普通／安全成功 **95.92% / 72.71%**，速度 **0.0883 m/s**；当前低频安全候选，但为 92D 特权 actor，仍需蒸馏 |

较新的 SmoothSafe run 开始于 **2026-09-08 13:26:12**，其末段普通／安全成功为 **97.84% / 0.41%**、接触步数比例 **32.42%**，本轮不将它列为安全跨越最优模型。

09-08 凌晨安全分支的候选比较：

| checkpoint | 前 3000 轮窗口普通成功 | 安全成功 | 平均速度 | 接触步数比例 | 包络违规步数比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `model_96000.pt` | 97.25% | **79.33%** | 0.0920 m/s | 0.92% | 0.37% |
| `model_96800.pt` | 96.93% | **79.25%** | 0.0926 m/s | 0.85% | 0.35% |

两者都已确认存在，但仍需使用各自训练配置做固定课程、多 seed、确定性 actor 评估。当前速度仍高于低频路线希望达到的 0.015–0.050 m/s，不能把“参考相位 0.20 Hz”当成“实际动作已经足够慢”。

另一个重要退化：DepthBar 在 08-12 晚间从 `145800` 继续的 run，末段成功率约 **0.92%**；08-14 run 约 **0.37%**、fall/tilt 约 **99.63%**。`model_175799.pt` 等更大编号不能因此取代旧部署基线。现有证据不足以将这些退化归因到某一个参数。

关键产物路径（相对仓库根目录）：

```text
# 历史 heightmap DepthBar 基线
logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar/
  2026-07-22_15-49-26_from_oracle10000_teacher36000/model_30000.pt

# 默认实机 DepthBar 基线和对应感知模型
logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar_sim2real/
  2026-08-05_04-55-55_realcam_crop_noise_l2far_render12hz_from42000/model_68500.pt
logs/rsl_rl/xy4leg_gripper_truss_v3_depth_crop_realcam/
  depth_crop_estimator_realcam_xmax060.pt

# 低频安全 Teacher 候选
logs/rsl_rl/xy4leg_gripper_truss_v3_teacher_lowfreq_nominal_safe_ft/
  2026-09-08_02-11-53_from69000_stable_frontsafe_pitchassist/model_96000.pt
  2026-09-08_02-11-53_from69000_stable_frontsafe_pitchassist/model_96800.pt

# Oracle 控制器参考
logs/rsl_rl/xy4leg_gripper_truss_v3_student_oracle_bar_sim2real_ppo/
  2026-08-28_03-08-45_ppo_ft_from7800_safe3cm_linear_fastgate/model_27000.pt
```

**问题 1：P0，旧指令可持续重发，退出没有明确安全停机动作**

位置：[发送线程](sim2real_v3/serial_threads.py#L59)、[共享状态](sim2real_v3/threads_shared.py#L11)、[退出流程](sim2real_v3/play_real.py#L1071)、[MotorSession.close](sim2real_v3/motor_session_gripper.py#L105)。

`SharedData` 只存 action，不存时间或序号；TX 线程只看固定的 `motor_enabled=True`，持续重发最后一次动作。相机等待、模型推理或主循环阻塞时，腿部旧位置和夹爪旧力矩仍可持续发送。离线 mock 中仅发布一次动作，已观察到 5 次重复发送；代码没有过期分支。

正常退出和异常退出只停止线程并关闭串口，没有发出机器人设计所要求的安全保持／夹紧／失能动作。关闭串口后的实际力矩取决于电机固件超时设置，当前代码未验证这些设置。`SIGINT` / `SIGTERM` 也只设置主循环标志，不能立即打断当前相机调用或启动流程。

建议增加独立的指令 watchdog、线程故障上报和硬件通信超时；将安全停机实现成明确状态转换。对于承载或依靠夹爪固定的机构，停机时是保持还是失能必须按实际机械约束设计，不能简单全电机置零。

**问题 2：P1，旧 checkpoint 使用当前配置，默认部署输入已漂移**

位置：[加载当前注册配置和模型](sim2real_v3/play_real.py#L760)、[读取当前 action / bar 配置](sim2real_v3/play_real.py#L821)、[相位构造](sim2real_v3/play_real.py#L968)。脚本没有恢复 checkpoint 同目录的 `params/env.yaml` 和 `params/agent.yaml`。

| 模型 | 训练保存值 | 当前部署值 |
| --- | --- | --- |
| 默认 DepthBar `68500` 步频 | 0.30 Hz | 当前任务类 0.18 Hz |
| 默认 DepthBar `68500` bar dy / dz scale | 0.30 / 0.50 | 当前任务类 0.60 / 1.00 |
| DepthBar `145800` 步频 | 0.22 Hz | 当前任务类 0.18 Hz |

这不是 shape 检查能发现的错误：旧模型仍接收 80D，y/z 却减半。读取默认 `68500` 的 normalizer 后，在训练均值处换用当前 dz scale，z 通道约偏移 **7.32 个归一化标准差单位**（按本地 normalizer 的 `std + 0.01` 计算）。这足以显著改变网络输入分布。

应从模型附带的部署契约恢复步频、观测顺序、scale、action 限位、normalizer 和 latch；实机有意调整的参数应显式记录。当前低通 alpha 另由 CLI 默认 0.30 控制，不能误认为会随任务 cfg 一起恢复。步频、scale 与模型不兼容时应在使能前拒绝运行。

**问题 3：P1，缺失、过期或故障反馈可被当成正常观测**

位置：[反馈缓存](sim2real_v3/motor_session_gripper.py#L86)、[初始零缓存](sim2real_v3/motor_session_gripper.py#L129)、[观测线程](sim2real_v3/serial_threads.py#L23)、[雷达数据](sim2real_v3/radar_receiver.py#L25)、[主循环默认值](sim2real_v3/play_real.py#L953)。

串口未读到某电机时，缓存保留旧值或初始零值；观测线程仍将其作为新的一整帧发布。没有逐电机接收时间、首次收到标志、完整帧判定；反馈中的故障码和温度也被丢弃。启用雷达后没有首帧就绪和超时检查；timestamp=0 的旧数据在 mock 中仍被接受。未启用雷达时则直接使用水平重力和零角速度。

此外，DepthBar 未开相机只给 warning，仍可带 `--enable_motors` 使用“横杆在远处”的 dummy 输入。实机闭环也没有像训练 `_get_dones()` 那样检查倾角或异常角速度。

建议在使能前要求所有必要传感器就绪；运行中按逐电机／姿态／相机时效和故障状态停止策略推进。调试时的默认值应有明确模式，不能自动成为带电运行的有效观测。

**问题 4：P1，启动确认和串口读取存在竞争**

位置：[启动顺序](sim2real_v3/play_real.py#L916)、[enable_all](sim2real_v3/motor_session_gripper.py#L27)、[send_targets](sim2real_v3/motor_session_gripper.py#L47)、[enable_motor](sim2real_v3/motor_control.py#L69)。

脚本先启动反馈线程，再调用 `enable_all()`；后者通过 `enable_motor()` 再次读取同一个串口，形成两个接收者。`enable_all()` 忽略返回反馈和状态码，所有返回 None 时仍会设为 enabled=True；异常时设为 False，主循环也没有据此拒绝启动 TX。`send_targets()` 自身不检查 session.enabled，mock 已复现 enabled=False 时发送全部 24 路命令。

启动还会直接发送 nominal target，未验证各电机当前位置与目标的差值，也没有在此入口执行姿态渐变。应保证一个串口只有一个接收者，按电机 ID 验证使能与反馈，完成姿态就绪后再进入策略状态。

**问题 5：P1，动作历史没有与训练保持相同的限幅和异常处理**

位置：[动作后处理](sim2real_v3/play_real.py#L1017)、[写 previous_action](sim2real_v3/play_real.py#L1060)、[训练侧限幅](truss_v3_env.py#L744)。

真实端只在 `_policy_action_to_hardware_command()` 内做 `nan_to_num` 和 `[-1,1]` 限幅，`previous_action` 记录的仍是原始网络输出。训练侧记录的 `_raw_actions` 已限幅。因此，输出 1.8 时，实机执行 1.0，但下一步观测记住 1.8；NaN 也会进入下一次动作历史。实测 mock 中两条路径均成立，非有限关节反馈也可直接进入观测。

应在推理出口统一校验 finite、维度和限幅，再把相同的策略动作写入历史并用于平滑／执行。遇到非有限输出应触发故障处理，不能只把它转换成一个继续执行的正常动作。额外平滑后的执行动作与策略历史应按训练定义处理。

**问题 6：P1，感知 NaN 可被判成“横杆可见”，训练感知链还使用真值辅助**

位置：[真实端 estimator 与 visibility](sim2real_v3/play_real.py#L590)、[训练 visibility](truss_v3_env.py#L1531)、[reset 感知状态](truss_v3_env.py#L1634)。

真实端先将 estimator 的 NaN/Inf 转成 `[0,0,0]`，再检查可见范围。零向量落在当前 x/y/z 范围内，只要有效深度比例够，就会被当成可见横杆并刷新 memory。mock 已复现“estimator 全 NaN，但 raw_visible=True、visible=1”。应在替换前校验 estimator 原始输出；无效输出只能进入带期限的失检路径。

训练端 `raw_visible` 使用真实 `bar_delta_b`，reset 时还以 GT 初始化滤波和 memory；真实端依据 estimator 输出范围判断，并以过期 fallback 初始化。即使 actor 输入声明可部署，感知预处理仍有真值依赖，尤其会高估首次检测和失检恢复能力。应使用不依赖横杆真值的可见性／置信度判定，并加入冷启动、无杆、错误估计、遮挡和记忆耗尽的闭环评估。

**问题 7：P1，特权 Teacher 可以被误当成可部署 crossing 策略**

位置：[obs_groups 分流](sim2real_v3/play_real.py#L804)、[policy 常量填充](sim2real_v3/play_real.py#L699)。

脚本接受任意 `['policy']` actor，为其填入“横杆在远处、四足未跨越、stage=approach”的常量，没有检查 `walk_only`。因此可选中 92D crossing Teacher 并使能电机，但输入完全不是训练时的 crossing 状态。相机也不能补齐它所需的 privileged 信息。应将该兼容入口限制为明确支持的 Walk，crossing 部署只允许已验证的 deployable actor。

**问题 8：P1，控制周期、相位和相机等待没有解耦**

位置：[相机阻塞读取](camera/realsense_camera.py#L66)、[相位计时](sim2real_v3/play_real.py#L968)、[超周期处理](sim2real_v3/play_real.py#L1065)。

相机读取／推理在主控制线程内；超周期仅打印 warning。相位以 `step * period` 推进，而 bar memory 和 latch 用实际时间。25 Hz 目标若实际降成 12.5 Hz，步态时钟对应的物理时间频率也会减半，但 TX 仍可按 25 Hz 重发旧动作。`--loop_hz` 也未与训练 `sim.dt * decimation = 0.04s` 校验。

应使用单调时钟、明确 deadline 和超时处理，将相机采集与控制解耦，并记录真实推理时延、观测年龄和控制周期。感知与动作延迟需要和训练配置一起验证。

**问题 9：P1／待实物确认，虚拟俯仰辅助和低重力缺少部署对应说明**

位置：[外力矩辅助](truss_v3_env.py#L831)、[训练仿真重力](truss_v3_env_cfg.py#L151)、[当前 Sim2Real 任务辅助配置](truss_v3_env_cfg.py#L1198)。

仿真直接对 base 施加外部俯仰力矩，不是经 20 个腿关节执行；部署循环只发送腿位置和夹爪力矩，未发现同等的 base assist 实现。多个训练任务使用 `gravity=-1.6`。如果实机确有等效吊挂减载和俯仰约束，应将其参数写入部署契约；如果没有，则这是需要重新验证的动力学差异。当前代码注释声称辅助会出现在实机控制器中，尚无对应代码证据。

**问题 10：P2，夹爪标定和辅助工具存在接口歧义**

位置：[夹爪角度换算](sim2real_v3/play_real.py#L342)、[标定文件](sim2real_v3/gripper_calibration.yaml)、[打开夹爪工具](sim2real_v3/open_all_grippers.py#L70)。

反馈换算把名为 `close_pos` 的值作为从零闭合位置到打开位置的 travel，不能表达实际闭合零偏；交接文档也只把它称为近似旧映射。应改为经实物测量的 `closed_pos` / `open_pos` 双端点，并验证四腿顺序。

辅助工具称 `torque=+1` 为打开，但 play 中“正 action=打开”配合 `torque_sign=-1` 会发送负力矩。代码层面存在定义冲突；没有实物标定，不能断言哪一个方向正确。辅助工具应复用每条腿的同一标定，而不是另设统一正力矩。

**问题 11：P2，串口和 UDP 缺少健壮性与身份约束**

- [反馈读取](sim2real_v3/motor_session_gripper.py#L89) 对不足 8 字节的读取直接丢弃，没有持久接收缓冲和重新对齐逻辑；断帧场景应按实际适配器协议验证。
- [串口设备绑定](sim2real_v3/motor_session_gripper.py#L115) 直接依赖 `ttyUSB0..3` 枚举顺序，未验证稳定设备身份与腿的对应。
- [UDP 接收](sim2real_v3/radar_receiver.py#L16) 监听所有网卡，忽略来源、包顺序和字段维度。错误来源或格式合法但内容不合法的包可进入控制观测；需验证发送端身份、字段、有限性和本机接收时间。
- [串口原始打印](sim2real_v3/motor_control.py#L56) 每次接收都打印，25 Hz、24 电机时可能达到每秒约 600 行；应节流并保留有结构的故障、时效和执行日志。

**问题 12：P2，交接文档和复现记录已不同步**

README 仍将 Walk `28800` 称为默认，实际代码默认是 DepthBar `68500`；CROSSING_SIM2REAL_PLAN 仍以 0.15 Hz SlowApproach 路线为当前推荐，历史记录已于 09-06 转到 0.20 Hz nominal Teacher 路线。当前整个 V3 目录在工作区 Git 状态中为 untracked，不能仅凭仓库 HEAD 复现它。

应维护一份模型发布清单，绑定 checkpoint、感知模型、任务、训练保存配置、部署覆盖参数、源码版本／哈希、标定和验证结果。本次 JSON 已保存所审源码的 SHA-256，便于判断报告是否仍适用于后续代码。

**确认正确的部分与验证边界**

- 默认不开电机；明确传入 `--enable_motors` 才打开串口。
- 主关节的 policy 腿顺序 `RF/LF/RB/LB` 到硬件 `LB/LF/RB/RF` 的重排，以及 Joint5 正反变换，离线 20 关节 round-trip 通过。
- 76D deployable + 4D bar 的尺寸、sin/cos 相位顺序和 0.8s latch 保持行为通过离线检查；主关节解码采用半宽 action scale，与训练一致。
- 所读 `68500` / `96000` checkpoint 分别为 80→24 / 92→24，均包含 normalizer；本地 RSL-RL 的 `get_inference_policy()` 返回 eval 模型，默认确定性推理，未发现遗漏 normalizer 或默认随机采样的问题。
- sim2real 8 个 Python 文件和 camera 4 个 Python 文件通过 AST 语法解析。11 个异常路径探针复现上述问题，3 个基础映射／形状／latch 检查通过。
- 原有 crossing logic 的 14 个无 fixture 测试函数全部直接执行通过。当前解释器的 pytest 入口缺少 `exceptiongroup`，因此未声称 pytest 整套运行通过。
- 本次没有启动完整 Isaac Sim rollout，也没有串口、相机、雷达或机器人端到端实验；静态检查和 mock 不能验证真实接线方向、机械限位、支撑约束或最终跨越成功率。

**建议的后续顺序**

1. `68500` 的部署契约恢复、watchdog 和传感器有效性校验已实现；安全停机动作按本轮要求暂不添加，详见修复记录。
2. 在模拟串口／传感器中验证断流、单电机缺失、主线程停顿、相机卡住、NaN、Ctrl+C 和部分使能失败，确保均进入预期安全状态。
3. 用保存配置复评 `96000` / `96800`，按固定课程、多 seed、无真值初始化的感知冷启动分别统计普通成功、安全成功、速度、冲击和姿态；确认实际支撑与 base assist 条件。
4. 若选择低频安全 Teacher，再蒸馏为新的 deployable OracleBar Student 并接回 DepthBar。现有 08 月学生不会因为 09 月 Teacher 进步而自动具备相同能力。
