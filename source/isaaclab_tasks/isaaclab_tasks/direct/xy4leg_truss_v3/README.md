# Truss V3 改动、测试与实机结论总结

更新日期：2026-11-09，北京时间。本文收录的实机数据截至 **2026-11-03 17:07 启动的 LB 正弦测试**；前馈保护参数沿用 11-04 更新值，11-09 统一运行入口的默认补偿。

本文是 `xy4leg_truss_v3` 的统一说明，合并训练设计与历史、部署修复、通信、静态标定、模型拟合和跟踪验证。依据当前源码、保留日志及历史记录整理；11-04 原工作区整理时未使用 Git 历史；本次发布在独立检出中以远端 `4e080ca` 为基线核对实际差异。11-04 完成说明/产物收敛；11-09 修改默认模型选择、接入回放前馈并补充回归测试，未修改模型参数、权重或实机原始记录，未驱动硬件或启动新的 Isaac Sim 实验。

## 当前结论

| 方向 | 已有结果 | 结论与边界 |
| --- | --- | --- |
| 训练与感知控制 | Teacher → OracleBar → DepthBar 已打通，有固定课程评估和实机初测 | 保留各路线模型；尚无当前整机实机跨越重复成功率 |
| 当前软件回归 | **13 个文件，397 项全部通过，39.65 s（11-09）** | CPU/mock；覆盖当前代码，不替代硬件精度验收 |
| 四串口＋相机＋策略 | 120 s 长测策略及整组软件发送均约 25 Hz | 稳态无持续队列积压；启动尖峰和一次遗漏周期仍待定位 |
| 四腿静态标定 | 14 个腿轮次、70 扫描、4270 有效停点 | 是停稳窗口结果，不是动态或任意组合姿态精度 |
| 旧随机补偿对照 | LB J1 中点 MAE 0.544°；五轴三模式按 ±1°均仅 1/10 姿态整腿通过 | 单轴改善不能推广为整腿达标 |
| 质量模型＋J1 残差 | 拟合、加载、独立初始化、整腿随机/正弦及留档流程已建立 | 等效质量不是称重；姿态/方向建模仍有局限 |
| LB 新随机实测 | 8 点全部执行；末次五轴角度≤2°为 5/8，结束时完整稳定验收为 1/8 | 完成计划不等于精度通过 |
| **LB 新正弦实测** | **5 周期完成；J1–J4 RMSE 3.215/1.167/2.003/1.793°；约 24.924 Hz** | **已有本工具正弦实机证据；仍未全程达到 ±2°，未认证 40 ms 到位** |
| 11-04 前馈保护 | 限幅和变化率已调整；本次回归及四腿模型/计划只读检查通过 | 以下 11-03 实测使用旧限幅，**新保护参数尚无实机复测** |

## 11-09 默认方案统一

按当前选用方案，以下四个日常运动入口默认均为 **`fitted_j1`：拟合节点质量＋J1 姿态/方向残差，无误差积分**：

| 入口 | 当前默认行为 |
| --- | --- |
| [init_default_joint_pose.py](sim2real_v3/init_default_joint_pose.py) | 不指定补偿时即使用 fitted_j1；`--preview` 仅预览，去掉会执行初始化。旧 `--enable_gravity_compensation` 也选择该方案 |
| [play_real.py](sim2real_v3/play_real.py) | 默认构造 fitted_j1，在启动插值/保持和策略循环中使用实测姿态计算前馈；仍须显式 `--enable_motors` 才执行 |
| [measure_model_leg_tracking.py](sim2real_v3/measure_model_leg_tracking.py) | 保持原 fitted_j1 默认，改为共用模型加载；仍默认仅预览 |
| [replay_simdata_real.py](simdata_replay/replay_simdata_real.py) | 默认 fitted_j1，软启动、保持、轨迹及退出时的目标均配套发布前馈；默认只检查轨迹，显式 `--enable_motors` 才执行 |

模型加载统一在 [make_compensation_model](sim2real_v3/initialization_models.py)。六节点质量为 `sim2real_v3/calibration/node_masses_static_20261102.json`，J1 残差为 `sim2real_v3/calibration/j1_residual_20261103.json`，使用同一份 URDF/符号/质量哈希校验。残差或质量缺失、不兼容时直接报错，不静默退回零前馈。部署时必须携带这两份参数文件。

初始化、策略和回放可显式指定 `--compensation none / nominal / fitted / fitted_j1`；纯 `fitted` 不自动加载残差，`fitted_j1` 自动加载默认残差，也可用 `--j1_residual_file` 覆盖。初始化额外保留 `table` 和旧积分入口。静态标定、固定前馈/中点/双分支对照脚本保留其明确的实验方案；仅操作 J6 的开爪工具及历史 copy 文件不接入腿部自重补偿。

默认力矩限幅 `[6.5,3.5,2,1.5,0.3] N·m`、变化率 `[0.8,0.4,0.2,0.15,0.03] N·m/s`、KP/KD、残差系数与原始实机数据保持原值。回放前馈按新鲜实测硬件关节角计算，假设机体水平和夹爪几何角 −30°；使能回复完成后才启动 RX，TX 也检查反馈新鲜度，线程故障传回并关闭资源。默认方案统一不构成新的硬件精度验收。

## GitHub 发布内容（2026-11-09）

本轮同步部署与通信改动、标定/模型/回放工具和 13 个 CPU/mock 回归文件，保持训练与奖励代码沿用远端版本。补偿参数均纳入 `sim2real_v3/calibration/`：节点质量、J1 残差、20 份静态方向候选及清单；候选数值和 SHA256 与原始拟合产物一致，仅改清单的文件位置。重力模型所需 URDF 随源码提供，网格和训练 USD 仍是外部资产。

默认 `model_68500.pt`、RealCam estimator 以及模型自己的 `params/env.yaml`、`params/agent.yaml` 随本轮一起发布，已有历史权重保留。原始实机 CSV/JSONL、失败快照、完整训练日志、NPZ 轨迹、预览图、缓存和临时验证输出留在原工作区。本地证据在文中以文字路径保留，不作为 GitHub 内文件链接；首次克隆不依赖本地 `logs/real_v3/` 即可加载默认补偿并运行全部 CPU/mock 回归。

本次发布的独立检出复测为 **397 passed in 39.74s**，初始化默认预览也已通过；真实 68500 权重已核对 80→24 的网络层结构，RealCam estimator 的 CPU 前向输出为有限的 `[1,3]`。没有启动完整 Isaac Sim、真实相机或电机。发布中保留远端低层驱动原有 `__main__` 入口，未同步本地 J6 归零调试改动。

## 运行结构与改动汇总

### 训练、感知和回放

| 模块 | 已形成的行为与主要改动 |
| --- | --- |
| [环境](truss_v3_env.py)、[配置](truss_v3_env_cfg.py)、[状态逻辑](crossing_logic.py) | V3 从 V2 的 AMP/skrl 路线独立为 PPO teacher/student；复用资产与关节定义。reset 由实际姿态反算动作历史；接触聚合 Link5 和两指端；逐脚 before→cleared→passed；进度只计历史最大值增量，失败步不获 crossing 正奖励；课程状态随 checkpoint 保存恢复 |
| 课程、奖励与诊断 | 后腿完成→前腿跨越→近/远 approach→随机完整跨越，按成功 EMA/失败降级；最终需四脚通过、base 通过并稳定落地。增加落足/支撑、clearance、冲击/滑动、姿态/能量、动作变化/饱和和碰杆/包络诊断；普通成功与安全成功分开统计 |
| [训练](train_v3.py)、[播放](play_v3.py)、[评估](eval_v3.py)、[PPO 配置](agents/rsl_rl_ppo_cfg.py)、[任务注册](__init__.py) | 支持 Walk、Teacher、DR、Oracle/Noisy/DepthBar 和 learned-gripper/Sim2Real 任务，迁移 actor/normalizer、蒸馏、resume、固定课程评估和模型导出。20D 与 24D actor 不可直接混载；旧 88D teacher 向 92D Sim2Real teacher 的适配保留。当前 `train_v3.py` 默认是 Student-OracleBar，旧说明中的“默认 Teacher”已过期，训练时应显式传 task |
| 深度感知 | 保留 [数据采集](collect_depth_bar_dataset_v3.py)、[估计器训练](train_depth_bar_estimator_v3.py)、[估计器](depth_bar_estimator.py)、[高度图检查](visualize_depth_heightmap_v3.py)。历史为前/后相机 heightmap，部署基线转为单前置 RealCam depth crop；两种感知接口不能混同 |
| [相机基础接口](camera/camera.py)、[RealSense](camera/realsense_camera.py) | 补齐基础驱动依赖，有限首帧等待/启动重试和清理；用设备 depth scale，最近邻缩放深度。相机只由采集线程读取，策略读取有时间戳和序号的最新缓存；同帧不重复低通，无效估计进入有期限的失检记忆 |
| [轨迹工具](simdata_replay/) | 仿真录制、NPZ 检查、PyBullet 足端 IK/FK/clearance、轨迹优化和实机开环回放入口均保留。录制实际 `hardware_command[24]`，同时保留 raw/applied action 和元数据，避免回放时重复解释动作映射；已有 NPZ 不能当成实机跨越成功证据 |
| Pot 环境（本地 `source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/pot_v3_env.py`）、配置（本地 `source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/pot_v3_env_cfg.py`） | 同 package 内独立圆弧板 Walk 场景，几何、reset、reward 和接触定义与 truss 隔离，详见文末；其运行代码和已有轨迹均保留 |

主干几何历史设定为纵梁顶高 0.50 m、中心距 0.296 m、横杆宽 0.04 m/长 0.54 m；多个任务使用低重力 −1.6 m/s²。常用策略周期 0.04 s。步频、课程距离、辅助力矩及 DR 随任务和训练快照变化，部署必须从目标模型配套配置恢复，不能使用这些历史值覆盖它。

### 部署、通信和初始化

| 模块 | 修复与当前行为 |
| --- | --- |
| [deployment_config.py](sim2real_v3/deployment_config.py) | 恢复模型配套 env/agent YAML 中的观测、步频、scale、latch、normalizer/网络、默认姿态、资产及 soft limit；缺少快照/周期不匹配则拒绝；限制错误模型共用扁平基线 params，支持旧 estimator 同名迁移。发布兼容时补齐旧配置九项部署接口声明 |
| [play_real.py](sim2real_v3/play_real.py) | 校验 24 个有限动作，先限幅再写策略历史/平滑；拒绝 privileged crossing Teacher 依靠常量观测带电部署；单调时钟推进相位；estimator 后重取本体观测。带电 crossing 要相机，默认不开雷达；四腿启动 S 曲线插值，默认 3 s；当前默认 `tx_mode=on_update` |
| [motor_control.py](sim2real_v3/motor_control.py) | 有界非阻塞接收与部分帧保留；校验合法 ID、参数范围、写入长度/错误；去掉逐帧打印，增加通信审计。MIT 命令仍为 10 字节，使能仍为独立 0xFC，115200 baud |
| [motor_session_gripper.py](sim2real_v3/motor_session_gripper.py) | 使能应答逐电机核验，完成前不启动第二个接收者；逐电机保存有效数据/接收时间；无包/故障/非法值不刷新；配置稳定端口路径并拒绝重复设备映射 |
| [threads_shared.py](sim2real_v3/threads_shared.py)、[serial_threads.py](sim2real_v3/serial_threads.py) | 指令序号/时间戳，位置与前馈同快照发布；陈旧指令/反馈和线程错误传到主循环。支持 periodic 重发与 on_update 唤醒；四路非阻塞轮询，队列溢出不伪造有效反馈 |
| [runtime_timing.py](sim2real_v3/runtime_timing.py) | 策略、RX 和 TX 绝对时刻调度，超周期统计遗漏；独立相机缓存；记录实际 Hz、工作耗时、缓存年龄、发布到 TX、队列与无效包，JSONL 可回溯 |
| [radar_receiver.py](sim2real_v3/radar_receiver.py) | 校验 q/w 维度、有限性和四元数；按主机接收时刻判陈旧，带源时间戳时拒绝重复/倒序，可限定源 IP。雷达仅在显式启用时使用 |
| [init_default_joint_pose.py](sim2real_v3/init_default_joint_pose.py) | 默认 fitted_j1；互斥 none/nominal/fitted/fitted_j1/table 及独立有界积分入口；四腿各一线程、每口单个读写者。只在使能/起步同步，删除运行阶段跨腿屏障；25 Hz 四腿起发在一周期内错开，最多 30 ms；聚合 REACHED 必须仍新鲜。一腿故障取消整组，未放宽超时 |
| 初始化反馈修复 | 模型计算前读与发送后读共享同一份逐电机缓存/真实接收时间，避免第一处消费回复导致第二处误报过期；空读不续命，移动转保持沿用原反馈年龄，旧 REACHED 不可使整组通过。运行中默认不打印/同步写盘；审计有界缓存后统一保存 |

`play_real.py`、初始化及整腿模型跟踪沿用 close-only 退出：停止软件发布/发送并关闭资源，没有新增失能、归零或保持命令；这不认证电机的机械停止行为。**该约定不代表其他工具也采用相同退出动作**：轨迹回放等脚本有各自的夹爪退出选项，以各自 CLI/源码为准。

默认不开雷达时，水平重力、零角速度与默认前向速度为回退观测，并非姿态实测。主程序的 feedback/command/radar 超时默认 0.5 s，相机帧缓存默认 250 ms，相机等待 200 ms；各测量工具使用自己的反馈阈值。协议仍未提供设备时间戳、执行 ACK 或任意丢字节后的可靠重同步保证。

### 标定、模型与验证工具

| 文件/组 | 作用与演变 |
| --- | --- |
| [measure_joint_tracking.py](sim2real_v3/measure_joint_tracking.py)、[compare_joint_tracking.py](sim2real_v3/compare_joint_tracking.py) | 阶跃/正弦/慢坡，25 Hz 目标和独立 100 Hz 重发、预热积分后冻结、配对反馈、覆盖范围及误差/响应分析 |
| [tracking_compensation.py](sim2real_v3/tracking_compensation.py)、[directional_compensation.py](sim2real_v3/directional_compensation.py) | 固定负载表、中点/方向半差、端点夹紧、支持域标记、滤波/限幅/变化率；不在线改写参数；旧慢坡 v1 与多轮静态候选分别保留 |
| [calibrate_joint_load.py](sim2real_v3/calibrate_joint_load.py)、[calibrate_leg_load.py](sim2real_v3/calibrate_leg_load.py)、[merge_static_load_batches.py](sim2real_v3/merge_static_load_batches.py) | 共用正反向网格、逐轴/整腿复位、稳定冻结后采样、异常留档与续采；指令方向和实测最终到达方向分别记录；按轮合并不混入失败尝试 |
| 静态采样判据修正 | 首轮 J5 原始速度量化使稳定门控失败，改为近期 0.2 s 和完整 1 s 窗口角度峰峰值≤0.1°，速度保留诊断/可选门控；被测轴默认 0.8°、其他轴 5°，稳定 0.5 s、冻结后等待 0.3 s，再采样 1 s。旧失败记录仍为失败，不回填成有效点 |
| [fit_static_directional_compensation.py](sim2real_v3/fit_static_directional_compensation.py) | 逐腿逐轴双分支稳健拟合、整轮留出、稀疏/低支持与方向符号保留；不能跨腿直接复制 |
| [measure_random_joint_positions.py](sim2real_v3/measure_random_joint_positions.py)、[measure_random_leg_positions.py](sim2real_v3/measure_random_leg_positions.py) | 同种子配对单轴/五轴目标，固定/中点/双分支比较；持续到位和逐轴/整腿评分、失败恢复与续测 |
| [gravity_compensation.py](sim2real_v3/gravity_compensation.py)、[benchmark_gravity_initialization.py](sim2real_v3/benchmark_gravity_initialization.py) | 独立 NumPy URDF 自重与节点/杆中点等效验证；限幅/变化率、姿态/夹爪假设、模拟 I/O 频率测量；主部署当前默认 fitted_j1 前馈 |
| [fit_static_node_masses.py](sim2real_v3/fit_static_node_masses.py)、[initialization_models.py](sim2real_v3/initialization_models.py) | 六节点非负等效质量辨识、不可辨识节点固定、方向/零偏消融、整轮留出；fitted 运行只应用导出的质量，核对几何/映射/夹爪质量和哈希，不偷用拟合 nuisance 项 |
| [fit_j1_residual.py](sim2real_v3/fit_j1_residual.py)、[j1_residual.py](sim2real_v3/j1_residual.py) | 在拟合质量上增加 J1 姿态/实测方向残差，无位置误差积分；方向静止锁存，角度超支持域时截断并留标记 |
| [measure_model_leg_tracking.py](sim2real_v3/measure_model_leg_tracking.py) | 单腿五轴硬件零位→随机目标→回零或平滑渐入/出正弦；五次过渡、逐周期位置/离散加速度及前馈保护；未达精度继续记录，通信/故障/越界仍中止；CSV/JSON/报告输出。已修复完整正弦中的浮点截止时刻零休眠循环，无 matplotlib 也保存数值结果 |

## 当前配置与部署接口

### 模型、腿序和观测

部署基线及配套文件实际存在于同一目录 [logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar_sim2real/](../../../../../logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar_sim2real/)：`model_68500.pt`、`depth_crop_estimator_realcam_xmax060.pt`、`params/env.yaml`、`params/agent.yaml`。这份 params **仅属于 68500**；其他 checkpoint 必须携带自己的 params，并用 `--training_config_dir` 显式指定。默认路径按仓库位置解析。

基线任务为 `Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Student-DepthBar-Sim2Real-Direct-v0`。80D actor 为 deployable 76D＋bar 4D，action 为主关节 20D＋夹爪 4D。deployable 顺序：关节位置 20、速度 20、projected gravity 3、角速度/5 3、gait clock 2、previous raw action 24、夹爪角 4；bar 为归一化 dx/dy/dz 和 visible。

| 接口 | 约定 |
| --- | --- |
| policy 腿序 | RF、LF、RB、LB |
| 硬件腿序/默认串口 | LB=/dev/ttyUSB0、LF=1、RB=2、RF=3；主程序可用 `--motor_ports` 传四路稳定路径，单腿工具用 `--port` |
| 硬件命令 | joint-major：LB1/LF1/RB1/RF1，…，LB6/LF6/RB6/RF6；J1–J5 位置 MIT，J6 夹爪力矩；J5 的仿真→硬件符号为负 |
| 动作解码 | midpoint＋0.5×(soft_upper−soft_lower)×clamp(action,−1,1)；使用半宽 |
| 68500 恢复值 | 25 Hz 控制、0.30 Hz gait、bar scale 1.2/0.3/0.5、latch 阈值 0.10/0.10、min hold 0.8 s、bar lowpass alpha 1.0 |
| 额外部署覆盖 | 例如 `--bar_estimate_lowpass_alpha 0.30`、latch open/close deadband 0.06/0.10 必须显式记录；不会回写模型或训练快照 |
| 夹爪映射 | [gripper_calibration.yaml](sim2real_v3/gripper_calibration.yaml) 保留旧 travel/close_pos：LB −6.96、LF −7.38、RB −10.15、RF −7.0；torque_sign 默认 −1。该映射不是已标定的双端点；辅助开爪工具方向冲突仍需实物核对 |

### 模型跟踪当前默认值与版本边界

| 设置 | 当前值 |
| --- | --- |
| 默认补偿 | 初始化、策略运行、整腿模型跟踪和轨迹回放统一为 `fitted_j1`；显式 `fitted` 为不带残差的质量模型对照 |
| KP / KD，J1–J5 | `[30,25,20,20,2]` / `[1,1,1,1,0.5]`，各腿统一，可 CLI 覆盖 |
| 硬件零位 | `[0,0,0,0,0]°`，与默认站立姿态不同 |
| 站立中心，硬件角 | LB/RF `[65,45,-20,24,-24]°`；LF/RB `[-65,-85,-20,24,-24]°` |
| 随机目标 | 8 点、seed=20261103，J1 中心±35°，J2–J5 中心±20°；每点 3 s 并回零 |
| 正弦 | 0.05 Hz、5 周期，首尾各一周期平滑渐变；中间 3 周期拟合响应 |
| 验收 | 所有轴绝对误差≤2°、速度≤2°/s、连续≥0.5 s、前馈爬升完成且有新反馈；零位/中心最多 25 s，未达标留 `accepted=false` 后继续 |
| 目标保护 | 15°/s、名义离散加速度 30°/s²、五次过渡；调度停顿不追发大跳变 |
| 故障 | 逐电机反馈超时 0.5 s、故障、非法数值、队列溢出或实测越界中止留档；J6 不使能、不发送 |
| **11-04 当前前馈保护** | **绝对限幅 `[6.5,3.5,2,1.5,0.3] N·m`；变化率 `[0.8,0.4,0.2,0.15,0.03] N·m/s`** |
| 11-03 两轮完成实测保护 | 限幅 `[5,3,2,1.5,0.3]`；变化率 `[0.5,0.3,0.2,0.15,0.03]`；以各轮 run.json 为准 |

前馈限幅只限制新增项，电机总输出还叠加 PD。11-04 新配置本次通过 CPU 回归和四腿 random/sine 共 8 组模型/计划只读加载检查；没有重新生成预览或进行新配置实机测试。

## 软件回归：11-09 共 397 项通过

现有 13 个 `source/isaaclab_tasks/test/test_xy4leg_truss_v3*.py` 全部执行，结果 **397 passed in 39.65s**，无失败、错误或跳过。相比 11-04 的 375 项增加 22 个用例，覆盖统一默认值、模型文件身份、模型缺失时在开端口前失败、实测姿态回放前馈、软启动/保持/退出整条发布路径、超时和线程异常清理；原零前馈对照测试改为显式指定 `--compensation none`。

| 测试文件后缀 | 项数 | 主要覆盖 |
| --- | ---: | --- |
| init_pose | 98 | 并联初始化、独立调度、共享新鲜度、默认/显式模型对照、异常取消 |
| sim2real | 75 | 快照恢复、默认补偿、动作/观测、使能/反馈、收发、相机、调度和清理 |
| static_load | 52 | 双向扫描、冻结窗口、异常留档、续采/合并 |
| tracking | 38 | 阶跃/正弦/慢坡分析、目标/重发分离、缺包/越界 |
| gravity | 27 | 质量模型数学、坐标映射、前馈及命令路径 |
| model_tracking | 24 | 四腿完整随机/正弦、变化率、停顿、未达标继续、故障中止留档 |
| replay | 12 | 默认模型/覆盖、模型缺失、实测姿态/新鲜度、完整回放前馈及故障清理 |
| load_friction | 17 | 支持域、负载/方向、限幅和切换 |
| random_positions | 14 | 单轴配对计划、补偿、持续到位和恢复 |
| random_leg_positions | 11 | 五轴共同目标、整腿评分、异常与续测 |
| directional_fit | 11 | 整轮隔离、稳健拟合、符号与支持域 |
| logic | 10 | 单调进度、逐脚状态、课程、成功和夹爪滞回 |
| node_mass_fit | 8 | 质量基函数、不可辨识节点、参数恢复、非负边界 |
| **合计** | **397** | **CPU/mock，不连接硬件或启动完整仿真器** |

复现命令在仓库根目录执行；只在当前进程追加已有 vla 依赖目录以补齐 exceptiongroup，未安装包或修改环境。`PYTHONDONTWRITEBYTECODE` 和关闭 pytest 缓存/外部插件避免产生本次清理对象。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/mel/anaconda3/envs/isaaclab/bin/python - <<'PY'
import sys
from pathlib import Path
sys.path.append('/home/mel/anaconda3/envs/vla/lib/python3.10/site-packages')
import pytest
files = sorted(str(p) for p in Path('source/isaaclab_tasks/test').glob('test_xy4leg_truss_v3*.py'))
raise SystemExit(pytest.main(['-q', '-o', 'addopts=', '-p', 'no:cacheprovider', *files]))
PY
```

本次 JUnit 留在 `/tmp/truss_v3_tests_2026-11-09.xml`。另直接执行初始化 `--preview` 和轨迹回放默认 dry-run，均核对加载相同的质量/残差哈希，未使能硬件。11-04 的 375 项用时 37.81 s；历史 334 项是 11 个文件的较早快照；295 项是 10 文件、未含依赖 torch 的 logic/sim2real 的一次工具验证；49/53/261 等为不同源码和范围，不作同版本数量累加。11-04 整理时曾重算两轮 LB 随机和一轮 LB 正弦 CSV，所有已保存误差/响应与发送频率指标一致；holds 验收沿用原记录，未用重算虚构新的硬件验收。

## 实机通信与控制频率

主要证据为 11-02 四路通信长测分析（本地 `logs/real_v3/timing_motors_on_update_ramp6s_long.jsonl`），原始文件为 `logs/real_v3/timing_motors_*ramp6s*.jsonl`。11-03 汇总曾再次从原始 JSONL 汇总总计数/总时长，并按样本数加权均值，结果与已有分析一致。

### 120 秒 on_update 长测

6 秒四腿启动插值后，共 60 个窗口、120.0023 秒：

| 指标 | 结果 |
| --- | ---: |
| 策略完成 | 3000 次，24.9995 Hz |
| 软件完成整组 24 命令发送 | 3000 组，24.9995 Hz；均计为新发布序号 |
| 最慢电机有效反馈计数增长 | 3000，24.9995 Hz |
| 反馈轮询 | 24000 次，199.9962 Hz |
| 相机采集 | 3599 次，29.9911 Hz |
| 发布到发送开始等待 | 均值 0.166 ms |
| 本体观测读取到发送开始 | 均值 1.073 ms |
| 整组软件发送耗时 | 均值 6.765 ms，最大 7.431 ms |
| 稳态最老电机反馈年龄 | 最大 33.215 ms |
| 稳态策略工作耗时 | 最大 3.511 ms |
| 首窗口异常 | 深度处理最大 103.540 ms，策略工作最大 104.546 ms，遗漏周期 1 次 |

“稳态”仅排除首个约 2 秒窗口。其余 59 窗口无遗漏周期；日志中无效 ID 和重复 TX 计数为 0，未见 RX/TX 队列持续增长。

### 两种发送模式对照

| 指标 | periodic，约 20 秒 | on_update，约 120 秒 |
| --- | ---: | ---: |
| 发布到发送等待均值 | 30.452 ms | 0.166 ms |
| 本体观测读取到发送开始均值 | 31.326 ms | 1.073 ms |
| 整组软件发送耗时均值 | 6.815 ms | 6.765 ms |

主要收益是减少独立周期 TX 的相位等待，串口整组写入仍需约 7 ms。不同运行的对照不足以证明机械响应缩短相同时间。

**当前源码默认 `tx_mode=on_update`**，见 [play_real.py](sim2real_v3/play_real.py)。10-22 以及 11-02 通信报告中“默认 periodic／本次不修改默认”的文字是各自记录当时的状态，不能覆盖当前源码。长测使用的启动插值为 6 秒，当前 CLI 默认仍是 3 秒，不应混同。

200 Hz 是主机轮询，不是电机反馈生成频率。最慢电机计数不是同步完整帧或动作 ACK，主机写入完成也不是机械执行确认。`on_update` 保留最新发布、没有周期心跳，忙时可能合并中间发布；总计数相等不能证明逐序号全部执行。

历史 10-22 CPU/mock 预算检查中，单线程真实 68500 actor 均值约 0.034 ms、estimator 约 0.172 ms；模拟 I/O 的实际主循环约 25.01 Hz。它们不含真实设备端到端延迟，不能代替上面的 11-02 硬件长测。每路六电机按 25 Hz/115200 baud/8N1 估算 TX/RX 负载约 13.0%/10.4%，只表明名义串口带宽余量，不保证 CAN、USB 或机械响应。

## 实机关节动态跟踪

### LB J1 小幅测试：25 Hz

见 小幅跟踪分析（本地 `logs/real_v3/analysis_LB_j1_20261027/analysis.json`）。中心 62°、幅值 ±2°，KP=30、KD=1，预热后冻结前馈。

| 测试 | RMSE | 绝对误差 p95 | 主要现象 |
| --- | ---: | ---: | --- |
| 阶跃 | 1.719° | 2.937° | 11 次切换只有 1 次达到 ±0.5°且低速度稳定段，约 285 ms |
| 0.3 Hz 正弦 | 1.826° | 3.107° | 首周期后约 16.6 秒角度在编码分辨率下不再形成周期运动 |

链路约 25 Hz、无遗漏发送 tick，但跟踪不合格。近零输出振幅下的拟合相位没有物理解释，不能把旧摘要中的 861 ms 当作电机延迟。

### LB J1 ±10°测试：25 Hz 目标，100 Hz 重发

见 ±10°/100 Hz 分析（本地 `logs/real_v3/analysis_LB_j1_10deg_100hz_20261027/analysis.json`）。中心 55°、目标 45–65°，KP=30、KD=1，其他关节保持目标，运动中不继续积分。

| 测试 | RMSE | 绝对误差 p95 |
| --- | ---: | ---: |
| 慢坡 1°/s | 2.368° | 4.039° |
| 10°阶跃 | 2.006° | 3.804° |
| 0.1 Hz 正弦 | 2.458° | 4.092° |
| 0.3 Hz 正弦 | 2.598° | 4.103° |

四次均完整结束，实收反馈包约 100 Hz。11 次阶跃中仅 6 次达到 ±0.5°且低速度稳定段，成功样本稳定时间约 435–1235 ms；目标切换约 45 ms 后仍有 6.31–8.56°误差。正弦基波幅值比约 0.77–0.79，存在换向平台和位置相关负载。

10-27 这两轮所用配置仍未达到 ±0.5°或 ±1°动态精度，未证明 40 ms 到位。两轮测试同时改变中心、幅度与重发频率，不能把运动变化全部归因于 100 Hz。现象支持继续研究负载和方向历史，但尚不足以唯一确定摩擦、传动或反馈中的某一个原因。

## 静态标定与逐腿补偿候选

### 四腿采集已完成

见 四腿静态数据分析（本地 `logs/real_v3/analysis_static_load_all_legs_20261030/analysis.json`）。扫描各腿初始目标 ±30°、步长 2°，逐轴扫描并整腿复位。

| 腿 | 轮数 | 扫描数 | 有效停点，含基线 | 接受窗口全关节 MAE | 最大静态误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LB | 5 | 25 | 1525 | 0.336° | 0.799° |
| LF | 3 | 15 | 915 | 0.361° | 0.800° |
| RB | 3 | 15 | 915 | 0.375° | 0.799° |
| RF | 3 | 15 | 915 | 0.325° | 0.795° |
| **合计** | **14** | **70** | **4270** | — | — |

这是采集控制器积分到位、稳定后冻结前馈的静态窗口结果，采用被测轴 0.8°容差；其他关节容差为 5°。不能把它解释成整腿动态 ±0.8°。

四腿完整扫描统计中，四次失败尝试的 105 个已采点整体排除，未混入上述完整轮次。失败包括反馈缺失、到位超时及 RB J1 实测越界；原始记录保留。RB J1 曾在调节阶段连续约 42.54 秒达到 −7 N·m 积分上限，虽最终有效窗口退出饱和，也不能宣称全程无饱和。

### 20 个逐腿逐轴方向候选已建立

各轴保留两支保持力矩曲线，以 `L(q)` 为中点、`H(q)` 为有符号半差，候选为 `L(q)+sH(q)`。方向差包含静摩擦、迟滞、姿态和控制历史，不能直接当作纯摩擦系数。

下表为**整轮留出力矩预测 MAE，单位 N·m**，不是补偿后的角度误差：

| 腿 | J1 | J2 | J3 | J4 | J5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LB | 0.2399 | 0.1495 | 0.0486 | 0.0349 | 0.0089 |
| LF | 0.2168 | 0.1867 | 0.0379 | 0.0344 | 0.0116 |
| RB | 0.2674 | 0.1530 | 0.0477 | 0.0505 | 0.0083 |
| RF | 0.2356 | 0.1258 | 0.0349 | 0.0460 | 0.0105 |

来源：LB 五轮候选（本地 `logs/real_v3/candidate_static_load_LB_directional_20261028_v1/index.json`）、LF/RB/RF 三轮候选（本地 `logs/real_v3/candidate_static_load_LF_RB_RF_directional_20261030_v1/index.json`）。

不能跨腿直接复制：例如 LB 表预测 RF J3 的 MAE 约 0.340 N·m，明显高于 LB 自身留出约 0.049 N·m。LF J3 方向效应弱且支持稀疏，双分支未优于忽略方向；RF J3也需保留低支持标记。所有候选仍是其他轴接近默认目标时的一维条件模型。

## 补偿实机对照：改善了什么，仍缺什么

### LB J1 单轴随机目标

见 单轴随机对照（本地 `logs/real_v3/analysis_random_LB_j1_20261030_141022/analysis.json`）。一轮 20 个相同目标，三种模式，2–10°跳变、每点 2 秒，取末 0.5 秒评价；KP=35、KD=1。

| 模式 | 稳态 MAE | ±0.5°持续到位/稳态达标 |
| --- | ---: | ---: |
| 固定预热前馈 | 1.834° | 2/20 |
| 中点 `L(q)` | **0.544°** | **11/20** |
| 完整双分支 `L(q)+sH(q)` | 1.121° | 3/20 |

中点比固定前馈 MAE 降低约 70.3%。完整双分支约 205 ms 时较好，但之后沿命令方向继续偏移，稳态反而更差：20 点都曾进入 ±0.5°，仅 3 点守住。

支持的结论是：**静态方向差有数据依据，但保持阶段全额施加该项未在本轮取得收益**。三组其他轴实际姿态、预热前馈不完全相同，非被测轴最大偏差约 3°；且仅一轮固定顺序，不能将差异全部归因于方向项。

### LB 五轴同时随机目标

见 五轴随机姿态分析（本地 `logs/real_v3/analysis_random_leg_LB_20261030_145002/analysis.json`）。一轮、三模式各 10 个相同姿态，允许域为初始角 ±20°；每轴跳变 2–10°、保持 2 秒。整腿达标要求同一姿态的五轴全部达标。

| 模式 | 五轴平均 MAE | ±0.5°逐轴 | ±1°逐轴 | ±0.5°整腿 | ±1°整腿 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 固定预热前馈 | 1.268° | 10/50 | 17/50 | 0/10 | 1/10 |
| 中点 | 0.903° | 15/50 | 31/50 | 0/10 | 1/10 |
| 完整双分支 | **0.866°** | **19/50** | **33/50** | 1/10 | 1/10 |

双分支平均值略优主要来自 J5；J1–J4 平均 MAE 为中点 0.803°、双分支 0.899°。J4 两种补偿均有 ±1°的 10/10 达标，J5 双分支从中点的 4/10 提升到 7/10；不能由总平均推断所有轴都适合同一方向幅值。

150 个关节停点中 148 个尾窗峰峰值不超过 0.1°，主要是稳定偏差。近似相同的 J2 目标和目标方向，在不同组合姿态下残差可反向，提示一维角度和方向不足以解释全部变化，但尚不能分离姿态耦合与历史影响。

“J1–J4 中点、J5 双分支”的混合方案只是下一轮候选，**尚未实测**，不能拼接现有逐轴数据宣称其整腿成功率。

## 重力模型、计算频率与节点质量拟合

### URDF 自重前馈

见 [重力模型说明](sim2real_v3/gravity_compensation.py)。已实现按五轴实测姿态计算的独立 NumPy 自重模型，运行入口默认在拟合质量上叠加 J1 残差。基础重力模型不包含接触承重、整机受力分配、摩擦和完整动态项。

离线频率的 [基准代码](sim2real_v3/benchmark_gravity_initialization.py) 保留；计算与模拟 I/O 结果见本节下方。重复临时输出清理，不将它们作为硬件频率证据。

这是 CPU 与模拟传输证据。`logs/real_v3/init_gravity_timing.jsonl` 另有记录，但本汇总未将其作为一项独立“真实带电初始化频率已验收”的结论；不能用模拟频率替代该项实机验收。

### 节点等效质量离线拟合

最新诊断见 11-02 v2 拟合报告（本地 `logs/real_v3/fit_static_node_masses_20261102_v2/fit.json`）。4270 点中排除基线/未知方向及被标注的被测轴停稳积分饱和点，纳入 4199、排除 71 点。

| 腿 | 纯质量模型留轮 MAE | 质量＋方向模型留轮 MAE |
| --- | ---: | ---: |
| LB | 0.3321 N·m | 0.1244 N·m |
| LF | 0.3460 N·m | 0.1439 N·m |
| RB | 0.3086 N·m | 0.1335 N·m |
| RF | 0.3425 N·m | 0.1646 N·m |

J1 节点质量不可辨识，固定名义值；LF/RB/RF 的 J6 与 RF 的 J4 出现零质量边界，加入自由零偏后部分节点质量变化很大。矩阵满秩不等于得到可信实物质量。留轮误差包含方向项贡献，不能读成纯重力误差或补偿后的角度精度。

机体水平、夹爪开度等仍是数据未记录的假设。这些结果保留为候选，拟合未覆盖原名义质量；后续 LB 动态验证见下文，接触承重仍未验证。

### 当前初始化工具已有显式模型对照入口

当前 [初始化脚本](sim2real_v3/init_default_joint_pose.py)支持四腿各用独立串口线程并联初始化，并提供互斥的 `--compensation none / nominal / fitted / fitted_j1 / table`。`fitted` 只替换六节点等效质量，不施加拟合中的方向/零偏项；`fitted_j1` 在此基础上加载独立 J1 残差文件；`table` 使用静态双分支及实测运动方向锁存。当前默认 `fitted_j1`。

模型选择、几何/哈希校验、预览不访问硬件，以及模型通过并联插值/保持/结果记录路径的行为已有 CPU/mock 覆盖。**具备显式对照入口不等于四种模式已完成实机配对比较**；后续 nominal/fitted 初始化与 fitted_j1 跟踪已实测，结果见下文，仍不据此宣称四模式配对验收通过。

### 质量模型的数学和使用范围

URDF 几何来自 [xy4legv5_gripper.urdf](../../../../../Collect_data_pybullet/xy4legv5_gripper/urdf/xy4legv5_gripper.urdf)，不直接使用其中原质量/惯量。名义六节点质量为 `[0.525,0.550,0.550,0.550,0.550,0.525] kg`，另有两指各 0.05 kg，总计每腿 3.35 kg；50 g 均匀直杆各半分到端点只保证重力势能/力矩等效，不保证转动惯量相同。

按全腿实测姿态累加下游质点重力并取关节轴投影的反力矩：`τᵢ=−aᵢ·Σ((pₖ−oᵢ)×mₖg)`。映射 `q_urdf=sign*q_hw+offset`，力矩同符号映射；J5 为 −1，其余 +1，默认零偏 0。未开雷达假设 base 水平，夹爪几何角默认 −30°。不包含接触承重、整机受力分配或完整动态项。

10-30 名义模型对 4270 点的力矩 MAE：J1 约 0.75–0.99、J2 0.56–0.77、J3 0.48–0.82、J4 0.35–0.47、J5 0.059–0.062 N·m。这是模型预测残差，不是补偿后的角度误差。

模型 CPU 微基准单腿/四腿平均 0.1611/0.6611 ms，p95 0.1656/1.4641 ms，最大 0.1892/1.8822 ms。实际初始化函数配名义线速模拟 I/O，保持 6.00037 s 完成 150 组、24.9984 Hz，工作 p95 6.9004 ms；每写入阻塞 10 ms 的慢 I/O 对照为 19.0396 Hz，58 次超预算、跳过 18 时刻。均是历史 CPU/mock，不是硬件频率；基准代码继续保留，重复 JSON/stdout 已合并到本文后清理。

### 静态采集的早期验证与保留异常

10-27 首次 65°点因速度门控超时，0 有效点；下一次只采到 65°，63°点超时，不能用于双向建表。后续 57–71°短扫完成 15/15 点，J1 MAE 0.369°、最大 0.774°；同角度支路半差约 0.417–1.101 N·m，但姿态耦合和重复数不足，未据此直接发布固定表。

随后 35–95°三次往返完成 181/181 点，约 866.4 s，J1 MAE 0.412°、最大 0.800°，积分峰值 6.392 N·m 低于本轮 7 N·m 上限。6 点实测最终到达方向反转，全部保留按实测方向归类；89°附近异常保持力矩不自动剔除。29 个内部目标有双向样本，覆盖本轮角度跨度 93.3%，不是训练/推理分布覆盖率。整体约 99.67 Hz，点间同步写盘有 287 次遗漏 tick、最大反馈间隔约 104 ms；采样窗口内无遗漏，不能称全程严格 100 Hz。

旧 [LB J1 慢坡负载 v1](sim2real_v3/calibration/lb_j1_load_friction_v1.json) 的支持为 49.5–63.5°，只覆盖 45–65°目标跨度的 70%；约 0.9504 N·m 半差是历史候选，方向项默认关闭，不能当作纯摩擦常数。范围外端点夹紧及内部稀疏支持标记均保留。

## 11-03 初始化、J1 残差和整腿实测

### nominal/fitted 与通信问题

原始 init_nominal.json（本地 `logs/real_v3/init_nominal.json`） 和 init_fitted.json（本地 `logs/real_v3/init_fitted.json`） 保留。相同增益、25 Hz、限幅/变化率下，末次 20 轴 MAE 为 1.325→1.129°，12/20 轴改善；J3 四腿平均 MAE 1.737→0.290°，J1 则 1.878→2.380°。nominal 保持 20 s 未到位，fitted 约 8.8 s 因 RB 反馈过期中止。初始姿态不同且没有完整时序，不能判定 fitted 全面更优，也不能将末次快照当成时间平均或稳定验收成功率。量化对照保留在 joint_comparison.csv（本地 `logs/real_v3/compare_nominal_fitted_20261103/joint_comparison.csv`）。

初始化双缓存/双读取导致误判的可复现软件路径已经修复，不能反推此前 RB 异常必由此引起。后续 14:12 记录的 LF 在约 0.520 s 内 26 次读取无完整帧，同时完成 13 组五电机写入，另外三腿正常；LF 比其他腿少 65 帧，定位到回复路径，但不足以唯一归因 USB、桥接/CAN、供电或其他进程占用。原始快照与时序审计保留在 analysis_lf_rx_gap_20261103_1412/（本地 `logs/real_v3/analysis_lf_rx_gap_20261103_1412`）。其他腿 BrokenBarrierError 为取消结果，不作为独立通信故障。运行屏障删除后的四腿同条件稳定性仍待实机对照。

### J1 残差模型

在固定拟合节点质量基础上使用：`τ1=gain*τ1_base(q1..q5)+bias+s*(gap+slope*(q1−center)/30°)`；J2–J5 不变。方向 s 由实测位移锁存，静止保留；残差另限幅 1.5 N·m，最终总前馈仍过共用保护，超支持域截断残差角度自变量。无误差积分，不宣称辨识出动态 Coulomb 摩擦。

| 腿 | 基础 J1 力矩 MAE | 留轮残差候选 MAE | gain / bias / gap / slope |
| --- | ---: | ---: | --- |
| LB | 0.707 | 0.238 | 0.964 / 0.075 / 0.690 / 0.464 |
| LF | 0.641 | 0.218 | 0.966 / −0.070 / 0.628 / −0.542 |
| RB | 0.561 | 0.253 | 0.972 / −0.078 / 0.545 / −0.508 |
| RF | 0.626 | 0.222 | 1.038 / −0.095 / 0.608 / 0.464 |

力矩单位 N·m，gain 无量纲；指标来自同一静态姿态族的整轮留出选型，不能当成独立跨姿态/动态精度认证，也不可与上文全轴质量＋方向 MAE 混用。模型保留于 [j1_residual.json](sim2real_v3/calibration/j1_residual_20261103.json)，离线验证保留于同目录 validation.json。

### LB 随机目标：两轮均保留

| 轮次/原始目录 | 完成情况 |
| --- | --- |
| 16:29 旧 KP 随机（本地 `logs/real_v3/model_tracking_LB_random_20261103_162910_872216`） | KP `[20,20,20,20,2]`；零位 J2 约 +2.24°、前馈已完成爬升且无限幅，旧流程中止，随机目标尚未执行 |
| 16:48 新 KP 随机（本地 `logs/real_v3/model_tracking_LB_random_20261103_164810_931899`） | KP `[30,25,20,20,2]`；8 点和回零全部执行，`complete=true,error=null`，主机命令记录约 317.11 s，约 24.935 Hz |

第二轮各点末次误差为“目标−实测”，单位°：

| 点号 | J1 | J2 | J3 | J4 | J5 | 五轴末次≤2° | 结束时完整稳定验收 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 1.259 | 0.716 | −0.227 | 1.831 | −1.904 | 是 | 否 |
| 2 | 0.423 | 0.829 | −0.032 | 1.700 | −1.724 | 是 | 否 |
| 3 | 1.236 | 1.278 | −0.285 | 1.930 | −1.761 | 是 | 是 |
| 4 | 1.391 | 1.067 | 0.021 | 2.131 | −1.449 | 否 | 否 |
| 5 | 6.150 | −0.044 | 0.608 | 1.785 | −0.754 | 否 | 否 |
| 6 | 0.755 | 1.106 | −0.325 | 1.419 | −1.356 | 是 | 否 |
| 7 | 0.683 | 2.723 | −0.970 | 1.539 | −1.289 | 否 | 否 |
| 8 | 1.151 | 1.637 | −1.511 | 1.544 | −1.326 | 是 | 否 |

末次角度 5/8，通过完整稳定验收 1/8。点 1/2/6/8 曾有合格窗口，但结束时连续稳定仅约 0.316/0.115/0.388/0.396 s，未满 0.5 s。因此 1/8 不是“只有一点曾到 2°内”，5/8 也不是全程合格。

| J1–J4 阶段 RMSE° | J1 | J2 | J3 | J4 |
| --- | ---: | ---: | ---: | ---: |
| 零位到目标 | 3.624 | 1.518 | 0.953 | 1.761 |
| 目标保持，含全部 3 s 收敛段 | 2.609 | 1.382 | 0.823 | 1.749 |
| 返回零位 | 2.632 | 2.159 | 1.184 | 1.744 |

点 5 的 J1 保持需求约 6.75–6.88 N·m，发送约 4.02–5.00 N·m，受当时变化率和 5 N·m 限幅约束，是 6.15°偏差的重要线索，尚不能唯一归因。J2 点 7 约 2.72°，多次归零仍约 2.11–2.22°；J4 点 4 约 2.13°。不据两轮起始条件不同的记录证明 KP 调整解决全部偏差。

本轮最大单周期目标增量约 0.600°、二阶差分 0.04745°，符合名义 25 Hz 下 0.600°/0.048°上限；前馈增量也受限。实际发送间隔约 34.86–45.33 ms，按时间戳最大目标速度约 15.15°/s，不能把逐周期限幅表述为每个实际间隔严格≤15°/s。

### **LB 正弦：最新已完成实测**

原始目录：model_tracking_LB_sine_20261103_170758_524715/（本地 `logs/real_v3/model_tracking_LB_sine_20261103_170758_524715`）。`run_type=hardware`、`complete=true`、`error=null`。拟合质量＋J1 残差，无误差积分；KP/KD 为当前默认值，**前馈保护为 11-03 旧值**。中心 `[65,45,-20,24,-24]°`，幅度 `[35,20,20,20,20]°`，名义 0.05 Hz、5 周期。主机命令时间约 126.143 s，包含归零、入场/中心、100 s 正弦及回零；命令和反馈 CSV 各 15725 行，为五轴总行数，不是 15725 组同步帧。

下表误差统计取整个 sine 阶段，每轴 2500 配对反馈；相位/幅值只拟合去掉首尾渐入/出后的中间三周期。目标采用实际发送的零阶保持参考，与未限速请求的 RMSE 一致；记录 `target_rate_limited_cycles=0`。

| 关节 | MAE° | RMSE° | P95绝对误差° | 最大绝对误差° | ±2°内比例 | 幅值比 | 观测相位延迟 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| J1 | 2.986 | 3.215 | 4.640 | 6.026 | 23.36% | 0.9473 | 405.7 |
| J2 | 0.953 | 1.167 | 2.241 | 3.345 | 91.68% | 1.0535 | 135.1 |
| J3 | 1.694 | 2.003 | 3.305 | 3.762 | 59.84% | 0.8852 | 401.5 |
| J4 | 1.718 | 1.793 | 2.448 | 3.065 | 78.28% | 0.9651 | 373.3 |
| J5 | 2.179 | 2.358 | 3.472 | 3.692 | 38.36% | 0.9660 | 566.6 |

实际响应拟合频率约 **0.049847 Hz**；逐轴平均发送 **24.9243 Hz**，全流程最大发送间隔 **40.763 ms**；逐轴最大目标速度约 14.996/10.387/6.331/6.334/6.336°/s。

| 保持验证 | 结果 | 最终“目标−实测”误差 J1–J5，° |
| --- | --- | --- |
| 开始零位 | 通过 | +0.208、−0.361、+0.033、−1.344、+1.694 |
| 进入正弦中心 | 通过 | +1.320、+1.078、+0.119、+1.717、−1.542 |
| 最终回零 | 通过 | −0.055、−1.803、+0.798、−1.716、+1.694 |

这三次保持均满足结束时完整稳定判据，**不能替代连续正弦精度验收**。J1–J4 仍有超 2°样本；该轮证明完整正弦流程和数值响应可测，未证明全范围动态±2°或设备 40 ms 内到位。相位延迟基于主机接收反馈相对发送参考的拟合，包含运动与链路等影响，不是电机 ACK 延迟。J5 数值保留原始统计，此处不扩展为后续问题优先项。

原 run.json（本地 `logs/real_v3/model_tracking_LB_sine_20261103_170758_524715/run.json`）、summary.json（本地 `logs/real_v3/model_tracking_LB_sine_20261103_170758_524715/summary.json`）、commands.csv（本地 `logs/real_v3/model_tracking_LB_sine_20261103_170758_524715/commands.csv`）、feedback.csv（本地 `logs/real_v3/model_tracking_LB_sine_20261103_170758_524715/feedback.csv`） 均保留。本次从 CSV 只读重算的误差、幅值/相位与发送频率和保存值一致；没有修改原始验收字段。

## 历史训练与跨越评估

主要来源为已有训练记录、10-22 审查和保留的 [logs/rsl_rl](../../../../../logs/rsl_rl/)。下表引用已有记录，**本次没有重新训练或 rollout**。

| 路线/模型 | 历史结果 | 能说明什么 |
| --- | --- | --- |
| 纯本体 Student | 尾段成功约 0.2%，behavior loss 已降低但闭环跌倒多 | 模仿损失低不足以证明闭环稳定 |
| OracleBar 小噪声/延迟消融 | NoiseOnly/Delayed 成功约 74.7%/75.3%；完整 noisy 约 0.8% | visible/FOV/dropout 引入的输入变化需单独处理，不能只看位置噪声 |
| 历史 learned-gripper DepthBar `model_30000.pt` | 固定 L2/L4 各 1000 episodes：77.3%/84.6% | 是独立固定课程评估；旧 76D、前后相机 heightmap 接口与当前不同 |
| 当前部署基线 RealCam DepthBar `68500` | 训练末段成功 79.03%、fall/tilt 16.21%；已有实机初测反馈 | 保留为部署参考，训练成功率不是实机跨越成功率 |
| 08-28 Noisy OracleBar PPO | 训练末段普通/安全成功 91.62%/79.28% | 带噪横杆真值控制器参考，不等于真实相机感知水平 |
| 09-08 低频安全 Teacher `96000/96800` | 邻近前 3000 轮窗口安全成功约 79.33%/79.25%，速度约 0.092 m/s | 历史低频安全候选；92D 特权 actor 仍需蒸馏，不能直接部署 |
| 较新 SmoothSafe | 训练末段普通成功 97.84%，安全成功仅 0.41% | 普通成功高不能替代安全成功；不据新编号替换旧基线 |

除明确标注 fixed-level 评估外，表中训练末段/窗口数值为抽样标量均值，不是独立确定性评估或按 episode 加权成功率。课程、DR、距离和输入不同，不作跨组统一排名。

当前部署基线实际位于 `logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar_sim2real/model_68500.pt`，对应 estimator `depth_crop_estimator_realcam_xmax060.pt` 也存在。当前本机 `logs/rsl_rl` 下未找到 `model_96000.pt` / `model_96800.pt`，相应低频安全 run 和历史 Oracle PPO run 不在本次可见日志目录中；10-22 报告“当时已确认存在”只能作为历史状态。相同编号的 Walk checkpoint 不能替代缺失的 crossing 模型。

[轨迹回放目录](simdata_replay/)已有录制/检查/回放工具和 NPZ 轨迹；这些产物证明具备实验入口，现有汇总证据不足以把它们计为实机开环跨越成功或闭环任务验收。

历史调参中需保留的判断：早期 reset/动作历史、接触/逐脚状态、落地课程修正后 Teacher 才能升阶；过强稳定惩罚及两次前脚 overshoot shaping 曾使课程卡 0，随后回退。纯本体蒸馏与过大 rollout 噪声曾造成 behavior loss 很低但闭环失败，增加低维横杆输入并降低 student std/LR 后改善。learned-gripper 增加 4D 动作、夹爪相位/latch 和安全 clearance；将碰杆接触代理直接强惩罚的试验受轨道接触混入影响，后续以低 clearance 等辅助约束。balanced-level 训练用 0.05/0.05/0.30/0.30/0.30 分配，改善 L2/L3 覆盖，不等于已固定评估最优。直接把 deployable PPO 叫作 privileged Teacher、以及仅从 Walk prior 重学 crossing 的两条失败路线均已纠正为三阶段，未用较大编号自动替换有效模型。

历史 Walk-Sim2Real `2026-07-24_18-58-21` 的 `28800` 是 Walk 部署/接口参考，末 5k 训练 timeout=1、fall=0、action saturation=0；不用于替代 crossing checkpoint。07-22 learned-gripper Teacher `36000`→Oracle `10000`→DepthBar `30000` 是早期有效传承。08-12 从 145800 续训与 08-14 训练曾退化到约 0.92%/0.37%普通成功；这些同样是训练窗口统计，不作确定性模型评估。

## 保留的操作入口

以下从仓库根目录执行。`BASE` 是本项目目录；需要 Isaac Sim 的训练、播放、评估使用既有 IsaacLab 环境。所有示例只说明入口，**本次整理没有执行新的训练/实机运动**。各工具完整参数以 `--help` 为准。

```bash
PY=/home/mel/anaconda3/envs/isaaclab/bin/python
BASE=source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3

# 初始化模型检查：显式 --preview，不打开串口
"$PY" "$BASE/sim2real_v3/init_default_joint_pose.py" --preview

# 整腿随机/正弦默认只保存计划；输出放 /tmp，避免项目再次堆积预览
"$PY" "$BASE/sim2real_v3/measure_model_leg_tracking.py" --leg LB --mode random --output_dir /tmp/truss_LB_random_preview
"$PY" "$BASE/sim2real_v3/measure_model_leg_tracking.py" --leg LB --mode sine --output_dir /tmp/truss_LB_sine_preview

# 静态采集计划，不使能
"$PY" "$BASE/sim2real_v3/calibrate_joint_load.py" --leg LB --joint 1 --min_deg 35 --max_deg 95 --cycles 1

# 整批静态路线，默认只预览，可用 --resume_batch 续采指定批次
"$PY" "$BASE/sim2real_v3/calibrate_leg_load.py" --leg LB --rounds 5 --range_deg 30

# 旧 NPZ 轨迹离线检查
"$PY" "$BASE/simdata_replay/inspect_trajectory.py" --trajectory /path/to/recorded_trajectory.npz
```

预览输出目录必须尚不存在。模型跟踪在上述命令追加 `--enable_motors` 才驱动指定腿 J1–J5；替换 `--leg` 选择 LF/RB/RF，同种子/频率/范围/增益做 nominal、fitted、fitted_j1 的重复对照。初始化工具和主程序的默认行为不同，不应省略其各自的预览开关。

纯软件部署 dry-run：`play_real.py --max_steps 10 --headless`；加 `--enable_camera` 可读真实相机但仍不使能电机。带电 crossing 要显式加 `--enable_motors --enable_camera`，默认不开雷达；`--timing_jsonl` 保存完整链路日志，`--tx_mode periodic` 可与当前默认 on_update 对照。相机可用 `camera/visualize_realsense_heightmap.py --list_devices` 检查设备，该工具按 s 保存带 depth_raw/heightmap/metadata 的 NPZ 及可视化。

训练应显式指定任务，例如：

```bash
"$PY" "$BASE/train_v3.py" --task Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Teacher-Sim2Real-Direct-v0 --headless
# 固定评估需显式指定所需 checkpoint、课程等参数
"$PY" "$BASE/eval_v3.py" --help
```

质量重新拟合可用 `fit_static_node_masses.py --points_csv logs/real_v3/analysis_static_load_all_legs_20261030/points_all_legs.csv --saturation_annotations logs/real_v3/analysis_static_load_all_legs_20261030/saturation_windows.csv --output_dir <新目录>`；保留 v1/v2 和参数文件，不覆盖旧来源。回放工具 `record_sim_play.py`、`pybullet_footpath_ik_collect.py`、`analyze_bar_clearance.py`、`optimize_sim_footpath_pybullet.py` 与 `replay_simdata_real.py` 保留，NPZ 至少记录 metadata/time/dt/loop_hz、raw/applied action、joint_target 和 hardware_command；轨迹速度缩放、夹爪退出行为以回放 CLI 为准。

## Pot 分支的必要记录

Pot 用独立 `pot_v3_env.py/cfg`，不含横杆、Truss 逐脚状态、DepthBar 或 AMP。圆弧板由 mesh 构造，历史尺寸长 2.0 m、宽 0.02 m、顶高 0.794 m、端高 0.495 m；reset 按所在 x 的板面高度加 clearance，而非固定世界 z。策略仍为 24D action；普通 Pot actor 72D、Sim2Real actor 76D，后者仅 deployable，不追加 bar_estimate；不能套用 crossing 80D 输入。

Soft-Assist 课程逐步移除 y/yaw/z 力，保留与绳索约束对应的 roll/pitch 辅助。Link5 普通足端和两个 out 指端接触拆开；同腿两 out 接触且在 pot 有效域内才记夹板接触，固定夹板 reward=1，课程 success 夹板比例阈值为 0/0.05/0.10/0.15/0.20。

历史 `2026-07-29_04-10-50` 的 40k 后训练窗口达到 level4/success=1、约 0.060 m/s、前进约 1.21 m，仍有约 0.75 N·m 姿态辅助；推荐 `43000` 作 Sim2Real 初始化候选而非只取更晚 `46000`。这些并非去辅助实机验收，当前 truss 的 CPU/mock 测试也不构成 Pot 闭环认证。Pot 运行文件、注册、权重/日志均保留。

## 尚未闭合的验证

1. **优先 J1–J4 执行精度。** LB 正弦已完成；下一轮以相同目标、增益、模型哈希和不同模式轮换顺序，复测 11-04 前馈保护，分开末次角度、曾稳定和结束时稳定；不要把新旧限幅结果拼成同组。
2. 扩展 LF/RB/RF 随机与正弦，并补组合姿态/方向历史对照；当前静态表是条件模型，质量零边界/零偏敏感性仍在，不直接整体放大质量或逐轴拼接模型宣称通用物理参数。
3. 四腿独立初始化改动后做同条件时序对照，继续定位 LF/RB 回复空窗与首窗口尖峰；逐序号发布/TX、设备采样年龄/ACK 和四腿执行时差仍缺证据，不用放宽超时掩盖异常。
4. 训练感知仍有真值 visibility/reset-memory 辅助，需无真值冷启动、遮挡/失检恢复和记忆耗尽闭环验证；低重力/base assist 要核对实物支撑条件，夹爪双端点和开爪方向需实物标定。
5. 找回安全 Teacher 候选及其配套 params，以固定课程、多 seed、确定性 actor 分别验普通/安全成功、速度、冲击和姿态，先蒸馏可部署 Student 再接 DepthBar。当前无整机跨越重复成功率或全部关节 40 ms 准确到位认证。

## 文件收敛与保留边界

活动说明统一到本文件。原设计/历史/交接、分项操作说明、10-22 diff 报告与补丁、11-02/03 汇总和证据、各次重复分析报告已提取到上文；运行脚本仍可生成未来实验报告，本次未改写输出行为。

| 清理类别 | 文件数 |
| --- | ---: |
| 项目旧 Markdown 说明 | 23 |
| logs/real_v3 重复 Markdown 报告 | 29 |
| logs/real_v3 派生 PNG/PDF/HTML 图表 | 58 |
| 旧轨迹 PNG 预览 | 1 |
| 三套 model_tracking_preview 目录内文件，含重复验证说明 | 22 |
| 重复 mock 重力计时 JSON/stdout | 9 |
| 历史审查/测试证据 JSON 与重复代码 patch | 5 |
| 项目 Python 字节码缓存 | 81 |
| **合计** | **228，18,289,624 bytes，约 17.44 MiB** |

**保留**全部 Python 运行/分析/回归代码（包括现有 copy 版本）、配置 YAML、静态候选和拟合 JSON/CSV、节点质量、J1 残差、URDF/资产、NPZ 轨迹、模型权重及配套 params。实机完成/失败两类原始 run.json、commands.csv、feedback.csv、summary.json、JSONL、异常快照和审计数值均保留；合并/拟合的数值产物也保留，不能把同名不同版本当成重复删除。训练 logs、outputs 和既有 backups 未清理。

11-04 清理前逐文件建立 **675 个保留文件**的 SHA256 清单（合计 1,710,293,724 bytes）；**当次清理后 675/675 哈希一致**，228 个清理目标全部移除。该清单覆盖项目运行/参数/轨迹、全部保留 real_v3 数值与分析代码、当时的 12 个回归文件及 fixtures 和所用 URDF，不含未触碰的整个训练权重库；11-09 已按本文记录修改运行/测试代码，不再将旧哈希称为当前源码哈希。当次清理的文件清单与恢复归档保留在原工作区的临时审计目录，不随代码发布。

### 关键版本 SHA256

| 文件 | SHA256 |
| --- | --- |
| [measure_model_leg_tracking.py](sim2real_v3/measure_model_leg_tracking.py) | `104241aa3c36c778c3fd991d9b5f91197884edfa536e796931f8a793902b3ea0` |
| [damiao.yaml](sim2real_v3/damiao.yaml) | `08487af6dc41b7f7ad3caea8ad110bf01b59f699d6ac3ca94ff7cf3a26607ef7` |
| [gravity_compensation.yaml](sim2real_v3/gravity_compensation.yaml) | `5c6abf5d680ccd26b078b575b72270c726024f31f54f6eb5793b87966636f04e` |
| [node_masses_static_20261102.json](sim2real_v3/calibration/node_masses_static_20261102.json) | `942b3c39e1b06da947dd0efcc0edc437be821db650bab67c0814c40bad5603a3` |
| [j1_residual.json](sim2real_v3/calibration/j1_residual_20261103.json) | `ac8bb94fab756f99c203aaec77d51390673b5133f420fd36b5e1d05f65127ec0` |
| [xy4legv5_gripper.urdf](../../../../../Collect_data_pybullet/xy4legv5_gripper/urdf/xy4legv5_gripper.urdf) | `588ac74bb48fbc335bdce1065816f06e5a6255b8d299b6fe3346466957c77c74` |
| [init_default_joint_pose.py](sim2real_v3/init_default_joint_pose.py) | `987cce7bfe9c4b177010521f38c1874266eb58d65c2ef74f8cc130164e9debf4` |
| [play_real.py](sim2real_v3/play_real.py) | `afdb1befaf27cf45e7629acdf8aa53ce047a149c13ddeb58f39accd77bdf58ea` |
| [initialization_models.py](sim2real_v3/initialization_models.py) | `0fddb991ccaaec07b2ba91cfa3dbcfbcf72867a2d2b1628749f1e126d1d2fbe5` |
| [replay_simdata_real.py](simdata_replay/replay_simdata_real.py) | `f565aefa771399ab5803d2d04a948af87dccae3e7bbfa4c8fb584ddbaec9a1d7` |
