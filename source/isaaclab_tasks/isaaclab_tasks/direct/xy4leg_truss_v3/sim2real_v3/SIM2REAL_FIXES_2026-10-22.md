# Sim2Real V3 修复记录（2026-10-22）

本轮只修复部署代码，不调整训练 reward、物理参数、模型权重或实物标定。模型推荐及其训练时间见 [项目进展表](../PROGRESS_AND_SIM2REAL_REVIEW_2026-10-22.md)。

同日后续的非阻塞通信、相机采集线程、实际频率与延迟统计见 [通信与频率检查](COMMUNICATION_TIMING_REVIEW_2026-10-22.md)。

**已修复的行为**

| 问题 | 当前行为 |
| --- | --- |
| 旧模型套用当前任务参数 | 从 checkpoint 同目录的 `params/env.yaml`、`params/agent.yaml` 恢复观测维度、步频、bar scale、latch、模型结构、关节默认姿态、USD 和 soft limit factor；缺少快照直接报错，可用 `--training_config_dir` 指定复制后的目录 |
| 不同频率运行旧模型 | 控制频率从训练 `dt × decimation` 推导；显式 `--loop_hz` 必须一致。步态相位使用单调时钟的实际流逝时间，避免超周期后时钟变慢 |
| 动作执行与历史不一致 | 网络输出必须是 24 个有限值；先限幅到 `[-1,1]`，再用于动作历史、平滑和夹爪 latch。NaN/Inf、错误尺寸直接报错 |
| 无效感知被当成零位置横杆 | 在替换数值前检查 estimator 输出；NaN/Inf 按失检处理，只能使用尚未过期的记忆，超期输出 finite fallback、visible=0；记忆超期后重新检测会重置旧低通状态 |
| 缺失单电机反馈被整帧刷新掩盖 | 记录每个电机的本机接收时间和有效状态；只有 ID 合法、状态为 enabled、位置速度有限的反馈才有效。无包或故障不会刷新有效时间 |
| 启动时两个串口读取者竞争 | 逐电机确认使能应答后才启动反馈线程；空应答、错误 ID、disabled 或故障状态拒绝进入控制；启动过程中支持取消后不再继续使能其他电机 |
| 发送线程重复执行过期动作 | 动作带单调时间戳，默认 0.5s 后报错并结束软件发送；反馈就绪后发送线程也检查逐电机反馈有效性。线程错误传回主循环 |
| UDP 旧包／坏数据继续当成姿态 | 校验 q/w 维度、有限性和有效四元数，以本机接收时间判断是否过期；有发送时间戳时拒绝重复／倒序包，可用 `--radar_source_ip` 限定来源 |
| 未就绪就启动实机策略 | 带电运行要求姿态雷达；crossing 带电运行还要求相机。等传感器就绪，拒绝以 privileged 常量运行 crossing Teacher |
| 串口分包与参数越界 | 缓冲不足 8 字节的部分帧，下次读取继续拼接；参数越界不再通过位运算回绕；发送错误和串口写超时会报告给主循环，移除每帧原始数据打印 |
| 相机无界等待与深度单位猜测 | 限制帧等待时间，使用设备报告的 depth scale；深度缩放采用最近邻避免生成插值距离；关闭各资源时一个清理失败不阻断其余资源清理 |

反馈第一字节的高 4 位是状态，`0` 表示 disabled、`1` 表示 enabled、`8..14` 为故障，尽管本地旧字段仍名为 `error_code`。状态解释参考 [python-damiao-driver 协议说明](https://jia-xie.github.io/python-damiao-driver/dev/concept/communication-protocol/#status-codes)。

**按要求不增加安全停机动作**

本轮没有新增失能、归零、零力矩或夹爪保持命令。异常发生时只报错、停止继续发布／发送软件指令并关闭资源；正常退出也只关闭资源。此前已有的启动 nominal hold 保留。

停止软件发送不等于电机已经失能或机械系统已经安全停止。实际电机在通信停止后的状态仍取决于其固件和既有设置；本轮未修改这些设置。部分电机使能后若后续应答失败，也不会新增自动失能动作。

**默认 `68500` 的恢复结果**

```text
训练 run 开始：2026-08-05 04:55:55（北京时间）
actor：76D deployable + 4D bar；24D action
gait_frequency：0.30 Hz
control loop：25 Hz
bar dx/dy/dz scale：1.2 / 0.3 / 0.5
latch open/close deadband：0.10 / 0.10（该旧快照使用单一阈值）
latch min hold：0.8 s
bar lowpass alpha：1.0（恢复训练值）
```

若实机需要额外去抖，可显式传 `--bar_estimate_lowpass_alpha 0.30`；若需要非对称夹爪阈值，可显式传 `--gripper_latch_open_deadband 0.06 --gripper_latch_close_deadband 0.10`。这些是部署覆盖值，会影响闭环行为，不会回写训练快照或模型。

拷贝模型到部署电脑时需要一起拷贝 `params/`；脚本不再静默回退到当前训练配置。旧机器路径下的 USD 若不存在，只允许继续使用当前配置中同名的资产，跨机器时仍需保证资产文件内容一致。

纯软件 dry run（不打开电机、相机、雷达）：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/play_real.py \
  --max_steps 10 --headless
```

真实感知 dry run（读取相机和雷达，不使能电机）：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/play_real.py \
  --enable_camera --enable_radar --max_steps 300 --headless
```

后续启用电机需要同时保留 `--enable_camera --enable_radar`。默认 `--feedback_timeout_s`、`--radar_timeout_s`、`--command_timeout_s` 均为 0.5s，初始就绪等待 `--sensor_ready_timeout_s` 为 5s，相机等待 `--camera_timeout_ms` 为 200ms。

**验证**

首次修复时 35 项 pytest 通过：21 项新增部署回归检查 + 14 项原有 crossing logic 测试。后续通信检查后合计 **53 项通过**，详细覆盖见通信记录。首次新增检查覆盖真实快照恢复、错误 YAML 标签拒绝、缺失/故障使能应答、逐电机过期、指令超时、NaN 感知、动作历史、串口分包，以及执行实际 `main()` 定义的 dry-run 和模拟带电启动流程。所有硬件、相机和 runner 均使用 CPU mock，没有启动真实硬件或完整 Isaac Sim。

GitHub 发布时以 V3 仓库原有 `main` 为基线，仅补齐相机基类与旧配置缺少的九个部署接口声明，没有同步本机其余训练/reward 修改。独立检出验证 **49 项通过**：39 项部署检查加远程原有的 10 项 crossing logic 检查。新增测试参数 fixture，使干净检出不再依赖被忽略的本机 `logs/`；fixture 仅用于 mock 测试，不包含权重，也不能替代实机模型配套的完整 `params/`。本次提交不含新模型、训练日志或夹爪标定修改。

本机测试解释器的 PyYAML 和 pytest 依赖不完整，本次只在测试进程中追加既有依赖搜索路径，没有安装包或修改 conda 环境。复现命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/lab202/miniconda3/envs/env_isaaclab/bin/python - <<'PY'
import sys
sys.path.append('/usr/lib/python3/dist-packages')
sys.path.append('/home/lab202/miniconda3/envs/lerobot/lib/python3.10/site-packages')
import pytest
raise SystemExit(pytest.main([
    '-q', '-p', 'no:cacheprovider',
    'source/isaaclab_tasks/test/test_xy4leg_truss_v3_sim2real.py',
    'source/isaaclab_tasks/test/test_xy4leg_truss_v3_logic.py',
]))
PY
```

尚未处理的事项：训练感知链仍有真值 visibility/reset 记忆辅助；虚拟 base assist 和低重力需核对实物支撑条件；夹爪双端点及辅助工具正反方向需实物标定；初始姿态渐变未实现。后续已支持显式配置稳定串口路径，但设备实际身份/接线仍需核对；串口缓冲可处理分包，不能据此声称已经解决任意丢字节后的重新同步。现有模型的真实成功率仍需独立实验验证。
