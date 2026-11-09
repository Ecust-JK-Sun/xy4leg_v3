# XY4Leg Truss V3

本仓库是 IsaacLab 的 Truss V3 overlay，包含任务源码、实机部署/标定/回放工具、选定模型及 CPU/mock 回归测试。将对应目录合入现有 IsaacLab 根目录后运行；训练/策略播放仍依赖 Isaac Sim 和外部机器人 USD。

2026-11-09 更新：初始化、策略运行、整腿模型跟踪与轨迹回放默认采用 **fitted_j1（拟合质量＋J1 残差）**，保留显式 `none / nominal / fitted` 对照。参数与几何依赖随源码发布，实机原始数据和完整训练日志不上传。

改动清单、完整测试结果、LB 随机/正弦实测、参数边界和使用入口统一见 [项目总结](source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/README.md)。模型文件清单见 [CHECKPOINTS.md](CHECKPOINTS.md)。

## 快速检查

在已安装 NumPy、SciPy、PyYAML、PyTorch、pytest 和 pyserial 的 Python 环境中：

```bash
BASE=source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3
# 只预览默认补偿，不打开电机串口
python "$BASE/sim2real_v3/init_default_joint_pose.py" --preview
# CPU/mock 回归，不启动 Isaac Sim 或真实硬件
python -m pytest -q -o addopts= source/isaaclab_tasks/test/test_xy4leg_truss_v3*.py
```

默认 `68500` 策略、对应 RealCam estimator 和完整 `params/` 位于 `logs/rsl_rl/xy4leg_gripper_truss_v3_student_depth_bar_sim2real/`；这份 params 只属于该模型。J1 残差、节点质量及静态候选位于任务的 `sim2real_v3/calibration/`，无需复制本地实机日志目录。

## 外部资产

训练和策略播放需要自行放置机器人 USD：

```text
source/isaaclab_tasks/isaaclab_tasks/direct/xy4legv2_amp/usd/xy4legv5_gripper_truss_v2.usd
```

该资产超过 GitHub 普通文件大小限制，未纳入 Git。本仓库内 URDF 用于不依赖网格的自重计算；PyBullet 可视化/物理回放若需要网格，应另行准备 URDF 所引用的 mesh。CPU/mock 通过不代表实机执行精度验收。
