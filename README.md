# XY4Leg Truss V3

This repository is an overlay for the IsaacLab workspace. It contains the
`xy4leg_truss_v3` direct task, sim2real helper scripts, selected training
checkpoints, and the V2 gripper actuator helper required by the V3 config.

## Layout

```text
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/
source/isaaclab_tasks/isaaclab_tasks/direct/xy4legv2_amp/controllers_5dof_gripper.py
source/isaaclab_tasks/test/test_xy4leg_truss_v3_logic.py
logs/rsl_rl/... selected checkpoints
```

Copy or clone this repository at the IsaacLab root, then install/run IsaacLab
as usual.

## External Asset

The robot USD is not included in git because the current file is larger than
GitHub's normal 100 MB file limit:

```text
source/isaaclab_tasks/isaaclab_tasks/direct/xy4legv2_amp/usd/xy4legv5_gripper_truss_v2.usd
```

Place that USD at the path above before running training or play commands.
If the asset should be versioned later, use Git LFS.

## Key Checkpoints

See `CHECKPOINTS.md` for the selected checkpoint list and purpose.

## Main Documents

```text
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/TRUSS_V3_DESIGN.md
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/TRUSS_V3_HISTORY.md
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/README.md
source/isaaclab_tasks/isaaclab_tasks/direct/xy4leg_truss_v3/sim2real_v3/HANDOVER.md
```
