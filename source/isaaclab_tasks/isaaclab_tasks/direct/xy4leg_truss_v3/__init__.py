"""XY4Leg Truss V3 crossing environments."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-XY4Leg-Gripper-Truss-V3-Walk-Direct-v0",
    entry_point=f"{__name__}.truss_v3_env:XY4LegTrussV3TeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.truss_v3_env_cfg:XY4LegTrussV3WalkEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:XY4LegTrussV3WalkPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Walk-Direct-v0",
    entry_point=f"{__name__}.truss_v3_env:XY4LegTrussV3TeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.truss_v3_env_cfg:XY4LegGripperTrussV3WalkEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:XY4LegGripperTrussV3WalkPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Walk-Sim2Real-Direct-v0",
    entry_point=f"{__name__}.truss_v3_env:XY4LegTrussV3TeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.truss_v3_env_cfg:XY4LegGripperTrussV3WalkSim2RealEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:XY4LegGripperTrussV3WalkSim2RealPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Isaac-XY4Leg-Gripper-Truss-V3-Teacher-Direct-v0",
    entry_point=f"{__name__}.truss_v3_env:XY4LegTrussV3TeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.truss_v3_env_cfg:XY4LegTrussV3TeacherEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:XY4LegTrussV3TeacherPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Teacher-Direct-v0",
    entry_point=f"{__name__}.truss_v3_env:XY4LegTrussV3TeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.truss_v3_env_cfg:XY4LegGripperTrussV3TeacherEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:XY4LegGripperTrussV3TeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Teacher-Sim2Real-Direct-v0",
    entry_point=f"{__name__}.truss_v3_env:XY4LegTrussV3TeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.truss_v3_env_cfg:XY4LegGripperTrussV3TeacherSim2RealEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:XY4LegGripperTrussV3TeacherSim2RealPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-XY4Leg-Gripper-Truss-V3-Teacher-DR-Direct-v0",
    entry_point=f"{__name__}.truss_v3_env:XY4LegTrussV3TeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.truss_v3_env_cfg:XY4LegTrussV3TeacherDREnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:XY4LegTrussV3TeacherDRPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-XY4Leg-Gripper-Truss-V3-Student-Direct-v0",
    entry_point=f"{__name__}.truss_v3_env:XY4LegTrussV3TeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.truss_v3_env_cfg:XY4LegTrussV3StudentEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:XY4LegTrussV3StudentDistillRunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-XY4Leg-Gripper-Truss-V3-Student-OracleBar-Direct-v0",
    entry_point=f"{__name__}.truss_v3_env:XY4LegTrussV3TeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.truss_v3_env_cfg:XY4LegTrussV3StudentOracleBarEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:XY4LegTrussV3StudentOracleBarDistillRunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Student-OracleBar-Direct-v0",
    entry_point=f"{__name__}.truss_v3_env:XY4LegTrussV3TeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.truss_v3_env_cfg:XY4LegGripperTrussV3StudentOracleBarEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:XY4LegGripperTrussV3StudentOracleBarDistillRunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Student-OracleBar-Sim2Real-Direct-v0",
    entry_point=f"{__name__}.truss_v3_env:XY4LegTrussV3TeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.truss_v3_env_cfg:XY4LegGripperTrussV3StudentOracleBarSim2RealEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "XY4LegGripperTrussV3StudentOracleBarSim2RealDistillRunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-XY4Leg-Gripper-Truss-V3-Student-NoisyBar-Direct-v0",
    entry_point=f"{__name__}.truss_v3_env:XY4LegTrussV3TeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.truss_v3_env_cfg:XY4LegTrussV3StudentNoisyBarEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:XY4LegTrussV3StudentNoisyBarDistillRunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-XY4Leg-Gripper-Truss-V3-Student-NoisyBar-NoiseOnly-Direct-v0",
    entry_point=f"{__name__}.truss_v3_env:XY4LegTrussV3TeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.truss_v3_env_cfg:XY4LegTrussV3StudentNoisyBarNoiseOnlyEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:XY4LegTrussV3StudentNoisyBarDistillRunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-XY4Leg-Gripper-Truss-V3-Student-NoisyBar-Delayed-Direct-v0",
    entry_point=f"{__name__}.truss_v3_env:XY4LegTrussV3TeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.truss_v3_env_cfg:XY4LegTrussV3StudentNoisyBarDelayedEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:XY4LegTrussV3StudentNoisyBarDistillRunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-XY4Leg-Gripper-Truss-V3-Student-DepthBar-Direct-v0",
    entry_point=f"{__name__}.truss_v3_env:XY4LegTrussV3TeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.truss_v3_env_cfg:XY4LegTrussV3StudentDepthBarEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:XY4LegTrussV3StudentDepthBarDistillRunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Student-DepthBar-Direct-v0",
    entry_point=f"{__name__}.truss_v3_env:XY4LegTrussV3TeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.truss_v3_env_cfg:XY4LegGripperTrussV3StudentDepthBarEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:XY4LegGripperTrussV3StudentDepthBarDistillRunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-XY4Leg-Gripper-Truss-V3-Gripper-Student-DepthBar-Sim2Real-Direct-v0",
    entry_point=f"{__name__}.truss_v3_env:XY4LegTrussV3TeacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.truss_v3_env_cfg:XY4LegGripperTrussV3StudentDepthBarSim2RealEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "XY4LegGripperTrussV3StudentDepthBarSim2RealDistillRunnerCfg"
        ),
    },
)
