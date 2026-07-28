from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class XY4LegTrussV3TeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = "xy4leg_truss_v3_teacher"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.2,
            std_type="log",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class XY4LegGripperTrussV3TeacherPPORunnerCfg(XY4LegTrussV3TeacherPPORunnerCfg):
    experiment_name = "xy4leg_gripper_truss_v3_teacher"
    run_name = "learn_gripper"


@configclass
class XY4LegTrussV3TeacherDRPPORunnerCfg(XY4LegTrussV3TeacherPPORunnerCfg):
    experiment_name = "xy4leg_truss_v3_teacher"
    run_name = "dr"
    max_iterations = 2500
    resume = True


@configclass
class XY4LegTrussV3WalkPPORunnerCfg(XY4LegTrussV3TeacherPPORunnerCfg):
    experiment_name = "xy4leg_truss_v3_walk"
    max_iterations = 50000
    save_interval = 500
    # Keep the same action interface as Crossing. The environment applies a
    # differentiable-through-reward soft limit instead of hiding saturation
    # behind a tighter wrapper clamp.
    clip_actions = 1.0


@configclass
class XY4LegGripperTrussV3WalkPPORunnerCfg(XY4LegTrussV3WalkPPORunnerCfg):
    experiment_name = "xy4leg_gripper_truss_v3_walk"
    run_name = "learn_gripper"


@configclass
class XY4LegGripperTrussV3WalkSim2RealPPORunnerCfg(XY4LegGripperTrussV3WalkPPORunnerCfg):
    experiment_name = "xy4leg_gripper_truss_v3_walk_sim2real"
    run_name = "deployable_interface"
    max_iterations = 30000
    save_interval = 500
    obs_groups = {"actor": ["deployable", "bar_estimate"], "critic": ["critic"]}


@configclass
class XY4LegGripperTrussV3TeacherSim2RealPPORunnerCfg(XY4LegGripperTrussV3TeacherPPORunnerCfg):
    experiment_name = "xy4leg_gripper_truss_v3_teacher_sim2real"
    run_name = "privileged_from_walksim"
    max_iterations = 50000
    save_interval = 500
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}


@configclass
class XY4LegTrussV3StudentDistillRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = "xy4leg_truss_v3_student"
    run_name = "nominal"
    obs_groups = {"student": ["deployable"], "teacher": ["policy"]}
    clip_actions = 1.0
    teacher_checkpoint_path = (
        "/home/lab202/IsaacLab/logs/rsl_rl/xy4leg_truss_v3_teacher/"
        "2026-07-14_02-51-07/model_49999.pt"
    )

    student = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.03,
            std_type="log",
        ),
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.0,
            std_type="log",
        ),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=3.0e-4,
        gradient_length=8,
        max_grad_norm=1.0,
        loss_type="mse",
    )


@configclass
class XY4LegTrussV3StudentOracleBarDistillRunnerCfg(XY4LegTrussV3StudentDistillRunnerCfg):
    experiment_name = "xy4leg_truss_v3_student_oracle_bar"
    run_name = "nominal"
    obs_groups = {"student": ["deployable", "bar_estimate"], "teacher": ["policy"]}


@configclass
class XY4LegGripperTrussV3StudentOracleBarDistillRunnerCfg(
    XY4LegTrussV3StudentOracleBarDistillRunnerCfg
):
    experiment_name = "xy4leg_gripper_truss_v3_student_oracle_bar"
    run_name = "teacher36000"
    teacher_checkpoint_path = (
        "/home/lab202/IsaacLab/logs/rsl_rl/xy4leg_gripper_truss_v3_teacher/"
        "2026-07-21_22-36-26_learn_gripper/model_36000.pt"
    )


@configclass
class XY4LegTrussV3StudentNoisyBarDistillRunnerCfg(XY4LegTrussV3StudentOracleBarDistillRunnerCfg):
    experiment_name = "xy4leg_truss_v3_student_noisy_bar"
    run_name = "eval_noise"


@configclass
class XY4LegTrussV3StudentDepthBarDistillRunnerCfg(XY4LegTrussV3StudentOracleBarDistillRunnerCfg):
    experiment_name = "xy4leg_truss_v3_student_depth_bar"
    run_name = "eval_depth"


@configclass
class XY4LegGripperTrussV3StudentDepthBarDistillRunnerCfg(
    XY4LegGripperTrussV3StudentOracleBarDistillRunnerCfg
):
    experiment_name = "xy4leg_gripper_truss_v3_student_depth_bar"
    run_name = "from_oracle10000_teacher36000"


@configclass
class XY4LegGripperTrussV3StudentOracleBarSim2RealDistillRunnerCfg(
    XY4LegGripperTrussV3StudentOracleBarDistillRunnerCfg
):
    experiment_name = "xy4leg_gripper_truss_v3_student_oracle_bar_sim2real"
    run_name = "teacher_sim2real"
    obs_groups = {"student": ["deployable", "bar_estimate"], "teacher": ["policy"]}
    teacher_checkpoint_path = ""


@configclass
class XY4LegGripperTrussV3StudentDepthBarSim2RealDistillRunnerCfg(
    XY4LegGripperTrussV3StudentOracleBarSim2RealDistillRunnerCfg
):
    experiment_name = "xy4leg_gripper_truss_v3_student_depth_bar_sim2real"
    run_name = "depth_from_priv_teacher_sim2real"
