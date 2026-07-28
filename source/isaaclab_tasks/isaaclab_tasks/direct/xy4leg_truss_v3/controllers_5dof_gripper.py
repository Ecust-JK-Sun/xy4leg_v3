from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg, ActuatorNetLSTMCfg, ActuatorBaseCfg,IdealPDActuatorCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

### 2.ImplicitActuator
actuators_implicit: dict[str, ActuatorBaseCfg]={
        "legControl": ImplicitActuatorCfg(
            joint_names_expr=["LB_Joint1", "LF_Joint1", "RB_Joint1", "RF_Joint1",
                              "LB_Joint2", "LF_Joint2", "RB_Joint2", "RF_Joint2",
                              "LB_Joint3", "LF_Joint3", "RB_Joint3", "RF_Joint3", 
                              "LB_Joint4", "LF_Joint4", "RB_Joint4", "RF_Joint4",
                              "LB_Joint5", "LF_Joint5", "RB_Joint5", "RF_Joint5",
                              "LB_1out_Joint", "LF_1out_Joint", "RB_1out_Joint", "RF_1out_Joint",
                              "LB_2out_Joint", "LF_2out_Joint", "RB_2out_Joint", "RF_2out_Joint",
                              ],
            stiffness={
                ".*[^G]_Joint1": 100.,      # 腿部关节1
                ".*[^G]_Joint2": 150.,      # 腿部关节2
                ".*[^G]_Joint3": 150.,      # 腿部关节3
                ".*[^G]_Joint4": 150.,      # 腿部关节4
                ".*[^G]_Joint5": 100.,      # 腿部关节5
                ".*[^G]_1out_Joint": 500.,  # 夹爪主动关节
                ".*[^G]_2out_Joint": 500.,  # 夹爪主动关节
            },
            damping={
                ".*[^G]_Joint1": 2.0,       # 腿部关节1
                ".*[^G]_Joint2": 2.0,       # 腿部关节2
                ".*[^G]_Joint3": 2.0,       # 腿部关节3
                ".*[^G]_Joint4": 2.0,       # 腿部关节4
                ".*[^G]_Joint5": 2.0,       # 腿部关节5
                ".*[^G]_1out_Joint": 0.0,   # 夹爪主动关节
                ".*[^G]_2out_Joint": 0.0,   # 夹爪主动关节
            },
        )
    }
    

actuators_ideal: dict[str, ActuatorBaseCfg]={
        "legControl": IdealPDActuatorCfg(
            joint_names_expr=["LB_Joint1", "LF_Joint1", "RB_Joint1", "RF_Joint1",
                              "LB_Joint2", "LF_Joint2", "RB_Joint2", "RF_Joint2",
                              "LB_Joint3", "LF_Joint3", "RB_Joint3", "RF_Joint3", 
                              "LB_Joint4", "LF_Joint4", "RB_Joint4", "RF_Joint4",
                              "LB_Joint5", "LF_Joint5", "RB_Joint5", "RF_Joint5",
                              ],
            effort_limit=28,
            velocity_limit=30.0,
            effort_limit_sim=28,
            velocity_limit_sim=30.0,
            stiffness={
                ".*[^G]_Joint1": 25, #500.,     # 腿部关节1
                ".*[^G]_Joint2": 25, #500.,      # 腿部关节2
                ".*[^G]_Joint3": 25, #500.,      # 腿部关节3
                ".*[^G]_Joint4": 25, #500.,      # 腿部关节4
                ".*[^G]_Joint5": 25, #500.,      # 腿部关节5
            },
            damping={
                ".*[^G]_Joint1": 0.5,#2.0,       # 腿部关节1
                ".*[^G]_Joint2": 0.5,#2.0,       # 腿部关节2
                ".*[^G]_Joint3": 0.5,#2.0,       # 腿部关节3
                ".*[^G]_Joint4": 0.5,#4.0,       # 腿部关节4
                ".*[^G]_Joint5": 0.5,#2.0,       # 腿部关节5
            },
            friction = 0.01
        )
    }

### 3.神经网络控制器
ANYDRIVE_3_LSTM_ACTUATOR_CFG = ActuatorNetLSTMCfg(
    joint_names_expr=["Left_Joint1", "Left_Joint2", "Left_Joint3", "Left_Joint4", "Left_Joint5", "Left_Joint6",
                    "Right_Joint1", "Right_Joint2", "Right_Joint3", "Right_Joint4", "Right_Joint5", "Right_Joint6"],
    network_file=f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt",
    saturation_effort=30.0,
    effort_limit=25.0, #80
    velocity_limit=7.5,)