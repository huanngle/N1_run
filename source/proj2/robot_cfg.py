from pathlib import Path
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

_THIS_DIR = Path(__file__).parent
USD_PATH = _THIS_DIR.parent.parent / "robots" / "N1" / "urdf" / "N1_main_body_raw" / "N1_main_body_raw.usd"

ACTION_JOINTS_13 = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_roll_joint", "left_ankle_pitch_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_roll_joint", "right_ankle_pitch_joint",
    "waist_yaw_joint",
]

HUAN_ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.0,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.70),
        joint_pos={
            "left_hip_pitch_joint": float(np.deg2rad(-14.0)),
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_pitch_joint": float(np.deg2rad(29.5)),
            "left_ankle_roll_joint": 0.0,
            "left_ankle_pitch_joint": float(np.deg2rad(-13.7)),
            "right_hip_pitch_joint": float(np.deg2rad(-14.0)),
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_pitch_joint": float(np.deg2rad(29.5)),
            "right_ankle_roll_joint": 0.0,
            "right_ankle_pitch_joint": float(np.deg2rad(-13.7)),
            "waist_yaw_joint": 0.0,
        },
    ),
    actuators={
        "legs_waist": ImplicitActuatorCfg(
            joint_names_expr=ACTION_JOINTS_13,
            stiffness={
                ".*hip_pitch.*": 180.0,
                ".*hip_roll.*": 120.0,
                ".*hip_yaw.*": 90.0,
                ".*knee.*": 120.0,
                ".*ankle.*": 45.0,
                "waist.*": 180.0,
            },
            damping={
                ".*hip_pitch.*": 10.0,
                ".*hip_roll.*": 10.0,
                ".*hip_yaw.*": 8.0,
                ".*knee.*": 8.0,
                ".*ankle.*": 2.5,
                "waist.*": 36.0,
            },
            effort_limit={
                ".*hip_pitch.*": 95.0,
                ".*hip_roll.*": 54.0,
                ".*hip_yaw.*": 54.0,
                ".*knee.*": 95.0,
                ".*ankle.*": 30.0,
                "waist.*": 54.0,
            },
        ),
    },
)