import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from . import mdp
from .robot_cfg import HUAN_ROBOT_CFG


ACTION_JOINTS_13 = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
]

LEFT_FOOT_BODY = "left_foot_pitch_link"
RIGHT_FOOT_BODY = "right_foot_pitch_link"
FOOT_BODIES = [LEFT_FOOT_BODY, RIGHT_FOOT_BODY]
UNDESIRED_CONTACT_BODIES = ["base_link"]

ACTIONS_MAX_13 = [2.618, 1.571, 1.571, 2.356, 0.436, 0.785, 2.618, 0.262, 1.571, 2.356, 0.436, 0.785, 2.618]
ACTIONS_MIN_13 = [-2.618, -0.262, -1.571, -0.087, -0.436, -0.785, -2.618, -1.571, -1.571, -0.087, -0.436, -0.785, -2.618]

DEFAULT_JOINT_ANGLES_DEG = [
    -14.0,
    0.0,
    0.0,
    29.5,
    0.0,
    -13.7,
    -14.0,
    0.0,
    0.0,
    29.5,
    0.0,
    -13.7,
    0.0,
]

ACTION_SCALE_BY_JOINT = {}
for i, joint_name in enumerate(ACTION_JOINTS_13):
    q0 = math.radians(DEFAULT_JOINT_ANGLES_DEG[i])
    s_pos = ACTIONS_MAX_13[i] - q0
    s_neg = q0 - ACTIONS_MIN_13[i]
    ACTION_SCALE_BY_JOINT[joint_name] = float(max(1e-6, min(s_pos, s_neg)))


@configclass
class WalkSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = HUAN_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=8,
        track_air_time=True,
        debug_vis=False,
    )


@configclass
class WalkCommandsCfg:
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.20,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 3.2),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class WalkActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTION_JOINTS_13,
        scale=ACTION_SCALE_BY_JOINT,
        use_default_offset=True,
    )


@configclass
class WalkObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.command_observation, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTION_JOINTS_13)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTION_JOINTS_13)},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.command_observation, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTION_JOINTS_13)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTION_JOINTS_13)},
        )
        actions = ObsTerm(func=mdp.last_action)
        base_height = ObsTerm(func=mdp.base_height_obs, params={"asset_cfg": SceneEntityCfg("robot")})
        feet_contact = ObsTerm(
            func=mdp.feet_contact_state,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
                "force_threshold": 1.0,
            },
        )
        feet_speed_xy = ObsTerm(
            func=mdp.feet_speed_xy_obs,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES)},
        )
        feet_height = ObsTerm(
            func=mdp.feet_height_obs,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES)},
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class WalkEventsCfg:
    curriculum = EventTerm(
        func=mdp.update_walk_run_curriculum,
        mode="reset",
        params={
            "command_name": "base_velocity",
            "stages": (
                {
                    # About 500 PPO iterations 
                    "max_common_step": 32_000,
                    "lin_vel_x": (0.0, 0.8),
                    "ang_vel_z": (0.0, 0.0),
                    "rel_standing_envs": 0.25,
                    "push_velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                },
                {
                    # About 500-1300 PPO iterations. Add only mild walking
                    # pushes so robustness does not dominate the gait.
                    "max_common_step": 83_200,
                    "lin_vel_x": (0.3, 1.5),
                    "ang_vel_z": (0.0, 0.0),
                    "rel_standing_envs": 0.10,
                    "push_velocity_range": {"x": (-0.10, 0.10), "y": (-0.08, 0.08)},
                },
                {
                    # About 1300-2400 PPO iterations. Keep pushes mild and
                    # limited to walking or fast walking speeds.
                    "max_common_step": 153_600,
                    "lin_vel_x": (1.0, 2.4),
                    "ang_vel_z": (0.0, 0.0),
                    "rel_standing_envs": 0.05,
                    "push_velocity_range": {"x": (-0.15, 0.15), "y": (-0.10, 0.10)},
                },
                {
                    # About 2400-4000 PPO iterations: learn high-speed running
                    # without pushes so the policy prioritizes tracking speed.
                    "max_common_step": 256_000,
                    "lin_vel_x": (1.8, 3.2),
                    "ang_vel_z": (0.0, 0.0),
                    "rel_standing_envs": 0.02,
                    "push_velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                },
                {
                    # About 4000-6000 PPO iterations: specialize on the
                    # assignment's running range. 
                    # robustness is covered by the walking stages above.
                    "max_common_step": 10_000_000,
                    "lin_vel_x": (2.5, 3.2),
                    "ang_vel_z": (0.0, 0.0),
                    "rel_standing_envs": 0.0,
                    "push_velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                },
            ),
        },
    )

    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )

    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=UNDESIRED_CONTACT_BODIES),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.05, 0.05),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTION_JOINTS_13),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
    )


@configclass
class WalkRewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=0.0)

    track_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=6.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.35,
        },
    )
    track_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.5,
        },
    )

    upright = RewTerm(func=mdp.upright_reward, weight=1.0, params={"asset_cfg": SceneEntityCfg("robot")})
    height = RewTerm(
        func=mdp.root_height_reward,
        weight=1.5,
        params={"target_height": 0.70, "asset_cfg": SceneEntityCfg("robot")},
    )
    pose = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTION_JOINTS_13)},
    )
    stand_still = RewTerm(
        func=mdp.stand_still_bonus,
        weight=0.35,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTION_JOINTS_13),
            "command_threshold": 0.125,
            "yaw_threshold": 0.125,
        },
    )
    feet_air_time = RewTerm(
        func=mdp.single_support_moving_reward,
        weight=0.6,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "speed_threshold": 0.2,
            "force_threshold": 1.0,
        },
    )
    double_support_slow = RewTerm(
        func=mdp.double_support_reward,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "speed_threshold": 0.125,
            "yaw_threshold": 0.125,
            "force_threshold": 1.0,
        },
    )
    fly = RewTerm(
        func=mdp.fly_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "command_threshold": 0.125,
            "force_threshold": 1.0,
        },
    )

    base_z_lin_vel = RewTerm(func=mdp.base_z_lin_vel_l2, weight=-0.25, params={"asset_cfg": SceneEntityCfg("robot")})
    base_xy_ang_vel = RewTerm(
        func=mdp.base_xy_ang_vel_l2,
        weight=-0.08,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limit_l1,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTION_JOINTS_13)},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    energy = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTION_JOINTS_13)},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide_penalty,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES),
            "force_threshold": 1.0,
        },
    )
    undesired_contact = RewTerm(
        func=mdp.undesired_contact_penalty,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=UNDESIRED_CONTACT_BODIES),
            "force_threshold": 1.0,
        },
    )
    waist_yaw_hold = RewTerm(
        func=mdp.waist_yaw_abs_l2,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])},
    )
    terminating = RewTerm(func=mdp.is_terminated, weight=-200.0)

    tracking_error = RewTerm(
        func=mdp.forward_velocity_tracking_error_yaw_frame,
        weight=-1.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "command_threshold": 0.1,
        },
    )
    no_progress = RewTerm(
        func=mdp.no_forward_progress_penalty,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "command_threshold": 0.1,
        },
    )
    slip_metric = RewTerm(
        func=mdp.feet_slide_penalty,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES),
            "force_threshold": 1.0,
        },
    )
    contact_ratio = RewTerm(
        func=mdp.feet_contact_ratio,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "force_threshold": 1.0,
        },
    )


@configclass
class WalkTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fall = DoneTerm(
        func=mdp.fall_by_height,
        params={"asset_cfg": SceneEntityCfg("robot"), "min_height": 0.42},
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 1.0},
    )
    undesired_contact = DoneTerm(
        func=mdp.undesired_contact_termination,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=UNDESIRED_CONTACT_BODIES),
            "force_threshold": 5.0,
        },
    )


@configclass
class HuanWalkEnvCfg(ManagerBasedRLEnvCfg):
    scene: WalkSceneCfg = WalkSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: WalkObservationsCfg = WalkObservationsCfg()
    actions: WalkActionsCfg = WalkActionsCfg()
    commands: WalkCommandsCfg = WalkCommandsCfg()
    rewards: WalkRewardsCfg = WalkRewardsCfg()
    events: WalkEventsCfg = WalkEventsCfg()
    terminations: WalkTerminationsCfg = WalkTerminationsCfg()

    def __post_init__(self):
        self.sim.dt = 0.005
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (8.0, 8.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class HuanWalkEnvCfg_PLAY(HuanWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 3.0