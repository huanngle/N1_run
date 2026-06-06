from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================================================
# OBSERVATION HELPERS
# ============================================================
def feet_height_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return robot.data.body_pos_w[:, asset_cfg.body_ids, 2]


def base_height_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return robot.data.root_pos_w[:, 2:3]


def feet_contact_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    if hasattr(contact_sensor.data, "net_forces_w_history"):
        forces = contact_sensor.data.net_forces_w_history
        force_mag = forces.norm(dim=-1).max(dim=1)[0]
        return (force_mag[:, sensor_cfg.body_ids] > float(force_threshold)).float()

    if hasattr(contact_sensor.data, "net_forces_w"):
        forces = contact_sensor.data.net_forces_w
        force_mag = forces.norm(dim=-1)
        return (force_mag[:, sensor_cfg.body_ids] > float(force_threshold)).float()

    return torch.zeros(
        (env.num_envs, len(sensor_cfg.body_ids)),
        device=env.device,
        dtype=torch.float32,
    )


def feet_speed_xy_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    vel_xy = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return vel_xy.reshape(env.num_envs, -1)

def feet_speed_xy_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    vel_xy = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(torch.square(vel_xy), dim=(1, 2))


def _get_command_tensor(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return torch.zeros((env.num_envs, 3), device=env.device)

    if hasattr(command_manager, "get_command"):
        return command_manager.get_command(command_name)

    if hasattr(command_manager, "_terms") and command_name in command_manager._terms:
        term = command_manager._terms[command_name]
        if hasattr(term, "command"):
            return term.command

    return torch.zeros((env.num_envs, 3), device=env.device)


def _get_named_manager_term(manager, term_name: str):
    if manager is None:
        return None
    if hasattr(manager, "_terms") and term_name in manager._terms:
        return manager._terms[term_name]
    if hasattr(manager, "terms") and term_name in manager.terms:
        return manager.terms[term_name]
    return None


def _log_scalar_metric(env: ManagerBasedRLEnv, name: str, value: torch.Tensor):
    if hasattr(env, "extras") and isinstance(env.extras, dict):
        env.extras.setdefault("log", {})
        env.extras["log"][name] = float(torch.mean(value).item())


def command_observation(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    return _get_command_tensor(env, command_name)



def height_scan(
    env: ManagerBasedRLEnv,
    num_rays: int = 8,
) -> torch.Tensor:
    return torch.zeros((env.num_envs, num_rays), device=env.device, dtype=torch.float32)


def surrounding_height_offsets(
    env: ManagerBasedRLEnv,
    num_points: int = 8,
) -> torch.Tensor:
    return torch.zeros((env.num_envs, num_points), device=env.device, dtype=torch.float32)


# ============================================================
# REWARDS
# ============================================================
def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)

def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.termination_manager.terminated.float()

def root_lin_vel_error_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.root_lin_vel_w[:, :3]), dim=1)

def root_ang_vel_error_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.root_ang_vel_w[:, :3]), dim=1)

def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    command = _get_command_tensor(env, command_name)
    vel_error = command[:, :2] - robot.data.root_lin_vel_b[:, :2]
    return torch.exp(-torch.sum(torch.square(vel_error), dim=1) / (float(std) ** 2))


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    command = _get_command_tensor(env, command_name)
    vel_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3])
    vel_error = command[:, :2] - vel_yaw[:, :2]
    return torch.exp(-torch.sum(torch.square(vel_error), dim=1) / (float(std) ** 2))


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    command = _get_command_tensor(env, command_name)
    ang_error = command[:, 2] - robot.data.root_ang_vel_b[:, 2]
    return torch.exp(-torch.square(ang_error) / (float(std) ** 2))


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    command = _get_command_tensor(env, command_name)
    ang_error = command[:, 2] - robot.data.root_ang_vel_w[:, 2]
    return torch.exp(-torch.square(ang_error) / (float(std) ** 2))


def forward_velocity_tracking_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    command = _get_command_tensor(env, command_name)
    return torch.abs(robot.data.root_lin_vel_b[:, 0] - command[:, 0])


def forward_velocity_tracking_error_yaw_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    command = _get_command_tensor(env, command_name)
    vel_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3])
    move_mask = torch.abs(command[:, 0]) > float(command_threshold)
    return torch.abs(vel_yaw[:, 0] - command[:, 0]) * move_mask.float()


def no_forward_progress_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    command = _get_command_tensor(env, command_name)
    vel_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3])
    move_mask = command[:, 0] > float(command_threshold)
    shortfall = torch.clamp(command[:, 0] - vel_yaw[:, 0], min=0.0)
    return shortfall * move_mask.float()


def upright_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    g_b = robot.data.projected_gravity_b
    err = torch.abs(g_b[:, 2] + 1.0)
    return torch.exp(-6.0 * err)

def base_yaw_ang_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return torch.square(robot.data.root_ang_vel_w[:, 2])


def base_xy_ang_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.root_ang_vel_w[:, :2]), dim=1)


def base_z_lin_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return torch.square(robot.data.root_lin_vel_w[:, 2])


def joint_pos_limit_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    lim = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, :]
    q_min = lim[..., 0]
    q_max = lim[..., 1]

    below = torch.clamp(q_min - q, min=0.0)
    above = torch.clamp(q - q_max, min=0.0)
    return torch.sum(below + above, dim=1)


def root_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    z = robot.data.root_pos_w[:, 2]
    err = torch.abs(z - float(target_height))
    return torch.exp(-12.0 * err)


def undesired_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    if hasattr(contact_sensor.data, "net_forces_w_history"):
        forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        force_mag = forces.norm(dim=-1).max(dim=1)[0]
    elif hasattr(contact_sensor.data, "net_forces_w"):
        forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
        force_mag = forces.norm(dim=-1)
    else:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    return torch.sum((force_mag > float(force_threshold)).float(), dim=1)

def joint_pos_target_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    err = torch.mean(torch.abs(q - q0), dim=1)
    return torch.exp(-1.0 * err)

def zero_velocity_command(env: ManagerBasedRLEnv, dim: int = 3) -> torch.Tensor:
    return torch.zeros((env.num_envs, dim), device=env.device)


def stand_still_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.1,
    yaw_threshold: float = 0.1,
) -> torch.Tensor:
    command = _get_command_tensor(env, command_name)
    stand_mask = (
        torch.norm(command[:, :2], dim=1) < float(command_threshold)
    ) & (torch.abs(command[:, 2]) < float(yaw_threshold))
    pose_reward = joint_pos_target_l2(env, asset_cfg)
    return pose_reward * stand_mask.float()

def waist_yaw_abs_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.square(q[:, 0])


def feet_slide_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    contact = feet_contact_state(env, sensor_cfg, force_threshold=force_threshold)
    robot = env.scene[asset_cfg.name]
    foot_speed = torch.norm(robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    value = torch.sum(contact * foot_speed, dim=1)
    _log_scalar_metric(env, "Metrics/slip_metric", value)
    _log_scalar_metric(env, "Metrics/contact_ratio", torch.mean(contact, dim=1))
    return value


def feet_contact_ratio(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    contact = feet_contact_state(env, sensor_cfg, force_threshold=force_threshold)
    return torch.mean(contact, dim=1)


def feet_air_time_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.2,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    command = _get_command_tensor(env, command_name)
    move_mask = torch.norm(command[:, :2], dim=1) > float(command_threshold)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if hasattr(contact_sensor.data, "last_air_time"):
        air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    elif hasattr(contact_sensor.data, "current_air_time"):
        air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    else:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    contact = feet_contact_state(env, sensor_cfg, force_threshold=force_threshold)
    touchdown_bonus = torch.sum(air_time * contact, dim=1)
    return touchdown_bonus * move_mask.float()


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.4,
    min_speed: float = 0.6,
) -> torch.Tensor:
    command = _get_command_tensor(env, command_name)
    speed = torch.norm(command[:, :2], dim=1)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if not hasattr(contact_sensor.data, "current_air_time") or not hasattr(contact_sensor.data, "current_contact_time"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=float(threshold))
    return reward * (speed > float(min_speed)).float()


def single_support_moving_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    speed_threshold: float = 0.2,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    command = _get_command_tensor(env, command_name)
    move_mask = torch.norm(command[:, :2], dim=1) > float(speed_threshold)
    contact = feet_contact_state(env, sensor_cfg, force_threshold=force_threshold)
    single_support = torch.sum(contact, dim=1) == 1.0
    return (move_mask & single_support).float()


def double_support_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    speed_threshold: float = 0.2,
    yaw_threshold: float = 0.2,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    command = _get_command_tensor(env, command_name)
    slow_mask = (
        torch.norm(command[:, :2], dim=1) < float(speed_threshold)
    ) & (torch.abs(command[:, 2]) < float(yaw_threshold))
    contact = feet_contact_state(env, sensor_cfg, force_threshold=force_threshold)
    both_feet = (torch.sum(contact, dim=1) >= 2.0).float()
    return both_feet * slow_mask.float()


def fly_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.2,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    command = _get_command_tensor(env, command_name)
    move_mask = torch.norm(command[:, :2], dim=1) > float(command_threshold)
    contact = feet_contact_state(env, sensor_cfg, force_threshold=force_threshold)
    no_contact = torch.sum(contact, dim=1) < 0.5
    return (move_mask & no_contact).float()


# ============================================================
# TERMINATIONS
# ============================================================

def fall_by_height(
    env: ManagerBasedRLEnv,
    min_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return robot.data.root_pos_w[:, 2] < float(min_height)


def bad_orientation(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    gravity_z = torch.clamp(robot.data.projected_gravity_b[:, 2], -1.0, 1.0)
    angle = torch.acos(-gravity_z)
    return angle > float(limit_angle)


def undesired_contact_termination(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    if hasattr(contact_sensor.data, "net_forces_w_history"):
        forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        force_mag = forces.norm(dim=-1).max(dim=1)[0]
    elif hasattr(contact_sensor.data, "net_forces_w"):
        forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
        force_mag = forces.norm(dim=-1)
    else:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    return torch.any(force_mag > float(force_threshold), dim=1)


# ============================================================
# EVENTS 
# ============================================================


def push_by_setting_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    velocity_range: dict,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    robot = env.scene[asset_cfg.name]
    vel = robot.data.root_vel_w[env_ids].clone()

    if hasattr(env, "_proj2_curriculum_push_velocity_range"):
        velocity_range = env._proj2_curriculum_push_velocity_range

    vel[:, 0] += torch.empty(len(env_ids), device=env.device).uniform_(
        velocity_range["x"][0], velocity_range["x"][1]
    )
    vel[:, 1] += torch.empty(len(env_ids), device=env.device).uniform_(
        velocity_range["y"][0], velocity_range["y"][1]
    )

    robot.write_root_velocity_to_sim(vel, env_ids)

def reset_joints_by_scale(
    env: ManagerBasedRLEnv, 
    env_ids: torch.Tensor, 
    position_range: tuple, 
    velocity_range: tuple, 
    asset_cfg: SceneEntityCfg
):
    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids]
    
    scales = torch.distributions.Uniform(position_range[0], position_range[1]).sample(joint_pos.shape).to(env.device)
    joint_pos *= scales
    
    joint_vel = torch.zeros_like(joint_pos) 
    robot.write_joint_state_to_sim(joint_pos, joint_vel, asset_cfg.joint_ids, env_ids)


def update_walk_run_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    stages: tuple,
):
    del env_ids

    common_step_counter = int(getattr(env, "common_step_counter", 0))
    active_stage = stages[-1]
    active_stage_idx = len(stages) - 1

    for idx, stage in enumerate(stages):
        if common_step_counter < int(stage["max_common_step"]):
            active_stage = stage
            active_stage_idx = idx
            break

    command_term = _get_named_manager_term(getattr(env, "command_manager", None), command_name)
    if command_term is not None and hasattr(command_term, "cfg") and hasattr(command_term.cfg, "ranges"):
        command_term.cfg.ranges.lin_vel_x = tuple(active_stage["lin_vel_x"])
        command_term.cfg.ranges.lin_vel_y = (0.0, 0.0)
        command_term.cfg.ranges.ang_vel_z = tuple(active_stage["ang_vel_z"])
        if "rel_standing_envs" in active_stage:
            command_term.cfg.rel_standing_envs = float(active_stage["rel_standing_envs"])

    env._proj2_curriculum_push_velocity_range = dict(active_stage["push_velocity_range"])
    env._proj2_curriculum_stage = active_stage_idx + 1

    if hasattr(env, "extras") and isinstance(env.extras, dict):
        env.extras.setdefault("log", {})
        env.extras["log"]["Curriculum/stage"] = float(active_stage_idx + 1)
