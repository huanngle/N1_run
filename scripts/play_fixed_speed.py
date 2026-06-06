"""Play a trained Project 2 RSL-RL policy at one fixed forward speed.

Example:
    python scripts/play_fixed_speed.py --task Isaac-HuanPJ2-v0 --num_envs 4 \
        --checkpoint /path/to/model_4500.pt --speed 0.8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ISAACLAB_DIR = Path(os.environ.get("ISAACLAB_PATH", Path.home() / "Huan" / "IsaacLab"))
RSL_RL_SCRIPT_DIR = ISAACLAB_DIR / "scripts" / "reinforcement_learning" / "rsl_rl"
if str(RSL_RL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))

from isaaclab.app import AppLauncher  
import cli_args  

parser = argparse.ArgumentParser(description="Visualize a Project 2 humanoid policy at a fixed speed.")
parser.add_argument("--task", type=str, default="Isaac-HuanPJ2-v0", help="Gym task id.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RSL-RL agent config entry point.")
parser.add_argument("--num_envs", type=int, default=4, help="Parallel envs to show.")
parser.add_argument("--speed", type=float, default=0.8, help="Fixed forward speed command in m/s.")
parser.add_argument("--yaw", type=float, default=0.0, help="Fixed yaw-rate command in rad/s.")
parser.add_argument(
    "--condition",
    choices=("nominal", "push"),
    default="nominal",
    help="Use push to include the final-stage disturbance push range.",
)
parser.add_argument("--seed", type=int, default=None, help="Evaluation seed.")
parser.add_argument(
    "--print_every",
    type=int,
    default=100,
    help="Print command/tracking diagnostics every N simulation steps. Use 0 to disable.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  
import torch  
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  
from isaaclab.utils.assets import retrieve_file_path  
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  

import isaaclab_tasks 
import proj2  
from isaaclab_tasks.utils import get_checkpoint_path  
from isaaclab_tasks.utils.hydra import hydra_task_config  


def _set_fixed_command(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, speed: float, yaw: float, condition: str):
    """Force every command sample to one speed/yaw command for visual inspection."""
    command = env_cfg.commands.base_velocity
    command.ranges.lin_vel_x = (speed, speed)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (yaw, yaw)
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.rel_standing_envs = 1.0 if abs(speed) < 1.0e-6 and abs(yaw) < 1.0e-6 else 0.0

    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "curriculum"):
        push_range = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
        if condition == "push":
            push_range = {"x": (-0.25, 0.25), "y": (-0.20, 0.20)}
        env_cfg.events.curriculum.params["stages"] = (
            {
                "max_common_step": 10_000_000,
                "lin_vel_x": (speed, speed),
                "ang_vel_z": (yaw, yaw),
                "rel_standing_envs": 1.0 if abs(speed) < 1.0e-6 and abs(yaw) < 1.0e-6 else 0.0,
                "push_velocity_range": push_range,
            },
        )


def _checkpoint_path(agent_cfg: RslRlBaseRunnerCfg) -> str:
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    return get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    _set_fixed_command(env_cfg, args_cli.speed, args_cli.yaw, args_cli.condition)

    resume_path = _checkpoint_path(agent_cfg)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    print(f"[INFO] Fixed command: vx={args_cli.speed:.3f} m/s, yaw={args_cli.yaw:.3f} rad/s")

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(gym_env.unwrapped, DirectMARLEnv):
        gym_env = multi_agent_to_single_agent(gym_env)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=base_env.device)
    policy_nn = getattr(runner.alg, "policy", getattr(runner.alg, "actor_critic", None))
    robot = base_env.scene["robot"]
    obs = env.get_observations()
    step = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if policy_nn is not None and hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)

            if args_cli.print_every > 0 and step % args_cli.print_every == 0:
                vx = robot.data.root_lin_vel_b[:, 0].mean().item()
                wz = robot.data.root_ang_vel_w[:, 2].mean().item()
                actual_cmd = base_env.command_manager.get_command("base_velocity")
                cmd_vx = actual_cmd[:, 0].mean().item()
                cmd_yaw = actual_cmd[:, 2].mean().item()
                print(
                    f"[step {step:06d}] vx={vx:+.3f} m/s "
                    f"(cmd {cmd_vx:+.3f}), yaw_rate={wz:+.3f} rad/s "
                    f"(cmd {cmd_yaw:+.3f})"
                )
            step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
