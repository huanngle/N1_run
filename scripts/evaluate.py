"""Evaluate a trained Project 2 RSL-RL policy at one fixed forward speed.

Example:
    python scripts/evaluate.py --task Isaac-HuanPJ2-v0 --num_envs 64 \
        --checkpoint /path/to/model_5000.pt --speed 1.5 --episodes_per_speed 100
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


ISAACLAB_DIR = Path(os.environ.get("ISAACLAB_PATH", Path.home() / "Huan" / "IsaacLab"))
RSL_RL_SCRIPT_DIR = ISAACLAB_DIR / "scripts" / "reinforcement_learning" / "rsl_rl"
if str(RSL_RL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))

from isaaclab.app import AppLauncher  

import cli_args  


parser = argparse.ArgumentParser(description="Evaluate Project 2 humanoid locomotion checkpoints.")
parser.add_argument("--task", type=str, default="Isaac-HuanPJ2-v0", help="Gym task id.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RSL-RL agent config entry point.")
parser.add_argument("--num_envs", type=int, default=64, help="Parallel envs for evaluation.")
parser.add_argument("--episodes_per_speed", type=int, default=100, help="Completed episodes to collect per speed.")
parser.add_argument("--speed", type=float, default=0.8, help="Fixed forward speed command in m/s.")
parser.add_argument(
    "--condition",
    choices=("nominal", "push"),
    default="nominal",
    help="Evaluation condition. Use push for disturbance-push robustness.",
)
parser.add_argument("--output", type=str, default=None, help="CSV output path.")
parser.add_argument("--seed", type=int, default=None, help="Evaluation seed.")
parser.add_argument(
    "--progress_every",
    type=int,
    default=500,
    help="Print evaluation progress every N env steps. Use 0 to disable.",
)
parser.add_argument(
    "--debug_contact",
    action="store_true",
    help="Print contact/slip debug statistics during evaluation.",
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
from isaaclab.managers import SceneEntityCfg  
from isaaclab.utils.assets import retrieve_file_path  
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  

import isaaclab_tasks  
import proj2  
from isaaclab_tasks.utils import get_checkpoint_path  
from isaaclab_tasks.utils.hydra import hydra_task_config  


def _set_fixed_command(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, speed: float, condition: str):
    """Force all command samples to a single forward speed during evaluation."""
    command = env_cfg.commands.base_velocity
    command.ranges.lin_vel_x = (speed, speed)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.rel_standing_envs = 1.0 if abs(speed) < 1.0e-6 else 0.0


    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "curriculum"):
        push_range = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
        if condition == "push":
            push_range = {"x": (-0.5, 0.5), "y": (-0.35, 0.35)}
        env_cfg.events.curriculum.params["stages"] = (
            {
                "max_common_step": 10_000_000,
                "lin_vel_x": (speed, speed),
                "ang_vel_z": (0.0, 0.0),
                "rel_standing_envs": 1.0 if abs(speed) < 1.0e-6 else 0.0,
                "push_velocity_range": push_range,
            },
        )


def _scene_body_ids(scene, entity_name: str, body_names: list[str]) -> list[int]:
    cfg = SceneEntityCfg(entity_name, body_names=body_names)
    cfg.resolve(scene)
    return list(cfg.body_ids)


def _contact_metrics(
    base_env,
    robot_foot_body_ids: list[int],
    sensor_foot_body_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sensor = base_env.scene.sensors["contact_forces"]
    robot = base_env.scene["robot"]
    if hasattr(sensor.data, "net_forces_w_history"):
        forces = sensor.data.net_forces_w_history[:, :, sensor_foot_body_ids, :]
        force_mag = forces.norm(dim=-1).max(dim=1)[0]
    else:
        forces = sensor.data.net_forces_w[:, sensor_foot_body_ids, :]
        force_mag = forces.norm(dim=-1)
    contact = (force_mag > 1.0).float()
    foot_speed = robot.data.body_lin_vel_w[:, robot_foot_body_ids, :2].norm(dim=-1)
    slip = (contact * foot_speed).sum(dim=1)
    contact_ratio = contact.mean(dim=1)
    return slip, contact_ratio, contact, foot_speed


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

    resume_path = _checkpoint_path(agent_cfg)
    output_path = args_cli.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(resume_path), f"evaluation_{args_cli.condition}.csv")

    rows: list[dict[str, float | int | str]] = []
    for speed_index, speed in enumerate((args_cli.speed,), start=1):
        print(
            f"[eval] starting speed {speed_index}/1: "
            f"{speed:.2f} m/s, condition={args_cli.condition}",
            flush=True,
        )
        _set_fixed_command(env_cfg, speed, args_cli.condition)
        print("[eval] creating environment...", flush=True)
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
        print(f"[eval] loading checkpoint: {resume_path}", flush=True)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=base_env.device)
        policy_nn = getattr(runner.alg, "policy", getattr(runner.alg, "actor_critic", None))

        robot = base_env.scene["robot"]
        foot_body_names = ["left_foot_pitch_link", "right_foot_pitch_link"]
        robot_foot_body_ids = _scene_body_ids(base_env.scene, "robot", foot_body_names)
        sensor_foot_body_ids = _scene_body_ids(base_env.scene, "contact_forces", foot_body_names)
        term_names = list(base_env.termination_manager.active_terms)
        term_counts = {name: 0 for name in term_names}

        ep_track_sum = torch.zeros(base_env.num_envs, device=base_env.device)
        ep_vx_sum = torch.zeros_like(ep_track_sum)
        ep_slip_sum = torch.zeros_like(ep_track_sum)
        ep_contact_sum = torch.zeros_like(ep_track_sum)
        ep_steps = torch.zeros_like(ep_track_sum)

        completed = 0
        successes = 0
        vx_total = 0.0
        episode_length_total = 0.0
        tracking_total = 0.0
        slip_total = 0.0
        contact_total = 0.0
        obs = env.get_observations()
        step_count = 0

        while simulation_app.is_running() and completed < args_cli.episodes_per_speed:
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                step_count += 1
                if policy_nn is not None and hasattr(policy_nn, "reset"):
                    policy_nn.reset(dones)

                vx = robot.data.root_lin_vel_b[:, 0]
                ep_vx_sum += vx
                ep_track_sum += torch.abs(vx - speed)
                slip, contact_ratio, contact, foot_speed = _contact_metrics(
                    base_env, robot_foot_body_ids, sensor_foot_body_ids
                )
                ep_slip_sum += slip
                ep_contact_sum += contact_ratio
                ep_steps += 1.0

                if args_cli.progress_every > 0 and step_count % args_cli.progress_every == 0:
                    vx_mean = robot.data.root_lin_vel_b[:, 0].mean().item()
                    print(
                        f"[eval] speed={speed:.2f} completed={completed}/"
                        f"{args_cli.episodes_per_speed} steps={step_count} mean_vx={vx_mean:.3f}",
                        flush=True,
                    )
                    if args_cli.debug_contact:
                        print(
                            f"[contact_debug] contact_mean={contact.mean().item():.4f} "
                            f"foot_speed_mean={foot_speed.mean().item():.4f} "
                            f"slip_mean={slip.mean().item():.4f}",
                            flush=True,
                        )

                done_ids = dones.nonzero(as_tuple=False).flatten()
                if done_ids.numel() == 0:
                    continue

                remaining = args_cli.episodes_per_speed - completed
                record_ids = done_ids[:remaining]
                safe_steps = torch.clamp(ep_steps[record_ids], min=1.0)
                vx_total += torch.sum(ep_vx_sum[record_ids] / safe_steps).item()
                episode_length_total += torch.sum(ep_steps[record_ids]).item()
                tracking_total += torch.sum(ep_track_sum[record_ids] / safe_steps).item()
                slip_total += torch.sum(ep_slip_sum[record_ids] / safe_steps).item()
                contact_total += torch.sum(ep_contact_sum[record_ids] / safe_steps).item()

                time_out = base_env.termination_manager.get_term("time_out")[record_ids]
                successes += int(time_out.sum().item())
                for name in term_names:
                    term_counts[name] += int(base_env.termination_manager.get_term(name)[record_ids].sum().item())

                completed += int(record_ids.numel())
                ep_track_sum[done_ids] = 0.0
                ep_vx_sum[done_ids] = 0.0
                ep_slip_sum[done_ids] = 0.0
                ep_contact_sum[done_ids] = 0.0
                ep_steps[done_ids] = 0.0

        env.close()

        denom = max(completed, 1)
        term_rates = {name: term_counts[name] / denom for name in term_names}
        main_termination_reason = max(term_rates, key=term_rates.get) if term_rates else ""
        row: dict[str, float | int | str] = {
            "condition": args_cli.condition,
            "speed_mps": speed,
            "episodes": completed,
            "success_rate": successes / denom,
            "success_rate_percent": 100.0 * successes / denom,
            "mean_vx": vx_total / denom,
            "mean_episode_length": episode_length_total / denom,
            "mean_episode_length_s": (episode_length_total / denom) * float(base_env.step_dt),
            "tracking_error_abs_vx": tracking_total / denom,
            "slip_metric": slip_total / denom,
            "contact_ratio": contact_total / denom,
            "main_termination_reason": main_termination_reason,
        }
        for name in term_names:
            row[f"termination_{name}"] = term_rates[name]
            row[f"termination_{name}_percent"] = 100.0 * term_rates[name]
        rows.append(row)
        print(row, flush=True)
        print(f"[eval] finished speed {speed:.2f} m/s", flush=True)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Wrote evaluation results to: {output_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
