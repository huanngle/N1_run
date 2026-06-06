from gymnasium.envs.registration import register

register(
    id="Isaac-HuanPJ2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "proj2.walk_env_cfg:HuanWalkEnvCfg",
        "rsl_rl_cfg_entry_point": "proj2.agents.rsl_rl_ppo_cfg:HumanoidN1PPORunnerCfg",
    },
)
