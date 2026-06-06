from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class HumanoidN1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    experiment_name = "PJ2"
    run_name = "walk_run_3p2_aggressive_v1"

    resume = False

    num_steps_per_env = 64
    max_iterations = 6000
    save_interval = 50

    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=8,
        num_mini_batches=16,
        learning_rate=3e-4,
        schedule="adaptive",
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.001,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
    )