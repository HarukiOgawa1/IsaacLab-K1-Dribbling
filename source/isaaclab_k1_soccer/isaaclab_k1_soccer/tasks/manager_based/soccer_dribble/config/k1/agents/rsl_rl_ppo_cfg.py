# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO runner configs for the K1 soccer-dribble task.

Re-uses configs from soccer_humanoid and overrides load_run/load_checkpoint
to point to the locally trained k1_flat policy for transfer-learning tasks.
"""

from soccer_humanoid.tasks.manager_based.soccer_dribble.config.k1.agents.rsl_rl_ppo_cfg import (
    K1DribblePPORunnerCfg,
    K1DribblePPORunnerCfg_PLAY,
)
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# Local k1_flat training run (2026-03-09)
_K1_FLAT_RUN = "2026-03-09_08-24-32"
_K1_FLAT_CHECKPOINT = "model_1499.pt"


@configclass
class K1WalkToDribblePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for walk-to-dribble transfer learning.

    Architecture matches K1FlatPPORunnerCfg ([256, 128, 128]) so the
    local flat-walking checkpoint can be loaded without shape mismatch.
    """

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "k1_flat"

    resume = True
    load_run = _K1_FLAT_RUN
    load_checkpoint = _K1_FLAT_CHECKPOINT

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
    )


@configclass
class K1WalkToDribblePPORunnerCfg_PLAY(K1WalkToDribblePPORunnerCfg):
    """Evaluation runner for walk-to-dribble."""
    load_run = ".*"
    load_checkpoint = "model_.*.pt"


@configclass
class K1BallDribbleNavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for the K1 ball-dribble navigation task.

    Navigation-style hierarchical control:
      - Low-level: frozen flat-walking policy
      - High-level: 9-dim obs → 3-dim velocity command [vx, vy, ω]
    Small network [128, 128] is sufficient (same architecture as navigation).
    """

    num_steps_per_env = 8
    max_iterations = 1500
    save_interval = 50
    experiment_name = "k1_ball_dribble_nav"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class K1BallDribbleNavPPORunnerCfg_PLAY(K1BallDribbleNavPPORunnerCfg):
    """Evaluation runner for the ball-dribble navigation task."""
    load_run = ".*"
    load_checkpoint = "model_.*.pt"


__all__ = [
    "K1DribblePPORunnerCfg",
    "K1DribblePPORunnerCfg_PLAY",
    "K1WalkToDribblePPORunnerCfg",
    "K1WalkToDribblePPORunnerCfg_PLAY",
    "K1BallDribbleNavPPORunnerCfg",
    "K1BallDribbleNavPPORunnerCfg_PLAY",
]
