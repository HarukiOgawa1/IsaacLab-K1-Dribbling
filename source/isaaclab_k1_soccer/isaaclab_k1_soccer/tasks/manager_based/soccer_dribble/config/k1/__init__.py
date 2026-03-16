# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# ---------------------------------------------------------------------------
# K1DribbleEnvCfg — Full dribble (54-dim obs, train from scratch)
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-Dribble-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dribble_env_cfg:K1DribbleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1DribblePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Dribble-K1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dribble_env_cfg:K1DribbleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1DribblePPORunnerCfg_PLAY",
    },
)

# ---------------------------------------------------------------------------
# K1WalkToDribbleEnvCfg — Transfer learning from flat-walking policy (48-dim)
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-WalkToDribble-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dribble_env_cfg:K1WalkToDribbleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkToDribblePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WalkToDribble-K1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dribble_env_cfg:K1WalkToDribbleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1WalkToDribblePPORunnerCfg_PLAY",
    },
)

# ---------------------------------------------------------------------------
# K1BallDribbleNavEnvCfg — Navigation-style ball dribble (9-dim obs)
# 凍結歩行方策 + ボールへのナビゲーション + 前方押し出し報酬
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-BallDribbleNav-K1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_nav_env_cfg:K1BallDribbleNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1BallDribbleNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BallDribbleNav-K1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_nav_env_cfg:K1BallDribbleNavEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:K1BallDribbleNavPPORunnerCfg_PLAY",
    },
)
