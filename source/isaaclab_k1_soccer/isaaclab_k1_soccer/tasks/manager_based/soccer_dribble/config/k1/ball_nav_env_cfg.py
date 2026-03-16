# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""K1 ボールドリブルナビゲーション環境。

navigation と同じ階層型制御アーキテクチャ：
  - 低レベル: 学習済み歩行方策（凍結）
  - 高レベル: ボールを前方向に押すための速度指令を出力

レイアウト (各環境):
    [Robot]  →  [Ball]  ·····  [Goal]
     x≈0         x=0.5~3.0      x=5.0

エピソード開始時にボールをランダム位置（前方 0.5~3.0 m）に配置し、
pose_command がボール位置を追跡。ロボットはボールに近づいて前方に押す。

報酬:
  + ball_forward_velocity      : ボールをロボットの向いている方向に押す
  - ball_non_forward_velocity  : 横・後ろ方向へのボール移動はペナルティ
  + position_tracking          : ボール（目標座標）への接近報酬
  - orientation_tracking       : ボール方向への向き追従
  - termination_penalty        : 転倒ペナルティ
"""

import math

import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from soccer_humanoid.tasks.manager_based.navigation.mdp import (
    BallGoalPose2dCommandCfg,
    ball_forward_velocity,
    ball_non_forward_velocity_penalty,
    reset_ball_to_goal_range,
)
from soccer_humanoid.tasks.manager_based.soccer_dribble.dribble_env_cfg import SoccerDribbleSceneCfg

from isaaclab_k1_soccer.tasks.manager_based.locomotion.config.k1.flat_env_cfg import K1FlatEnvCfg

# ---------------------------------------------------------------------------
# 低レベル歩行方策（凍結）
# ---------------------------------------------------------------------------
LOW_LEVEL_ENV_CFG = K1FlatEnvCfg()

K1_FLAT_POLICY_PATH = (
    "/workspace/isaaclab_k1_soccer/logs/rsl_rl/k1_flat/2026-03-09_08-24-32/exported/policy.pt"
)

# ボール配置範囲（env origin からの相対位置）
_GOAL_RANGE_X = (0.5, 3.0)   # 前方 0.5~3.0 m
_GOAL_RANGE_Y = (-1.5, 1.5)  # 横 ±1.5 m
_BALL_RADIUS = 0.11


# ---------------------------------------------------------------------------
# シーン: K1 ロボット + ボール + ゴール（ビジュアル）
# ---------------------------------------------------------------------------


@configclass
class K1BallNavSceneCfg(SoccerDribbleSceneCfg):
    """K1 ロボット + ボール（ナビゲーション目標）+ ゴール（視覚的）を含むシーン。

    SoccerDribbleSceneCfg をそのまま継承:
      - terrain       : フラット地面
      - robot         : __post_init__ で K1FlatEnvCfg から設定
      - ball          : 動的剛体球（ナビゲーション目標）
      - goal          : 静的ビジュアル（前方 5m）
      - contact_forces: コンタクトセンサー
    """
    pass


# ---------------------------------------------------------------------------
# MDP コンポーネント
# ---------------------------------------------------------------------------


@configclass
class EventCfg:
    """リセットイベント設定。"""

    # ロボット初期位置をランダム化（向きはゴール方向＋X に固定）
    reset_base = EventTerm(
        func=nav_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.3, 0.3)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # ボールを目標座標（ランダム）に配置（command_manager.reset() より前に実行）
    reset_ball = EventTerm(
        func=reset_ball_to_goal_range,
        mode="reset",
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "pos_range_x": _GOAL_RANGE_X,
            "pos_range_y": _GOAL_RANGE_Y,
            "ball_radius": _BALL_RADIUS,
        },
    )


@configclass
class ActionsCfg:
    """アクション設定: 凍結歩行方策への速度指令。"""

    pre_trained_policy_action: nav_mdp.PreTrainedPolicyActionCfg = nav_mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=K1_FLAT_POLICY_PATH,
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )


@configclass
class ObservationsCfg:
    """観測設定 (9次元): 速度 + 重力 + ボール目標位置。"""

    @configclass
    class PolicyCfg(ObsGroup):
        # ロボット線速度 (3次元)
        base_lin_vel = ObsTerm(func=nav_mdp.base_lin_vel)
        # 重力方向 (3次元)
        projected_gravity = ObsTerm(func=nav_mdp.projected_gravity)
        # ボール目標位置コマンド: ロボット相対 (x, y, heading) (3次元)
        pose_command = ObsTerm(func=nav_mdp.generated_commands, params={"command_name": "pose_command"})

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """報酬設定。"""

    # 転倒ペナルティ
    termination_penalty = RewTerm(func=nav_mdp.is_terminated, weight=-400.0)

    # ボール（目標座標）への接近報酬（粗い + 細かい）
    position_tracking = RewTerm(
        func=nav_mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=nav_mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 0.2, "command_name": "pose_command"},
    )

    # ボール方向への向き追従
    orientation_tracking = RewTerm(
        func=nav_mdp.heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "pose_command"},
    )

    # ボールをロボットの向いている方向（前方）に押す: +報酬
    ball_forward_vel = RewTerm(
        func=ball_forward_velocity,
        weight=2.0,
        params={"ball_cfg": SceneEntityCfg("ball")},
    )

    # ボールを横・後ろに動かす: -ペナルティ
    ball_non_forward_vel_penalty = RewTerm(
        func=ball_non_forward_velocity_penalty,
        weight=-3.0,
        params={"ball_cfg": SceneEntityCfg("ball")},
    )


@configclass
class CommandsCfg:
    """コマンド設定: pose_command がボール位置を毎ステップ追跡。"""

    pose_command: BallGoalPose2dCommandCfg = BallGoalPose2dCommandCfg(
        asset_name="robot",
        ball_name="ball",
        # simple_heading=True: ロボットの向きをボール方向に自動計算
        simple_heading=True,
        resampling_time_range=(20.0, 20.0),
        debug_vis=True,
        ranges=BallGoalPose2dCommandCfg.Ranges(
            pos_x=_GOAL_RANGE_X,
            pos_y=_GOAL_RANGE_Y,
            heading=(0.0, 0.0),  # simple_heading=True のため未使用
        ),
    )


@configclass
class TerminationsCfg:
    """終了条件。"""

    time_out = DoneTerm(func=nav_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=nav_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Trunk"),
            "threshold": 1.0,
        },
    )


# ---------------------------------------------------------------------------
# トップレベル環境設定
# ---------------------------------------------------------------------------


@configclass
class K1BallDribbleNavEnvCfg(ManagerBasedRLEnvCfg):
    """K1 ボールドリブルナビゲーション環境。

    凍結歩行方策 + 高レベル RL:
      観測 (9次元): base_lin_vel + projected_gravity + pose_command
      行動 (3次元): [vx, vy, ω] 速度指令 → 歩行方策に渡す
    """

    scene: K1BallNavSceneCfg = K1BallNavSceneCfg(num_envs=4096, env_spacing=8.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # K1FlatEnvCfg のロボットとコンタクトセンサーをシーンに設定
        self.scene.robot = LOW_LEVEL_ENV_CFG.scene.robot
        self.scene.contact_forces = LOW_LEVEL_ENV_CFG.scene.contact_forces

        # フラット地形（高さスキャン不要）
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # タイムステップ設定（歩行方策と同期）
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class K1BallDribbleNavEnvCfg_PLAY(K1BallDribbleNavEnvCfg):
    """評価用設定: 小規模、観測ノイズなし。"""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 8.0
        self.observations.policy.enable_corruption = False
