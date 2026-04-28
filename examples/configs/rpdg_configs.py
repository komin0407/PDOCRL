from dataclasses import dataclass
from typing import List, Optional, Tuple

from pyrallis import field


@dataclass
class RPDGTrainConfig:
    # wandb params
    project: str = "OSRL-baselines"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "RPDG"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
    # dataset params
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float, float, float], ...] = None
    epsilon: float = None
    density: float = 1.0
    # training params
    task: str = "OfflineCarCircle-v0"
    dataset: str = None
    seed: int = 0
    device: str = "cpu"
    threads: int = 4
    reward_scale: float = 0.1
    cost_scale: float = 1.0
    actor_lr: float = 3e-4
    q_lr: float = 3e-4
    w_lr: float = 3e-4
    lambda_lr: float = 1e-3
    cost_limit: int = 10
    episode_len: int = 300
    batch_size: int = 512
    update_steps: int = 100_000
    num_workers: int = 8
    # model params
    a_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    c_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    w_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    gamma: float = 0.99
    tau: float = 0.005
    num_q: int = 2
    kappa: float = 0.1  # entropy bonus temperature
    rho_w: float = 1e-3  # quadratic regularization for w
    rho_Q: float = 1e-3  # quadratic regularization for Q
    rho_lambda: float = 1e-3  # quadratic regularization for λ
    slater_phi: float = 0.1
    init_lambda: float = 1.0
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 2500


# -----------------------------------------------------------------------
# per-task default configs (episode_len from bcql_configs)
# -----------------------------------------------------------------------

@dataclass
class RPDGCarCircleConfig(RPDGTrainConfig):
    pass


@dataclass
class RPDGAntRunConfig(RPDGTrainConfig):
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200


@dataclass
class RPDGDroneRunConfig(RPDGTrainConfig):
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200


@dataclass
class RPDGDroneCircleConfig(RPDGTrainConfig):
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300


@dataclass
class RPDGCarRunConfig(RPDGTrainConfig):
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200


@dataclass
class RPDGAntCircleConfig(RPDGTrainConfig):
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500


@dataclass
class RPDGBallRunConfig(RPDGTrainConfig):
    task: str = "OfflineBallRun-v0"
    episode_len: int = 100


@dataclass
class RPDGBallCircleConfig(RPDGTrainConfig):
    task: str = "OfflineBallCircle-v0"
    episode_len: int = 200


@dataclass
class RPDGCarButton1Config(RPDGTrainConfig):
    task: str = "OfflineCarButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class RPDGCarButton2Config(RPDGTrainConfig):
    task: str = "OfflineCarButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class RPDGCarCircle1Config(RPDGTrainConfig):
    task: str = "OfflineCarCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class RPDGCarCircle2Config(RPDGTrainConfig):
    task: str = "OfflineCarCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class RPDGCarGoal1Config(RPDGTrainConfig):
    task: str = "OfflineCarGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class RPDGCarGoal2Config(RPDGTrainConfig):
    task: str = "OfflineCarGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class RPDGCarPush1Config(RPDGTrainConfig):
    task: str = "OfflineCarPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class RPDGCarPush2Config(RPDGTrainConfig):
    task: str = "OfflineCarPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class RPDGPointButton1Config(RPDGTrainConfig):
    task: str = "OfflinePointButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class RPDGPointButton2Config(RPDGTrainConfig):
    task: str = "OfflinePointButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class RPDGPointCircle1Config(RPDGTrainConfig):
    task: str = "OfflinePointCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class RPDGPointCircle2Config(RPDGTrainConfig):
    task: str = "OfflinePointCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class RPDGPointGoal1Config(RPDGTrainConfig):
    task: str = "OfflinePointGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class RPDGPointGoal2Config(RPDGTrainConfig):
    task: str = "OfflinePointGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class RPDGPointPush1Config(RPDGTrainConfig):
    task: str = "OfflinePointPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class RPDGPointPush2Config(RPDGTrainConfig):
    task: str = "OfflinePointPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class RPDGAntVelocityConfig(RPDGTrainConfig):
    task: str = "OfflineAntVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class RPDGHalfCheetahVelocityConfig(RPDGTrainConfig):
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class RPDGHopperVelocityConfig(RPDGTrainConfig):
    task: str = "OfflineHopperVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class RPDGSwimmerVelocityConfig(RPDGTrainConfig):
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class RPDGWalker2dVelocityConfig(RPDGTrainConfig):
    task: str = "OfflineWalker2dVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class RPDGEasySparseConfig(RPDGTrainConfig):
    task: str = "OfflineMetadrive-easysparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class RPDGEasyMeanConfig(RPDGTrainConfig):
    task: str = "OfflineMetadrive-easymean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class RPDGEasyDenseConfig(RPDGTrainConfig):
    task: str = "OfflineMetadrive-easydense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class RPDGMediumSparseConfig(RPDGTrainConfig):
    task: str = "OfflineMetadrive-mediumsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class RPDGMediumMeanConfig(RPDGTrainConfig):
    task: str = "OfflineMetadrive-mediummean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class RPDGMediumDenseConfig(RPDGTrainConfig):
    task: str = "OfflineMetadrive-mediumdense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class RPDGHardSparseConfig(RPDGTrainConfig):
    task: str = "OfflineMetadrive-hardsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class RPDGHardMeanConfig(RPDGTrainConfig):
    task: str = "OfflineMetadrive-hardmean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class RPDGHardDenseConfig(RPDGTrainConfig):
    task: str = "OfflineMetadrive-harddense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


RPDG_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": RPDGCarCircleConfig,
    "OfflineAntRun-v0": RPDGAntRunConfig,
    "OfflineDroneRun-v0": RPDGDroneRunConfig,
    "OfflineDroneCircle-v0": RPDGDroneCircleConfig,
    "OfflineCarRun-v0": RPDGCarRunConfig,
    "OfflineAntCircle-v0": RPDGAntCircleConfig,
    "OfflineBallCircle-v0": RPDGBallCircleConfig,
    "OfflineBallRun-v0": RPDGBallRunConfig,
    # safety_gymnasium: car
    "OfflineCarButton1Gymnasium-v0": RPDGCarButton1Config,
    "OfflineCarButton2Gymnasium-v0": RPDGCarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": RPDGCarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": RPDGCarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": RPDGCarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": RPDGCarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": RPDGCarPush1Config,
    "OfflineCarPush2Gymnasium-v0": RPDGCarPush2Config,
    # safety_gymnasium: point
    "OfflinePointButton1Gymnasium-v0": RPDGPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": RPDGPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": RPDGPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": RPDGPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": RPDGPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": RPDGPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": RPDGPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": RPDGPointPush2Config,
    # safety_gymnasium: velocity
    "OfflineAntVelocityGymnasium-v1": RPDGAntVelocityConfig,
    "OfflineHalfCheetahVelocityGymnasium-v1": RPDGHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": RPDGHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": RPDGSwimmerVelocityConfig,
    "OfflineWalker2dVelocityGymnasium-v1": RPDGWalker2dVelocityConfig,
    # safe_metadrive
    "OfflineMetadrive-easysparse-v0": RPDGEasySparseConfig,
    "OfflineMetadrive-easymean-v0": RPDGEasyMeanConfig,
    "OfflineMetadrive-easydense-v0": RPDGEasyDenseConfig,
    "OfflineMetadrive-mediumsparse-v0": RPDGMediumSparseConfig,
    "OfflineMetadrive-mediummean-v0": RPDGMediumMeanConfig,
    "OfflineMetadrive-mediumdense-v0": RPDGMediumDenseConfig,
    "OfflineMetadrive-hardsparse-v0": RPDGHardSparseConfig,
    "OfflineMetadrive-hardmean-v0": RPDGHardMeanConfig,
    "OfflineMetadrive-harddense-v0": RPDGHardDenseConfig,
}
