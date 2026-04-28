from dataclasses import dataclass
from typing import List, Optional, Tuple

from pyrallis import field


@dataclass
class PDOCRLTrainConfig:
    # wandb params
    project: str = "OSRL-baselines"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "PDOCRL"
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
    slater_phi: float = 0.1
    init_lambda: float = 1.0
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 2500


# -----------------------------------------------------------------------
# per-task default configs (episode_len from bcql_configs)
# -----------------------------------------------------------------------

@dataclass
class PDOCRLCarCircleConfig(PDOCRLTrainConfig):
    pass


@dataclass
class PDOCRLAntRunConfig(PDOCRLTrainConfig):
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200


@dataclass
class PDOCRLDroneRunConfig(PDOCRLTrainConfig):
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200


@dataclass
class PDOCRLDroneCircleConfig(PDOCRLTrainConfig):
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300


@dataclass
class PDOCRLCarRunConfig(PDOCRLTrainConfig):
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200


@dataclass
class PDOCRLAntCircleConfig(PDOCRLTrainConfig):
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500


@dataclass
class PDOCRLBallRunConfig(PDOCRLTrainConfig):
    task: str = "OfflineBallRun-v0"
    episode_len: int = 100


@dataclass
class PDOCRLBallCircleConfig(PDOCRLTrainConfig):
    task: str = "OfflineBallCircle-v0"
    episode_len: int = 200


@dataclass
class PDOCRLCarButton1Config(PDOCRLTrainConfig):
    task: str = "OfflineCarButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRLCarButton2Config(PDOCRLTrainConfig):
    task: str = "OfflineCarButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRLCarCircle1Config(PDOCRLTrainConfig):
    task: str = "OfflineCarCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class PDOCRLCarCircle2Config(PDOCRLTrainConfig):
    task: str = "OfflineCarCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class PDOCRLCarGoal1Config(PDOCRLTrainConfig):
    task: str = "OfflineCarGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRLCarGoal2Config(PDOCRLTrainConfig):
    task: str = "OfflineCarGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRLCarPush1Config(PDOCRLTrainConfig):
    task: str = "OfflineCarPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRLCarPush2Config(PDOCRLTrainConfig):
    task: str = "OfflineCarPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRLPointButton1Config(PDOCRLTrainConfig):
    task: str = "OfflinePointButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRLPointButton2Config(PDOCRLTrainConfig):
    task: str = "OfflinePointButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRLPointCircle1Config(PDOCRLTrainConfig):
    task: str = "OfflinePointCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class PDOCRLPointCircle2Config(PDOCRLTrainConfig):
    task: str = "OfflinePointCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class PDOCRLPointGoal1Config(PDOCRLTrainConfig):
    task: str = "OfflinePointGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRLPointGoal2Config(PDOCRLTrainConfig):
    task: str = "OfflinePointGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRLPointPush1Config(PDOCRLTrainConfig):
    task: str = "OfflinePointPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRLPointPush2Config(PDOCRLTrainConfig):
    task: str = "OfflinePointPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRLAntVelocityConfig(PDOCRLTrainConfig):
    task: str = "OfflineAntVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class PDOCRLHalfCheetahVelocityConfig(PDOCRLTrainConfig):
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class PDOCRLHopperVelocityConfig(PDOCRLTrainConfig):
    task: str = "OfflineHopperVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class PDOCRLSwimmerVelocityConfig(PDOCRLTrainConfig):
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class PDOCRLWalker2dVelocityConfig(PDOCRLTrainConfig):
    task: str = "OfflineWalker2dVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class PDOCRLEasySparseConfig(PDOCRLTrainConfig):
    task: str = "OfflineMetadrive-easysparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDOCRLEasyMeanConfig(PDOCRLTrainConfig):
    task: str = "OfflineMetadrive-easymean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDOCRLEasyDenseConfig(PDOCRLTrainConfig):
    task: str = "OfflineMetadrive-easydense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDOCRLMediumSparseConfig(PDOCRLTrainConfig):
    task: str = "OfflineMetadrive-mediumsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDOCRLMediumMeanConfig(PDOCRLTrainConfig):
    task: str = "OfflineMetadrive-mediummean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDOCRLMediumDenseConfig(PDOCRLTrainConfig):
    task: str = "OfflineMetadrive-mediumdense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDOCRLHardSparseConfig(PDOCRLTrainConfig):
    task: str = "OfflineMetadrive-hardsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDOCRLHardMeanConfig(PDOCRLTrainConfig):
    task: str = "OfflineMetadrive-hardmean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDOCRLHardDenseConfig(PDOCRLTrainConfig):
    task: str = "OfflineMetadrive-harddense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


PDOCRL_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": PDOCRLCarCircleConfig,
    "OfflineAntRun-v0": PDOCRLAntRunConfig,
    "OfflineDroneRun-v0": PDOCRLDroneRunConfig,
    "OfflineDroneCircle-v0": PDOCRLDroneCircleConfig,
    "OfflineCarRun-v0": PDOCRLCarRunConfig,
    "OfflineAntCircle-v0": PDOCRLAntCircleConfig,
    "OfflineBallCircle-v0": PDOCRLBallCircleConfig,
    "OfflineBallRun-v0": PDOCRLBallRunConfig,
    # safety_gymnasium: car
    "OfflineCarButton1Gymnasium-v0": PDOCRLCarButton1Config,
    "OfflineCarButton2Gymnasium-v0": PDOCRLCarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": PDOCRLCarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": PDOCRLCarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": PDOCRLCarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": PDOCRLCarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": PDOCRLCarPush1Config,
    "OfflineCarPush2Gymnasium-v0": PDOCRLCarPush2Config,
    # safety_gymnasium: point
    "OfflinePointButton1Gymnasium-v0": PDOCRLPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": PDOCRLPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": PDOCRLPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": PDOCRLPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": PDOCRLPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": PDOCRLPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": PDOCRLPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": PDOCRLPointPush2Config,
    # safety_gymnasium: velocity
    "OfflineAntVelocityGymnasium-v1": PDOCRLAntVelocityConfig,
    "OfflineHalfCheetahVelocityGymnasium-v1": PDOCRLHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": PDOCRLHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": PDOCRLSwimmerVelocityConfig,
    "OfflineWalker2dVelocityGymnasium-v1": PDOCRLWalker2dVelocityConfig,
    # safe_metadrive
    "OfflineMetadrive-easysparse-v0": PDOCRLEasySparseConfig,
    "OfflineMetadrive-easymean-v0": PDOCRLEasyMeanConfig,
    "OfflineMetadrive-easydense-v0": PDOCRLEasyDenseConfig,
    "OfflineMetadrive-mediumsparse-v0": PDOCRLMediumSparseConfig,
    "OfflineMetadrive-mediummean-v0": PDOCRLMediumMeanConfig,
    "OfflineMetadrive-mediumdense-v0": PDOCRLMediumDenseConfig,
    "OfflineMetadrive-hardsparse-v0": PDOCRLHardSparseConfig,
    "OfflineMetadrive-hardmean-v0": PDOCRLHardMeanConfig,
    "OfflineMetadrive-harddense-v0": PDOCRLHardDenseConfig,
}
