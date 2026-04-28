from dataclasses import dataclass
from typing import List, Optional, Tuple

from pyrallis import field


@dataclass
class PDOCRL_WTrainConfig:
    # wandb params
    project: str = "OSRL-baselines"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "PDOCRL-W"
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
    v_lr: float = 3e-4
    w_lr: float = 3e-4
    lambda_lr: float = 1e-3
    cost_limit: int = 10
    episode_len: int = 300
    batch_size: int = 512
    update_steps: int = 100_000
    num_workers: int = 8
    # model params
    a_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    v_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    w_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    gamma: float = 0.99
    tau: float = 0.005
    slater_phi: float = 0.1
    init_lambda: float = 1.0
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 2500


# -----------------------------------------------------------------------
# per-task default configs
# -----------------------------------------------------------------------

@dataclass
class PDOCRL_WCarCircleConfig(PDOCRL_WTrainConfig):
    pass


@dataclass
class PDOCRL_WAntRunConfig(PDOCRL_WTrainConfig):
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200


@dataclass
class PDOCRL_WDroneRunConfig(PDOCRL_WTrainConfig):
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200


@dataclass
class PDOCRL_WDroneCircleConfig(PDOCRL_WTrainConfig):
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300


@dataclass
class PDOCRL_WCarRunConfig(PDOCRL_WTrainConfig):
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200


@dataclass
class PDOCRL_WAntCircleConfig(PDOCRL_WTrainConfig):
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500


@dataclass
class PDOCRL_WBallRunConfig(PDOCRL_WTrainConfig):
    task: str = "OfflineBallRun-v0"
    episode_len: int = 100


@dataclass
class PDOCRL_WBallCircleConfig(PDOCRL_WTrainConfig):
    task: str = "OfflineBallCircle-v0"
    episode_len: int = 200


@dataclass
class PDOCRL_WCarButton1Config(PDOCRL_WTrainConfig):
    task: str = "OfflineCarButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRL_WCarButton2Config(PDOCRL_WTrainConfig):
    task: str = "OfflineCarButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRL_WCarCircle1Config(PDOCRL_WTrainConfig):
    task: str = "OfflineCarCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class PDOCRL_WCarCircle2Config(PDOCRL_WTrainConfig):
    task: str = "OfflineCarCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class PDOCRL_WCarGoal1Config(PDOCRL_WTrainConfig):
    task: str = "OfflineCarGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRL_WCarGoal2Config(PDOCRL_WTrainConfig):
    task: str = "OfflineCarGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRL_WCarPush1Config(PDOCRL_WTrainConfig):
    task: str = "OfflineCarPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRL_WCarPush2Config(PDOCRL_WTrainConfig):
    task: str = "OfflineCarPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRL_WPointButton1Config(PDOCRL_WTrainConfig):
    task: str = "OfflinePointButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRL_WPointButton2Config(PDOCRL_WTrainConfig):
    task: str = "OfflinePointButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRL_WPointCircle1Config(PDOCRL_WTrainConfig):
    task: str = "OfflinePointCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class PDOCRL_WPointCircle2Config(PDOCRL_WTrainConfig):
    task: str = "OfflinePointCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class PDOCRL_WPointGoal1Config(PDOCRL_WTrainConfig):
    task: str = "OfflinePointGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRL_WPointGoal2Config(PDOCRL_WTrainConfig):
    task: str = "OfflinePointGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRL_WPointPush1Config(PDOCRL_WTrainConfig):
    task: str = "OfflinePointPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRL_WPointPush2Config(PDOCRL_WTrainConfig):
    task: str = "OfflinePointPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class PDOCRL_WAntVelocityConfig(PDOCRL_WTrainConfig):
    task: str = "OfflineAntVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class PDOCRL_WHalfCheetahVelocityConfig(PDOCRL_WTrainConfig):
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class PDOCRL_WHopperVelocityConfig(PDOCRL_WTrainConfig):
    task: str = "OfflineHopperVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class PDOCRL_WSwimmerVelocityConfig(PDOCRL_WTrainConfig):
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class PDOCRL_WWalker2dVelocityConfig(PDOCRL_WTrainConfig):
    task: str = "OfflineWalker2dVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class PDOCRL_WEasySparseConfig(PDOCRL_WTrainConfig):
    task: str = "OfflineMetadrive-easysparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDOCRL_WEasyMeanConfig(PDOCRL_WTrainConfig):
    task: str = "OfflineMetadrive-easymean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDOCRL_WEasyDenseConfig(PDOCRL_WTrainConfig):
    task: str = "OfflineMetadrive-easydense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDOCRL_WMediumSparseConfig(PDOCRL_WTrainConfig):
    task: str = "OfflineMetadrive-mediumsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDOCRL_WMediumMeanConfig(PDOCRL_WTrainConfig):
    task: str = "OfflineMetadrive-mediummean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDOCRL_WMediumDenseConfig(PDOCRL_WTrainConfig):
    task: str = "OfflineMetadrive-mediumdense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDOCRL_WHardSparseConfig(PDOCRL_WTrainConfig):
    task: str = "OfflineMetadrive-hardsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDOCRL_WHardMeanConfig(PDOCRL_WTrainConfig):
    task: str = "OfflineMetadrive-hardmean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class PDOCRL_WHardDenseConfig(PDOCRL_WTrainConfig):
    task: str = "OfflineMetadrive-harddense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


PDOCRL_W_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": PDOCRL_WCarCircleConfig,
    "OfflineAntRun-v0": PDOCRL_WAntRunConfig,
    "OfflineDroneRun-v0": PDOCRL_WDroneRunConfig,
    "OfflineDroneCircle-v0": PDOCRL_WDroneCircleConfig,
    "OfflineCarRun-v0": PDOCRL_WCarRunConfig,
    "OfflineAntCircle-v0": PDOCRL_WAntCircleConfig,
    "OfflineBallCircle-v0": PDOCRL_WBallCircleConfig,
    "OfflineBallRun-v0": PDOCRL_WBallRunConfig,
    # safety_gymnasium: car
    "OfflineCarButton1Gymnasium-v0": PDOCRL_WCarButton1Config,
    "OfflineCarButton2Gymnasium-v0": PDOCRL_WCarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": PDOCRL_WCarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": PDOCRL_WCarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": PDOCRL_WCarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": PDOCRL_WCarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": PDOCRL_WCarPush1Config,
    "OfflineCarPush2Gymnasium-v0": PDOCRL_WCarPush2Config,
    # safety_gymnasium: point
    "OfflinePointButton1Gymnasium-v0": PDOCRL_WPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": PDOCRL_WPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": PDOCRL_WPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": PDOCRL_WPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": PDOCRL_WPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": PDOCRL_WPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": PDOCRL_WPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": PDOCRL_WPointPush2Config,
    # safety_gymnasium: velocity
    "OfflineAntVelocityGymnasium-v1": PDOCRL_WAntVelocityConfig,
    "OfflineHalfCheetahVelocityGymnasium-v1": PDOCRL_WHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": PDOCRL_WHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": PDOCRL_WSwimmerVelocityConfig,
    "OfflineWalker2dVelocityGymnasium-v1": PDOCRL_WWalker2dVelocityConfig,
    # safe_metadrive
    "OfflineMetadrive-easysparse-v0": PDOCRL_WEasySparseConfig,
    "OfflineMetadrive-easymean-v0": PDOCRL_WEasyMeanConfig,
    "OfflineMetadrive-easydense-v0": PDOCRL_WEasyDenseConfig,
    "OfflineMetadrive-mediumsparse-v0": PDOCRL_WMediumSparseConfig,
    "OfflineMetadrive-mediummean-v0": PDOCRL_WMediumMeanConfig,
    "OfflineMetadrive-mediumdense-v0": PDOCRL_WMediumDenseConfig,
    "OfflineMetadrive-hardsparse-v0": PDOCRL_WHardSparseConfig,
    "OfflineMetadrive-hardmean-v0": PDOCRL_WHardMeanConfig,
    "OfflineMetadrive-harddense-v0": PDOCRL_WHardDenseConfig,
}
