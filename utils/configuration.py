from typing import TypedDict, Optional, Union

from utils.dataset import ClipClasses, ClipConditions, ClipViews


class SystemConfiguration(TypedDict):
    disable_acc: bool
    CUDA_VISIBLE_DEVICES: str
    save_dir: str
    image_log_on: bool
    val_size: int


class DatasetConfiguration(TypedDict):
    name: str
    root_dir: str
    train_size: int
    num_sampled_frames: int
    truncate_threshold: int
    discard_threshold: int
    selector: Optional[dict[str, Union[ClipClasses, ClipConditions, ClipViews]]]
    num_input_channels: int
    frame_size: tuple[int, int]
    cache_on: bool


class DataloaderConfiguration(TypedDict):
    batch_size: tuple[int, int]
    num_workers: int
    pin_memory: bool


class ModelHPConfiguration(TypedDict):
    ae_feature_channels: int
    f_a_c_p_dims: tuple[int, int, int]
    hpm_scales: tuple[int, ...]
    hpm_use_avg_pool: bool
    hpm_use_max_pool: bool
    tfa_num_parts: int
    tfa_squeeze_ratio: int
    embedding_dims: tuple[int]
    triplet_is_hard: bool
    triplet_is_mean: bool
    triplet_margins: tuple[float, float]


class SubOptimizerHPConfiguration(TypedDict):
    lr: int
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool


class OptimizerHPConfiguration(TypedDict):
    lr: int
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool
    auto_encoder: SubOptimizerHPConfiguration
    hpm: SubOptimizerHPConfiguration
    part_net: SubOptimizerHPConfiguration


class SubSchedulerHPConfiguration(TypedDict):
    start_step: int
    final_gamma: float


class SchedulerHPConfiguration(TypedDict):
    start_step: int
    final_gamma: float
    auto_encoder: SubSchedulerHPConfiguration
    hpm: SubSchedulerHPConfiguration
    part_net: SubSchedulerHPConfiguration


class HyperparameterConfiguration(TypedDict):
    model: ModelHPConfiguration
    optimizer: OptimizerHPConfiguration
    scheduler: SchedulerHPConfiguration


class ModelConfiguration(TypedDict):
    name: str
    restore_iter: int
    total_iter: int
    restore_iters: tuple[int, ...]
    total_iters: tuple[int, ...]


class Configuration(TypedDict):
    system: SystemConfiguration
    dataset: DatasetConfiguration
    dataloader: DataloaderConfiguration
    hyperparameter: HyperparameterConfiguration
    model: ModelConfiguration
