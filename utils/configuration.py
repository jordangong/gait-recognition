from typing import TypedDict, Optional, Union, Tuple, Dict

from utils.dataset import ClipClasses, ClipConditions, ClipViews


class SystemConfiguration(TypedDict):
    disable_acc: bool
    CUDA_VISIBLE_DEVICES: str
    save_dir: str
    image_log_on: bool


class DatasetConfiguration(TypedDict):
    name: str
    root_dir: str
    train_size: int
    num_sampled_frames: int
    truncate_threshold: int
    discard_threshold: int
    selector: Optional[Dict[str, Union[ClipClasses, ClipConditions, ClipViews]]]
    num_input_channels: int
    frame_size: Tuple[int, int]
    cache_on: bool


class DataloaderConfiguration(TypedDict):
    batch_size: Tuple[int, int]
    num_workers: int
    pin_memory: bool


class ModelHPConfiguration(TypedDict):
    ae_feature_channels: int
    f_a_c_p_dims: Tuple[int, int, int]
    hpm_scales: Tuple[int, ...]
    hpm_use_1x1conv: bool
    hpm_use_avg_pool: bool
    hpm_use_max_pool: bool
    fpfe_feature_channels: int
    fpfe_kernel_sizes: Tuple[Tuple, ...]
    fpfe_paddings: Tuple[Tuple, ...]
    fpfe_halving: Tuple[int, ...]
    tfa_squeeze_ratio: int
    tfa_num_parts: int
    embedding_dims: int
    triplet_margins: Tuple[float, float]


class SubOptimizerHPConfiguration(TypedDict):
    lr: int
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool


class OptimizerHPConfiguration(TypedDict):
    start_iter: int
    lr: int
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool
    auto_encoder: SubOptimizerHPConfiguration
    part_net: SubOptimizerHPConfiguration
    hpm: SubOptimizerHPConfiguration
    fc: SubOptimizerHPConfiguration


class SchedulerHPConfiguration(TypedDict):
    step_size: int
    gamma: float


class HyperparameterConfiguration(TypedDict):
    model: ModelHPConfiguration
    optimizer: OptimizerHPConfiguration
    scheduler: SchedulerHPConfiguration


class ModelConfiguration(TypedDict):
    name: str
    restore_iter: int
    total_iter: int
    restore_iters: Tuple[int, ...]
    total_iters: Tuple[int, ...]


class Configuration(TypedDict):
    system: SystemConfiguration
    dataset: DatasetConfiguration
    dataloader: DataloaderConfiguration
    hyperparameter: HyperparameterConfiguration
    model: ModelConfiguration
