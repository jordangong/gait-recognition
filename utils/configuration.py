from typing import TypedDict, Optional, Union

from utils.dataset import ClipClasses, ClipConditions, ClipViews


class SystemConfiguration(TypedDict):
    disable_acc: bool
    CUDA_VISIBLE_DEVICES: str
    save_dir: str


class DatasetConfiguration(TypedDict):
    name: str
    root_dir: str
    train_size: int
    num_sampled_frames: int
    discard_threshold: int
    selector: Optional[dict[str, Union[ClipClasses, ClipConditions, ClipViews]]]
    num_input_channels: int
    frame_size: tuple[int, int]
    cache_on: bool


class DataloaderConfiguration(TypedDict):
    batch_size: tuple[int, int]
    num_workers: int
    pin_memory: bool


class HyperparameterConfiguration(TypedDict):
    ae_feature_channels: int
    f_a_c_p_dims: tuple[int, int, int]
    hpm_scales: tuple[int, ...]
    hpm_use_avg_pool: bool
    hpm_use_max_pool: bool
    fpfe_feature_channels: int
    fpfe_kernel_sizes: tuple[tuple, ...]
    fpfe_paddings: tuple[tuple, ...]
    fpfe_halving: tuple[int, ...]
    tfa_squeeze_ratio: int
    tfa_num_parts: int
    embedding_dims: int
    triplet_margin: float
    lr: int
    betas: tuple[float, float]


class ModelConfiguration(TypedDict):
    name: str
    restore_iter: int
    total_iter: int


class Configuration(TypedDict):
    system: SystemConfiguration
    dataset: DatasetConfiguration
    dataloader: DataloaderConfiguration
    hyperparameter: HyperparameterConfiguration
    model: ModelConfiguration
