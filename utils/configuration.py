from typing import TypedDict

import torch


class SystemConfiguration(TypedDict):
    device: torch.device
    CUDA_VISIBLE_DEVICES: str
    save_path: str


class DatasetConfiguration(TypedDict):
    name: str
    path: str
    train_size: int
    num_sampled_frames: int
    discard_threshold: int
    num_input_channels: int
    frame_size: tuple[int, int]
    cache_on: bool


class DataloaderConfiguration(TypedDict):
    batch_size: tuple[int, int]
    num_workers: int
    pin_memory: bool


class HyperparameterConfiguration(TypedDict):
    hidden_dim: int
    lr: int
    betas: tuple[float, float]
    hard_or_all: str
    margin: float


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
