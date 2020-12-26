import torch

from utils.configuration import Configuration

config: Configuration = {
    'system': {
        # Device(s) used in training and testing (CPU or CUDA)
        'device': torch.device('cuda'),
        # GPU(s) used in training or testing, if CUDA enabled
        'CUDA_VISIBLE_DEVICES': '0',
        # Directory used in training or testing for temporary storage
        'save_path': 'runs',
    },
    # Dataset settings
    'dataset': {
        # Name of dataset (CASIA-B or FVG)
        'name': 'CASIA-B',
        # Path to dataset root
        'path': 'dataset/output/CASIA-B',
        # The number of subjects for training
        'train_size': 74,
        # Number of sampled frames per sequence (Training only)
        'num_sampled_frames': 30,
        # Discard clips shorter than `discard_threshold`
        'discard_threshold': 15,
        # Number of input channels of model
        'num_input_channels': 3,
        # Resolution after resize, height : width should be 2 : 1
        'frame_size': (64, 32),
        # Cache dataset or not
        'cache_on': False,
    },
    # Dataloader settings
    'dataloader': {
        # Batch size (pr, k)
        # `pr` denotes number of persons
        # `k` denotes number of sequences per person
        'batch_size': (8, 16),
        # Number of workers of Dataloader
        'num_workers': 4,
        # Faster data transfer from RAM to GPU if enabled
        'pin_memory': True,
    },
    # Hyperparameter tuning
    'hyperparameter': {
        # Hidden dimension of FC
        'hidden_dim': 256,
        # Initial learning rate of Adam Optimizer
        'lr': 1e-4,
        # Betas of Adam Optimizer
        'betas': (0.9, 0.999),
        # Batch Hard or Batch Full Triplet loss
        # `hard` for BH, `all` for BA
        'hard_or_all': 'all',
        # Triplet loss margin
        'margin': 0.2,
    },
    # Model metadata
    'model': {
        # Model name, used for naming checkpoint
        'name': 'RGB-GaitPart',
        # Restoration iteration from checkpoint
        'restore_iter': 0,
        # Total iteration for training
        'total_iter': 80000,
    },
}
