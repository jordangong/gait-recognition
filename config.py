from utils.configuration import Configuration

config: Configuration = {
    'system': {
        # Disable accelerator
        'disable_acc': False,
        # GPU(s) used in training or testing if available
        'CUDA_VISIBLE_DEVICES': '0',
        # Directory used in training or testing for temporary storage
        'save_dir': 'runs',
    },
    # Dataset settings
    'dataset': {
        # Name of dataset (CASIA-B or FVG)
        'name': 'CASIA-B',
        # Path to dataset root
        'root_dir': 'data/CASIA-B-MRCNN/SEG',
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
        # Auto-encoder feature channels coefficient
        'ae_feature_channels': 64,
        # Appearance, canonical and pose feature dimensions
        'f_a_c_p_dims': (128, 128, 64),
        # HPM pyramid scales, of which sum is number of parts
        'hpm_scales': (1, 2, 4),
        # Global pooling method
        'hpm_use_avg_pool': True,
        'hpm_use_max_pool': True,
        # FConv feature channels coefficient
        'fpfe_feature_channels': 32,
        # FConv blocks kernel sizes
        'fpfe_kernel_sizes': ((5, 3), (3, 3), (3, 3)),
        # FConv blocks paddings
        'fpfe_paddings': ((2, 1), (1, 1), (1, 1)),
        # FConv blocks halving
        'fpfe_halving': (0, 2, 3),
        # Attention squeeze ratio
        'tfa_squeeze_ratio': 4,
        # Number of parts after Part Net
        'tfa_num_parts': 16,
        # Embedding dimension for each part
        'embedding_dims': 256,
        # Triplet loss margin
        'triplet_margin': 0.2,
        # Initial learning rate of Adam Optimizer
        'lr': 1e-4,
        # Betas of Adam Optimizer
        'betas': (0.9, 0.999),
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
