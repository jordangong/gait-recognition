from utils.configuration import Configuration

config: Configuration = {
    'system': {
        # Disable accelerator
        'disable_acc': False,
        # GPU(s) used in training or testing if available
        'CUDA_VISIBLE_DEVICES': '0,1',
        # Directory used in training or testing for temporary storage
        'save_dir': 'runs',
        # Recorde disentangled image or not
        'image_log_on': False
    },
    # Dataset settings
    'dataset': {
        # Name of dataset (CASIA-B or FVG)
        'name': 'CASIA-B',
        # Path to dataset root (required)
        'root_dir': 'data/CASIA-B-MRCNN-V2/SEG',
        # The number of subjects for training
        'train_size': 74,
        # Number of sampled frames per sequence (Training only)
        'num_sampled_frames': 30,
        # Truncate clips longer than `truncate_threshold`
        'truncate_threshold': 40,
        # Discard clips shorter than `discard_threshold`
        'discard_threshold': 15,
        # Number of input channels of model
        'num_input_channels': 3,
        # Resolution after resize, can be divided 16
        'frame_size': (64, 48),
        # Cache dataset or not
        'cache_on': True,
    },
    # Dataloader settings
    'dataloader': {
        # Batch size (pr, k)
        # `pr` denotes number of persons
        # `k` denotes number of sequences per person
        'batch_size': (6, 8),
        # Number of workers of Dataloader
        'num_workers': 4,
        # Faster data transfer from RAM to GPU if enabled
        'pin_memory': True,
    },
    # Hyperparameter tuning
    'hyperparameter': {
        'model': {
            # Auto-encoder feature channels coefficient
            'ae_feature_channels': 64,
            # Appearance, canonical and pose feature dimensions
            'f_a_c_p_dims': (192, 192, 96),
            # Use 1x1 convolution in dimensionality reduction
            'hpm_use_1x1conv': False,
            # HPM pyramid scales, of which sum is number of parts
            'hpm_scales': (1, 2, 4, 8),
            # Global pooling method
            'hpm_use_avg_pool': True,
            'hpm_use_max_pool': True,
            # Attention squeeze ratio
            'tfa_squeeze_ratio': 4,
            # Number of parts after Part Net
            'tfa_num_parts': 16,
            # Embedding dimension for each part
            'embedding_dims': 256,
            # Batch Hard or Batch All
            'triplet_is_hard': True,
            # Use non-zero mean or sum
            'triplet_is_mean': True,
            # Triplet loss margins for HPM and PartNet, None for soft margin
            'triplet_margins': None,
        },
        'optimizer': {
            # Global parameters
            # Initial learning rate of Adam Optimizer
            'lr': 1e-4,
            # Coefficients used for computing running averages of
            #   gradient and its square
            # 'betas': (0.9, 0.999),
            # Term added to the denominator
            # 'eps': 1e-8,
            # Weight decay (L2 penalty)
            'weight_decay': 0.001,
            # Use AMSGrad or not
            # 'amsgrad': False,

            # Local parameters (override global ones)
            # 'auto_encoder': {
            #     'weight_decay': 0.001
            # },
        },
        'scheduler': {
            # Step start to decay
            'start_step': 15_000,
            # Multiplicative factor of decay in the end
            'final_gamma': 0.001,
        }
    },
    # Model metadata
    'model': {
        # Model name, used for naming checkpoint
        'name': 'RGB-GaitPart',
        # Restoration iteration from checkpoint (single model)
        # 'restore_iter': 0,
        # Total iteration for training (single model)
        # 'total_iter': 80000,
        # Restoration iteration (multiple models, e.g. nm, bg and cl)
        'restore_iters': (0, 0, 0),
        # Total iteration for training (multiple models)
        'total_iters': (25_000, 25_000, 25_000),
    },
}
