config = {
    'system': {
        # Disable accelerator
        'disable_acc': False,
        # GPU(s) used in training or testing if available
        'CUDA_VISIBLE_DEVICES': '0',
        # Directory used in training or testing for temporary storage
        'save_dir': 'runs/dis_only',
        # Recorde disentangled image or not
        'image_log_on': True
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
        'cache_on': False,
    },
    # Dataloader settings
    'dataloader': {
        # Batch size (pr, k)
        # `pr` denotes number of persons
        # `k` denotes number of sequences per person
        'batch_size': (2, 2),
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
            # 'weight_decay': 0,
            # Use AMSGrad or not
            # 'amsgrad': False,
        },
        'scheduler': {
            # Period of learning rate decay
            'step_size': 500,
            # Multiplicative factor of decay
            'gamma': 0.9,
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
        'total_iters': (80_000, 80_000, 80_000),
    },
}
