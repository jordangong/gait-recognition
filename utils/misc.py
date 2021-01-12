import os
from typing import Dict


def set_visible_cuda(config: Dict):
    """Set environment variable CUDA device(s)"""
    CUDA_VISIBLE_DEVICES = config.get('CUDA_VISIBLE_DEVICES', None)
    if CUDA_VISIBLE_DEVICES:
        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
