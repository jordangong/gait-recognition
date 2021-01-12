import os

from utils.configuration import SystemConfiguration


def set_visible_cuda(config: SystemConfiguration):
    """Set environment variable CUDA device(s)"""
    CUDA_VISIBLE_DEVICES = config.get('CUDA_VISIBLE_DEVICES', None)
    if CUDA_VISIBLE_DEVICES:
        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
