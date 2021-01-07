import os

from config import config
from models import Model
from utils.dataset import ClipConditions

# Set environment variable CUDA device(s)
CUDA_VISIBLE_DEVICES = config['system'].get('CUDA_VISIBLE_DEVICES', None)
if CUDA_VISIBLE_DEVICES:
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

model = Model(config['system'], config['model'], config['hyperparameter'])

# 3 models for different conditions
dataset_selectors = [
    {'conditions': ClipConditions({r'nm-0\d'})},
    {'conditions': ClipConditions({r'nm-0\d', r'bg-0\d'})},
    {'conditions': ClipConditions({r'nm-0\d', r'cl-0\d'})},
]
for selector in dataset_selectors:
    model.fit(
        dict(**config['dataset'], **{'selector': selector}),
        config['dataloader']
    )
