import os

from config import config
from models import Model

# Set environment variable CUDA device(s)
CUDA_VISIBLE_DEVICES = config['system'].get('CUDA_VISIBLE_DEVICES', None)
if CUDA_VISIBLE_DEVICES:
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

model = Model(config['system'], config['model'], config['hyperparameter'])
model.fit(config['dataset'], config['dataloader'])
