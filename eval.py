from config import config
from models import Model
from utils.dataset import ClipConditions
from utils.misc import set_visible_cuda

set_visible_cuda(config['system'])
model = Model(config['system'], config['model'], config['hyperparameter'])

dataset_selectors = {
    'nm': {'conditions': ClipConditions({r'nm-0\d'})},
    'bg': {'conditions': ClipConditions({r'nm-0\d', r'bg-0\d'})},
    'cl': {'conditions': ClipConditions({r'nm-0\d', r'cl-0\d'})},
}

accuracy = model.predict_all(config['model']['total_iter'], config['dataset'],
                             dataset_selectors, config['dataloader'])
