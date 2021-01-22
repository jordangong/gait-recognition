import numpy as np

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

accuracy = model.predict_all(config['model']['total_iters'], config['dataset'],
                             dataset_selectors, config['dataloader'])
rank = 5
np.set_printoptions(formatter={'float': '{:5.2f}'.format})
for n in range(rank):
    print(f'===Rank-{n + 1} Accuracy===')
    for (condition, accuracy_c) in accuracy.items():
        acc_excl_identical_view = accuracy_c[:, :, n].fill_diagonal_(0)
        num_gallery_views = (acc_excl_identical_view != 0).sum()
        acc_each_angle = acc_excl_identical_view.sum(0) / num_gallery_views
        print('{0}: {1} mean: {2:5.2f}'.format(
            condition, acc_each_angle.cpu().numpy() * 100,
            acc_each_angle.mean() * 100)
        )
