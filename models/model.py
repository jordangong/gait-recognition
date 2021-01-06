import os
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter

from models import RGBPartNet
from utils.configuration import DataloaderConfiguration, \
    HyperparameterConfiguration, DatasetConfiguration, ModelConfiguration, \
    SystemConfiguration
from utils.dataset import CASIAB
from utils.sampler import TripletSampler


class Model:
    def __init__(
            self,
            system_config: SystemConfiguration,
            model_config: ModelConfiguration,
            hyperparameter_config: HyperparameterConfiguration
    ):
        self.device = system_config['device']
        self.save_dir = system_config['save_dir']
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        self.log_dir = os.path.join(self.save_dir, 'logs')
        for dir_ in (self.save_dir, self.log_dir, self.checkpoint_dir):
            if not os.path.exists(dir_):
                os.mkdir(dir_)

        self.meta = model_config
        self.hp = hyperparameter_config
        self.curr_iter = self.meta['restore_iter']
        self.total_iter = self.meta['total_iter']

        self.is_train: bool = True
        self.train_size: int = 74
        self.in_channels: int = 3
        self.pr: Optional[int] = None
        self.k: Optional[int] = None

        self._model_sig: str = self._make_signature(self.meta, ['restore_iter'])
        self._hp_sig: str = self._make_signature(self.hp)
        self._dataset_sig: str = 'undefined'
        self._log_sig: str = '_'.join((self._model_sig, self._hp_sig))
        self.log_name: str = os.path.join(self.log_dir, self._log_sig)

        self.rgb_pn: Optional[RGBPartNet] = None
        self.optimizer: Optional[optim.Adam] = None
        self.scheduler: Optional[optim.lr_scheduler.StepLR] = None
        self.writer: Optional[SummaryWriter] = None

    @property
    def signature(self) -> str:
        return '_'.join((self._model_sig, str(self.curr_iter), self._hp_sig,
                         self._dataset_sig, str(self.pr), str(self.k)))

    @property
    def checkpoint_name(self) -> str:
        return os.path.join(self.checkpoint_dir, self.signature)

    def fit(
            self,
            dataset_config: DatasetConfiguration,
            dataloader_config: DataloaderConfiguration,
    ):
        self.is_train = True
        dataset = self._parse_dataset_config(dataset_config)
        dataloader = self._parse_dataloader_config(dataset, dataloader_config)
        # Prepare for model, optimizer and scheduler
        hp = self.hp.copy()
        lr, betas = hp.pop('lr', 1e-4), hp.pop('betas', (0.9, 0.999))
        self.rgb_pn = RGBPartNet(self.train_size, self.in_channels, **hp)
        self.optimizer = optim.Adam(self.rgb_pn.parameters(), lr, betas)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 500, 0.9)
        self.writer = SummaryWriter(self.log_name)

        self.rgb_pn.train()
        # Init weights at first iter
        if self.curr_iter == 0:
            self.rgb_pn.apply(self.init_weights)
        else:  # Load saved state dicts
            checkpoint = torch.load(self.checkpoint_name)
            iter_, loss = checkpoint['iter'], checkpoint['loss']
            print('{0:5d} loss: {1:.3f}'.format(iter_, loss))
            self.rgb_pn.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optim_state_dict'])

        for (batch_c1, batch_c2) in dataloader:
            self.curr_iter += 1
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            # forward + backward + optimize
            loss, metrics = self.rgb_pn(
                batch_c1['clip'], batch_c2['clip'], batch_c1['label']
            )
            loss.backward()
            self.optimizer.step()
            # Step scheduler
            self.scheduler.step(self.curr_iter)

            # Write losses to TensorBoard
            self.writer.add_scalar('Loss/all', loss.item(), self.curr_iter)
            self.writer.add_scalars('Loss/details', dict(zip([
                'Cross reconstruction loss', 'Pose similarity loss',
                'Canonical consistency loss', 'Batch All triplet loss'
            ], metrics)), self.curr_iter)

            if self.curr_iter % 100 == 0:
                print('{0:5d} loss: {1:.3f}'.format(self.curr_iter, loss),
                      '(xrecon = {:f}, pose_sim = {:f},'
                      ' cano_cons = {:f}, ba_trip = {:f})'.format(*metrics),
                      'lr:', self.scheduler.get_last_lr())

            if self.curr_iter % 1000 == 0:
                torch.save({
                    'iter': self.curr_iter,
                    'model_state_dict': self.rgb_pn.state_dict(),
                    'optim_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                }, self.checkpoint_name)

            if self.curr_iter == self.total_iter:
                self.writer.close()
                break

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.modules.conv._ConvNd):
            nn.init.normal_(m.weight, 0.0, 0.01)
        elif isinstance(m, nn.modules.batchnorm._NormBase):
            nn.init.normal_(m.weight, 1.0, 0.01)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, RGBPartNet):
            nn.init.xavier_uniform_(m.fc_mat)

    def _parse_dataset_config(
            self,
            dataset_config: DatasetConfiguration
    ) -> Union[CASIAB]:
        self.train_size = dataset_config['train_size']
        self.in_channels = dataset_config['num_input_channels']
        self._dataset_sig = self._make_signature(
            dataset_config,
            popped_keys=['root_dir', 'cache_on']
        )
        self.log_name = '_'.join((self.log_name, self._dataset_sig))
        config: dict = dataset_config.copy()
        name = config.pop('name')
        if name == 'CASIA-B':
            return CASIAB(**config, is_train=self.is_train)
        elif name == 'FVG':
            # TODO
            pass
        raise ValueError('Invalid dataset: {0}'.format(name))

    def _parse_dataloader_config(
            self,
            dataset: Union[CASIAB],
            dataloader_config: DataloaderConfiguration
    ) -> DataLoader:
        config: dict = dataloader_config.copy()
        if self.is_train:
            (self.pr, self.k) = config.pop('batch_size')
            self.log_name = '_'.join((self.log_name, str(self.pr), str(self.k)))
            triplet_sampler = TripletSampler(dataset, (self.pr, self.k))
            return DataLoader(dataset,
                              batch_sampler=triplet_sampler,
                              collate_fn=self._batch_splitter,
                              **config)
        else:  # is_test
            config.pop('batch_size')
            return DataLoader(dataset, **config)

    def _batch_splitter(
            self,
            batch: list[dict[str, Union[np.int64, str, torch.Tensor]]]
    ) -> tuple[dict[str, Union[list[str], torch.Tensor]],
               dict[str, Union[list[str], torch.Tensor]]]:
        """
        Disentanglement need two random conditions, this function will
        split pr * k * 2 samples to 2 dicts each containing pr * k
        samples. labels and clip data are tensor, and others are list.
        """
        _batch = [[], []]
        for i in range(0, self.pr * self.k * 2, self.k * 2):
            _batch[0] += batch[i:i + self.k]
            _batch[1] += batch[i + self.k:i + self.k * 2]

        return default_collate(_batch[0]), default_collate(_batch[1])

    def _make_signature(self,
                        config: dict,
                        popped_keys: Optional[list] = None) -> str:
        _config = config.copy()
        if popped_keys:
            for key in popped_keys:
                _config.pop(key)

        return self._gen_sig(list(_config.values()))

    def _gen_sig(self, values: Union[tuple, list, str, int, float]) -> str:
        strings = []
        for v in values:
            if isinstance(v, str):
                strings.append(v)
            elif isinstance(v, (tuple, list)):
                strings.append(self._gen_sig(v))
            else:
                strings.append(str(v))
        return '_'.join(strings)
