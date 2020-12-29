from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from utils.configuration import DataloaderConfiguration, \
    HyperparameterConfiguration, DatasetConfiguration, ModelConfiguration
from utils.dataset import CASIAB
from utils.sampler import TripletSampler


class Model:
    def __init__(
            self,
            model_config: ModelConfiguration,
            hyperparameter_config: HyperparameterConfiguration
    ):
        self.meta = model_config
        self.hp = hyperparameter_config
        self.curr_iter = self.meta['restore_iter']

        self.is_train: bool = True
        self.dataset_metadata: Optional[DatasetConfiguration] = None
        self.pr: Optional[int] = None
        self.k: Optional[int] = None

        self._model_sig: str = self._make_signature(self.meta, ['restore_iter'])
        self._hp_sig: str = self._make_signature(self.hp)
        self._dataset_sig: str = 'undefined'

    @property
    def signature(self) -> str:
        return '_'.join((self._model_sig, str(self.curr_iter), self._hp_sig,
                         self._dataset_sig, str(self.batch_size)))

    @property
    def batch_size(self) -> int:
        if self.is_train:
            if self.pr and self.k:
                return self.pr * self.k
            raise AttributeError('No dataset loaded')
        else:
            return 1

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

    def fit(
            self,
            dataset_config: DatasetConfiguration,
            dataloader_config: DataloaderConfiguration,
    ):
        self.is_train = True
        dataset = self._parse_dataset_config(dataset_config)
        dataloader = self._parse_dataloader_config(dataset, dataloader_config)
        for iter_i, (samples_c1, samples_c2) in enumerate(dataloader):
            pass

            if iter_i == 0:
                break

    def _parse_dataset_config(
            self,
            dataset_config: DatasetConfiguration
    ) -> Union[CASIAB]:
        self._dataset_sig = self._make_signature(
            dataset_config,
            popped_keys=['root_dir', 'cache_on']
        )

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
            triplet_sampler = TripletSampler(dataset, (self.pr, self.k))
            return DataLoader(dataset,
                              batch_sampler=triplet_sampler,
                              collate_fn=self._batch_splitter,
                              **config)
        else:  # is_test
            config.pop('batch_size')
            return DataLoader(dataset, **config)

    @staticmethod
    def _make_signature(config: dict,
                        popped_keys: Optional[list] = None) -> str:
        _config = config.copy()
        for (key, value) in config.items():
            if popped_keys and key in popped_keys:
                _config.pop(key)
                continue
            if isinstance(value, str):
                pass
            elif isinstance(value, (tuple, list)):
                _config[key] = '_'.join([str(v) for v in value])
            else:
                _config[key] = str(value)

        return '_'.join(_config.values())
