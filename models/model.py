from typing import Union, Optional

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
            batch: list[dict[str, Union[str, torch.Tensor]]]
    ) -> list[tuple[dict[str, list[Union[str, torch.Tensor]]],
                    dict[str, list[Union[str, torch.Tensor]]]]]:
        """
        Disentanglement cannot be processed on different subjects at the
        same time, we need to load `pr` subjects one by one. The batch
        splitter will return a pr-length list of tuples (with 2 dicts
        containing k-length lists of labels, conditions, view and
        k-length tensor of clip data, representing condition 1 and
        condition 2 respectively).
        """
        _batch = []
        for i in range(0, self.pr * self.k * 2, self.k * 2):
            _batch.append((default_collate(batch[i:i + self.k]),
                           default_collate(batch[i + self.k:i + self.k * 2])))

        return _batch

    def fit(
            self,
            dataset_config: DatasetConfiguration,
            dataloader_config: DataloaderConfiguration,
    ):
        self.is_train = True
        dataset = self._parse_dataset_config(dataset_config)
        dataloader = self._parse_dataloader_config(dataset, dataloader_config)
        for iter_i, samples_batched in enumerate(dataloader):
            for sub_i, (subject_c1, subject_c2) in enumerate(samples_batched):
                pass

                if sub_i == 0:
                    break
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
