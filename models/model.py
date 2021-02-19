import os
from datetime import datetime
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.rgb_part_net import RGBPartNet
from utils.configuration import DataloaderConfiguration, \
    HyperparameterConfiguration, DatasetConfiguration, ModelConfiguration, \
    SystemConfiguration
from utils.dataset import CASIAB, ClipConditions, ClipViews, ClipClasses
from utils.sampler import TripletSampler


class Model:
    def __init__(
            self,
            system_config: SystemConfiguration,
            model_config: ModelConfiguration,
            hyperparameter_config: HyperparameterConfiguration
    ):
        self.disable_acc = system_config.get('disable_acc', False)
        if self.disable_acc:
            self.device = torch.device('cpu')
        else:  # Enable accelerator
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                print('No accelerator available, fallback to CPU.')
                self.device = torch.device('cpu')

        self.save_dir = system_config.get('save_dir', 'runs')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        self.log_dir = os.path.join(self.save_dir, 'logs')
        for dir_ in (self.log_dir, self.checkpoint_dir):
            if not os.path.exists(dir_):
                os.mkdir(dir_)

        self.meta = model_config
        self.hp = hyperparameter_config
        self.curr_iter = self.meta.get('restore_iter', 0)
        self.total_iter = self.meta.get('total_iter', 80_000)
        self.curr_iters = self.meta.get('restore_iters', (0, 0, 0))
        self.total_iters = self.meta.get('total_iters', (80000, 80000, 80000))

        self.is_train: bool = True
        self.in_channels: int = 3
        self.in_size: tuple[int, int] = (64, 48)
        self.pr: Optional[int] = None
        self.k: Optional[int] = None

        self._gallery_dataset_meta: Optional[dict[str, list]] = None
        self._probe_datasets_meta: Optional[dict[str, dict[str, list]]] = None

        self._model_name: str = self.meta.get('name', 'RGB-GaitPart')
        self._hp_sig: str = self._make_signature(self.hp)
        self._dataset_sig: str = 'undefined'

        self.rgb_pn: Optional[RGBPartNet] = None
        self.optimizer: Optional[optim.Adam] = None
        self.scheduler: Optional[optim.lr_scheduler.StepLR] = None
        self.writer: Optional[SummaryWriter] = None
        self.image_log_on = system_config.get('image_log_on', False)

        self.CASIAB_GALLERY_SELECTOR = {
            'selector': {'conditions': ClipConditions({r'nm-0[1-4]'})}
        }
        self.CASIAB_PROBE_SELECTORS = {
            'nm': {'selector': {'conditions': ClipConditions({r'nm-0[5-6]'})}},
            'bg': {'selector': {'conditions': ClipConditions({r'bg-0[1-2]'})}},
            'cl': {'selector': {'conditions': ClipConditions({r'cl-0[1-2]'})}},
        }

    @property
    def _model_sig(self) -> str:
        return '_'.join(
            (self._model_name, str(self.curr_iter), str(self.total_iter))
        )

    @property
    def _checkpoint_sig(self) -> str:
        return '_'.join((self._model_sig, self._hp_sig, self._dataset_sig,
                         str(self.pr), str(self.k)))

    @property
    def _checkpoint_name(self) -> str:
        return os.path.join(self.checkpoint_dir, self._checkpoint_sig)

    @property
    def _log_sig(self) -> str:
        return '_'.join((self._model_name, str(self.total_iter), self._hp_sig,
                         self._dataset_sig, str(self.pr), str(self.k)))

    @property
    def _log_name(self) -> str:
        return os.path.join(self.log_dir, self._log_sig)

    def fit_all(
            self,
            dataset_config: DatasetConfiguration,
            dataset_selectors: dict[
                str, dict[str, Union[ClipClasses, ClipConditions, ClipViews]]
            ],
            dataloader_config: DataloaderConfiguration,
    ):
        for (curr_iter, total_iter, (condition, selector)) in zip(
                self.curr_iters, self.total_iters, dataset_selectors.items()
        ):
            print(f'Training model {condition} ...')
            # Skip finished model
            if curr_iter == total_iter:
                continue
            # Check invalid restore iter
            elif curr_iter > total_iter:
                raise ValueError("Restore iter '{}' should less than total "
                                 "iter '{}'".format(curr_iter, total_iter))
            self.curr_iter = curr_iter
            self.total_iter = total_iter
            self.fit(
                dict(**dataset_config, **{'selector': selector}),
                dataloader_config
            )

    def fit(
            self,
            dataset_config: DatasetConfiguration,
            dataloader_config: DataloaderConfiguration,
    ):
        self.is_train = True
        dataset = self._parse_dataset_config(dataset_config)
        dataloader = self._parse_dataloader_config(dataset, dataloader_config)
        # Prepare for model, optimizer and scheduler
        model_hp = self.hp.get('model', {})
        optim_hp: dict = self.hp.get('optimizer', {}).copy()
        sched_hp = self.hp.get('scheduler', {})
        self.rgb_pn = RGBPartNet(self.in_channels, self.in_size, **model_hp,
                                 image_log_on=self.image_log_on)
        # Try to accelerate computation using CUDA or others
        self.rgb_pn = self.rgb_pn.to(self.device)
        self.optimizer = optim.Adam(self.rgb_pn.parameters(), **optim_hp)
        sched_gamma = sched_hp.get('gamma', 0.9)
        sched_step_size = sched_hp.get('step_size', 500)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=[
            lambda epoch: sched_gamma ** (epoch // sched_step_size),
        ])
        self.writer = SummaryWriter(self._log_name)

        self.rgb_pn.train()
        # Init weights at first iter
        if self.curr_iter == 0:
            self.rgb_pn.apply(self.init_weights)
        else:  # Load saved state dicts
            checkpoint = torch.load(self._checkpoint_name)
            iter_, loss = checkpoint['iter'], checkpoint['loss']
            print('{0:5d} loss: {1:.3f}'.format(iter_, loss))
            self.rgb_pn.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
            self.scheduler.load_state_dict(checkpoint['sched_state_dict'])

        # Training start
        start_time = datetime.now()
        running_loss = torch.zeros(3, device=self.device)
        print(f"{'Time':^8} {'Iter':^5} {'Loss':^6}",
              f"{'Xrecon':^8} {'CanoCons':^8} {'PoseSim':^8}",
              f"{'LR':^9}")
        for (batch_c1, batch_c2) in dataloader:
            self.curr_iter += 1
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            # forward + backward + optimize
            x_c1 = batch_c1['clip'].to(self.device)
            x_c2 = batch_c2['clip'].to(self.device)
            losses, images = self.rgb_pn(x_c1, x_c2)
            loss = losses.sum()
            loss.backward()
            self.optimizer.step()

            # Statistics and checkpoint
            running_loss += losses.detach()
            # Write losses to TensorBoard
            self.writer.add_scalar('Loss/all', loss, self.curr_iter)
            self.writer.add_scalars('Loss/details', dict(zip([
                'Cross reconstruction loss',
                'Canonical consistency loss',
                'Pose similarity loss'
            ], losses)), self.curr_iter)

            if self.curr_iter % 100 == 0:
                lr = self.scheduler.get_last_lr()[0]
                # Write learning rates
                self.writer.add_scalar(
                    'Learning rate/Auto-encoder', lr, self.curr_iter
                )
                # Write disentangled images
                if self.image_log_on:
                    i_a, i_c, i_p = images
                    self.writer.add_images(
                        'Appearance image', i_a, self.curr_iter
                    )
                    self.writer.add_images(
                        'Canonical image', i_c, self.curr_iter
                    )
                    for i, (o, p) in enumerate(zip(x_c1, i_p)):
                        self.writer.add_images(
                            f'Original image/batch {i}', o, self.curr_iter
                        )
                        self.writer.add_images(
                            f'Pose image/batch {i}', p, self.curr_iter
                        )
                time_used = datetime.now() - start_time
                remaining_minute, second = divmod(time_used.seconds, 60)
                hour, minute = divmod(remaining_minute, 60)
                print(f'{hour:02}:{minute:02}:{second:02}',
                      f'{self.curr_iter:5d} {running_loss.sum() / 100:6.3f}',
                      '{:f} {:f} {:f}'.format(*running_loss / 100),
                      f'{lr:.3e}')
                running_loss.zero_()

            # Step scheduler
            self.scheduler.step()

            if self.curr_iter % 1000 == 0:
                torch.save({
                    'iter': self.curr_iter,
                    'model_state_dict': self.rgb_pn.state_dict(),
                    'optim_state_dict': self.optimizer.state_dict(),
                    'sched_state_dict': self.scheduler.state_dict(),
                    'loss': loss,
                }, self._checkpoint_name)

            if self.curr_iter == self.total_iter:
                self.writer.close()
                break

    def transform(
            self,
            iters: tuple[int],
            dataset_config: DatasetConfiguration,
            dataset_selectors: dict[
                str, dict[str, Union[ClipClasses, ClipConditions, ClipViews]]
            ],
            dataloader_config: DataloaderConfiguration
    ):
        self.is_train = False
        # Split gallery and probe dataset
        gallery_dataloader, probe_dataloaders = self._split_gallery_probe(
            dataset_config, dataloader_config
        )
        # Get pretrained models at iter_
        checkpoints = self._load_pretrained(
            iters, dataset_config, dataset_selectors
        )

        # Init models
        model_hp = self.hp.get('model', {})
        self.rgb_pn = RGBPartNet(self.in_channels, self.in_size, **model_hp)
        # Try to accelerate computation using CUDA or others
        self.rgb_pn = self.rgb_pn.to(self.device)
        self.rgb_pn.eval()

        gallery_samples, probe_samples = [], {}
        # Gallery
        checkpoint = torch.load(list(checkpoints.values())[0])
        self.rgb_pn.load_state_dict(checkpoint['model_state_dict'])
        for sample in tqdm(gallery_dataloader,
                           desc='Transforming gallery', unit='clips'):
            gallery_samples.append(self._get_eval_sample(sample))
        gallery_samples = default_collate(gallery_samples)
        # Probe
        for (condition, dataloader) in probe_dataloaders.items():
            checkpoint = torch.load(checkpoints[condition])
            self.rgb_pn.load_state_dict(checkpoint['model_state_dict'])
            probe_samples_c = []
            for sample in tqdm(dataloader,
                               desc=f'Transforming probe {condition}',
                               unit='clips'):
                probe_samples_c.append(self._get_eval_sample(sample))
            probe_samples[condition] = default_collate(probe_samples_c)

        return gallery_samples, probe_samples

    def _get_eval_sample(self, sample: dict[str, Union[list, torch.Tensor]]):
        label = sample.pop('label').item()
        clip = sample.pop('clip').to(self.device)
        x_c, x_p = self.rgb_pn(clip).detach()
        return {
            **{'label': label},
            **sample,
            **{'cano_feature': x_c, 'pose_feature': x_p}
        }

    def _load_pretrained(
            self,
            iters: tuple[int],
            dataset_config: DatasetConfiguration,
            dataset_selectors: dict[
                str, dict[str, Union[ClipClasses, ClipConditions, ClipViews]]
            ]
    ) -> dict[str, str]:
        checkpoints = {}
        for (iter_, (condition, selector)) in zip(
                iters, dataset_selectors.items()
        ):
            self.curr_iter = iter_
            self._dataset_sig = self._make_signature(
                dict(**dataset_config, **selector),
                popped_keys=['root_dir', 'cache_on']
            )
            checkpoints[condition] = self._checkpoint_name
        return checkpoints

    def _split_gallery_probe(
            self,
            dataset_config: DatasetConfiguration,
            dataloader_config: DataloaderConfiguration,
    ) -> tuple[DataLoader, dict[str, DataLoader]]:
        dataset_name = dataset_config.get('name', 'CASIA-B')
        if dataset_name == 'CASIA-B':
            gallery_dataset = self._parse_dataset_config(
                dict(**dataset_config, **self.CASIAB_GALLERY_SELECTOR)
            )
            self._gallery_dataset_meta = gallery_dataset.metadata
            gallery_dataloader = self._parse_dataloader_config(
                gallery_dataset, dataloader_config
            )
            probe_datasets = {
                condition: self._parse_dataset_config(
                    dict(**dataset_config, **selector)
                )
                for (condition, selector) in self.CASIAB_PROBE_SELECTORS.items()
            }
            self._probe_datasets_meta = {
                condition: dataset.metadata
                for (condition, dataset) in probe_datasets.items()
            }
            probe_dataloaders = {
                condition: self._parse_dataloader_config(
                    dataset, dataloader_config
                )
                for (condition, dataset) in probe_datasets.items()
            }
        elif dataset_name == 'FVG':
            # TODO
            gallery_dataloader = None
            probe_dataloaders = None
        else:
            raise ValueError('Invalid dataset: {0}'.format(dataset_name))

        return gallery_dataloader, probe_dataloaders

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.modules.conv._ConvNd):
            nn.init.normal_(m.weight, 0.0, 0.01)
        elif isinstance(m, nn.modules.batchnorm._NormBase):
            nn.init.normal_(m.weight, 1.0, 0.01)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def _parse_dataset_config(
            self,
            dataset_config: DatasetConfiguration
    ) -> Union[CASIAB]:
        self.in_channels = dataset_config.get('num_input_channels', 3)
        self.in_size = dataset_config.get('frame_size', (64, 48))
        self._dataset_sig = self._make_signature(
            dataset_config,
            popped_keys=['root_dir', 'cache_on']
        )
        config: dict = dataset_config.copy()
        name = config.pop('name', 'CASIA-B')
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
        (self.pr, self.k) = config.pop('batch_size', (8, 16))
        if self.is_train:
            triplet_sampler = TripletSampler(dataset, (self.pr, self.k))
            return DataLoader(dataset,
                              batch_sampler=triplet_sampler,
                              collate_fn=self._batch_splitter,
                              **config)
        else:  # is_test
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
                _config.pop(key, None)

        return self._gen_sig(list(_config.values()))

    def _gen_sig(self, values: Union[tuple, list, set, str, int, float]) -> str:
        strings = []
        for v in values:
            if isinstance(v, str):
                strings.append(v)
            elif isinstance(v, (tuple, list)):
                strings.append(self._gen_sig(v))
            elif isinstance(v, set):
                strings.append(self._gen_sig(sorted(list(v))))
            elif isinstance(v, dict):
                strings.append(self._gen_sig(list(v.values())))
            else:
                strings.append(str(v))
        return '_'.join(strings)
