import os
from datetime import datetime
from typing import Union, Optional, Tuple, List, Dict, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.rgb_part_net import RGBPartNet
from utils.dataset import CASIAB, ClipConditions, ClipViews, ClipClasses
from utils.sampler import TripletSampler


class Model:
    def __init__(
            self,
            system_config: Dict,
            model_config: Dict,
            hyperparameter_config: Dict
    ):
        self.nprocs = system_config['nprocs']
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

        self._gallery_dataset_meta: Optional[Dict[str, List]] = None
        self._probe_datasets_meta: Optional[Dict[str, Dict[str, List]]] = None

        self._model_sig: str = self._make_signature(self.meta, ['restore_iter'])
        self._hp_sig: str = self._make_signature(self.hp)
        self._dataset_sig: str = 'undefined'
        self._log_sig: str = '_'.join((self._model_sig, self._hp_sig))
        self._log_name: str = os.path.join(self.log_dir, self._log_sig)

        self.CASIAB_GALLERY_SELECTOR = {
            'selector': {'conditions': ClipConditions({r'nm-0[1-4]'})}
        }
        self.CASIAB_PROBE_SELECTORS = {
            'nm': {'selector': {'conditions': ClipConditions({r'nm-0[5-6]'})}},
            'bg': {'selector': {'conditions': ClipConditions({r'bg-0[1-2]'})}},
            'cl': {'selector': {'conditions': ClipConditions({r'cl-0[1-2]'})}},
        }

    @property
    def _signature(self) -> str:
        return '_'.join((self._model_sig, str(self.curr_iter), self._hp_sig,
                         self._dataset_sig, str(self.pr), str(self.k)))

    @property
    def _checkpoint_name(self) -> str:
        return os.path.join(self.checkpoint_dir, self._signature)

    def fit_all(
            self,
            dataset_config: Dict,
            dataset_selectors: Dict[
                str, Dict[str, Union[ClipClasses, ClipConditions, ClipViews]]
            ],
            dataloader_config: Dict,
    ):
        for (condition, selector) in dataset_selectors.items():
            print(f'Training model {condition} ...')
            self.fit(
                dict(**dataset_config, **{'selector': selector}),
                dataloader_config
            )

    def fit(
            self,
            dataset_config: Dict,
            dataloader_config: Dict,
    ):
        # Only instantiate model weights once in memory.
        model_hp = self.hp.get('model', {})
        rgb_pn = RGBPartNet(self.train_size, self.in_channels, **model_hp)
        wrapped_rgb_pn = xmp.MpModelWrapper(rgb_pn)

        xmp.spawn(
            self._fit_map_fn,
            args=(wrapped_rgb_pn, dataset_config, dataloader_config),
            nprocs=self.nprocs,
            start_method='fork'
        )

    def _fit_map_fn(
            self,
            rank: int,
            wrapped_rgb_pn: xmp.MpModelWrapper,
            dataset_config: Dict,
            dataloader_config: Dict,
    ):
        self.is_train = True
        dataset = self._parse_dataset_config(dataset_config)
        dataloader = self._parse_dataloader_config(dataset, dataloader_config)
        # Prepare for optimizer and scheduler
        optim_hp = self.hp.get('optimizer', {})
        # Scale learning rate to world size
        lr = optim_hp.get('lr', '1-e3')
        optim_hp['lr'] = lr * xm.xrt_world_size()
        sched_hp = self.hp.get('scheduler', {})
        device = xm.xla_device()
        rgb_pn = wrapped_rgb_pn.to(device)
        optimizer = optim.Adam(rgb_pn.parameters(), **optim_hp)
        scheduler = optim.lr_scheduler.StepLR(optimizer, **sched_hp)
        writer = SummaryWriter(self._log_name)

        para_loader = pl.ParallelLoader(dataloader, [device])
        self._train_loop(
            rank,
            para_loader.per_device_loader(device),
            rgb_pn, optimizer, scheduler, writer
        )

    def _train_loop(
            self,
            rank: int,
            dataloader: pl.PerDeviceLoader,
            rgb_pn: RGBPartNet,
            optimizer: optim.Adam,
            scheduler: optim.lr_scheduler.StepLR,
            writer: SummaryWriter
    ):
        rgb_pn.train()
        # Init weights at first iter
        if self.curr_iter == 0:
            rgb_pn.apply(self.init_weights)
        else:  # Load saved state dicts
            checkpoint = torch.load(self._checkpoint_name)
            iter_, loss = checkpoint['iter'], checkpoint['loss']
            print('{0:5d} loss: {1:.3f}'.format(iter_, loss))
            rgb_pn.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
        # Training start
        start_time = datetime.now()
        for (iter_i, (batch_c1, batch_c2)) in enumerate(dataloader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            x_c1 = batch_c1['clip']
            x_c2 = batch_c2['clip']
            y = batch_c1['label']
            loss, metrics = rgb_pn(x_c1, x_c2, y)
            loss.backward()
            xm.optimizer_step(optimizer)
            # Step scheduler
            scheduler.step()

            # Write losses to TensorBoard
            writer.add_scalar(
                f'[xla:{rank}]Loss/all', loss.item(), iter_i + 1
            )
            writer.add_scalars(
                f'[xla:{rank}]Loss/details', dict(zip([
                    'Cross reconstruction loss', 'Pose similarity loss',
                    'Canonical consistency loss', 'Batch All triplet loss'
                ], metrics)),
                iter_i + 1
            )

            if iter_i % 100 == 99:
                print('[xla:{0}]({1:5d})'.format(rank, iter_i + 1),
                      'loss: {:6.3f}'.format(loss),
                      '(xrecon = {:f}, pose_sim = {:f},'
                      ' cano_cons = {:f}, ba_trip = {:f})'.format(*metrics),
                      'lr:', scheduler.get_last_lr()[0])
                xm.master_print(iter_i + 1, 'iteration finished')

            if iter_i % 1000 == 999 and xm.is_master_ordinal():
                self.curr_iter = iter_i + 1
                torch.save({
                    'iter': self.curr_iter,
                    'model_state_dict': rgb_pn.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, self._checkpoint_name)
                print(datetime.now() - start_time, 'used')
                start_time = datetime.now()

            if iter_i == self.total_iter - 1:
                if xm.is_master_ordinal():
                    self.curr_iter = 0
                writer.close()
                break

    def predict_all(
            self,
            iter_: int,
            dataset_config: Dict,
            dataset_selectors: Dict[
                str, Dict[str, Union[ClipClasses, ClipConditions, ClipViews]]
            ],
            dataloader_config: Dict,
    ) -> Dict[str, torch.Tensor]:
        self.is_train = False
        device = xm.xla_device()
        # Split gallery and probe dataset
        gallery_dataloader, probe_dataloaders = self._split_gallery_probe(
            dataset_config, dataloader_config
        )
        # Get pretrained models at iter_
        checkpoints = self._load_pretrained(
            iter_, dataset_config, dataset_selectors
        )
        # Init models
        model_hp = self.hp.get('model', {})
        rgb_pn = RGBPartNet(ae_in_channels=self.in_channels, **model_hp)

        rgb_pn.eval()
        gallery_samples, probe_samples = [], {}

        # Gallery
        checkpoint = torch.load(list(checkpoints.values())[0])
        rgb_pn.load_state_dict(checkpoint['model_state_dict'])
        for sample in tqdm(gallery_dataloader,
                           desc='Transforming gallery', unit='clips'):
            label = sample.pop('label').item()
            clip = sample.pop('clip').to(device)
            feature = rgb_pn(clip).detach()
            gallery_samples.append({
                **{'label': label},
                **sample,
                **{'feature': feature}
            })
        gallery_samples = default_collate(gallery_samples)

        # Probe
        for (condition, dataloader) in probe_dataloaders.items():
            checkpoint = torch.load(checkpoints[condition])
            rgb_pn.load_state_dict(checkpoint['model_state_dict'])
            probe_samples[condition] = []
            for sample in tqdm(dataloader,
                               desc=f'Transforming probe {condition}',
                               unit='clips'):
                label = sample.pop('label').item()
                clip = sample.pop('clip').to(device)
                feature = rgb_pn(clip).detach()
                probe_samples[condition].append({
                    **{'label': label},
                    **sample,
                    **{'feature': feature}
                })
        for (k, v) in probe_samples.items():
            probe_samples[k] = default_collate(v)

        return self._evaluate(gallery_samples, probe_samples)

    def _evaluate(
            self,
            gallery_samples: Dict[str, Union[List[str], torch.Tensor]],
            probe_samples: Dict[str, Dict[str, Union[List[str], torch.Tensor]]],
            num_ranks: int = 5
    ) -> Dict[str, torch.Tensor]:
        probe_conditions = self._probe_datasets_meta.keys()
        gallery_views_meta = self._gallery_dataset_meta['views']
        probe_views_meta = list(self._probe_datasets_meta.values())[0]['views']
        accuracy = {
            condition: torch.empty(
                len(gallery_views_meta), len(probe_views_meta), num_ranks
            )
            for condition in self._probe_datasets_meta.keys()
        }

        (labels_g, _, views_g, features_g) = gallery_samples.values()
        views_g = np.asarray(views_g)
        for (v_g_i, view_g) in enumerate(gallery_views_meta):
            gallery_view_mask = (views_g == view_g)
            f_g = features_g[gallery_view_mask]
            y_g = labels_g[gallery_view_mask]
            for condition in probe_conditions:
                probe_samples_c = probe_samples[condition]
                accuracy_c = accuracy[condition]
                (labels_p, _, views_p, features_p) = probe_samples_c.values()
                views_p = np.asarray(views_p)
                for (v_p_i, view_p) in enumerate(probe_views_meta):
                    probe_view_mask = (views_p == view_p)
                    f_p = features_p[probe_view_mask]
                    y_p = labels_p[probe_view_mask]
                    # Euclidean distance
                    f_p_squared_sum = torch.sum(f_p ** 2, dim=1).unsqueeze(1)
                    f_g_squared_sum = torch.sum(f_g ** 2, dim=1).unsqueeze(0)
                    f_p_times_f_g_sum = f_p @ f_g.T
                    dist = torch.sqrt(F.relu(
                        f_p_squared_sum - 2*f_p_times_f_g_sum + f_g_squared_sum
                    ))
                    # Ranked accuracy
                    rank_mask = dist.argsort(1)[:, :num_ranks]
                    positive_mat = torch.eq(y_p.unsqueeze(1),
                                            y_g[rank_mask]).cumsum(1).gt(0)
                    positive_counts = positive_mat.sum(0)
                    total_counts, _ = dist.size()
                    accuracy_c[v_g_i, v_p_i, :] = positive_counts / total_counts

        return accuracy

    def _load_pretrained(
            self,
            iter_: int,
            dataset_config: Dict,
            dataset_selectors: Dict[
                str, Dict[str, Union[ClipClasses, ClipConditions, ClipViews]]
            ]
    ) -> Dict[str, str]:
        checkpoints = {}
        self.curr_iter = iter_
        for (k, v) in dataset_selectors.items():
            self._dataset_sig = self._make_signature(
                dict(**dataset_config, **v),
                popped_keys=['root_dir', 'cache_on']
            )
            checkpoints[k] = self._checkpoint_name
        return checkpoints

    def _split_gallery_probe(
            self,
            dataset_config: Dict,
            dataloader_config: Dict,
    ) -> Tuple[DataLoader, Dict[str, DataLoader]]:
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
                condtion: self._parse_dataloader_config(
                    dataset, dataloader_config
                )
                for (condtion, dataset) in probe_datasets.items()
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
        elif isinstance(m, RGBPartNet):
            nn.init.xavier_uniform_(m.fc_mat)

    def _parse_dataset_config(
            self,
            dataset_config: Dict
    ) -> Union[CASIAB]:
        self.train_size = dataset_config.get('train_size', 74)
        self.in_channels = dataset_config.get('num_input_channels', 3)
        self._dataset_sig = self._make_signature(
            dataset_config,
            popped_keys=['root_dir', 'cache_on']
        )
        self._log_name = '_'.join((self._log_name, self._dataset_sig))
        config: Dict = dataset_config.copy()
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
            dataloader_config: Dict
    ) -> DataLoader:
        config: Dict = dataloader_config.copy()
        (self.pr, self.k) = config.pop('batch_size')
        if self.is_train:
            self._log_name = '_'.join(
                (self._log_name, str(self.pr), str(self.k)))
            triplet_sampler = TripletSampler(dataset, (self.pr, self.k))
            return DataLoader(dataset,
                              batch_sampler=triplet_sampler,
                              collate_fn=self._batch_splitter,
                              **config)
        else:  # is_test
            return DataLoader(dataset, **config)

    def _batch_splitter(
            self,
            batch: List[Dict[str, Union[np.int64, str, torch.Tensor]]]
    ) -> Tuple[Dict[str, Union[List[str], torch.Tensor]],
               Dict[str, Union[List[str], torch.Tensor]]]:
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
                        config: Dict,
                        popped_keys: Optional[List] = None) -> str:
        _config = config.copy()
        if popped_keys:
            for key in popped_keys:
                _config.pop(key)

        return self._gen_sig(list(_config.values()))

    def _gen_sig(self, values: Union[Tuple, List, Set, str, int, float]) -> str:
        strings = []
        for v in values:
            if isinstance(v, str):
                strings.append(v)
            elif isinstance(v, (Tuple, List)):
                strings.append(self._gen_sig(v))
            elif isinstance(v, Set):
                strings.append(self._gen_sig(sorted(list(v))))
            elif isinstance(v, Dict):
                strings.append(self._gen_sig(list(v.values())))
            else:
                strings.append(str(v))
        return '_'.join(strings)
