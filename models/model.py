import copy
import os
import random
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.hpm import HorizontalPyramidMatching
from models.part_net import PartNet
from models.rgb_part_net import RGBPartNet
from utils.configuration import DataloaderConfiguration, \
    HyperparameterConfiguration, DatasetConfiguration, ModelConfiguration, \
    SystemConfiguration
from utils.dataset import CASIAB, ClipConditions, ClipViews, ClipClasses
from utils.sampler import TripletSampler
from utils.triplet_loss import BatchTripletLoss


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
        self.restore_iter = self.curr_iter = self.meta.get('restore_iter', 0)
        self.total_iter = self.meta.get('total_iter', 80_000)
        self.restore_iters = self.meta.get('restore_iters', (self.curr_iter,))
        self.total_iters = self.meta.get('total_iters', (self.total_iter,))

        self.is_train: bool = True
        self.in_channels: int = 3
        self.in_size: tuple[int, int] = (64, 48)
        self.pr: Optional[int] = None
        self.k: Optional[int] = None
        self.num_pairs: Optional[int] = None
        self.num_pos_pairs: Optional[int] = None

        self._gallery_dataset_meta: Optional[dict[str, list]] = None
        self._probe_datasets_meta: Optional[dict[str, dict[str, list]]] = None

        self._model_name: str = self.meta.get('name', 'RGB-GaitPart')
        self._hp_sig: str = self._make_signature(self.hp)
        self._dataset_sig: str = 'undefined'

        self.rgb_pn: Optional[RGBPartNet] = None
        self.triplet_loss_hpm: Optional[BatchTripletLoss] = None
        self.triplet_loss_pn: Optional[BatchTripletLoss] = None
        self.optimizer: Optional[optim.Adam] = None
        self.scheduler: Optional[optim.lr_scheduler.StepLR] = None
        self.writer: Optional[SummaryWriter] = None
        self.image_log_on = system_config.get('image_log_on', False)
        self.image_log_steps = system_config.get('image_log_steps', 100)
        self.val_size = system_config.get('val_size', 10)

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
            (self._model_name, str(self.curr_iter + 1), str(self.total_iter))
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
        for (restore_iter, total_iter, (condition, selector)) in zip(
                self.restore_iters, self.total_iters, dataset_selectors.items()
        ):
            print(f'Training model {condition} ...')
            # Skip finished model
            if restore_iter == total_iter:
                continue
            # Check invalid restore iter
            elif restore_iter > total_iter:
                raise ValueError("Restore iter '{}' should less than total "
                                 "iter '{}'".format(restore_iter, total_iter))
            self.restore_iter = self.curr_iter = restore_iter
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
        # Validation dataset
        # (the first `val_size` subjects from evaluation set)
        val_dataset_config = copy.deepcopy(dataset_config)
        train_size = dataset_config.get('train_size', 74)
        val_dataset_config['train_size'] = train_size + self.val_size
        val_dataset_config['selector']['classes'] = ClipClasses({
            str(c).zfill(3)
            for c in range(train_size + 1, train_size + self.val_size + 1)
        })
        val_dataset = self._parse_dataset_config(val_dataset_config)
        val_dataloader = iter(self._parse_dataloader_config(
            val_dataset, dataloader_config
        ))
        # Training dataset
        train_dataset = self._parse_dataset_config(dataset_config)
        train_dataloader = iter(self._parse_dataloader_config(
            train_dataset, dataloader_config
        ))
        # Prepare for model, optimizer and scheduler
        model_hp: dict = self.hp.get('model', {}).copy()
        triplet_is_hard = model_hp.pop('triplet_is_hard', True)
        triplet_is_mean = model_hp.pop('triplet_is_mean', True)
        triplet_margins = model_hp.pop('triplet_margins', None)
        optim_hp: dict = self.hp.get('optimizer', {}).copy()
        ae_optim_hp = optim_hp.pop('auto_encoder', {})
        hpm_optim_hp = optim_hp.pop('hpm', {})
        pn_optim_hp = optim_hp.pop('part_net', {})
        sched_hp = self.hp.get('scheduler', {})
        ae_sched_hp = sched_hp.get('auto_encoder', {})
        hpm_sched_hp = sched_hp.get('hpm', {})
        pn_sched_hp = sched_hp.get('part_net', {})

        self.rgb_pn = RGBPartNet(self.in_channels, self.in_size, **model_hp,
                                 image_log_on=self.image_log_on)
        # Hard margins
        if triplet_margins:
            self.triplet_loss_hpm = BatchTripletLoss(
                triplet_is_hard, triplet_is_mean, triplet_margins[0]
            )
            self.triplet_loss_pn = BatchTripletLoss(
                triplet_is_hard, triplet_is_mean, triplet_margins[1]
            )
        else:  # Soft margins
            self.triplet_loss_hpm = BatchTripletLoss(
                triplet_is_hard, triplet_is_mean, None
            )
            self.triplet_loss_pn = BatchTripletLoss(
                triplet_is_hard, triplet_is_mean, None
            )

        self.num_pairs = (self.pr * self.k - 1) * (self.pr * self.k) // 2
        self.num_pos_pairs = (self.k * (self.k - 1) // 2) * self.pr

        # Try to accelerate computation using CUDA or others
        self.rgb_pn = self.rgb_pn.to(self.device)
        self.triplet_loss_hpm = self.triplet_loss_hpm.to(self.device)
        self.triplet_loss_pn = self.triplet_loss_pn.to(self.device)

        self.optimizer = optim.Adam([
            {'params': self.rgb_pn.ae.parameters(), **ae_optim_hp},
            {'params': self.rgb_pn.hpm.parameters(), **hpm_optim_hp},
            {'params': self.rgb_pn.pn.parameters(), **pn_optim_hp},
        ], **optim_hp)

        # Scheduler
        start_step = sched_hp.get('start_step', 0)
        stop_step = sched_hp.get('stop_step', self.total_iter)
        final_gamma = sched_hp.get('final_gamma', 0.001)
        ae_start_step = ae_sched_hp.get('start_step', start_step)
        ae_stop_step = ae_sched_hp.get('stop_step', stop_step)
        ae_final_gamma = ae_sched_hp.get('final_gamma', final_gamma)
        ae_all_step = ae_stop_step - ae_start_step
        hpm_start_step = hpm_sched_hp.get('start_step', start_step)
        hpm_stop_step = hpm_sched_hp.get('stop_step', stop_step)
        hpm_final_gamma = hpm_sched_hp.get('final_gamma', final_gamma)
        hpm_all_step = hpm_stop_step - hpm_start_step
        pn_start_step = pn_sched_hp.get('start_step', start_step)
        pn_stop_step = pn_sched_hp.get('stop_step', stop_step)
        pn_final_gamma = pn_sched_hp.get('final_gamma', final_gamma)
        pn_all_step = pn_stop_step - pn_start_step
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=[
            lambda t: 1 if t <= ae_start_step
            else ae_final_gamma ** ((t - ae_start_step) / ae_all_step)
            if ae_start_step < t <= ae_stop_step else ae_final_gamma,
            lambda t: 1 if t <= hpm_start_step
            else hpm_final_gamma ** ((t - hpm_start_step) / hpm_all_step)
            if hpm_start_step < t <= hpm_stop_step else hpm_final_gamma,
            lambda t: 1 if t <= pn_start_step
            else pn_final_gamma ** ((t - pn_start_step) / pn_all_step)
            if pn_start_step < t <= pn_stop_step else pn_final_gamma,
        ])

        self.writer = SummaryWriter(self._log_name)

        # Set seeds for reproducibility
        random.seed(0)
        torch.manual_seed(0)
        self.rgb_pn.train()
        # Init weights at first iter
        if self.curr_iter == 0:
            self.rgb_pn.apply(self.init_weights)
        else:  # Load saved state dicts
            # Offset a iter to load last checkpoint
            self.curr_iter -= 1
            checkpoint = torch.load(self._checkpoint_name)
            random.setstate(checkpoint['rand_states'][0])
            torch.set_rng_state(checkpoint['rand_states'][1])
            self.rgb_pn.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
            self.scheduler.load_state_dict(checkpoint['sched_state_dict'])

        # Training start
        for self.curr_iter in tqdm(range(self.restore_iter, self.total_iter),
                                   desc='Training'):
            batch_c1, batch_c2 = next(train_dataloader)
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            # forward + backward + optimize
            x_c1 = batch_c1['clip'].to(self.device)
            x_c2 = batch_c2['clip'].to(self.device)
            embed_c, embed_p, ae_losses, images = self.rgb_pn(x_c1, x_c2)
            y = batch_c1['label'].to(self.device)
            losses, hpm_result, pn_result = self._classification_loss(
                embed_c, embed_p, ae_losses, y
            )
            loss = losses.sum()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Learning rate
            self.writer.add_scalars('Learning rate', dict(zip((
                'Auto-encoder', 'HPM', 'PartNet'
            ), self.scheduler.get_last_lr())), self.curr_iter)
            # Other stats
            self._write_stat(
                'Train', embed_c, embed_p, hpm_result, pn_result, loss, losses
            )

            # Write disentangled images
            if self.image_log_on and self.curr_iter % self.image_log_steps \
                    == self.image_log_steps - 1:
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

            if self.curr_iter % 100 == 99:
                # Validation
                embed_c = self._flatten_embedding(embed_c)
                embed_p = self._flatten_embedding(embed_p)
                self._write_embedding('HPM Train', embed_c, x_c1, y)
                self._write_embedding('PartNet Train', embed_p, x_c1, y)

                # Calculate losses on testing batch
                batch_c1, batch_c2 = next(val_dataloader)
                x_c1 = batch_c1['clip'].to(self.device)
                x_c2 = batch_c2['clip'].to(self.device)
                with torch.no_grad():
                    embed_c, embed_p, ae_losses, _ = self.rgb_pn(x_c1, x_c2)
                y = batch_c1['label'].to(self.device)
                losses, hpm_result, pn_result = self._classification_loss(
                    embed_c, embed_p, ae_losses, y
                )
                loss = losses.sum()

                self._write_stat(
                    'Val', embed_c, embed_p, hpm_result, pn_result, loss, losses
                )
                embed_c = self._flatten_embedding(embed_c)
                embed_p = self._flatten_embedding(embed_p)
                self._write_embedding('HPM Val', embed_c, x_c1, y)
                self._write_embedding('PartNet Val', embed_p, x_c1, y)

            # Checkpoint
            if self.curr_iter % 1000 == 999:
                torch.save({
                    'rand_states': (random.getstate(), torch.get_rng_state()),
                    'model_state_dict': self.rgb_pn.state_dict(),
                    'optim_state_dict': self.optimizer.state_dict(),
                    'sched_state_dict': self.scheduler.state_dict(),
                }, self._checkpoint_name)

        self.writer.close()

    def _classification_loss(self, embed_c, embed_p, ae_losses, y):
        # Duplicate labels for each part
        y_triplet = y.repeat(self.rgb_pn.num_parts, 1)
        hpm_result = self.triplet_loss_hpm(
            embed_c, y_triplet[:self.rgb_pn.hpm.num_parts]
        )
        pn_result = self.triplet_loss_pn(
            embed_p, y_triplet[self.rgb_pn.hpm.num_parts:]
        )
        losses = torch.stack((
            *ae_losses,
            hpm_result.pop('loss').mean(),
            pn_result.pop('loss').mean()
        ))
        return losses, hpm_result, pn_result

    def _write_embedding(self, tag, embed, x, y):
        frame = x[:, 0, :, :, :].cpu()
        n, c, h, w = frame.size()
        padding = torch.zeros(n, c, h, (h - w) // 2)
        padded_frame = torch.cat((padding, frame, padding), dim=-1)
        self.writer.add_embedding(
            embed,
            metadata=y.cpu().tolist(),
            label_img=padded_frame,
            global_step=self.curr_iter,
            tag=tag
        )

    def _flatten_embedding(self, embed):
        return embed.detach().transpose(0, 1).reshape(self.k * self.pr, -1)

    def _write_stat(
            self, postfix, embed_c, embed_p, hpm_result, pn_result, loss, losses
    ):
        # Write losses to TensorBoard
        self.writer.add_scalar(f'Loss/all {postfix}', loss, self.curr_iter)
        self.writer.add_scalars(f'Loss/disentanglement {postfix}', dict(zip((
            'Cross reconstruction loss', 'Canonical consistency loss',
            'Pose similarity loss'
        ), losses[:3])), self.curr_iter)
        self.writer.add_scalars(f'Loss/triplet loss {postfix}', {
            'HPM': losses[3],
            'PartNet': losses[4]
        }, self.curr_iter)
        # None-zero losses in batch
        if hpm_result['counts'] is not None and pn_result['counts'] is not None:
            self.writer.add_scalars(f'Loss/non-zero counts {postfix}', {
                'HPM': hpm_result['counts'].mean(),
                'PartNet': pn_result['counts'].mean()
            }, self.curr_iter)
        # Embedding distance
        mean_hpm_dist = hpm_result['dist'].mean(0)
        self._add_ranked_scalars(
            f'Embedding/HPM distance {postfix}', mean_hpm_dist,
            self.num_pos_pairs, self.num_pairs, self.curr_iter
        )
        mean_pn_dist = pn_result['dist'].mean(0)
        self._add_ranked_scalars(
            f'Embedding/ParNet distance {postfix}', mean_pn_dist,
            self.num_pos_pairs, self.num_pairs, self.curr_iter
        )
        # Embedding norm
        mean_hpm_embedding = embed_c.mean(0)
        mean_hpm_norm = mean_hpm_embedding.norm(dim=-1)
        self._add_ranked_scalars(
            f'Embedding/HPM norm {postfix}', mean_hpm_norm,
            self.k, self.pr * self.k, self.curr_iter
        )
        mean_pa_embedding = embed_p.mean(0)
        mean_pa_norm = mean_pa_embedding.norm(dim=-1)
        self._add_ranked_scalars(
            f'Embedding/PartNet norm {postfix}', mean_pa_norm,
            self.k, self.pr * self.k, self.curr_iter
        )

    def _add_ranked_scalars(
            self,
            main_tag: str,
            metric: torch.Tensor,
            num_pos: int,
            num_all: int,
            global_step: int
    ):
        rank = metric.argsort()
        pos_ile = 100 - (num_pos - 1) * 100 // num_all
        self.writer.add_scalars(main_tag, {
            '0%-ile': metric[rank[-1]],
            f'{100 - pos_ile}%-ile': metric[rank[-num_pos]],
            '50%-ile': metric[rank[num_all // 2 - 1]],
            f'{pos_ile}%-ile': metric[rank[num_pos - 1]],
            '100%-ile': metric[rank[0]]
        }, global_step)

    def predict_all(
            self,
            iters: tuple[int],
            dataset_config: DatasetConfiguration,
            dataset_selectors: dict[
                str, dict[str, Union[ClipClasses, ClipConditions, ClipViews]]
            ],
            dataloader_config: DataloaderConfiguration,
    ) -> dict[str, torch.Tensor]:
        # Transform data to features
        gallery_samples, probe_samples = self.transform(
            iters, dataset_config, dataset_selectors, dataloader_config
        )
        # Evaluate features
        accuracy = self.evaluate(gallery_samples, probe_samples)

        return accuracy

    def transform(
            self,
            iters: tuple[int],
            dataset_config: DatasetConfiguration,
            dataset_selectors: dict[
                str, dict[str, Union[ClipClasses, ClipConditions, ClipViews]]
            ],
            dataloader_config: DataloaderConfiguration,
            is_train: bool = False
    ):
        # Split gallery and probe dataset
        gallery_dataloader, probe_dataloaders = self._split_gallery_probe(
            dataset_config, dataloader_config, is_train
        )
        # Get pretrained models at iter_
        checkpoints = self._load_pretrained(
            iters, dataset_config, dataset_selectors
        )

        # Init models
        model_hp: dict = self.hp.get('model', {}).copy()
        model_hp.pop('triplet_is_hard', True)
        model_hp.pop('triplet_is_mean', True)
        model_hp.pop('triplet_margins', None)
        self.rgb_pn = RGBPartNet(self.in_channels, self.in_size, **model_hp)
        # Try to accelerate computation using CUDA or others
        self.rgb_pn = self.rgb_pn.to(self.device)
        self.rgb_pn.eval()

        gallery_samples, probe_samples = {}, {}
        for (condition, probe_dataloader) in probe_dataloaders.items():
            checkpoint = torch.load(checkpoints[condition])
            self.rgb_pn.load_state_dict(checkpoint['model_state_dict'])
            # Gallery
            gallery_samples_c = []
            for sample in tqdm(gallery_dataloader,
                               desc=f'Transforming gallery {condition}',
                               unit='clips'):
                gallery_samples_c.append(self._get_eval_sample(sample))
            gallery_samples[condition] = default_collate(gallery_samples_c)
            # Probe
            probe_samples_c = []
            for sample in tqdm(probe_dataloader,
                               desc=f'Transforming probe {condition}',
                               unit='clips'):
                probe_samples_c.append(self._get_eval_sample(sample))
            probe_samples_c = default_collate(probe_samples_c)
            probe_samples_c['meta'] = self._probe_datasets_meta[condition]
            probe_samples[condition] = probe_samples_c
        gallery_samples['meta'] = self._gallery_dataset_meta

        return gallery_samples, probe_samples

    def _get_eval_sample(self, sample: dict[str, Union[list, torch.Tensor]]):
        label, condition, view, clip = sample.values()
        with torch.no_grad():
            feature_c, feature_p = self.rgb_pn(clip.to(self.device))
        return {
            'label': label.item(),
            'condition': condition[0],
            'view': view[0],
            'feature': torch.cat((feature_c, feature_p)).view(-1)
        }

    @staticmethod
    def evaluate(
            gallery_samples: dict[str, dict[str, Union[list, torch.Tensor]]],
            probe_samples: dict[str, dict[str, Union[list, torch.Tensor]]],
            num_ranks: int = 5
    ) -> dict[str, torch.Tensor]:
        conditions = list(probe_samples.keys())
        gallery_views_meta = gallery_samples['meta']['views']
        probe_views_meta = probe_samples[conditions[0]]['meta']['views']
        accuracy = {
            condition: torch.empty(
                len(gallery_views_meta), len(probe_views_meta), num_ranks
            )
            for condition in conditions
        }

        for condition in conditions:
            gallery_samples_c = gallery_samples[condition]
            (labels_g, _, views_g, features_g) = gallery_samples_c.values()
            views_g = np.asarray(views_g)
            probe_samples_c = probe_samples[condition]
            (labels_p, _, views_p, features_p, _) = probe_samples_c.values()
            views_p = np.asarray(views_p)
            accuracy_c = accuracy[condition]
            for (v_g_i, view_g) in enumerate(gallery_views_meta):
                gallery_view_mask = (views_g == view_g)
                f_g = features_g[gallery_view_mask]
                y_g = labels_g[gallery_view_mask]
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
            iters: tuple[int],
            dataset_config: DatasetConfiguration,
            dataset_selectors: dict[
                str, dict[str, Union[ClipClasses, ClipConditions, ClipViews]]
            ]
    ) -> dict[str, str]:
        checkpoints = {}
        for (iter_, total_iter, (condition, selector)) in zip(
                iters, self.total_iters, dataset_selectors.items()
        ):
            self.curr_iter = iter_ - 1
            self.total_iter = total_iter
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
            is_train: bool = False
    ) -> tuple[DataLoader, dict[str, DataLoader]]:
        dataset_name = dataset_config.get('name', 'CASIA-B')
        if dataset_name == 'CASIA-B':
            self.is_train = is_train
            gallery_dataset = self._parse_dataset_config(
                dict(**dataset_config, **self.CASIAB_GALLERY_SELECTOR)
            )
            probe_datasets = {
                condition: self._parse_dataset_config(
                    dict(**dataset_config, **selector)
                )
                for (condition, selector) in self.CASIAB_PROBE_SELECTORS.items()
            }
            self._gallery_dataset_meta = gallery_dataset.metadata
            self._probe_datasets_meta = {
                condition: dataset.metadata
                for (condition, dataset) in probe_datasets.items()
            }
            self.is_train = False
            gallery_dataloader = self._parse_dataloader_config(
                gallery_dataset, dataloader_config
            )
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
        elif isinstance(m, (HorizontalPyramidMatching, PartNet)):
            nn.init.xavier_uniform_(m.fc_mat)

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
