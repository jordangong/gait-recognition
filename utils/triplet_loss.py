from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchTripletLoss(nn.Module):
    def __init__(
            self,
            is_hard: bool = True,
            is_mean: bool = True,
            margin: Optional[float] = 0.2,
    ):
        super().__init__()
        self.is_hard = is_hard
        self.is_mean = is_mean
        self.margin = margin

    def forward(self, x, y):
        p, n, c = x.size()
        dist = self._batch_distance(x)
        flat_dist_mask = torch.tril_indices(n, n, offset=-1, device=dist.device)
        flat_dist = dist[:, flat_dist_mask[0], flat_dist_mask[1]]

        if self.is_hard:
            positive_negative_dist = self._hard_distance(dist, y, p, n)
        else:  # is_all
            positive_negative_dist = self._all_distance(dist, y, p, n)

        if self.margin:
            losses = F.relu(self.margin + positive_negative_dist).view(p, -1)
            non_zero_counts = (losses != 0).sum(1).float()
            if self.is_mean:
                loss_metric = self._none_zero_mean(losses, non_zero_counts)
            else:  # is_sum
                loss_metric = losses.sum(1)
            return loss_metric, flat_dist, non_zero_counts
        else:  # Soft margin
            losses = F.softplus(positive_negative_dist).view(p, -1)
            if self.is_mean:
                loss_metric = losses.mean(1)
            else:  # is_sum
                loss_metric = losses.sum(1)
            return loss_metric, flat_dist, None

    @staticmethod
    def _batch_distance(x):
        # Euclidean distance p x n x n
        x_squared_sum = torch.sum(x ** 2, dim=2)
        x1_squared_sum = x_squared_sum.unsqueeze(2)
        x2_squared_sum = x_squared_sum.unsqueeze(1)
        x1_times_x2_sum = x @ x.transpose(1, 2)
        dist = torch.sqrt(
            F.relu(x1_squared_sum - 2 * x1_times_x2_sum + x2_squared_sum)
        )
        return dist

    @staticmethod
    def _hard_distance(dist, y, p, n):
        positive_mask = y.unsqueeze(1) == y.unsqueeze(2)
        negative_mask = y.unsqueeze(1) != y.unsqueeze(2)
        hard_positive = dist[positive_mask].view(p, n, -1).max(-1).values
        hard_negative = dist[negative_mask].view(p, n, -1).min(-1).values
        positive_negative_dist = hard_positive - hard_negative

        return positive_negative_dist

    @staticmethod
    def _all_distance(dist, y, p, n):
        # Unmask identical samples
        positive_mask = torch.eye(
            n, dtype=torch.bool, device=y.device
        ) ^ (y.unsqueeze(1) == y.unsqueeze(2))
        negative_mask = y.unsqueeze(1) != y.unsqueeze(2)
        all_positive = dist[positive_mask].view(p, n, -1, 1)
        all_negative = dist[negative_mask].view(p, n, 1, -1)
        positive_negative_dist = all_positive - all_negative

        return positive_negative_dist

    @staticmethod
    def _none_zero_mean(losses, non_zero_counts):
        # Non-zero parted mean
        non_zero_mean = losses.sum(1) / non_zero_counts
        non_zero_mean[non_zero_counts == 0] = 0
        return non_zero_mean


class JointBatchTripletLoss(BatchTripletLoss):
    def __init__(
            self,
            hpm_num_parts: int,
            is_hard: bool = True,
            is_mean: bool = True,
            margins: Tuple[float, float] = (0.2, 0.2)
    ):
        super().__init__(is_hard, is_mean)
        self.hpm_num_parts = hpm_num_parts
        self.margin_hpm, self.margin_pn = margins

    def forward(self, x, y):
        p, n, c = x.size()
        dist = self._batch_distance(x)
        flat_dist_mask = torch.tril_indices(n, n, offset=-1, device=dist.device)
        flat_dist = dist[:, flat_dist_mask[0], flat_dist_mask[1]]

        if self.is_hard:
            positive_negative_dist = self._hard_distance(dist, y, p, n)
        else:  # is_all
            positive_negative_dist = self._all_distance(dist, y, p, n)

        hpm_part_loss = F.relu(
            self.margin_hpm + positive_negative_dist[:self.hpm_num_parts]
        )
        pn_part_loss = F.relu(
            self.margin_pn + positive_negative_dist[self.hpm_num_parts:]
        )
        losses = torch.cat((hpm_part_loss, pn_part_loss)).view(p, -1)

        non_zero_counts = (losses != 0).sum(1).float()
        if self.is_mean:
            loss_metric = self._none_zero_mean(losses, non_zero_counts)
        else:  # is_sum
            loss_metric = losses.sum(1)

        return loss_metric, flat_dist, non_zero_counts
