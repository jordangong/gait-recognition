from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchTripletLoss(nn.Module):
    def __init__(
            self,
            is_hard: bool = True,
            margin: Optional[float] = 0.2,
    ):
        super().__init__()
        self.is_hard = is_hard
        self.margin = margin

    def forward(self, x, y):
        p, n, c = x.size()
        dist = self._batch_distance(x)

        if self.is_hard:
            positive_negative_dist = self._hard_distance(dist, y, p, n)
        else:  # is_all
            positive_negative_dist = self._all_distance(dist, y, p, n)

        if self.margin:
            all_loss = F.relu(self.margin + positive_negative_dist).view(p, -1)
        else:
            all_loss = F.softplus(positive_negative_dist).view(p, -1)
        non_zero_mean, non_zero_counts = self._none_zero_parted_mean(all_loss)

        return non_zero_mean, dist.mean((1, 2)), non_zero_counts

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
        positive_mask = y.unsqueeze(1) == y.unsqueeze(2)
        negative_mask = y.unsqueeze(1) != y.unsqueeze(2)
        all_positive = dist[positive_mask].view(p, n, -1, 1)
        all_negative = dist[negative_mask].view(p, n, 1, -1)
        positive_negative_dist = all_positive - all_negative

        return positive_negative_dist

    @staticmethod
    def _none_zero_parted_mean(all_loss):
        # Non-zero parted mean
        non_zero_counts = (all_loss != 0).sum(1).float()
        non_zero_mean = all_loss.sum(1) / non_zero_counts
        non_zero_mean[non_zero_counts == 0] = 0

        return non_zero_mean, non_zero_counts


class JointBatchTripletLoss(BatchTripletLoss):
    def __init__(
            self,
            hpm_num_parts: int,
            is_hard: bool = True,
            margins: Tuple[float, float] = (0.2, 0.2)
    ):
        super().__init__(is_hard)
        self.hpm_num_parts = hpm_num_parts
        self.margin_hpm, self.margin_pn = margins

    def forward(self, x, y):
        p, n, c = x.size()
        dist = self._batch_distance(x)

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
        all_loss = torch.cat((hpm_part_loss, pn_part_loss)).view(p, -1)
        non_zero_mean, non_zero_counts = self._none_zero_parted_mean(all_loss)

        return non_zero_mean, dist.mean((1, 2)), non_zero_counts
