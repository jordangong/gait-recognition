import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchAllTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, x, y):
        p, n, c = x.size()

        dist = self._batch_distance(x)
        positive_negative_dist = self._hard_distance(dist, y, p, n)
        all_loss = F.relu(self.margin + positive_negative_dist).view(p, -1)
        parted_loss_mean = self._none_zero_parted_mean(all_loss)

        return parted_loss_mean

    @staticmethod
    def _hard_distance(dist, y, p, n):
        hard_positive_mask = y.unsqueeze(1) == y.unsqueeze(2)
        hard_negative_mask = y.unsqueeze(1) != y.unsqueeze(2)
        all_hard_positive = dist[hard_positive_mask].view(p, n, -1, 1)
        all_hard_negative = dist[hard_negative_mask].view(p, n, 1, -1)
        positive_negative_dist = all_hard_positive - all_hard_negative

        return positive_negative_dist

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
    def _none_zero_parted_mean(all_loss):
        # Non-zero parted mean
        non_zero_counts = (all_loss != 0).sum(1)
        parted_loss_mean = all_loss.sum(1) / non_zero_counts
        parted_loss_mean[non_zero_counts == 0] = 0

        return parted_loss_mean


class JointBatchAllTripletLoss(BatchAllTripletLoss):
    def __init__(
            self,
            hpm_num_parts: int,
            margins: tuple[float, float] = (0.2, 0.2)
    ):
        super().__init__()
        self.hpm_num_parts = hpm_num_parts
        self.margin_hpm, self.margin_pn = margins

    def forward(self, x, y):
        p, n, c = x.size()

        dist = self._batch_distance(x)
        positive_negative_dist = self._hard_distance(dist, y, p, n)
        hpm_part_loss = F.relu(
            self.margin_hpm + positive_negative_dist[:self.hpm_num_parts]
        ).view(self.hpm_num_parts, -1)
        pn_part_loss = F.relu(
            self.margin_pn + positive_negative_dist[self.hpm_num_parts:]
        ).view(p - self.hpm_num_parts, -1)
        all_loss = torch.cat((hpm_part_loss, pn_part_loss)).view(p, -1)
        parted_loss_mean = self._none_zero_parted_mean(all_loss)

        return parted_loss_mean
