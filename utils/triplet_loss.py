import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchAllTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, x, y):
        # Duplicate labels for each part
        p, n, c = x.size()
        y = y.repeat(p, 1)

        # Euclidean distance p x n x n
        x_squared_sum = torch.sum(x ** 2, dim=2)
        x1_squared_sum = x_squared_sum.unsqueeze(2)
        x2_squared_sum = x_squared_sum.unsqueeze(1)
        x1_times_x2_sum = x @ x.transpose(1, 2)
        dist = torch.sqrt(
            F.relu(x1_squared_sum - 2 * x1_times_x2_sum + x2_squared_sum)
        )

        hard_positive_mask = y.unsqueeze(1) == y.unsqueeze(2)
        hard_negative_mask = y.unsqueeze(1) != y.unsqueeze(2)
        all_hard_positive = dist[hard_positive_mask].view(p, n, -1, 1)
        all_hard_negative = dist[hard_negative_mask].view(p, n, 1, -1)
        positive_negative_dist = all_hard_positive - all_hard_negative
        all_loss = F.relu(self.margin + positive_negative_dist).view(p, -1)

        # Non-zero parted mean
        non_zero_counts = (all_loss != 0).sum(1)
        parted_loss_mean = all_loss.sum(1) / non_zero_counts
        parted_loss_mean[non_zero_counts == 0] = 0

        loss = parted_loss_mean.sum()
        return loss
