import copy

import torch
from torch import nn as nn


class TemporalFeatureAggregator(nn.Module):
    def __init__(
            self,
            in_channels: int,
            squeeze: int = 4,
            num_part: int = 16
    ):
        super(TemporalFeatureAggregator, self).__init__()
        hidden_dim = in_channels // squeeze
        self.num_part = num_part

        # MTB1
        conv3x1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim,
                      kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(hidden_dim, in_channels,
                      kernel_size=1, padding=0, bias=False)
        )
        self.conv1d3x1 = self._parted(conv3x1)
        self.avg_pool3x1 = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.max_pool3x1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        # MTB2
        conv3x3 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim,
                      kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(hidden_dim, in_channels,
                      kernel_size=3, padding=1, bias=False)
        )
        self.conv1d3x3 = self._parted(conv3x3)
        self.avg_pool3x3 = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.max_pool3x3 = nn.MaxPool1d(kernel_size=5, stride=1, padding=2)

    def _parted(self, module: nn.Module):
        """Duplicate module `part_num` times."""
        return nn.ModuleList([copy.deepcopy(module)
                              for _ in range(self.num_part)])

    def forward(self, x):
        """
          Input: x, [p, n, c, s]
        """
        p, n, c, s = x.size()
        feature = x.split(1, 0)
        x = x.view(-1, c, s)

        # MTB1: ConvNet1d & Sigmoid
        logits3x1 = torch.cat(
            [conv(_.squeeze(0)).unsqueeze(0)
             for conv, _ in zip(self.conv1d3x1, feature)], dim=0
        )
        scores3x1 = torch.sigmoid(logits3x1)
        # MTB1: Template Function
        feature3x1 = self.avg_pool3x1(x) + self.max_pool3x1(x)
        feature3x1 = feature3x1.view(p, n, c, s)
        feature3x1 = feature3x1 * scores3x1

        # MTB2: ConvNet1d & Sigmoid
        logits3x3 = torch.cat(
            [conv(_.squeeze(0)).unsqueeze(0)
             for conv, _ in zip(self.conv1d3x3, feature)], dim=0
        )
        scores3x3 = torch.sigmoid(logits3x3)
        # MTB2: Template Function
        feature3x3 = self.avg_pool3x3(x) + self.max_pool3x3(x)
        feature3x3 = feature3x3.view(p, n, c, s)
        feature3x3 = feature3x3 * scores3x3

        # Temporal Pooling
        ret = (feature3x1 + feature3x3).max(-1)[0]
        return ret