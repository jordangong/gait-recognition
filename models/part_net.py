import copy
from typing import Tuple

import torch
import torch.nn as nn

from models.layers import BasicConv1d, FocalConv2dBlock


class FrameLevelPartFeatureExtractor(nn.Module):

    def __init__(
            self,
            in_channels: int = 3,
            feature_channels: int = 32,
            kernel_sizes: Tuple[Tuple, ...] = ((5, 3), (3, 3), (3, 3)),
            paddings: Tuple[Tuple, ...] = ((2, 1), (1, 1), (1, 1)),
            halving: Tuple[int, ...] = (0, 2, 3)
    ):
        super().__init__()
        num_blocks = len(kernel_sizes)
        out_channels = [feature_channels * 2 ** i for i in range(num_blocks)]
        in_channels = [in_channels] + out_channels[:-1]
        use_pools = [True] * (num_blocks - 1) + [False]
        params = (in_channels, out_channels, kernel_sizes,
                  paddings, halving, use_pools)

        self.fconv_blocks = nn.ModuleList([
            FocalConv2dBlock(*_params) for _params in zip(*params)
        ])

    def forward(self, x):
        # Flatten frames in all batches
        n, t, c, h, w = x.size()
        x = x.view(n * t, c, h, w)

        for fconv_block in self.fconv_blocks:
            x = fconv_block(x)
        return x


class TemporalFeatureAggregator(nn.Module):
    def __init__(
            self,
            in_channels: int,
            squeeze_ratio: int = 4,
            num_part: int = 16
    ):
        super().__init__()
        hidden_dim = in_channels // squeeze_ratio
        self.num_part = num_part

        # MTB1
        conv3x1 = nn.Sequential(
            BasicConv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            BasicConv1d(hidden_dim, in_channels, kernel_size=1, padding=0)
        )
        self.conv1d3x1 = self._parted(conv3x1)
        self.avg_pool3x1 = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.max_pool3x1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        # MTB2
        conv3x3 = nn.Sequential(
            BasicConv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            BasicConv1d(hidden_dim, in_channels, kernel_size=3, padding=1)
        )
        self.conv1d3x3 = self._parted(conv3x3)
        self.avg_pool3x3 = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.max_pool3x3 = nn.MaxPool1d(kernel_size=5, stride=1, padding=2)

    def _parted(self, module: nn.Module):
        """Duplicate module `part_num` times."""
        return nn.ModuleList([copy.deepcopy(module)
                              for _ in range(self.num_part)])

    def forward(self, x):
        # p, n, t, c
        x = x.transpose(2, 3)
        p, n, c, t = x.size()
        feature = x.split(1, dim=0)
        feature = [f.squeeze(0) for f in feature]
        x = x.view(-1, c, t)

        # MTB1: ConvNet1d & Sigmoid
        logits3x1 = torch.stack(
            [conv(f) for conv, f in zip(self.conv1d3x1, feature)]
        )
        scores3x1 = torch.sigmoid(logits3x1)
        # MTB1: Template Function
        feature3x1 = self.avg_pool3x1(x) + self.max_pool3x1(x)
        feature3x1 = feature3x1.view(p, n, c, t)
        feature3x1 = feature3x1 * scores3x1

        # MTB2: ConvNet1d & Sigmoid
        logits3x3 = torch.stack(
            [conv(f) for conv, f in zip(self.conv1d3x3, feature)]
        )
        scores3x3 = torch.sigmoid(logits3x3)
        # MTB2: Template Function
        feature3x3 = self.avg_pool3x3(x) + self.max_pool3x3(x)
        feature3x3 = feature3x3.view(p, n, c, t)
        feature3x3 = feature3x3 * scores3x3

        # Temporal Pooling
        ret = (feature3x1 + feature3x3).max(-1)[0]
        return ret


class PartNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 128,
            squeeze_ratio: int = 4,
            num_part: int = 16
    ):
        super().__init__()
        self.num_part = num_part
        self.tfa = TemporalFeatureAggregator(
            in_channels, squeeze_ratio, self.num_part
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        n, t, c, h, w = x.size()
        x = x.view(n * t, c, h, w)
        # n * t x c x h x w

        # Horizontal Pooling
        _, c, h, w = x.size()
        split_size = h // self.num_part
        x = x.split(split_size, dim=2)
        x = [self.avg_pool(x_) + self.max_pool(x_) for x_ in x]
        x = [x_.view(n, t, c) for x_ in x]
        x = torch.stack(x)

        # p, n, t, c
        x = self.tfa(x)
        return x
