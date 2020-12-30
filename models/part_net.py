import copy

import torch
import torch.nn as nn

from models.layers import BasicConv1d, FocalConv2dBlock


class FrameLevelPartFeatureExtractor(nn.Module):

    def __init__(
            self,
            in_channels: int = 3,
            feature_channels: int = 32,
            kernel_sizes: tuple[tuple, ...] = ((5, 3), (3, 3), (3, 3)),
            paddings: tuple[tuple, ...] = ((2, 1), (1, 1), (1, 1)),
            halving: tuple[int, ...] = (0, 2, 3)
    ):
        super().__init__()
        num_blocks = len(kernel_sizes)
        out_channels = [feature_channels * 2 ** i for i in range(num_blocks)]
        in_channels = [in_channels] + out_channels[:-1]
        use_pools = [True] * (num_blocks - 1) + [False]
        params = (in_channels, out_channels, kernel_sizes,
                  paddings, halving, use_pools)

        self.fconv_blocks = [FocalConv2dBlock(*_params)
                             for _params in zip(*params)]

    def forward(self, x):
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


class PartNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            feature_channels: int = 32,
            kernel_sizes: tuple[tuple, ...] = ((5, 3), (3, 3), (3, 3)),
            paddings: tuple[tuple, ...] = ((2, 1), (1, 1), (1, 1)),
            halving: tuple[int, ...] = (0, 2, 3),
            squeeze_ratio: int = 4,
            num_part: int = 16
    ):
        super().__init__()
        self.num_part = num_part
        self.fpfe = FrameLevelPartFeatureExtractor(
            in_channels, feature_channels, kernel_sizes, paddings, halving
        )

        num_fconv_blocks = len(self.fpfe.fconv_blocks)
        tfa_in_channels = feature_channels * 2 ** (num_fconv_blocks - 1)
        self.tfa = TemporalFeatureAggregator(
            tfa_in_channels, squeeze_ratio, self.num_part
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = self.fpfe(x)

        # Horizontal Pooling
        n, t, c, h, w = x.size()
        split_size = h // self.num_part
        x = x.split(split_size, dim=3)
        x = [self.avg_pool(x_) + self.max_pool(x_) for x_ in x]
        x = [x_.view(n, t, c, -1) for x_ in x]
        x = torch.cat(x, dim=3)

        x = self.tfa(x)
        return x
