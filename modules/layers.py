import copy
from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels: int, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        self.em_dim = opt.em_dim
        nf = 64

        # Cx[HxW]
        # Conv1 3x64x32 -> 64x64x32
        self.conv1 = nn.Conv2d(in_channels, nf, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(nf)
        # MaxPool1 64x64x32 -> 64x32x16
        self.max_pool1 = nn.AdaptiveMaxPool2d((32, 16))
        # Conv2 64x32x16 -> 256x32x16
        self.conv2 = nn.Conv2d(nf, nf * 4, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(nf * 4)
        # MaxPool2 256x32x16 -> 256x16x8
        self.max_pool2 = nn.AdaptiveMaxPool2d((16, 8))
        # Conv3 256x16x8 -> 512x16x8
        self.conv3 = nn.Conv2d(nf * 4, nf * 8, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(nf * 8)
        # Conv4 512x16x8 -> 512x16x8 (for large dataset)
        self.conv4 = nn.Conv2d(nf * 8, nf * 8, kernel_size=3, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(nf * 8)
        # MaxPool3 512x16x8 -> 512x4x2
        self.max_pool3 = nn.AdaptiveMaxPool2d((4, 2))
        # FC 512*4*2 -> 320
        self.fc = nn.Linear(nf * 8 * 2 * 4, self.em_dim)
        self.batch_norm_fc = nn.BatchNorm1d(self.em_dim)

    def forward(self, x):
        x = F.leaky_relu(self.batch_norm1(self.conv1(x)), 0.2)
        x = self.max_pool1(x)
        x = F.leaky_relu(self.batch_norm2(self.conv2(x)), 0.2)
        x = self.max_pool2(x)
        x = F.leaky_relu(self.batch_norm3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.batch_norm4(self.conv4(x)), 0.2)
        x = self.max_pool3(x)
        x = x.view(-1, (64 * 8) * 2 * 4)
        embedding = self.batch_norm_fc(self.fc(x))

        fa, fgs, fgd = embedding.split(
            (self.opt.fa_dim, self.opt.fg_dim / 2, self.opt.fg_dim / 2), dim=1
        )
        return fa, fgs, fgd


class Decoder(nn.Module):
    def __init__(self, out_channels: int, opt):
        super(Decoder, self).__init__()
        self.em_dim = opt.em_dim
        nf = 64

        # Cx[HxW]
        # FC 320 -> 512*4*2
        self.fc = nn.Linear(self.em_dim, nf * 8 * 2 * 4)
        self.batch_norm_fc = nn.BatchNorm1d(nf * 8 * 2 * 4)
        # TransConv1 512x4x2 -> 256x8x4
        self.trans_conv1 = nn.ConvTranspose2d(nf * 8, nf * 4, kernel_size=4,
                                              stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(nf * 4)
        # TransConv2 256x8x4 -> 128x16x8
        self.trans_conv2 = nn.ConvTranspose2d(nf * 4, nf * 2, kernel_size=4,
                                              stride=2, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(nf * 2)
        # TransConv3 128x16x8 -> 64x32x16
        self.trans_conv3 = nn.ConvTranspose2d(nf * 2, nf, kernel_size=4,
                                              stride=2, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(nf)
        # TransConv4 3x32x16
        self.trans_conv4 = nn.ConvTranspose2d(nf, out_channels, kernel_size=4,
                                              stride=2, padding=1)

    def forward(self, fa, fgs, fgd):
        x = torch.cat((fa, fgs, fgd), dim=1).view(-1, self.em_dim)
        x = F.leaky_relu(self.batch_norm_fc(self.fc(x)), 0.2)
        x = F.leaky_relu(self.batch_norm1(self.trans_conv1(x)), 0.2)
        x = F.leaky_relu(self.batch_norm2(self.trans_conv2(x)), 0.2)
        x = F.leaky_relu(self.batch_norm3(self.trans_conv3(x)), 0.2)
        x = F.sigmoid(self.trans_conv4(x))

        return x


class FocalConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            halving: int,
            **kwargs
    ):
        super(FocalConv2d, self).__init__()
        self.halving = halving
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              bias=False, **kwargs)

    def forward(self, x):
        h = x.size(2)
        split_size = h // 2 ** self.halving
        z = x.split(split_size, dim=2)
        z = torch.cat([self.conv(_) for _ in z], dim=2)
        return F.leaky_relu(z, inplace=True)


class FrameLevelPartFeatureExtractor(nn.Module):

    def __init__(self, in_channels: int):
        super(FrameLevelPartFeatureExtractor, self).__init__()
        nf = 32

        self.focal_conv1 = FocalConv2d(in_channels, nf, kernel_size=5,
                                       padding=2, halving=1)
        self.focal_conv2 = FocalConv2d(nf, nf, kernel_size=3,
                                       padding=1, halving=1)
        self.focal_conv3 = FocalConv2d(nf, nf * 2, kernel_size=3,
                                       padding=1, halving=4)
        self.focal_conv4 = FocalConv2d(nf * 2, nf * 2, kernel_size=3,
                                       padding=1, halving=4)
        self.focal_conv5 = FocalConv2d(nf * 2, nf * 4, kernel_size=3,
                                       padding=1, halving=8)
        self.focal_conv6 = FocalConv2d(nf * 4, nf * 4, kernel_size=3,
                                       padding=1, halving=8)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.leaky_relu(self.focal_conv1(x))
        x = F.leaky_relu(self.focal_conv2(x))
        x = self.max_pool(x)
        x = F.leaky_relu(self.focal_conv3(x))
        x = F.leaky_relu(self.focal_conv4(x))
        x = self.max_pool(x)
        x = F.leaky_relu(self.focal_conv5(x))
        x = F.leaky_relu(self.focal_conv6(x))

        return x


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
