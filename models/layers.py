from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, tuple[int, int]],
            **kwargs
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class VGGConv2d(BasicConv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, tuple[int, int]] = 3,
            padding: int = 1,
            **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size,
                         padding=padding, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.leaky_relu(x, 0.2, inplace=True)


class BasicConvTranspose2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, tuple[int, int]],
            **kwargs
    ):
        super().__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels,
                                             kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.trans_conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class DCGANConvTranspose2d(BasicConvTranspose2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, tuple[int, int]] = 4,
            stride: int = 2,
            padding: int = 1,
            is_last_layer: bool = False,
            **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, **kwargs)
        self.is_last_layer = is_last_layer

    def forward(self, x):
        if self.is_last_layer:
            return self.trans_conv(x)
        else:
            x = self.trans_conv(x)
            x = self.bn(x)
            return F.leaky_relu(x, 0.2, inplace=True)


class BasicLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
    ):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return x


class FocalConv2d(BasicConv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, tuple[int, int]],
            halving: int,
            **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.halving = halving

    def forward(self, x):
        h = x.size(2)
        split_size = h // 2 ** self.halving
        z = x.split(split_size, dim=2)
        z = torch.cat([self.conv(_) for _ in z], dim=2)
        return F.leaky_relu(z, inplace=True)


class FocalConv2dBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_sizes: tuple[int, int],
            paddings: tuple[int, int],
            halving: int,
            use_pool: bool = True,
            **kwargs
    ):
        super().__init__()
        self.use_pool = use_pool
        self.fconv1 = FocalConv2d(in_channels, out_channels, kernel_sizes[0],
                                  halving, padding=paddings[0], **kwargs)
        self.fconv2 = FocalConv2d(out_channels, out_channels, kernel_sizes[1],
                                  halving, padding=paddings[1], **kwargs)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.fconv1(x)
        x = self.fconv2(x)
        if self.use_pool:
            x = self.max_pool(x)
        return x


class BasicConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, tuple[int]],
            **kwargs
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              bias=False, **kwargs)

    def forward(self, x):
        return self.conv(x)


class HorizontalPyramidPooling(nn.Module):
    def __init__(
            self,
            use_avg_pool: bool = True,
            use_max_pool: bool = False,
    ):
        super().__init__()
        self.use_avg_pool = use_avg_pool
        self.use_max_pool = use_max_pool
        assert use_avg_pool or use_max_pool, 'Pooling layer(s) required.'
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        if self.use_avg_pool and self.use_max_pool:
            x = self.avg_pool(x) + self.max_pool(x)
        elif self.use_avg_pool and not self.use_max_pool:
            x = self.avg_pool(x)
        elif not self.use_avg_pool and self.use_max_pool:
            x = self.max_pool(x)
        return x
