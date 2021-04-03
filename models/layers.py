from typing import Union

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
