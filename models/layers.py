from typing import Union, Tuple

import torch
import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            **kwargs
    ):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              bias=False, **kwargs)

    def forward(self, x):
        return self.conv(x)


class VGGConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            padding: int = 1,
            **kwargs
    ):
        super(VGGConv2d, self).__init__()
        self.conv = BasicConv2d(in_channels, out_channels, kernel_size,
                                padding=padding, **kwargs)

    def forward(self, x):
        return self.conv(x)


class BasicConvTranspose2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            **kwargs
    ):
        super(BasicConvTranspose2d, self).__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels,
                                             kernel_size, bias=False, **kwargs)

    def forward(self, x):
        return self.trans_conv(x)


class DCGANConvTranspose2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]] = 4,
            stride: int = 2,
            padding: int = 1,
            **kwargs
    ):
        super(DCGANConvTranspose2d).__init__()
        self.trans_conv = BasicConvTranspose2d(in_channels, out_channels,
                                               kernel_size, stride=stride,
                                               padding=padding, **kwargs)

    def forward(self, x):
        return self.trans_conv(x)


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
        return z


class BasicConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int]],
            **kwargs
    ):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              bias=False, **kwargs)

    def forward(self, x):
        return self.conv(x)
