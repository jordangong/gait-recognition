from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
