from typing import Tuple

import torch
import torch.nn as nn

from models.layers import HorizontalPyramidPooling


class HorizontalPyramidMatching(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int = 128,
            scales: Tuple[int, ...] = (1, 2, 4),
            use_avg_pool: bool = True,
            use_max_pool: bool = True,
            **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.use_avg_pool = use_avg_pool
        self.use_max_pool = use_max_pool

        self.pyramids = nn.ModuleList([
            self._make_pyramid(scale, **kwargs) for scale in self.scales
        ])

    def _make_pyramid(self, scale: int, **kwargs):
        pyramid = nn.ModuleList([
            HorizontalPyramidPooling(self.in_channels,
                                     self.out_channels,
                                     use_avg_pool=self.use_avg_pool,
                                     use_max_pool=self.use_max_pool,
                                     **kwargs)
            for _ in range(scale)
        ])
        return pyramid

    def forward(self, x):
        # Flatten canonical features in all batches
        t, n, c, h, w = x.size()
        x = x.view(t * n, c, h, w)

        feature = []
        for pyramid_index, pyramid in enumerate(self.pyramids):
            h_per_hpp = h // self.scales[pyramid_index]
            for hpp_index, hpp in enumerate(pyramid):
                h_filter = torch.arange(hpp_index * h_per_hpp,
                                        (hpp_index + 1) * h_per_hpp)
                x_slice = x[:, :, h_filter, :]
                x_slice = hpp(x_slice)
                x_slice = x_slice.view(t * n, -1)
                feature.append(x_slice)
        x = torch.stack(feature)

        # Unfold frames to original batch
        p, _, c = x.size()
        x = x.view(p, t, n, c)
        return x
