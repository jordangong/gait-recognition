import torch
import torch.nn as nn

from models.layers import HorizontalPyramidPooling


class HorizontalPyramidMatching(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int = 128,
            scales: tuple[int, ...] = (1, 2, 4),
            use_avg_pool: bool = True,
            use_max_pool: bool = False,
    ):
        super().__init__()
        self.scales = scales
        self.num_parts = sum(scales)
        self.use_avg_pool = use_avg_pool
        self.use_max_pool = use_max_pool

        self.pyramids = nn.ModuleList([
            self._make_pyramid(scale) for scale in scales
        ])
        self.fc_mat = nn.Parameter(
            torch.empty(self.num_parts, in_channels, out_channels)
        )

    def _make_pyramid(self, scale: int):
        pyramid = nn.ModuleList([
            HorizontalPyramidPooling(self.use_avg_pool, self.use_max_pool)
            for _ in range(scale)
        ])
        return pyramid

    def forward(self, x):
        n, c, h, w = x.size()
        feature = []
        for scale, pyramid in zip(self.scales, self.pyramids):
            h_per_hpp = h // scale
            for hpp_index, hpp in enumerate(pyramid):
                h_filter = torch.arange(hpp_index * h_per_hpp,
                                        (hpp_index + 1) * h_per_hpp)
                x_slice = x[:, :, h_filter, :]
                x_slice = hpp(x_slice)
                x_slice = x_slice.view(n, -1)
                feature.append(x_slice)
        x = torch.stack(feature)

        # p, n, c
        x = x @ self.fc_mat
        # p, n, d

        return x
