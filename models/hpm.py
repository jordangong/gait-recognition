import torch
import torch.nn as nn
from torchvision.models import resnet50

from models.layers import HorizontalPyramidPooling


class HorizontalPyramidMatching(nn.Module):
    def __init__(
            self,
            scales: tuple[int, ...] = (1, 2, 4, 8),
            out_channels: int = 256,
            use_avg_pool: bool = True,
            use_max_pool: bool = True,
            **kwargs
    ):
        super().__init__()
        self.scales = scales
        self.out_channels = out_channels
        self.use_avg_pool = use_avg_pool
        self.use_max_pool = use_max_pool

        self.backbone = resnet50(pretrained=True)
        self.in_channels = self.backbone.layer4[-1].conv1.in_channels

        self.pyramids = [
            self._make_pyramid(scale, **kwargs) for scale in self.scales
        ]

    def _make_pyramid(self, scale: int, **kwargs):
        pyramid = [HorizontalPyramidPooling(self.in_channels,
                                            self.out_channels,
                                            use_avg_pool=self.use_avg_pool,
                                            use_max_pool=self.use_max_pool,
                                            **kwargs)
                   for _ in range(scale)]
        return pyramid

    def forward(self, x):
        x = self.backbone(x)
        n, c, h, w = x.size()

        feature = []
        for pyramid_index, pyramid in enumerate(self.pyramids):
            h_per_hpp = h // self.scales[pyramid_index]
            for hpp_index, hpp in enumerate(pyramid):
                h_filter = torch.arange(hpp_index * h_per_hpp,
                                        (hpp_index + 1) * h_per_hpp)
                x_slice = x[:, :, h_filter, :]
                x_slice = hpp(x_slice)
                x_slice = x_slice.view(n, -1)
                feature.append(x_slice)

        feature = torch.cat(feature, dim=1)
        return feature
