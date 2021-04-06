from typing import Tuple

import torch
import torch.nn as nn

from models.auto_encoder import AutoEncoder
from models.hpm import HorizontalPyramidMatching
from models.part_net import PartNet


class RGBPartNet(nn.Module):
    def __init__(
            self,
            ae_in_channels: int = 3,
            ae_in_size: Tuple[int, int] = (64, 48),
            ae_feature_channels: int = 64,
            f_a_c_p_dims: Tuple[int, int, int] = (192, 192, 128),
            hpm_scales: Tuple[int, ...] = (1, 2, 4),
            hpm_use_avg_pool: bool = True,
            hpm_use_max_pool: bool = True,
            tfa_squeeze_ratio: int = 4,
            tfa_num_parts: int = 16,
            embedding_dims: Tuple[int] = (256, 256),
            image_log_on: bool = False
    ):
        super().__init__()
        self.h, self.w = ae_in_size
        self.image_log_on = image_log_on

        self.ae = AutoEncoder(
            ae_in_channels, ae_in_size, ae_feature_channels, f_a_c_p_dims
        )
        self.hpm = HorizontalPyramidMatching(
            f_a_c_p_dims[1], embedding_dims[0], hpm_scales,
            hpm_use_avg_pool, hpm_use_max_pool
        )
        self.pn = PartNet(
            f_a_c_p_dims[2], embedding_dims[1], tfa_num_parts, tfa_squeeze_ratio
        )

        self.num_parts = self.hpm.num_parts + tfa_num_parts

    def forward(self, x_c1, x_c2=None):
        # Step 1: Disentanglement
        # n, t, c, h, w
        (f_a, f_c, f_p), ae_losses = self._disentangle(x_c1, x_c2)

        # Step 2.a: Static Gait Feature Aggregation & HPM
        # n, c, h, w
        f_c_mean = f_c.mean(1)
        x_c = self.hpm(f_c_mean)
        # p, n, d

        # Step 2.b: FPFE & TFA (Dynamic Gait Feature Aggregation)
        # n, t, c, h, w
        x_p = self.pn(f_p)
        # p, n, d

        if self.training:
            i_a, i_c, i_p = None, None, None
            if self.image_log_on:
                with torch.no_grad():
                    f_a_mean = f_a.mean(1)
                    i_a = self.ae.decoder(
                        f_a_mean,
                        torch.zeros_like(f_c_mean),
                        torch.zeros_like(f_p[:, 0])
                    )
                    i_c = self.ae.decoder(
                        torch.zeros_like(f_a_mean),
                        f_c_mean,
                        torch.zeros_like(f_p[:, 0])
                    )
                    f_p_size = f_p.size()
                    i_p = self.ae.decoder(
                        torch.zeros(f_p_size[0] * f_p_size[1], *f_a.shape[2:],
                                    device=f_a.device),
                        torch.zeros(f_p_size[0] * f_p_size[1], *f_c.shape[2:],
                                    device=f_c.device),
                        f_p.view(-1, *f_p_size[2:])
                    ).view(x_c1.size())
            return x_c, x_p, ae_losses, (i_a, i_c, i_p)
        else:
            return x_c, x_p

    def _disentangle(self, x_c1_t2, x_c2_t2=None):
        if self.training:
            x_c1_t1 = x_c1_t2[:, torch.randperm(x_c1_t2.size(1)), :, :, :]
            features, losses = self.ae(x_c1_t2, x_c1_t1, x_c2_t2)
            return features, losses
        else:  # evaluating
            features = self.ae(x_c1_t2)
            return features, None
