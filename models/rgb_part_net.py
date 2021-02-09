from typing import Tuple

import torch
import torch.nn as nn

from models.auto_encoder import AutoEncoder
from models.hpm import HorizontalPyramidMatching
from models.part_net import PartNet
from utils.triplet_loss import BatchAllTripletLoss


class RGBPartNet(nn.Module):
    def __init__(
            self,
            ae_in_channels: int = 3,
            ae_feature_channels: int = 64,
            f_a_c_p_dims: Tuple[int, int, int] = (128, 128, 64),
            hpm_use_1x1conv: bool = False,
            hpm_scales: Tuple[int, ...] = (1, 2, 4),
            hpm_use_avg_pool: bool = True,
            hpm_use_max_pool: bool = True,
            fpfe_feature_channels: int = 32,
            fpfe_kernel_sizes: Tuple[Tuple, ...] = ((5, 3), (3, 3), (3, 3)),
            fpfe_paddings: Tuple[Tuple, ...] = ((2, 1), (1, 1), (1, 1)),
            fpfe_halving: Tuple[int, ...] = (0, 2, 3),
            tfa_squeeze_ratio: int = 4,
            tfa_num_parts: int = 16,
            embedding_dims: int = 256,
            triplet_margins: Tuple[float, float] = (0.2, 0.2),
            image_log_on: bool = False
    ):
        super().__init__()
        (self.f_a_dim, self.f_c_dim, self.f_p_dim) = f_a_c_p_dims
        self.hpm_num_parts = sum(hpm_scales)
        self.image_log_on = image_log_on

        self.ae = AutoEncoder(
            ae_in_channels, ae_feature_channels, f_a_c_p_dims
        )
        self.pn = PartNet(
            ae_in_channels, fpfe_feature_channels, fpfe_kernel_sizes,
            fpfe_paddings, fpfe_halving, tfa_squeeze_ratio, tfa_num_parts
        )
        out_channels = self.pn.tfa_in_channels
        self.hpm = HorizontalPyramidMatching(
            ae_feature_channels * 2, out_channels, hpm_use_1x1conv,
            hpm_scales, hpm_use_avg_pool, hpm_use_max_pool
        )
        empty_fc = torch.empty(self.hpm_num_parts + tfa_num_parts,
                               out_channels, embedding_dims)
        self.fc_mat = nn.Parameter(empty_fc)

        (hpm_margin, pn_margin) = triplet_margins
        self.hpm_ba_trip = BatchAllTripletLoss(hpm_margin)
        self.pn_ba_trip = BatchAllTripletLoss(pn_margin)

    def fc(self, x):
        return x @ self.fc_mat

    def forward(self, x_c1, x_c2=None, y=None):
        # Step 1: Disentanglement
        # n, t, c, h, w
        ((x_c, x_p), losses, images) = self._disentangle(x_c1, x_c2)

        # Step 2.a: Static Gait Feature Aggregation & HPM
        # n, c, h, w
        x_c = self.hpm(x_c)
        # p, n, c

        # Step 2.b: FPFE & TFA (Dynamic Gait Feature Aggregation)
        # n, t, c, h, w
        x_p = self.pn(x_p)
        # p, n, c

        # Step 3: Cat feature map together and fc
        x = torch.cat((x_c, x_p))
        x = self.fc(x)

        if self.training:
            hpm_ba_trip = self.hpm_ba_trip(x[:self.hpm_num_parts], y)
            pn_ba_trip = self.pn_ba_trip(x[self.hpm_num_parts:], y)
            losses = torch.stack((*losses, hpm_ba_trip, pn_ba_trip))
            return losses, images
        else:
            return x.unsqueeze(1).view(-1)

    def _disentangle(self, x_c1_t2, x_c2_t2=None):
        n, t, c, h, w = x_c1_t2.size()
        device = x_c1_t2.device
        x_c1_t1 = x_c1_t2[:, torch.randperm(t), :, :, :]
        if self.training:
            ((f_a_, f_c_, f_p_), losses) = self.ae(x_c1_t2, x_c1_t1, x_c2_t2)
            # Decode features
            with torch.no_grad():
                x_c = self._decode_cano_feature(f_c_, n, t, device)
                x_p = self._decode_pose_feature(f_p_, n, t, c, h, w, device)

                i_a, i_c, i_p = None, None, None
                if self.image_log_on:
                    i_a = self._decode_appr_feature(f_a_, n, t, c, h, w, device)
                    # Continue decoding canonical features
                    i_c = self.ae.decoder.trans_conv3(x_c)
                    i_c = torch.sigmoid(self.ae.decoder.trans_conv4(i_c))
                    i_p = x_p

            return (x_c, x_p), losses, (i_a, i_c, i_p)

        else:  # evaluating
            f_c_, f_p_ = self.ae(x_c1_t2)
            x_c = self._decode_cano_feature(f_c_, n, t, device)
            x_p = self._decode_pose_feature(f_p_, n, t, c, h, w, device)
            return (x_c, x_p), None, None

    def _decode_appr_feature(self, f_a_, n, t, c, h, w, device):
        # Decode appearance features
        x_a_ = self.ae.decoder(
            f_a_,
            torch.zeros((n * t, self.f_c_dim), device=device),
            torch.zeros((n * t, self.f_p_dim), device=device)
        )
        x_a = x_a_.view(n, t, c, h, w)
        return x_a

    def _decode_cano_feature(self, f_c_, n, t, device):
        # Decode average canonical features to higher dimension
        f_c = f_c_.view(n, t, -1)
        x_c = self.ae.decoder(
            torch.zeros((n, self.f_a_dim), device=device),
            f_c.mean(1),
            torch.zeros((n, self.f_p_dim), device=device),
            cano_only=True
        )
        return x_c

    def _decode_pose_feature(self, f_p_, n, t, c, h, w, device):
        # Decode pose features to images
        x_p_ = self.ae.decoder(
            torch.zeros((n * t, self.f_a_dim), device=device),
            torch.zeros((n * t, self.f_c_dim), device=device),
            f_p_
        )
        x_p = x_p_.view(n, t, c, h, w)
        return x_p
