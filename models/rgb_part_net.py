import random
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # Step 0: Swap batch_size and time dimensions for next step
        # n, t, c, h, w
        x_c1 = x_c1.transpose(0, 1)
        if self.training:
            x_c2 = x_c2.transpose(0, 1)

        # Step 1: Disentanglement
        # t, n, c, h, w
        ((x_c_c1, x_p_c1), images, losses) = self._disentangle(x_c1, x_c2)

        # Step 2.a: Static Gait Feature Aggregation & HPM
        # n, c, h, w
        x_c = self.hpm(x_c_c1)
        # p, n, c

        # Step 2.b: FPFE & TFA (Dynamic Gait Feature Aggregation)
        # t, n, c, h, w
        x_p = self.pn(x_p_c1)
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

    def _disentangle(self, x_c1, x_c2=None):
        t, n, c, h, w = x_c1.size()
        device = x_c1.device
        if self.training:
            # Encoded appearance, canonical and pose features
            f_a_c1, f_c_c1, f_p_c1 = [], [], []
            # Features required to calculate losses
            f_p_c2 = []
            xrecon_loss, cano_cons_loss = [], []
            for t2 in range(t):
                t1 = random.randrange(t)
                output = self.ae(x_c1[t2], x_c1[t1], x_c2[t2])
                (f_c1_t2, f_p_t2, losses) = output

                (f_a_c1_t2, f_c_c1_t2, f_p_c1_t2) = f_c1_t2
                if self.image_log_on:
                    f_a_c1.append(f_a_c1_t2)
                # Save canonical features and pose features
                f_c_c1.append(f_c_c1_t2)
                f_p_c1.append(f_p_c1_t2)

                # Losses per time step
                # Used in pose similarity loss
                (_, f_p_c2_t2) = f_p_t2
                f_p_c2.append(f_p_c2_t2)

                # Cross reconstruction loss and canonical loss
                (xrecon_loss_t2, cano_cons_loss_t2) = losses
                xrecon_loss.append(xrecon_loss_t2)
                cano_cons_loss.append(cano_cons_loss_t2)
            if self.image_log_on:
                f_a_c1 = torch.stack(f_a_c1)
            f_c_c1_mean = torch.stack(f_c_c1).mean(0)
            f_p_c1 = torch.stack(f_p_c1)
            f_p_c2 = torch.stack(f_p_c2)

            # Decode features
            appearance_image, canonical_image, pose_image = None, None, None
            with torch.no_grad():
                # Decode average canonical features to higher dimension
                x_c_c1 = self.ae.decoder(
                    torch.zeros((n, self.f_a_dim), device=device),
                    f_c_c1_mean,
                    torch.zeros((n, self.f_p_dim), device=device),
                    cano_only=True
                )
                # Decode pose features to images
                f_p_c1_ = f_p_c1.view(t * n, -1)
                x_p_c1_ = self.ae.decoder(
                    torch.zeros((t * n, self.f_a_dim), device=device),
                    torch.zeros((t * n, self.f_c_dim), device=device),
                    f_p_c1_
                )
                x_p_c1 = x_p_c1_.view(t, n, c, h, w)

                if self.image_log_on:
                    # Decode appearance features
                    f_a_c1_ = f_a_c1.view(t * n, -1)
                    appearance_image_ = self.ae.decoder(
                        f_a_c1_,
                        torch.zeros((t * n, self.f_c_dim), device=device),
                        torch.zeros((t * n, self.f_p_dim), device=device)
                    )
                    appearance_image = appearance_image_.view(t, n, c, h, w)
                    # Continue decoding canonical features
                    canonical_image = self.ae.decoder.trans_conv3(x_c_c1)
                    canonical_image = torch.sigmoid(
                        self.ae.decoder.trans_conv4(canonical_image)
                    )
                    pose_image = x_p_c1

            # Losses
            xrecon_loss = torch.sum(torch.stack(xrecon_loss))
            pose_sim_loss = self._pose_sim_loss(f_p_c1, f_p_c2) * 10
            cano_cons_loss = torch.mean(torch.stack(cano_cons_loss))

            return ((x_c_c1, x_p_c1),
                    (appearance_image, canonical_image, pose_image),
                    (xrecon_loss, pose_sim_loss, cano_cons_loss))

        else:  # evaluating
            x_c1_ = x_c1.view(t * n, c, h, w)
            (f_c_c1_, f_p_c1_) = self.ae(x_c1_)

            # Canonical features
            f_c_c1 = f_c_c1_.view(t, n, -1)
            f_c_c1_mean = f_c_c1.mean(0)
            x_c_c1 = self.ae.decoder(
                torch.zeros((n, self.f_a_dim)),
                f_c_c1_mean,
                torch.zeros((n, self.f_p_dim)),
                cano_only=True
            )

            # Pose features
            x_p_c1_ = self.ae.decoder(
                torch.zeros((t * n, self.f_a_dim)),
                torch.zeros((t * n, self.f_c_dim)),
                f_p_c1_
            )
            x_p_c1 = x_p_c1_.view(t, n, c, h, w)

            return (x_c_c1, x_p_c1), None, None

    @staticmethod
    def _pose_sim_loss(f_p_c1: torch.Tensor,
                       f_p_c2: torch.Tensor) -> torch.Tensor:
        f_p_c1_mean = f_p_c1.mean(dim=0)
        f_p_c2_mean = f_p_c2.mean(dim=0)
        return F.mse_loss(f_p_c1_mean, f_p_c2_mean)
