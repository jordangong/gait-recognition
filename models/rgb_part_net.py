import random

import torch
import torch.nn as nn

from models import AutoEncoder, HorizontalPyramidMatching, PartNet


class RGBPartNet(nn.Module):
    def __init__(
            self,
            num_class: int = 74,
            ae_in_channels: int = 3,
            ae_feature_channels: int = 64,
            f_a_c_p_dims: tuple[int, int, int] = (128, 128, 64),
            hpm_scales: tuple[int, ...] = (1, 2, 4),
            hpm_use_avg_pool: bool = True,
            hpm_use_max_pool: bool = True,
            fpfe_feature_channels: int = 32,
            fpfe_kernel_sizes: tuple[tuple, ...] = ((5, 3), (3, 3), (3, 3)),
            fpfe_paddings: tuple[tuple, ...] = ((2, 1), (1, 1), (1, 1)),
            fpfe_halving: tuple[int, ...] = (0, 2, 3),
            tfa_squeeze_ratio: int = 4,
            tfa_num_part: int = 16,
    ):
        super().__init__()
        self.ae = AutoEncoder(
            num_class, ae_in_channels, ae_feature_channels, f_a_c_p_dims
        )
        self.pn = PartNet(
            ae_in_channels, fpfe_feature_channels, fpfe_kernel_sizes,
            fpfe_paddings, fpfe_halving, tfa_squeeze_ratio, tfa_num_part
        )
        self.hpm = HorizontalPyramidMatching(
            ae_feature_channels * 8, self.pn.tfa_in_channels, hpm_scales,
            hpm_use_avg_pool, hpm_use_max_pool
        )

        self.mse_loss = nn.MSELoss()

        # TODO Weight inti here

    def pose_sim_loss(self, f_p_c1: torch.Tensor,
                      f_p_c2: torch.Tensor) -> torch.Tensor:
        f_p_c1_mean = f_p_c1.mean(dim=0)
        f_p_c2_mean = f_p_c2.mean(dim=0)
        return self.mse_loss(f_p_c1_mean, f_p_c2_mean)

    def forward(self, x_c1, x_c2, y):
        # Step 0: Swap batch_size and time dimensions for next step
        # n, t, c, h, w
        x_c1, x_c2 = x_c1.transpose(0, 1), x_c2.transpose(0, 1)

        # Step 1: Disentanglement
        # t, n, c, h, w
        num_frames = len(x_c1)
        # Decoded canonical features and Pose images
        x_c_c1, x_p_c1 = [], []
        # Features required to calculate losses
        f_p_c1, f_p_c2 = [], []
        xrecon_loss, cano_cons_loss = torch.zeros(1), torch.zeros(1)
        for t2 in range(num_frames):
            t1 = random.randrange(num_frames)
            output = self.ae(x_c1[t1], x_c1[t2], x_c2[t2], y)
            (x_c1_t2, f_p_t2, losses) = output

            # Decoded features or image
            (x_c_c1_t2, x_p_c1_t2) = x_c1_t2
            # Canonical Features for HPM
            x_c_c1.append(x_c_c1_t2)
            # Pose image for Part Net
            x_p_c1.append(x_p_c1_t2)

            # Losses per time step
            # Used in pose similarity loss
            (f_p_c1_t2, f_p_c2_t2) = f_p_t2
            f_p_c1.append(f_p_c1_t2)
            f_p_c2.append(f_p_c2_t2)
            # Cross reconstruction loss and canonical loss
            (xrecon_loss_t2, cano_cons_loss_t2) = losses
            xrecon_loss += xrecon_loss_t2
            cano_cons_loss += cano_cons_loss_t2

        x_c_c1 = torch.stack(x_c_c1)
        x_p_c1 = torch.stack(x_p_c1)

        # Step 2.a: HPM & Static Gait Feature Aggregation
        # t, n, c, h, w
        x_c = self.hpm(x_c_c1)
        # p, t, n, c
        x_c = x_c.mean(dim=1)
        # p, n, c

        # Step 2.b: FPFE & TFA (Dynamic Gait Feature Aggregation)
        # t, n, c, h, w
        x_p = self.pn(x_p_c1)
        # p, n, c

        # Step 3: Cat feature map together and calculate losses
        x = torch.cat([x_c, x_p])
        # Losses
        f_p_c1 = torch.stack(f_p_c1)
        f_p_c2 = torch.stack(f_p_c2)
        pose_sim_loss = self.pose_sim_loss(f_p_c1, f_p_c2)
        cano_cons_loss /= num_frames
        # TODO Implement Batch All triplet loss function
        batch_all_triplet_loss = 0
        loss = (xrecon_loss + pose_sim_loss + cano_cons_loss
                + batch_all_triplet_loss)

        return x, loss
