from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import VGGConv2d, DCGANConvTranspose2d, BasicLinear


class Encoder(nn.Module):
    """Squeeze input feature map to lower dimension"""

    def __init__(
            self,
            in_channels: int = 3,
            feature_channels: int = 64,
            output_dims: Tuple[int, int, int] = (128, 128, 64)
    ):
        super().__init__()
        self.feature_channels = feature_channels
        # Appearance features, canonical features, pose features
        (self.f_a_dim, self.f_c_dim, self.f_p_dim) = output_dims

        # Conv1      in_channels x 64 x 32
        #    -> feature_map_size x 64 x 32
        self.conv1 = VGGConv2d(in_channels, feature_channels)
        # MaxPool1 feature_map_size x 64 x 32
        #       -> feature_map_size x 32 x 16
        self.max_pool1 = nn.AdaptiveMaxPool2d((32, 16))
        # Conv2 feature_map_size    x 32 x 16
        #   -> (feature_map_size*4) x 32 x 16
        self.conv2 = VGGConv2d(feature_channels, feature_channels * 4)
        # MaxPool2 (feature_map_size*4) x 32 x 16
        #       -> (feature_map_size*4) x 16 x 8
        self.max_pool2 = nn.AdaptiveMaxPool2d((16, 8))
        # Conv3 (feature_map_size*4) x 16 x 8
        #    -> (feature_map_size*8) x 16 x 8
        self.conv3 = VGGConv2d(feature_channels * 4, feature_channels * 8)
        # Conv4 (feature_map_size*8) x 16 x 8
        #    -> (feature_map_size*8) x 16 x 8 (for large dataset)
        self.conv4 = VGGConv2d(feature_channels * 8, feature_channels * 8)
        # MaxPool3 (feature_map_size*8) x 16 x 8
        # ->       (feature_map_size*8) x  4 x 2
        self.max_pool3 = nn.AdaptiveMaxPool2d((4, 2))

        embedding_dim = sum(output_dims)
        # FC (feature_map_size*8) * 4 * 2 -> 320
        self.fc = BasicLinear(feature_channels * 8 * 2 * 4, embedding_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool3(x)
        x = x.view(-1, (self.feature_channels * 8) * 2 * 4)
        embedding = self.fc(x)

        f_appearance, f_canonical, f_pose = embedding.split(
            (self.f_a_dim, self.f_c_dim, self.f_p_dim), dim=1
        )
        return f_appearance, f_canonical, f_pose


class Decoder(nn.Module):
    """Upscale embedding to original image"""

    def __init__(
            self,
            input_dims: Tuple[int, int, int] = (128, 128, 64),
            feature_channels: int = 64,
            out_channels: int = 3,
    ):
        super().__init__()
        self.feature_channels = feature_channels

        embedding_dim = sum(input_dims)
        # FC 320 -> (feature_map_size*8) * 4 * 2
        self.fc = BasicLinear(embedding_dim, feature_channels * 8 * 2 * 4)

        # TransConv1 (feature_map_size*8) x 4 x 2
        #         -> (feature_map_size*4) x 8 x 4
        self.trans_conv1 = DCGANConvTranspose2d(feature_channels * 8,
                                                feature_channels * 4)
        # TransConv2 (feature_map_size*4) x  8 x 4
        #         -> (feature_map_size*2) x 16 x 8
        self.trans_conv2 = DCGANConvTranspose2d(feature_channels * 4,
                                                feature_channels * 2)
        # TransConv3 (feature_map_size*2) x 16 x  8
        #         ->  feature_map_size    x 32 x 16
        self.trans_conv3 = DCGANConvTranspose2d(feature_channels * 2,
                                                feature_channels)
        # TransConv4 feature_map_size x 32 x 16
        #         ->      in_channels x 64 x 32
        self.trans_conv4 = DCGANConvTranspose2d(feature_channels, out_channels,
                                                is_last_layer=True)

    def forward(self, f_appearance, f_canonical, f_pose, cano_only=False):
        x = torch.cat((f_appearance, f_canonical, f_pose), dim=1)
        x = self.fc(x)
        x = F.relu(x.view(-1, self.feature_channels * 8, 4, 2), inplace=True)
        x = self.trans_conv1(x)
        x = self.trans_conv2(x)
        if cano_only:
            return x
        x = self.trans_conv3(x)
        x = torch.sigmoid(self.trans_conv4(x))

        return x


class AutoEncoder(nn.Module):
    def __init__(
            self,
            channels: int = 3,
            feature_channels: int = 64,
            embedding_dims: Tuple[int, int, int] = (128, 128, 64)
    ):
        super().__init__()
        self.f_c_c1_t2_ = None
        self.f_p_c1_t2_ = None
        self.f_c_c1_t1_ = None
        self.encoder = Encoder(channels, feature_channels, embedding_dims)
        self.decoder = Decoder(embedding_dims, feature_channels, channels)

    def forward(self, x_t2, is_c1=True):
        n, t, c, h, w = x_t2.size()
        if is_c1:  # condition 1
            # x_c1_t2 is the frame for later module
            x_c1_t2_ = x_t2.view(n * t, c, h, w)
            (f_a_c1_t2_, self.f_c_c1_t2_, self.f_p_c1_t2_) \
                = self.encoder(x_c1_t2_)

            if self.training:
                # t1 is random time step
                x_c1_t1 = x_t2[:, torch.randperm(t), :, :, :]
                x_c1_t1_ = x_c1_t1.view(n * t, c, h, w)
                (f_a_c1_t1_, self.f_c_c1_t1_, _) = self.encoder(x_c1_t1_)

                x_c1_t2_pred_ = self.decoder(
                    f_a_c1_t1_, self.f_c_c1_t1_, self.f_p_c1_t2_
                )
                x_c1_t2_pred = x_c1_t2_pred_.view(n, t, c, h, w)

                xrecon_loss = torch.stack([
                    F.mse_loss(x_t2[:, i, :, :, :], x_c1_t2_pred[:, i, :, :, :])
                    for i in range(t)
                ]).sum()

                return ((f_a_c1_t2_, self.f_c_c1_t2_, self.f_p_c1_t2_),
                        xrecon_loss)
            else:  # evaluating
                return self.f_c_c1_t2_, self.f_p_c1_t2_
        else:  # condition 2
            # c2 is another condition
            x_c2_t2_ = x_t2.view(n * t, c, h, w)
            (_, f_c_c2_t2_, f_p_c2_t2_) = self.encoder(x_c2_t2_)

            f_c_c1_t1 = self.f_c_c1_t1_.view(n, t, -1)
            f_c_c1_t2 = self.f_c_c1_t2_.view(n, t, -1)
            f_c_c2_t2 = f_c_c2_t2_.view(n, t, -1)
            cano_cons_loss = torch.stack([
                F.mse_loss(f_c_c1_t1[:, i, :], f_c_c1_t2[:, i, :])
                + F.mse_loss(f_c_c1_t2[:, i, :], f_c_c2_t2[:, i, :])
                for i in range(t)
            ]).mean()

            f_p_c1_t2 = self.f_p_c1_t2_.view(n, t, -1)
            f_p_c2_t2 = f_p_c2_t2_.view(n, t, -1)
            pose_sim_loss = F.mse_loss(f_p_c1_t2.mean(1), f_p_c2_t2.mean(1))

            return cano_cons_loss, pose_sim_loss * 10
