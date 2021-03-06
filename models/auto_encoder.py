import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import VGGConv2d, DCGANConvTranspose2d


class Encoder(nn.Module):
    """Squeeze input feature map to lower dimension"""

    def __init__(
            self,
            in_channels: int = 3,
            frame_size: tuple[int, int] = (64, 48),
            feature_channels: int = 32,
            output_dims: tuple[int, int, int] = (48, 48, 32)
    ):
        super().__init__()
        h_0, w_0 = frame_size
        h_1, w_1 = h_0 // 2, w_0 // 2
        h_2, w_2 = h_1 // 4, w_1 // 4
        # Appearance features, canonical features, pose features
        (self.f_a_dim, self.f_c_dim, self.f_p_dim) = output_dims

        # Conv1      in_channels x H x W
        #    -> feature_map_size x H x W
        self.conv1 = VGGConv2d(in_channels,
                               feature_channels,
                               kernel_size=5,
                               padding=2)
        # Conv2 feature_map_size x H x W
        #    -> feature_map_size x H x W
        self.conv2 = VGGConv2d(feature_channels, feature_channels)
        # MaxPool1 feature_map_size x    H x    W
        #       -> feature_map_size x H//2 x W//2
        self.max_pool1 = nn.AdaptiveMaxPool2d((h_1, w_1))
        # Conv3 feature_map_size   x H//2 x W//2
        #   ->  feature_map_size*2 x H//2 x W//2
        self.conv3 = VGGConv2d(feature_channels, feature_channels * 2)
        # Conv4 feature_map_size*2 x H//2 x W//2
        #   ->  feature_map_size*2 x H//2 x W//2
        self.conv4 = VGGConv2d(feature_channels * 2, feature_channels * 2)
        # MaxPool2 feature_map_size*2 x H//2 x W//2
        #       -> feature_map_size*2 x H//8 x W//8
        self.max_pool2 = nn.AdaptiveMaxPool2d((h_2, w_2))
        # Conv5 feature_map_size*2 x H//8 x W//8
        #    -> feature_map_size*4 x H//8 x W//8
        self.conv5 = VGGConv2d(feature_channels * 2, feature_channels * 4)
        # Conv6 feature_map_size*4 x H//8 x W//8
        #    -> feature_map_size*4 x H//8 x W//8
        self.conv6 = VGGConv2d(feature_channels * 4, feature_channels * 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        f_appearance, f_canonical, f_pose = x.split(
            (self.f_a_dim, self.f_c_dim, self.f_p_dim), dim=1
        )
        return f_appearance, f_canonical, f_pose


class Decoder(nn.Module):
    """Upscale embedding to original image"""

    def __init__(
            self,
            feature_channels: int = 32,
            out_channels: int = 3,
    ):
        super().__init__()
        self.feature_channels = feature_channels

        # TransConv1 feature_map_size*4 x H x W
        #         -> feature_map_size*4 x H x W
        self.trans_conv1 = DCGANConvTranspose2d(feature_channels * 4,
                                                feature_channels * 4,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1)
        # TransConv2 feature_map_size*4 x   H x   W
        #         -> feature_map_size*2 x H*2 x W*2
        self.trans_conv2 = DCGANConvTranspose2d(feature_channels * 4,
                                                feature_channels * 2)
        # TransConv3 feature_map_size*2 x H*2 x W*2
        #         -> feature_map_size*2 x H*2 x W*2
        self.trans_conv3 = DCGANConvTranspose2d(feature_channels * 2,
                                                feature_channels * 2,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1)
        # TransConv4 feature_map_size*2 x H*2 x W*2
        #         -> feature_map_size   x H*4 x W*4
        self.trans_conv4 = DCGANConvTranspose2d(feature_channels * 2,
                                                feature_channels)

        # TransConv5 feature_map_size x H*4 x W*4
        #         -> feature_map_size x H*4 x W*4
        self.trans_conv5 = DCGANConvTranspose2d(feature_channels,
                                                feature_channels,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1)

        # TransConv6 feature_map_size x H*4 x W*4
        #         ->     out_channels x H*8 x W*8
        self.trans_conv6 = DCGANConvTranspose2d(feature_channels,
                                                out_channels)

    def forward(self, f_appearance, f_canonical, f_pose):
        x = torch.cat((f_appearance, f_canonical, f_pose), dim=1)
        x = self.trans_conv1(x)
        x = self.trans_conv2(x)
        x = self.trans_conv3(x)
        x = self.trans_conv4(x)
        x = self.trans_conv5(x)
        x = torch.sigmoid(self.trans_conv6(x))

        return x


class AutoEncoder(nn.Module):
    def __init__(
            self,
            channels: int = 3,
            frame_size: tuple[int, int] = (64, 48),
            feature_channels: int = 32,
            embedding_dims: tuple[int, int, int] = (48, 48, 32)
    ):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.encoder = Encoder(channels, frame_size,
                               feature_channels, embedding_dims)
        self.decoder = Decoder(feature_channels, channels)

    def forward(self, x_c1_t2, x_c1_t1=None, x_c2_t2=None):
        n, t, c, h, w = x_c1_t2.size()
        # x_c1_t2 is the frame for later module
        x_c1_t2_ = x_c1_t2.view(n * t, c, h, w)
        (f_a_c1_t2_, f_c_c1_t2_, f_p_c1_t2_) = self.encoder(x_c1_t2_)

        f_size = [torch.Size([n, t, embedding_dim, h // 8, w // 8])
                  for embedding_dim in self.embedding_dims]
        f_a_c1_t2 = f_a_c1_t2_.view(f_size[0])
        f_c_c1_t2 = f_c_c1_t2_.view(f_size[1])
        f_p_c1_t2 = f_p_c1_t2_.view(f_size[2])

        if self.training:
            # t1 is random time step, c2 is another condition
            x_c1_t1_ = x_c1_t1.view(n * t, c, h, w)
            (f_a_c1_t1_, f_c_c1_t1_, _) = self.encoder(x_c1_t1_)
            x_c2_t2_ = x_c2_t2.view(n * t, c, h, w)
            (_, f_c_c2_t2_, f_p_c2_t2_) = self.encoder(x_c2_t2_)

            x_c1_t2_pred_ = self.decoder(f_a_c1_t1_, f_c_c1_t1_, f_p_c1_t2_)
            x_c1_t2_pred = x_c1_t2_pred_.view(n, t, c, h, w)

            f_c_c1_t1 = f_c_c1_t1_.view(f_size[1])
            f_c_c2_t2 = f_c_c2_t2_.view(f_size[1])
            f_p_c2_t2 = f_p_c2_t2_.view(f_size[2])

            return (
                (f_a_c1_t2, f_c_c1_t2, f_p_c1_t2),
                (x_c1_t2_pred, (f_c_c1_t1, f_c_c2_t2), f_p_c2_t2)
            )
        else:  # evaluating
            return f_a_c1_t2, f_c_c1_t2, f_p_c1_t2
