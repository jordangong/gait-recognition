import torch
from torch import nn as nn
from torch.nn import functional as F

from models.layers import VGGConv2d, DCGANConvTranspose2d


class Encoder(nn.Module):
    def __init__(self, in_channels: int, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        self.em_dim = opt.em_dim
        nf = 64

        # Cx[HxW]
        # Conv1 3x64x32 -> 64x64x32
        self.conv1 = VGGConv2d(in_channels, nf)
        # MaxPool1 64x64x32 -> 64x32x16
        self.max_pool1 = nn.AdaptiveMaxPool2d((32, 16))
        # Conv2 64x32x16 -> 256x32x16
        self.conv2 = VGGConv2d(nf, nf * 4)
        # MaxPool2 256x32x16 -> 256x16x8
        self.max_pool2 = nn.AdaptiveMaxPool2d((16, 8))
        # Conv3 256x16x8 -> 512x16x8
        self.conv3 = VGGConv2d(nf * 4, nf * 8)
        # Conv4 512x16x8 -> 512x16x8 (for large dataset)
        self.conv4 = VGGConv2d(nf * 8, nf * 8)
        # MaxPool3 512x16x8 -> 512x4x2
        self.max_pool3 = nn.AdaptiveMaxPool2d((4, 2))
        # FC 512*4*2 -> 320
        self.fc = nn.Linear(nf * 8 * 2 * 4, self.em_dim, bias=False)
        self.bn_fc = nn.BatchNorm1d(self.em_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool3(x)
        x = x.view(-1, (64 * 8) * 2 * 4)
        embedding = self.bn_fc(self.fc(x))

        fa, fgs, fgd = embedding.split(
            (self.opt.fa_dim, self.opt.fg_dim / 2, self.opt.fg_dim / 2), dim=1
        )
        return fa, fgs, fgd


class Decoder(nn.Module):
    def __init__(self, out_channels: int, opt):
        super(Decoder, self).__init__()
        self.em_dim = opt.em_dim
        nf = 64

        # Cx[HxW]
        # FC 320 -> 512*4*2
        self.fc = nn.Linear(self.em_dim, nf * 8 * 2 * 4)
        self.bn_fc = nn.BatchNorm1d(nf * 8 * 2 * 4)
        # TransConv1 512x4x2 -> 256x8x4
        self.trans_conv1 = DCGANConvTranspose2d(nf * 8, nf * 4)
        # TransConv2 256x8x4 -> 128x16x8
        self.trans_conv2 = DCGANConvTranspose2d(nf * 4, nf * 2)
        # TransConv3 128x16x8 -> 64x32x16
        self.trans_conv3 = DCGANConvTranspose2d(nf * 2, nf)
        # TransConv4 3x32x16
        self.trans_conv4 = DCGANConvTranspose2d(nf, out_channels,
                                                is_last_layer=True)

    def forward(self, fa, fgs, fgd):
        x = torch.cat((fa, fgs, fgd), dim=1).view(-1, self.em_dim)
        x = self.bn_fc(self.fc(x))
        x = F.relu(x.view(-1, 64 * 8, 4, 2), True)
        x = self.trans_conv1(x)
        x = self.trans_conv2(x)
        x = self.trans_conv3(x)
        x = F.sigmoid(self.trans_conv4(x))

        return x
