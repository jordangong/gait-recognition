import torch.nn as nn

from models.layers import FocalConv2d


class FrameLevelPartFeatureExtractor(nn.Module):

    def __init__(self, in_channels: int):
        super(FrameLevelPartFeatureExtractor, self).__init__()
        nf = 32

        self.focal_conv1 = FocalConv2d(in_channels, nf, kernel_size=5,
                                       padding=2, halving=1)
        self.focal_conv2 = FocalConv2d(nf, nf, kernel_size=3,
                                       padding=1, halving=1)
        self.focal_conv3 = FocalConv2d(nf, nf * 2, kernel_size=3,
                                       padding=1, halving=4)
        self.focal_conv4 = FocalConv2d(nf * 2, nf * 2, kernel_size=3,
                                       padding=1, halving=4)
        self.focal_conv5 = FocalConv2d(nf * 2, nf * 4, kernel_size=3,
                                       padding=1, halving=8)
        self.focal_conv6 = FocalConv2d(nf * 4, nf * 4, kernel_size=3,
                                       padding=1, halving=8)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.focal_conv1(x)
        x = self.focal_conv2(x)
        x = self.max_pool(x)

        x = self.focal_conv3(x)
        x = self.focal_conv4(x)
        x = self.max_pool(x)

        x = self.focal_conv5(x)
        x = self.focal_conv6(x)

        return x
