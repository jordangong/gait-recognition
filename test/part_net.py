import torch

from models.part_net import FrameLevelPartFeatureExtractor, \
    TemporalFeatureAggregator, PartNet

T, N, C, H, W = 15, 4, 3, 64, 32


def test_default_fpfe():
    fpfe = FrameLevelPartFeatureExtractor()
    x = torch.rand(T, N, C, H, W)
    x = fpfe(x)

    assert tuple(x.size()) == (T * N, 32 * 4, 16, 8)


def test_custom_fpfe():
    feature_channels = 64
    fpfe = FrameLevelPartFeatureExtractor(
        in_channels=1,
        feature_channels=feature_channels,
        kernel_sizes=((5, 3), (3, 3), (3, 3), (3, 3)),
        paddings=((2, 1), (1, 1), (1, 1), (1, 1)),
        halving=(1, 1, 3, 3)
    )
    x = torch.rand(T, N, 1, H, W)
    x = fpfe(x)

    assert tuple(x.size()) == (T * N, feature_channels * 8, 8, 4)


def test_default_tfa():
    in_channels = 32 * 4
    tfa = TemporalFeatureAggregator(in_channels)
    x = torch.rand(16, T, N, in_channels)
    x = tfa(x)

    assert tuple(x.size()) == (16, N, in_channels)


def test_custom_tfa():
    in_channels = 64 * 8
    num_part = 8
    tfa = TemporalFeatureAggregator(in_channels=in_channels,
                                    squeeze_ratio=8, num_part=num_part)
    x = torch.rand(num_part, T, N, in_channels)
    x = tfa(x)

    assert tuple(x.size()) == (num_part, N, in_channels)


def test_default_part_net():
    pa = PartNet()
    x = torch.rand(T, N, C, H, W)
    x = pa(x)

    assert tuple(x.size()) == (16, N, 32 * 4)


def test_custom_part_net():
    feature_channels = 64
    pa = PartNet(in_channels=1, feature_channels=feature_channels,
                 kernel_sizes=((5, 3), (3, 3), (3, 3), (3, 3)),
                 paddings=((2, 1), (1, 1), (1, 1), (1, 1)),
                 halving=(1, 1, 3, 3),
                 squeeze_ratio=8,
                 num_part=8)
    x = torch.rand(T, N, 1, H, W)
    x = pa(x)

    assert tuple(x.size()) == (8, N, pa.tfa_in_channels)
