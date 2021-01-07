import torch

from models.hpm import HorizontalPyramidMatching

T, N, C, H, W = 15, 4, 256, 32, 16


def test_default_hpm():
    hpm = HorizontalPyramidMatching(in_channels=C)
    x = torch.rand(T, N, C, H, W)
    x = hpm(x)
    assert tuple(x.size()) == (1 + 2 + 4, T, N, 128)


def test_custom_hpm():
    hpm = HorizontalPyramidMatching(in_channels=2048,
                                    out_channels=256,
                                    scales=(1, 2, 4, 8),
                                    use_avg_pool=True,
                                    use_max_pool=False)
    x = torch.rand(T, N, 2048, H, W)
    x = hpm(x)
    assert tuple(x.size()) == (1 + 2 + 4 + 8, T, N, 256)
