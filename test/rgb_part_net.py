import torch

from models.rgb_part_net import RGBPartNet

P, K = 2, 4
N, T, C, H, W = P * K, 10, 3, 64, 32


def rand_x1_x2_y(n, t, c, h, w):
    x1 = torch.rand(n, t, c, h, w)
    x2 = torch.rand(n, t, c, h, w)
    y = []
    for p in range(P):
        y += [p] * K
    y = torch.as_tensor(y)
    return x1, x2, y


def test_default_rgb_part_net():
    rgb_pa = RGBPartNet()
    x1, x2, y = rand_x1_x2_y(N, T, C, H, W)

    rgb_pa.train()
    loss, metrics = rgb_pa(x1, x2, y)
    _, _, _, _ = metrics
    assert tuple(loss.size()) == ()
    assert isinstance(_, float)

    rgb_pa.eval()
    x = rgb_pa(x1, x2)
    assert tuple(x.size()) == (23, N, 256)


def test_custom_rgb_part_net():
    hpm_scales = (1, 2, 4, 8)
    tfa_num_parts = 8
    embedding_dims = 1024
    rgb_pa = RGBPartNet(num_class=10,
                        ae_in_channels=1,
                        ae_feature_channels=32,
                        f_a_c_p_dims=(64, 64, 32),
                        hpm_scales=hpm_scales,
                        hpm_use_avg_pool=True,
                        hpm_use_max_pool=False,
                        fpfe_feature_channels=64,
                        fpfe_kernel_sizes=((5, 3), (3, 3), (3, 3), (3, 3)),
                        fpfe_paddings=((2, 1), (1, 1), (1, 1), (1, 1)),
                        fpfe_halving=(1, 1, 3, 3),
                        tfa_squeeze_ratio=8,
                        tfa_num_parts=tfa_num_parts,
                        embedding_dims=1024,
                        triplet_margin=0.4)
    x1, x2, y = rand_x1_x2_y(N, T, 1, H, W)

    rgb_pa.train()
    loss, metrics = rgb_pa(x1, x2, y)
    _, _, _, _ = metrics
    assert tuple(loss.size()) == ()
    assert isinstance(_, float)

    rgb_pa.eval()
    x = rgb_pa(x1, x2)
    assert tuple(x.size()) == (
        sum(hpm_scales) + tfa_num_parts, N, embedding_dims
    )
