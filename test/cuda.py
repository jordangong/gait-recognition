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


def test_default_rgb_part_net_cuda():
    rgb_pa = RGBPartNet()
    rgb_pa = rgb_pa.cuda()
    x1, x2, y = rand_x1_x2_y(N, T, C, H, W)
    x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()

    rgb_pa.train()
    loss, metrics = rgb_pa(x1, x2, y)
    _, _, _, _ = metrics
    assert loss.device == torch.device('cuda', torch.cuda.current_device())
    assert tuple(loss.size()) == ()
    assert isinstance(_, float)

    rgb_pa.eval()
    x = rgb_pa(x1, x2)
    assert x.device == torch.device('cuda', torch.cuda.current_device())
    assert tuple(x.size()) == (23, N, 256)
