import torch

from models.auto_encoder import Encoder, Decoder, AutoEncoder

N, C, H, W = 128, 3, 64, 32


def test_default_encoder():
    encoder = Encoder()
    x = torch.rand(N, C, H, W)
    f_a, f_c, f_p = encoder(x)

    assert tuple(f_a.size()) == (N, 128)
    assert tuple(f_c.size()) == (N, 128)
    assert tuple(f_p.size()) == (N, 64)


def test_custom_encoder():
    output_dims = (64, 64, 32)
    encoder = Encoder(in_channels=1,
                      feature_channels=32,
                      output_dims=output_dims)
    x = torch.rand(N, 1, H, W)
    f_a, f_c, f_p = encoder(x)

    assert tuple(f_a.size()) == (N, output_dims[0])
    assert tuple(f_c.size()) == (N, output_dims[1])
    assert tuple(f_p.size()) == (N, output_dims[2])


def test_default_decoder():
    decoder = Decoder()
    f_a, f_c, f_p = torch.rand(N, 128), torch.rand(N, 128), torch.rand(N, 64)

    x_trans_conv = decoder(f_a, f_c, f_p)
    assert tuple(x_trans_conv.size()) == (N, C, H, W)
    x_no_trans_conv = decoder(f_a, f_c, f_p, no_trans_conv=True)
    assert tuple(x_no_trans_conv.size()) == (N, 64 * 8, 4, 2)


def test_custom_decoder():
    embedding_dims = (64, 64, 32)
    feature_channels = 32
    decoder = Decoder(input_dims=embedding_dims,
                      feature_channels=feature_channels,
                      out_channels=1)
    f_a, f_c, f_p = (torch.rand(N, embedding_dims[0]),
                     torch.rand(N, embedding_dims[1]),
                     torch.rand(N, embedding_dims[2]))

    x_trans_conv = decoder(f_a, f_c, f_p)
    assert tuple(x_trans_conv.size()) == (N, 1, H, W)
    x_no_trans_conv = decoder(f_a, f_c, f_p, no_trans_conv=True)
    assert tuple(x_no_trans_conv.size()) == (N, feature_channels * 8, 4, 2)


def test_default_auto_encoder():
    ae = AutoEncoder()
    x = torch.rand(N, C, H, W)
    y = torch.randint(74, (N,))

    ae.train()
    ((x_c, x_p), (f_p_c1, f_p_c2), (xrecon, cano)) = ae(x, x, x, y)
    assert tuple(x_c.size()) == (N, 64 * 8, 4, 2)
    assert tuple(x_p.size()) == (N, C, H, W)
    assert tuple(f_p_c1.size()) == tuple(f_p_c2.size()) == (N, 64)
    assert tuple(xrecon.size()) == tuple(cano.size()) == ()

    ae.eval()
    (x_c, x_p) = ae(x, x, x)
    assert tuple(x_c.size()) == (N, 64 * 8, 4, 2)
    assert tuple(x_p.size()) == (N, C, H, W)


def test_custom_auto_encoder():
    num_class = 10
    channels = 1
    embedding_dims = (64, 64, 32)
    feature_channels = 32
    ae = AutoEncoder(num_class=num_class,
                     channels=channels,
                     feature_channels=feature_channels,
                     embedding_dims=embedding_dims)
    x = torch.rand(N, 1, H, W)
    y = torch.randint(num_class, (N,))

    ae.train()
    ((x_c, x_p), (f_p_c1, f_p_c2), (xrecon, cano)) = ae(x, x, x, y)
    assert tuple(x_c.size()) == (N, feature_channels * 8, 4, 2)
    assert tuple(x_p.size()) == (N, 1, H, W)
    assert tuple(f_p_c1.size()) \
           == tuple(f_p_c2.size()) \
           == (N, embedding_dims[2])
    assert tuple(xrecon.size()) == tuple(cano.size()) == ()

    ae.eval()
    (x_c, x_p) = ae(x, x, x)
    assert tuple(x_c.size()) == (N, feature_channels * 8, 4, 2)
    assert tuple(x_p.size()) == (N, 1, H, W)
