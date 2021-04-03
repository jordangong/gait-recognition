from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.auto_encoder import AutoEncoder


class RGBPartNet(nn.Module):
    def __init__(
            self,
            ae_in_channels: int = 3,
            ae_in_size: Tuple[int, int] = (64, 48),
            ae_feature_channels: int = 64,
            f_a_c_p_dims: Tuple[int, int, int] = (128, 128, 64),
            image_log_on: bool = False
    ):
        super().__init__()
        self.h, self.w = ae_in_size
        (self.f_a_dim, self.f_c_dim, self.f_p_dim) = f_a_c_p_dims
        self.image_log_on = image_log_on

        self.ae = AutoEncoder(
            ae_in_channels, ae_in_size, ae_feature_channels, f_a_c_p_dims
        )

    def forward(self, x_c1, x_c2=None):
        losses, features, images = self._disentangle(x_c1, x_c2)

        if self.training:
            losses = torch.stack(losses)
            return losses, features, images
        else:
            return features

    def _disentangle(self, x_c1_t2, x_c2_t2=None):
        n, t, c, h, w = x_c1_t2.size()
        if self.training:
            x_c1_t1 = x_c1_t2[:, torch.randperm(t), :, :, :]
            ((f_a_, f_c_, f_p_), losses) = self.ae(x_c1_t2, x_c1_t1, x_c2_t2)
            f_a = f_a_.view(n, t, -1)
            f_c = f_c_.view(n, t, -1)
            f_p = f_p_.view(n, t, -1)

            i_a, i_c, i_p = None, None, None
            if self.image_log_on:
                with torch.no_grad():
                    x_a, i_a = self._separate_decode(
                        f_a.mean(1),
                        torch.zeros_like(f_c[:, 0, :]),
                        torch.zeros_like(f_p[:, 0, :])
                    )
                    x_c, i_c = self._separate_decode(
                        torch.zeros_like(f_a[:, 0, :]),
                        f_c.mean(1),
                        torch.zeros_like(f_p[:, 0, :]),
                    )
                    x_p_, i_p_ = self._separate_decode(
                        torch.zeros_like(f_a_),
                        torch.zeros_like(f_c_),
                        f_p_
                    )
                    x_p = tuple(_x_p.view(n, t, *_x_p.size()[1:]) for _x_p in x_p_)
                    i_p = i_p_.view(n, t, c, h, w)

            return losses, (x_a, x_c, x_p), (i_a, i_c, i_p)

        else:  # evaluating
            f_c_, f_p_ = self.ae(x_c1_t2)
            f_c = f_c_.view(n, t, -1)
            f_p = f_p_.view(n, t, -1)
            return (f_c, f_p), None, None

    def _separate_decode(self, f_a, f_c, f_p):
        x_1 = torch.cat((f_a, f_c, f_p), dim=1)
        x_1 = self.ae.decoder.fc(x_1).view(
            -1,
            self.ae.decoder.feature_channels * 8,
            self.ae.decoder.h_0,
            self.ae.decoder.w_0
        )
        x_1 = F.relu(x_1, inplace=True)
        x_2 = self.ae.decoder.trans_conv1(x_1)
        x_3 = self.ae.decoder.trans_conv2(x_2)
        x_4 = self.ae.decoder.trans_conv3(x_3)
        image = torch.sigmoid(self.ae.decoder.trans_conv4(x_4))
        x = (x_1, x_2, x_3, x_4)
        return x, image
