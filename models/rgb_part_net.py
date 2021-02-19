import torch
import torch.nn as nn

from models.auto_encoder import AutoEncoder


class RGBPartNet(nn.Module):
    def __init__(
            self,
            ae_in_channels: int = 3,
            ae_in_size: tuple[int, int] = (64, 48),
            ae_feature_channels: int = 64,
            f_a_c_p_dims: tuple[int, int, int] = (128, 128, 64),
            image_log_on: bool = False
    ):
        super().__init__()
        (self.f_a_dim, self.f_c_dim, self.f_p_dim) = f_a_c_p_dims
        self.image_log_on = image_log_on

        self.ae = AutoEncoder(
            ae_in_channels, ae_in_size, ae_feature_channels, f_a_c_p_dims
        )

    def forward(self, x_c1, x_c2=None):
        # Step 1: Disentanglement
        # n, t, c, h, w
        ((x_c, x_p), losses, images) = self._disentangle(x_c1, x_c2)

        if self.training:
            losses = torch.stack(losses)
            return losses, images
        else:
            return x_c, x_p

    def _disentangle(self, x_c1_t2, x_c2_t2=None):
        n, t, c, h, w = x_c1_t2.size()
        device = x_c1_t2.device
        x_c1_t1 = x_c1_t2[:, torch.randperm(t), :, :, :]
        if self.training:
            ((f_a_, f_c_, f_p_), losses) = self.ae(x_c1_t2, x_c1_t1, x_c2_t2)
            # Decode features
            with torch.no_grad():
                x_c = self._decode_cano_feature(f_c_, n, t, device)
                x_p = self._decode_pose_feature(f_p_, n, t, c, h, w, device)

                i_a, i_c, i_p = None, None, None
                if self.image_log_on:
                    i_a = self._decode_appr_feature(f_a_, n, t, device)
                    # Continue decoding canonical features
                    i_c = self.ae.decoder.trans_conv3(x_c)
                    i_c = torch.sigmoid(self.ae.decoder.trans_conv4(i_c))
                    i_p = x_p

            return (x_c, x_p), losses, (i_a, i_c, i_p)

        else:  # evaluating
            f_c_, f_p_ = self.ae(x_c1_t2)
            x_c = self._decode_cano_feature(f_c_, n, t, device)
            x_p = self._decode_pose_feature(f_p_, n, t, c, h, w, device)
            return (x_c, x_p), None, None

    def _decode_appr_feature(self, f_a_, n, t, device):
        # Decode appearance features
        f_a = f_a_.view(n, t, -1)
        x_a = self.ae.decoder(
            f_a.mean(1),
            torch.zeros((n, self.f_c_dim), device=device),
            torch.zeros((n, self.f_p_dim), device=device)
        )
        return x_a

    def _decode_cano_feature(self, f_c_, n, t, device):
        # Decode average canonical features to higher dimension
        f_c = f_c_.view(n, t, -1)
        x_c = self.ae.decoder(
            torch.zeros((n, self.f_a_dim), device=device),
            f_c.mean(1),
            torch.zeros((n, self.f_p_dim), device=device),
            cano_only=True
        )
        return x_c

    def _decode_pose_feature(self, f_p_, n, t, c, h, w, device):
        # Decode pose features to images
        x_p_ = self.ae.decoder(
            torch.zeros((n * t, self.f_a_dim), device=device),
            torch.zeros((n * t, self.f_c_dim), device=device),
            f_p_
        )
        x_p = x_p_.view(n, t, c, h, w)
        return x_p
