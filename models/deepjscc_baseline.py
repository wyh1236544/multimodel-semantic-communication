from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.channel import SemanticChannel


class GDN(nn.Module):
    """Compact generalized divisive normalization used by learned codecs."""

    def __init__(self, channels: int, inverse: bool = False):
        super().__init__()
        self.inverse = inverse
        self.beta_reparam = nn.Parameter(torch.ones(channels))
        self.gamma_reparam = nn.Parameter(torch.eye(channels) * (0.1 ** 0.5))

    def forward(self, inputs: Tensor) -> Tensor:
        beta = self.beta_reparam.square() + 1e-6
        gamma = self.gamma_reparam.square().view(
            inputs.size(1), inputs.size(1), 1, 1
        )
        norm = F.conv2d(inputs.square(), gamma, beta)
        return inputs * (torch.sqrt(norm) if self.inverse else torch.rsqrt(norm))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + 0.1 * self.block(inputs)


class DeepJSCCBaseline(nn.Module):
    """DeepJSCC-style convolutional image transmission baseline.

    For a 128x128 input and 72 channel features, the transmitted tensor is
    [B, 72, 8, 8], or exactly 4608 real channel symbols per image.
    """

    def __init__(
        self,
        channel_type: str = "awgn",
        channel_channels: int = 72,
        gamma_gamma_alpha: float = 4.0,
        gamma_gamma_beta: float = 2.0,
    ):
        super().__init__()
        widths = [64, 96, 128]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, widths[0], 5, stride=2, padding=2),
            GDN(widths[0]),
            nn.Conv2d(widths[0], widths[1], 5, stride=2, padding=2),
            GDN(widths[1]),
            nn.Conv2d(widths[1], widths[2], 5, stride=2, padding=2),
            GDN(widths[2]),
            ResidualBlock(widths[2]),
            nn.Conv2d(widths[2], channel_channels, 5, stride=2, padding=2),
        )
        self.channel = SemanticChannel(
            channel_type,
            gamma_gamma_alpha=gamma_gamma_alpha,
            gamma_gamma_beta=gamma_gamma_beta,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                channel_channels, widths[2], 5, stride=2, padding=2,
                output_padding=1,
            ),
            GDN(widths[2], inverse=True),
            ResidualBlock(widths[2]),
            nn.ConvTranspose2d(
                widths[2], widths[1], 5, stride=2, padding=2, output_padding=1
            ),
            GDN(widths[1], inverse=True),
            nn.ConvTranspose2d(
                widths[1], widths[0], 5, stride=2, padding=2, output_padding=1
            ),
            GDN(widths[0], inverse=True),
            nn.ConvTranspose2d(
                widths[0], 3, 5, stride=2, padding=2, output_padding=1
            ),
            nn.Sigmoid(),
        )
        self.channel_channels = int(channel_channels)

    def configure_channel(self, channel_type: str) -> None:
        self.channel.configure(channel_type)

    def num_channel_uses(self, image_size: int = 128) -> int:
        spatial = image_size // 16
        return self.channel_channels * spatial * spatial

    def forward(
        self,
        images: Tensor,
        snr_db: Union[float, int, Tensor],
        standard_noise: Optional[Tensor] = None,
        channel_gain: Optional[Tensor] = None,
        channel_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        latent = self.encoder(images)
        noisy = self.channel(
            latent,
            snr_db,
            standard_noise=standard_noise,
            channel_gain=channel_gain,
        )
        # Baseline robustness evaluation uses post-channel symbol erasures.
        # The mask is applied after AWGN so the comparison does not change
        # the transmit power normalization of the baseline.
        if channel_mask is not None:
            noisy = noisy * channel_mask.to(dtype=noisy.dtype)
        return {"image": self.decoder(noisy), "latent": latent, "latent_noisy": noisy}
