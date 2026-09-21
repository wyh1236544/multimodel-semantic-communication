from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, features: Tensor) -> Tensor:
        return self.activation(features + self.block(features))


class DownsampleBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            ResidualBlock(out_channels),
        )


class ImageEncoder(nn.Module):
    """Encode [B, 3, H, W] images into [B, N, D] image tokens."""

    downsample_factor = 16

    def __init__(self, in_channels: int = 3, hidden_dim: int = 256):
        super().__init__()
        widths = [32, 64, 128, hidden_dim]
        blocks = []
        current_channels = in_channels
        for width in widths:
            blocks.append(DownsampleBlock(current_channels, width))
            current_channels = width
        self.encoder = nn.Sequential(*blocks)

    def forward(self, images: Tensor) -> Tensor:
        features = self.encoder(images)
        return features.flatten(2).transpose(1, 2)
