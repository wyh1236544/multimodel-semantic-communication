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


class UpsampleBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            ResidualBlock(out_channels),
        )


class ImageDecoder(nn.Module):
    """Decode [B, N, D] tokens into [B, 3, H, W] images in [0, 1]."""

    def __init__(self, hidden_dim: int, output_channels: int, feature_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_size = feature_size
        self.decoder = nn.Sequential(
            ResidualBlock(hidden_dim),
            UpsampleBlock(hidden_dim, 128),
            UpsampleBlock(128, 64),
            UpsampleBlock(64, 32),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, semantic_tokens: Tensor) -> Tensor:
        batch_size = semantic_tokens.size(0)
        features = semantic_tokens.transpose(1, 2).reshape(
            batch_size, self.hidden_dim, self.feature_size, self.feature_size
        )
        return self.decoder(features)
