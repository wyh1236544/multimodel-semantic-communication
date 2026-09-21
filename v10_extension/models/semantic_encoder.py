from torch import Tensor, nn


class SemanticEncoder(nn.Module):
    """Compress [B, H*W, D] tokens into spatial latents [B, h*w, C]."""

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        feature_size: int,
        downsample_factor: int,
        num_layers: int,
        num_heads: int,
    ):
        super().__init__()
        if downsample_factor not in {1, 2}:
            raise ValueError("latent downsample_factor must be 1 or 2")
        if feature_size % downsample_factor != 0:
            raise ValueError("feature_size must be divisible by downsample_factor")

        self.hidden_dim = hidden_dim
        self.feature_size = feature_size
        self.latent_size = feature_size // downsample_factor

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.token_norm = nn.LayerNorm(hidden_dim)
        if downsample_factor == 1:
            self.spatial_compressor = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, latent_dim, kernel_size=1),
            )
        else:
            self.spatial_compressor = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(
                    hidden_dim,
                    latent_dim,
                    kernel_size=4,
                    stride=downsample_factor,
                    padding=1,
                ),
            )

    def forward(self, fused_tokens: Tensor) -> Tensor:
        batch_size, num_tokens, hidden_dim = fused_tokens.shape
        expected_tokens = self.feature_size**2
        if num_tokens != expected_tokens or hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected [B, {expected_tokens}, {self.hidden_dim}], "
                f"received {tuple(fused_tokens.shape)}"
            )

        encoded = self.token_norm(self.transformer(fused_tokens))
        feature_map = encoded.transpose(1, 2).reshape(
            batch_size, hidden_dim, self.feature_size, self.feature_size
        )
        latent_map = self.spatial_compressor(feature_map)
        return latent_map.flatten(2).transpose(1, 2)
