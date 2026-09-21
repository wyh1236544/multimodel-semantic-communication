from torch import Tensor, nn


class SemanticDecoder(nn.Module):
    """Expand spatial [B, h*w, C] latents into [B, H*W, D] image tokens."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        feature_size: int,
        downsample_factor: int,
        num_layers: int,
        num_heads: int,
    ):
        super().__init__()
        if downsample_factor not in {1, 2}:
            raise ValueError("latent downsample_factor must be 1 or 2")

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.feature_size = feature_size
        self.latent_size = feature_size // downsample_factor
        if downsample_factor == 1:
            self.spatial_expander = nn.Sequential(
                nn.Conv2d(latent_dim, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
        else:
            self.spatial_expander = nn.Sequential(
                nn.ConvTranspose2d(
                    latent_dim,
                    hidden_dim,
                    kernel_size=4,
                    stride=downsample_factor,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.refiner = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, noisy_latent: Tensor) -> Tensor:
        batch_size, num_tokens, latent_dim = noisy_latent.shape
        expected_tokens = self.latent_size**2
        if num_tokens != expected_tokens or latent_dim != self.latent_dim:
            raise ValueError(
                f"Expected [B, {expected_tokens}, {self.latent_dim}], "
                f"received {tuple(noisy_latent.shape)}"
            )

        latent_map = noisy_latent.transpose(1, 2).reshape(
            batch_size, latent_dim, self.latent_size, self.latent_size
        )
        feature_map = self.spatial_expander(latent_map)
        tokens = feature_map.flatten(2).transpose(1, 2)
        return self.output_norm(self.refiner(tokens))
