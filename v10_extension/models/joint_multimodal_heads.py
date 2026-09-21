from __future__ import annotations

import torch
from torch import Tensor, nn


class LatentResidualBlock(nn.Module):
    """Refine local structure in the recovered VAE latent grid."""

    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return features + self.block(features)


class ImageReconstructionHead(nn.Module):
    """Recover all 16x16 VAE latent tokens from post-channel semantic tokens."""

    def __init__(
        self,
        hidden_dim: int,
        latent_channels: int,
        token_grid_size: int,
        num_heads: int,
        num_layers: int = 1,
        refinement_blocks: int = 0,
        refinement_channels: int | None = None,
        spatial_direct: bool = False,
        symmetric_cross_attention: bool = False,
    ):
        super().__init__()
        self.token_grid_size = token_grid_size
        self.spatial_direct = spatial_direct
        self.symmetric_cross_attention = symmetric_cross_attention
        num_queries = token_grid_size**2
        self.queries = nn.Parameter(
            torch.randn(1, num_queries, hidden_dim) * 0.02
        )
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.direct_norm = (
            nn.LayerNorm(hidden_dim) if spatial_direct else nn.Identity()
        )
        if refinement_blocks > 0:
            refine_channels = int(refinement_channels or hidden_dim)
            self.pre_refine = nn.Conv2d(
                hidden_dim,
                refine_channels,
                kernel_size=3,
                padding=1,
            )
            self.refinement = nn.Sequential(
                *[
                    LatentResidualBlock(refine_channels)
                    for _ in range(refinement_blocks)
                ]
            )
            self.unprojection = nn.Conv2d(
                refine_channels,
                latent_channels,
                kernel_size=3,
                padding=1,
            )
        else:
            self.pre_refine = nn.Identity()
            self.refinement = nn.Identity()
            self.unprojection = nn.Conv2d(
                hidden_dim,
                latent_channels,
                kernel_size=1,
            )

    def forward(
        self,
        memory: Tensor,
        spatial_tokens: Tensor | None = None,
        spatial_reliability: Tensor | None = None,
    ) -> Tensor:
        batch_size = memory.size(0)
        if self.symmetric_cross_attention:
            expected_tokens = self.token_grid_size**2
            if spatial_tokens is None or spatial_tokens.size(1) != expected_tokens:
                raise ValueError(
                    "Symmetric image head requires the received spatial tokens"
                )
            queries = self.queries.expand(batch_size, -1, -1)
            direct_tokens = self.direct_norm(spatial_tokens)
            if spatial_reliability is not None:
                reliability = spatial_reliability.to(
                    device=direct_tokens.device,
                    dtype=direct_tokens.dtype,
                ).reshape(batch_size, 1, 1)
                direct_tokens = direct_tokens * reliability
            tokens = self.decoder(
                direct_tokens + queries,
                memory,
            )
        elif self.spatial_direct:
            expected_tokens = self.token_grid_size**2
            if memory.size(1) != expected_tokens:
                raise ValueError(
                    f"Spatial image head expected {expected_tokens} tokens, "
                    f"got {memory.size(1)}"
                )
            tokens = self.direct_norm(memory)
        else:
            queries = self.queries.expand(batch_size, -1, -1)
            tokens = self.decoder(queries, memory)
        feature_map = tokens.transpose(1, 2).reshape(
            batch_size,
            tokens.size(-1),
            self.token_grid_size,
            self.token_grid_size,
        )
        refined = self.refinement(self.pre_refine(feature_map))
        return self.unprojection(refined)


class CommonCoarseLatentHead(nn.Module):
    """Predict a low-frequency VAE latent using only received common symbols.

    This branch deliberately bypasses the joint decoder, so image/text-private
    channel symbols cannot leak into the common regression prediction.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        latent_channels: int,
        token_grid_size: int,
        num_heads: int,
        num_layers: int = 1,
        zero_init: bool = True,
    ):
        super().__init__()
        self.input_projection = nn.Linear(latent_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.common_decoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.reconstruction = ImageReconstructionHead(
            hidden_dim=hidden_dim,
            latent_channels=latent_channels,
            token_grid_size=token_grid_size,
            num_heads=num_heads,
            num_layers=num_layers,
            refinement_blocks=0,
            spatial_direct=False,
            symmetric_cross_attention=False,
        )
        if zero_init:
            nn.init.zeros_(self.reconstruction.unprojection.weight)
            nn.init.zeros_(self.reconstruction.unprojection.bias)

    def forward(self, received_common: Tensor) -> Tensor:
        common = self.output_norm(
            self.common_decoder(self.input_projection(received_common))
        )
        return self.reconstruction(common)


class CommonBaseSemanticDecoder(nn.Module):
    """Decode a physically received common-only payload into semantic tokens.

    The module has no connection to image/text-private channel positions.  It
    is therefore suitable for independently testable base-layer tasks such as
    short-caption recovery and receiver-side semantic retrieval.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(latent_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, received_common: Tensor) -> Tensor:
        return self.output_norm(
            self.decoder(self.input_projection(received_common))
        )
