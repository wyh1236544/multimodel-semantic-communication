"""Lightweight spatial residual correction for recovered VAE latents."""

from __future__ import annotations

import math
import torch
from torch import Tensor, nn


class SpatialResidualAdapter(nn.Module):
    def __init__(self, hidden_dim: int, latent_channels: int = 4, initial_alpha: float = 0.05):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden_dim, latent_channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(float(initial_alpha)))

    def forward(self, image_tokens: Tensor, recovered_latents: Tensor) -> tuple[Tensor, Tensor]:
        b, n, _ = image_tokens.shape
        side = int(math.sqrt(n))
        if side * side != n:
            raise ValueError(f"Residual adapter expects square image tokens, got {n}")
        features = self.norm(image_tokens).transpose(1, 2).reshape(b, -1, side, side)
        residual = self.net(features)
        return recovered_latents, self.alpha * residual


class ResidualReliabilityGate(nn.Module):
    """Learned global and spatial reliability gate for residual injection."""

    def __init__(self, hidden_dim: int, latent_channels: int = 4, snr_prior_strength: float = 0.25):
        super().__init__()
        self.snr_prior_strength = float(snr_prior_strength)
        # Mean/std image-private features plus latent/residual statistics,
        # damage rate and normalized SNR form a compact global reliability cue.
        global_dim = hidden_dim * 2 + latent_channels * 4 + 3
        self.global_net = nn.Sequential(
            nn.Linear(global_dim, 64), nn.GELU(), nn.Linear(64, 1)
        )
        # Spatial gate sees local image-private features and local latent/residual.
        spatial_dim = hidden_dim + latent_channels * 2 + 3
        self.spatial_net = nn.Sequential(
            nn.Conv2d(spatial_dim, 32, 1), nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )
        # Start usable on damaged samples; the clean-sample suppression loss
        # drives this toward zero when no image tokens are missing.
        nn.init.constant_(self.global_net[-1].bias, -0.2)
        nn.init.constant_(self.spatial_net[-1].bias, -0.2)

    def forward(
        self,
        image_tokens: Tensor,
        base_latents: Tensor,
        residual: Tensor,
        damage_rate: Tensor,
        snr_db: Tensor,
        reliability: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        b, n, _ = image_tokens.shape
        side = int(math.sqrt(n))
        features = image_tokens.transpose(1, 2).reshape(b, -1, side, side)
        base_stats = torch.cat([base_latents.mean((2, 3)), base_latents.std((2, 3))], dim=1)
        residual_stats = torch.cat([residual.mean((2, 3)), residual.std((2, 3))], dim=1)
        image_stats = torch.cat([features.mean((2, 3)), features.std((2, 3))], dim=1)
        scalar = torch.cat([damage_rate.view(b, 1), (snr_db / 20.0).view(b, 1), reliability.view(b, 1)], dim=1)
        global_gate = torch.sigmoid(self.global_net(torch.cat([image_stats, base_stats, residual_stats, scalar], dim=1)))
        scalar_map = scalar[:, :, None, None].expand(-1, -1, side, side)
        spatial_gate = torch.sigmoid(self.spatial_net(torch.cat([features, base_latents, residual, scalar_map], dim=1)))
        # A small monotonic reliability prior makes the learned gate more
        # conservative at high SNR and more willing to use residuals at low
        # SNR. The learned image/base/residual cues remain the main signal.
        snr_prior = 1.0 + self.snr_prior_strength * (1.0 - snr_db.clamp(0.0, 20.0) / 20.0)
        gate = global_gate[:, :, None, None] * spatial_gate * snr_prior.view(b, 1, 1, 1)
        return gate, global_gate, spatial_gate
