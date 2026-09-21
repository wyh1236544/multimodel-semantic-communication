from __future__ import annotations

import torch
from torch import Tensor, nn


class JointMultimodalDecoder(nn.Module):
    """Decode one noisy shared latent and split it into modality token groups."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_common_tokens: int,
        num_img_tokens: int,
        num_txt_tokens: int,
        num_heads: int,
        num_layers: int,
        transmission_mode: str,
        preserve_spatial_image_tokens: bool = False,
    ):
        super().__init__()
        self.num_common_tokens = num_common_tokens
        self.num_img_tokens = num_img_tokens
        self.num_txt_tokens = num_txt_tokens
        self.transmission_mode = transmission_mode
        self.preserve_spatial_image_tokens = preserve_spatial_image_tokens
        self.input_projection = nn.Linear(latent_dim, hidden_dim)
        self.image_position = (
            nn.Parameter(torch.randn(1, num_img_tokens, hidden_dim) * 0.02)
            if preserve_spatial_image_tokens
            else None
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
        if transmission_mode == "separate":
            self.common_decoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.image_decoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.text_decoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.shared_decoder = None
        else:
            self.shared_decoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.common_decoder = None
            self.image_decoder = None
            self.text_decoder = None
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, noisy_shared: Tensor) -> dict[str, Tensor]:
        tokens = self.input_projection(noisy_shared)
        common_end = self.num_common_tokens
        image_end = common_end + self.num_img_tokens
        if self.image_position is not None:
            tokens = tokens.clone()
            tokens[:, common_end:image_end] = (
                tokens[:, common_end:image_end] + self.image_position
            )
        if self.shared_decoder is not None:
            decoded = self.output_norm(self.shared_decoder(tokens))
            common = decoded[:, :common_end]
            image = decoded[:, common_end:image_end]
            text = decoded[:, image_end:]
        else:
            common = self.output_norm(
                self.common_decoder(tokens[:, :common_end])
            )
            image = self.output_norm(
                self.image_decoder(tokens[:, common_end:image_end])
            )
            text = self.output_norm(
                self.text_decoder(tokens[:, image_end:])
            )
        return {
            "common_tokens": common,
            "image_tokens": image,
            "text_tokens": text,
        }
