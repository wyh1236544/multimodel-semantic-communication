from __future__ import annotations

import torch
from torch import Tensor, nn


class ConcatMLPFusion(nn.Module):
    """Fuse [B, N, D] image tokens with pooled [B, D] text features."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, image_tokens: Tensor, text_features: Tensor) -> Tensor:
        text_expanded = text_features.unsqueeze(1).expand(-1, image_tokens.size(1), -1)
        return self.mlp(torch.cat([image_tokens, text_expanded], dim=-1))


class CrossAttentionFusion(nn.Module):
    """Use image tokens as queries over token-level text features."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList(
            _CrossAttentionBlock(hidden_dim, num_heads)
            for _ in range(num_layers)
        )

    def forward(
        self,
        image_tokens: Tensor,
        text_features: Tensor,
        text_attention_mask: Tensor | None = None,
    ) -> Tensor:
        if text_features.ndim == 2:
            text_features = text_features.unsqueeze(1)
        key_padding_mask = None
        if text_attention_mask is not None:
            key_padding_mask = ~text_attention_mask.bool()
        fused = image_tokens
        for layer in self.layers:
            fused = layer(fused, text_features, key_padding_mask)
        return fused


class GatedTextToImageFusion(nn.Module):
    """Inject token-level text context into each spatial image token."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        initial_gate_bias: float = -2.0,
    ):
        super().__init__()
        self.image_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, initial_gate_bias)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        image_tokens: Tensor,
        text_tokens: Tensor,
        text_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        key_padding_mask = (
            ~text_attention_mask.bool()
            if text_attention_mask is not None
            else None
        )
        text_context, _ = self.cross_attention(
            self.image_norm(image_tokens),
            self.text_norm(text_tokens),
            self.text_norm(text_tokens),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        gate = torch.sigmoid(
            self.gate(torch.cat([image_tokens, text_context], dim=-1))
        )
        fused = image_tokens + gate * text_context
        fused = fused + gate * self.ffn(self.ffn_norm(fused))
        return fused, gate


class _CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        image_tokens: Tensor,
        text_tokens: Tensor,
        key_padding_mask: Tensor | None,
    ) -> Tensor:
        attended, _ = self.attention(
            self.query_norm(image_tokens),
            self.context_norm(text_tokens),
            self.context_norm(text_tokens),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        fused = self.attention_norm(image_tokens + attended)
        return self.output_norm(fused + self.ffn(fused))


def build_fusion(
    fusion_type: str,
    hidden_dim: int,
    num_heads: int,
    num_layers: int = 1,
) -> nn.Module:
    if fusion_type == "concat_mlp":
        return ConcatMLPFusion(hidden_dim)
    if fusion_type == "cross_attention":
        return CrossAttentionFusion(hidden_dim, num_heads, num_layers)
    raise ValueError(f"Unsupported fusion type: {fusion_type}")
