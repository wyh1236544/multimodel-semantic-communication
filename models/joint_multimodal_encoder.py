from __future__ import annotations

import torch
from torch import Tensor, nn

from .shared_bottleneck import QueryTokenCompressor


class JointMultimodalEncoder(nn.Module):
    """Fuse image/text tokens and compress them into three semantic groups."""

    def __init__(
        self,
        hidden_dim: int,
        num_visual_tokens: int,
        text_max_length: int,
        num_common_tokens: int,
        num_img_tokens: int,
        num_txt_tokens: int,
        num_heads: int,
        num_layers: int,
        preserve_spatial_image_tokens: bool = False,
        explicit_shared_semantics: bool = False,
        common_residual_gate_initial_bias: float = -2.0,
        private_common_gate_initial_bias: float = -2.0,
    ):
        super().__init__()
        self.num_visual_tokens = num_visual_tokens
        self.text_max_length = text_max_length
        self.num_common_tokens = num_common_tokens
        self.preserve_spatial_image_tokens = preserve_spatial_image_tokens
        self.explicit_shared_semantics = bool(explicit_shared_semantics)
        if preserve_spatial_image_tokens and num_img_tokens != num_visual_tokens:
            raise ValueError(
                "Spatial image transmission requires "
                "num_img_tokens == num_visual_tokens"
            )
        self.common_tokens = nn.Parameter(
            torch.randn(1, num_common_tokens, hidden_dim) * 0.02
        )
        self.visual_position = (
            nn.Parameter(
                torch.randn(1, num_visual_tokens, hidden_dim) * 0.02
            )
            if preserve_spatial_image_tokens
            else None
        )
        self.image_type = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.text_type = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.joint_transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )
        self.image_transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )
        self.text_transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.image_compressor = QueryTokenCompressor(
            hidden_dim,
            num_img_tokens,
            num_heads,
        )
        self.text_compressor = QueryTokenCompressor(
            hidden_dim,
            num_txt_tokens,
            num_heads,
        )
        self.common_image_extractor = QueryTokenCompressor(
            hidden_dim,
            num_common_tokens,
            num_heads,
        )
        self.common_text_extractor = QueryTokenCompressor(
            hidden_dim,
            num_common_tokens,
            num_heads,
        )
        self.common_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.common_output_norm = nn.LayerNorm(hidden_dim)
        self.common_to_image = nn.Linear(hidden_dim, hidden_dim)
        self.common_to_text = nn.Linear(hidden_dim, hidden_dim)
        self.common_residual_gate_logit = nn.Parameter(
            torch.tensor(float(common_residual_gate_initial_bias))
        )
        self.private_common_gate_logit = nn.Parameter(
            torch.tensor(float(private_common_gate_initial_bias))
        )
        self.separate_common_image = QueryTokenCompressor(
            hidden_dim,
            num_common_tokens // 2,
            num_heads,
        )
        self.separate_common_text = QueryTokenCompressor(
            hidden_dim,
            num_common_tokens - num_common_tokens // 2,
            num_heads,
        )

    def _joint_forward(
        self,
        visual_tokens: Tensor,
        text_tokens: Tensor,
        text_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size = visual_tokens.size(0)
        common = self.common_tokens.expand(batch_size, -1, -1)
        visual = visual_tokens + self.image_type
        if self.visual_position is not None:
            visual = visual + self.visual_position
        text = text_tokens + self.text_type
        tokens = torch.cat([common, visual, text], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros(
                    batch_size,
                    self.num_common_tokens + self.num_visual_tokens,
                    dtype=torch.bool,
                    device=tokens.device,
                ),
                ~text_attention_mask.bool(),
            ],
            dim=1,
        )
        encoded = self.output_norm(
            self.joint_transformer(
                tokens,
                src_key_padding_mask=padding_mask,
            )
        )
        common_query_out = encoded[:, : self.num_common_tokens]
        image_start = self.num_common_tokens
        text_start = image_start + self.num_visual_tokens
        visual_out = encoded[:, image_start:text_start]
        text_out = encoded[:, text_start:]
        # Build unimodal views before cross-modal attention. Their agreement is
        # therefore evidence of learned shared content, not information leakage.
        common_image = self.common_image_extractor(visual)
        common_text = self.common_text_extractor(
            text,
            key_padding_mask=~text_attention_mask.bool(),
        )
        if self.explicit_shared_semantics:
            common_delta = self.common_fusion(
                torch.cat([common_query_out, common_image, common_text], dim=-1)
            )
            common_gate = torch.sigmoid(self.common_residual_gate_logit).to(
                dtype=common_query_out.dtype
            )
            common_out = self.common_output_norm(
                common_query_out + common_gate * common_delta
            )
            private_gate = torch.sigmoid(self.private_common_gate_logit).to(
                dtype=visual_out.dtype
            )
            common_summary = common_out.mean(dim=1, keepdim=True)
            visual_private = visual_out - private_gate * self.common_to_image(
                common_summary
            )
            text_private = text_out - private_gate * self.common_to_text(
                common_summary
            )
        else:
            common_out = common_query_out
            visual_private = visual_out
            text_private = text_out
        z_img_tokens = (
            visual_private
            if self.preserve_spatial_image_tokens
            else self.image_compressor(visual_private)
        )
        z_txt_tokens = self.text_compressor(
            text_private,
            key_padding_mask=~text_attention_mask.bool(),
        )
        return (
            common_out,
            z_img_tokens,
            z_txt_tokens,
            common_image,
            common_text,
        )

    def _separate_forward(
        self,
        visual_tokens: Tensor,
        text_tokens: Tensor,
        text_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        visual = visual_tokens + self.image_type
        if self.visual_position is not None:
            visual = visual + self.visual_position
        visual_out = self.output_norm(self.image_transformer(visual))
        text_padding = ~text_attention_mask.bool()
        text_out = self.output_norm(
            self.text_transformer(
                text_tokens + self.text_type,
                src_key_padding_mask=text_padding,
            )
        )
        common_image = self.separate_common_image(visual_out)
        common_text = self.separate_common_text(
            text_out,
            key_padding_mask=text_padding,
        )
        common_image_view = self.common_image_extractor(visual_out)
        common_text_view = self.common_text_extractor(
            text_out,
            key_padding_mask=text_padding,
        )
        return (
            torch.cat([common_image, common_text], dim=1),
            (
                visual_out
                if self.preserve_spatial_image_tokens
                else self.image_compressor(visual_out)
            ),
            self.text_compressor(text_out, key_padding_mask=text_padding),
            common_image_view,
            common_text_view,
        )

    def forward(
        self,
        visual_tokens: Tensor,
        text_tokens: Tensor,
        text_attention_mask: Tensor,
        transmission_mode: str,
    ) -> dict[str, Tensor]:
        if transmission_mode == "joint":
            common, image, text, common_image, common_text = self._joint_forward(
                visual_tokens,
                text_tokens,
                text_attention_mask,
            )
        elif transmission_mode == "separate":
            common, image, text, common_image, common_text = self._separate_forward(
                visual_tokens,
                text_tokens,
                text_attention_mask,
            )
        elif transmission_mode == "image_only":
            common, image, text, common_image, common_text = self._separate_forward(
                visual_tokens,
                torch.zeros_like(text_tokens),
                torch.ones_like(text_attention_mask),
            )
            common = torch.zeros_like(common)
            text = torch.zeros_like(text)
            common_text = torch.zeros_like(common_text)
        elif transmission_mode == "text_only":
            common, image, text, common_image, common_text = self._separate_forward(
                torch.zeros_like(visual_tokens),
                text_tokens,
                text_attention_mask,
            )
            common = torch.zeros_like(common)
            image = torch.zeros_like(image)
            common_image = torch.zeros_like(common_image)
        else:
            raise ValueError(
                "transmission_mode must be joint, separate, image_only, or text_only"
            )
        return {
            "common_tokens": common,
            "image_tokens": image,
            "text_tokens": text,
            "common_image_tokens": common_image,
            "common_text_tokens": common_text,
        }
