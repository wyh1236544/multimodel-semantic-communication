from __future__ import annotations

import torch
from torch import Tensor, nn


class QueryTokenCompressor(nn.Module):
    """Compress a variable-length token sequence with learnable queries."""

    def __init__(self, hidden_dim: int, num_queries: int, num_heads: int):
        super().__init__()
        self.queries = nn.Parameter(
            torch.randn(1, num_queries, hidden_dim) * 0.02
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        context: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        queries = self.queries.expand(context.size(0), -1, -1)
        attended, _ = self.attention(
            self.query_norm(queries),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.output_norm(queries + attended)


class SharedBottleneck(nn.Module):
    """Project common, image, and text semantic tokens to one channel latent."""

    def __init__(self, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.common_projection = nn.Linear(hidden_dim, latent_dim)
        self.image_projection = nn.Linear(hidden_dim, latent_dim)
        self.text_projection = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        common_tokens: Tensor,
        image_tokens: Tensor,
        text_tokens: Tensor,
    ) -> dict[str, Tensor]:
        z_common = self.common_projection(common_tokens)
        z_img = self.image_projection(image_tokens)
        z_txt = self.text_projection(text_tokens)
        return {
            "z_common": z_common,
            "z_img": z_img,
            "z_txt": z_txt,
            "z_shared": torch.cat([z_common, z_img, z_txt], dim=1),
        }


class FixedBudgetTokenPacker(nn.Module):
    """Pack a variable image/text token mix into a fixed-size channel payload."""

    supported_strategies = {"fixed", "length", "learned", "utility"}

    def __init__(
        self,
        hidden_dim: int,
        modality_budget: int,
        min_text_tokens: int,
        max_text_tokens: int,
        fixed_text_tokens: int,
        strategy: str = "fixed",
        text_selection_strategy: str = "importance",
    ):
        super().__init__()
        self.modality_budget = int(modality_budget)
        self.min_text_tokens = int(min_text_tokens)
        self.max_text_tokens = int(max_text_tokens)
        self.fixed_text_tokens = int(fixed_text_tokens)
        self.strategy = str(strategy).lower()
        self.text_selection_strategy = str(text_selection_strategy).lower()
        if self.text_selection_strategy not in {
            "importance",
            "valid_prefix",
            "valid_prefix_repeat",
        }:
            raise ValueError(
                "Unsupported text_selection_strategy: "
                f"{self.text_selection_strategy}"
            )
        if self.strategy not in self.supported_strategies:
            raise ValueError(f"Unsupported token allocation strategy: {strategy}")
        self.image_importance = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1)
        )
        self.text_importance = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1)
        )
        self.text_length_bias = nn.Parameter(torch.tensor(1.0))
        self.text_snr_bias = nn.Parameter(torch.tensor(-0.1))
        self.allocation_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _length_quota(self, valid_lengths: Tensor, text_max_length: int) -> Tensor:
        fraction = valid_lengths.float() / max(int(text_max_length), 1)
        quota = self.min_text_tokens + fraction * (
            self.max_text_tokens - self.min_text_tokens
        )
        return quota.round().long()

    def _quota_bounds(self, num_image: int, num_text: int) -> tuple[int, int]:
        low = max(self.min_text_tokens, self.modality_budget - num_image, 0)
        high = min(self.max_text_tokens, num_text, self.modality_budget)
        if low > high:
            raise ValueError(
                "Fixed channel budget cannot be satisfied with configured token pools"
            )
        return low, high

    def pack(
        self,
        z_common: Tensor,
        z_image: Tensor,
        z_text: Tensor,
        image_features: Tensor,
        text_features: Tensor,
        valid_text_lengths: Tensor,
        text_max_length: int,
        snr_db: Tensor | float | int,
        utility_weight: Tensor | float | None = None,
        text_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        batch, num_image, latent_dim = z_image.shape
        num_text = z_text.size(1)
        low, high = self._quota_bounds(num_image, num_text)
        image_scores = self.image_importance(image_features).squeeze(-1)
        text_scores = self.text_importance(text_features).squeeze(-1)
        snr = torch.as_tensor(snr_db, device=z_image.device, dtype=torch.float32)
        if snr.ndim == 0:
            snr = snr.expand(batch)
        snr = snr.reshape(batch, -1)[:, 0]
        length_fraction = valid_text_lengths.float() / max(int(text_max_length), 1)
        text_scores = text_scores + self.text_length_bias * length_fraction[:, None]
        text_scores = text_scores + self.text_snr_bias * (snr[:, None] / 20.0)
        pooled_image = image_features.mean(dim=1).float()
        pooled_text = text_features.mean(dim=1).float()
        allocation_features = torch.cat(
            [pooled_image, pooled_text, length_fraction[:, None], snr[:, None] / 20.0],
            dim=1,
        )
        predicted_fraction = torch.sigmoid(
            self.allocation_predictor(allocation_features).squeeze(-1)
        )
        predicted_quota = low + predicted_fraction * (high - low)
        # Long descriptions deserve more slots, while high-SNR images need less
        # semantic compensation. This weak target prevents quota collapse.
        snr_factor = 1.0 - 0.35 * (snr.clamp(0, 20) / 20.0)
        target_fraction = (length_fraction * snr_factor).clamp(0.0, 1.0)
        target_quota = low + target_fraction * (high - low)

        packed_rows = []
        image_indices = []
        text_indices = []
        image_counts = []
        text_counts = []
        for index in range(batch):
            if self.strategy == "fixed":
                text_count = min(max(self.fixed_text_tokens, low), high)
                img_count = self.modality_budget - text_count
                img_idx = torch.topk(image_scores[index], img_count).indices
                txt_idx = torch.topk(text_scores[index], text_count).indices
            elif self.strategy == "length":
                requested = int(
                    self._length_quota(
                        valid_text_lengths[index:index + 1], text_max_length
                    )[0].item()
                )
                text_count = min(max(requested, low), high)
                img_count = self.modality_budget - text_count
                img_idx = torch.topk(image_scores[index], img_count).indices
                txt_idx = torch.topk(text_scores[index], text_count).indices
            elif self.strategy == "learned":
                text_count = int(predicted_quota[index].round().item())
                text_count = min(max(text_count, low), high)
                img_count = self.modality_budget - text_count
                img_idx = torch.topk(image_scores[index], img_count).indices
                txt_idx = torch.topk(text_scores[index], text_count).indices
            else:
                utility = torch.as_tensor(
                    0.5 if utility_weight is None else utility_weight,
                    device=z_image.device,
                    dtype=torch.float32,
                )
                if utility.ndim == 0:
                    utility = utility.expand(batch)
                utility = utility.reshape(batch, -1)[:, 0].clamp(0.0, 1.0)
                text_count = int(
                    (low + utility[index] * (high - low)).round().item()
                )
                text_count = min(max(text_count, low), high)
                img_count = self.modality_budget - text_count
                img_idx = torch.topk(image_scores[index], img_count).indices
                txt_idx = torch.topk(text_scores[index], text_count).indices

            if self.text_selection_strategy in {
                "valid_prefix",
                "valid_prefix_repeat",
            }:
                if text_attention_mask is None:
                    raise ValueError(
                        "text_attention_mask is required for valid-prefix selection"
                    )
                valid = torch.nonzero(
                    text_attention_mask[index].bool(), as_tuple=False
                ).flatten()
                invalid = torch.nonzero(
                    ~text_attention_mask[index].bool(), as_tuple=False
                ).flatten()
                if self.text_selection_strategy == "valid_prefix":
                    txt_idx = torch.cat([valid, invalid], dim=0)[:text_count]
                else:
                    # Extra text channel slots must carry useful redundancy,
                    # not padding.  Send every valid token once, then repeat
                    # high-importance valid tokens.  Independent channel noise
                    # on the repeated copies is averaged by ``unpack`` below,
                    # implementing text unequal-error protection (UEP).
                    if valid.numel() == 0:
                        valid = torch.zeros(
                            1, device=z_text.device, dtype=torch.long
                        )
                    prefix = valid[:text_count]
                    remaining = text_count - int(prefix.numel())
                    if remaining > 0:
                        priority_order = torch.argsort(
                            text_scores[index, valid], descending=True
                        )
                        priority = valid[priority_order]
                        repeat_positions = torch.arange(
                            remaining, device=z_text.device
                        ) % int(priority.numel())
                        prefix = torch.cat(
                            [prefix, priority[repeat_positions]], dim=0
                        )
                    txt_idx = prefix

            img_values = z_image[index, img_idx]
            txt_values = z_text[index, txt_idx]
            if self.strategy in {"learned", "utility"}:
                img_values = img_values * torch.sigmoid(
                    image_scores[index, img_idx]
                ).unsqueeze(-1)
                txt_values = txt_values * torch.sigmoid(
                    text_scores[index, txt_idx]
                ).unsqueeze(-1)
            packed_rows.append(torch.cat([z_common[index], img_values, txt_values], 0))
            image_indices.append(img_idx)
            text_indices.append(txt_idx)
            image_counts.append(img_count)
            text_counts.append(text_count)

        text_count_tensor = torch.tensor(text_counts, device=z_image.device)
        valid_count_tensor = valid_text_lengths.to(
            device=z_image.device, dtype=torch.long
        ).clamp(min=1, max=num_text)
        if self.text_selection_strategy == "valid_prefix_repeat":
            redundant_slots = (text_count_tensor - valid_count_tensor).clamp_min(0)
            represented_tokens = torch.minimum(
                text_count_tensor, valid_count_tensor
            ).clamp_min(1)
            repetition_factor = text_count_tensor.float() / represented_tokens.float()
        else:
            redundant_slots = torch.zeros_like(text_count_tensor)
            repetition_factor = torch.ones_like(text_count_tensor, dtype=torch.float32)
        metadata = {
            "image_indices": image_indices,
            "text_indices": text_indices,
            "image_counts": torch.tensor(image_counts, device=z_image.device),
            "text_counts": text_count_tensor,
            "valid_text_counts": valid_count_tensor,
            "text_redundant_slots": redundant_slots,
            "text_repetition_factor": repetition_factor,
            "num_common": torch.tensor(z_common.size(1), device=z_image.device),
            "num_image": torch.tensor(num_image, device=z_image.device),
            "num_text": torch.tensor(num_text, device=z_image.device),
            "predicted_text_quota": predicted_quota,
            "target_text_quota": target_quota,
        }
        return torch.stack(packed_rows), metadata

    @staticmethod
    def unpack(
        packed: Tensor,
        metadata: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, _, latent_dim = packed.shape
        num_common = int(metadata["num_common"].item())
        num_image = int(metadata["num_image"].item())
        num_text = int(metadata["num_text"].item())
        common = packed[:, :num_common]
        image = packed.new_zeros(batch, num_image, latent_dim)
        text = packed.new_zeros(batch, num_text, latent_dim)
        offset = num_common
        for index in range(batch):
            image_count = int(metadata["image_counts"][index].item())
            text_count = int(metadata["text_counts"][index].item())
            image[index, metadata["image_indices"][index]] = packed[
                index, offset:offset + image_count
            ]
            selected_text = packed[
                index, offset + image_count:offset + image_count + text_count
            ]
            selected_indices = metadata["text_indices"][index]
            # ``valid_prefix_repeat`` may transmit multiple independently
            # corrupted copies of one logical token.  Scatter-add and average
            # them.  For all legacy strategies every count is one, so this is
            # exactly equivalent to the previous indexed assignment.
            text[index].index_add_(0, selected_indices, selected_text)
            repeat_counts = packed.new_zeros(num_text)
            repeat_counts.index_add_(
                0,
                selected_indices,
                packed.new_ones(selected_indices.numel()),
            )
            text[index] = text[index] / repeat_counts.clamp_min(1.0).unsqueeze(-1)
        return common, image, text
