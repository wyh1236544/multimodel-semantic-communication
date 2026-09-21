"""Receiver-only policy for V6.4 ControlNet recovery and conservative fusion."""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .quality_adaptive_recovery import FEATURE_NAMES


class ControlNetFusionPolicy(nn.Module):
    """Predict bandwidth, route, diffusion strength, fusion type and alpha."""

    def __init__(
        self,
        hidden_dim: int = 96,
        min_strength: float = 0.20,
        max_strength: float = 0.70,
        min_blend_alpha: float = 0.10,
        max_blend_alpha: float = 0.55,
    ):
        super().__init__()
        self.min_strength = float(min_strength)
        self.max_strength = float(max_strength)
        self.min_blend_alpha = float(min_blend_alpha)
        self.max_blend_alpha = float(max_blend_alpha)
        feature_dim = len(FEATURE_NAMES)
        self.normalizer = nn.LayerNorm(feature_dim)
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mode_head = nn.Linear(hidden_dim, 1)
        self.text_head = nn.Linear(hidden_dim, 1)
        self.strength_head = nn.Linear(hidden_dim, 1)
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.detail_head = nn.Linear(hidden_dim, 1)
        self.quality_head = nn.Linear(hidden_dim, 1)
        for layer in (
            self.mode_head,
            self.text_head,
            self.strength_head,
            self.alpha_head,
            self.detail_head,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def _receiver_summaries(features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        snr = features[:, 0]
        text_confidence = (
            0.45 * features[:, 1]
            + 0.35 * features[:, 2]
            + 0.20 * features[:, 4]
        ).clamp(0.0, 1.0)
        image_quality = (
            0.20 * features[:, 5]
            + 0.25 * features[:, 18]
            + 0.25 * features[:, 6]
            + 0.10 * features[:, 7]
            + 0.10 * features[:, 8]
            + 0.10 * features[:, 10]
        ).clamp(0.0, 1.0)
        return snr, text_confidence, image_quality

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        hidden = self.backbone(self.normalizer(features))
        snr, text_confidence, image_quality = self._receiver_summaries(features)
        mode_logit = (
            6.0 * (0.52 - image_quality)
            + 3.0 * (text_confidence - 0.48)
            + 1.0 * (0.45 - snr)
            + self.mode_head(hidden).squeeze(-1)
        )
        text_logit = (
            5.0 * (0.62 - text_confidence)
            + 1.5 * (0.50 - image_quality)
            + 0.8 * (0.40 - snr)
            + self.text_head(hidden).squeeze(-1)
        )
        strength_fraction = torch.sigmoid(
            2.0 * (0.48 - image_quality)
            + 0.8 * (text_confidence - 0.50)
            + self.strength_head(hidden).squeeze(-1)
        )
        # The V6.3 pilot found conservative alpha around 0.25.  The learned
        # residual starts from that safe operating point.
        alpha_fraction = torch.sigmoid(
            -0.69 + self.alpha_head(hidden).squeeze(-1)
        )
        return {
            "generation_probability": torch.sigmoid(mode_logit),
            "text_utility": torch.sigmoid(text_logit),
            "diffusion_strength": self.min_strength
            + (self.max_strength - self.min_strength) * strength_fraction,
            "blend_alpha": self.min_blend_alpha
            + (self.max_blend_alpha - self.min_blend_alpha) * alpha_fraction,
            "detail_probability": torch.sigmoid(
                self.detail_head(hidden).squeeze(-1)
            ),
            "predicted_regression_quality": torch.sigmoid(
                self.quality_head(hidden).squeeze(-1)
            ),
        }


def _masked_smooth_l1(
    prediction: Tensor, target: Tensor, mask: Tensor
) -> Tensor:
    values = F.smooth_l1_loss(prediction, target.float(), reduction="none")
    mask = mask.float()
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def controlnet_policy_loss(
    probe_predictions: Mapping[str, Tensor],
    route_predictions: Mapping[str, Tensor],
    generation_target: Tensor,
    text_utility_target: Tensor,
    strength_target: Tensor,
    alpha_target: Tensor,
    detail_target: Tensor,
    regression_quality_target: Tensor,
    mode_positive_weight: Union[float, Tensor] = 1.0,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    generation_target = generation_target.float()
    positive_mask = generation_target > 0.5
    positive_weight = torch.as_tensor(
        mode_positive_weight,
        device=generation_target.device,
        dtype=generation_target.dtype,
    )
    mode_weights = torch.where(
        positive_mask, positive_weight, torch.ones_like(generation_target)
    )
    mode = F.binary_cross_entropy(
        route_predictions["generation_probability"],
        generation_target,
        weight=mode_weights,
    ) / mode_weights.mean().clamp_min(1e-6)
    text = F.smooth_l1_loss(
        probe_predictions["text_utility"], text_utility_target.float()
    )
    strength = _masked_smooth_l1(
        route_predictions["diffusion_strength"], strength_target, positive_mask
    )
    alpha = _masked_smooth_l1(
        route_predictions["blend_alpha"], alpha_target, positive_mask
    )
    detail_values = F.binary_cross_entropy(
        route_predictions["detail_probability"],
        detail_target.float(),
        reduction="none",
    )
    detail = (
        detail_values * positive_mask.float()
    ).sum() / positive_mask.float().sum().clamp_min(1.0)
    quality = F.smooth_l1_loss(
        route_predictions["predicted_regression_quality"],
        regression_quality_target.float(),
    )
    total = mode + 2.0 * text + strength + alpha + 0.5 * detail + 0.5 * quality
    return total, {
        "mode": mode,
        "text": text,
        "strength": strength,
        "alpha": alpha,
        "detail": detail,
        "quality": quality,
    }


def load_controlnet_policy(
    path: str, device: torch.device
) -> ControlNetFusionPolicy:
    checkpoint = torch.load(path, map_location=device)
    policy = ControlNetFusionPolicy(**checkpoint.get("policy_config", {})).to(device)
    policy.load_state_dict(checkpoint["policy"])
    policy.policy_data_image_ids = {
        int(value) for value in checkpoint.get("policy_data_image_ids", [])
    }
    policy.policy_train_image_ids = {
        int(value) for value in checkpoint.get("train_image_ids", [])
    }
    policy.policy_val_image_ids = {
        int(value) for value in checkpoint.get("val_image_ids", [])
    }
    return policy.eval()
