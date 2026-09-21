"""Learned post-generation verifier for V6.5 selective recovery."""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .quality_adaptive_recovery import FEATURE_NAMES
from utils.v65_selective import VERIFIER_FEATURE_NAMES


class SelectiveGenerationVerifier(nn.Module):
    """Predict source-supervised gains from receiver-only candidate evidence."""

    def __init__(self, hidden_dim: int = 96):
        super().__init__()
        route_dim = len(FEATURE_NAMES)
        verifier_dim = len(VERIFIER_FEATURE_NAMES)
        self.route_encoder = nn.Sequential(
            nn.LayerNorm(route_dim),
            nn.Linear(route_dim, hidden_dim),
            nn.SiLU(),
        )
        self.verifier_encoder = nn.Sequential(
            nn.LayerNorm(verifier_dim),
            nn.Linear(verifier_dim, hidden_dim),
            nn.SiLU(),
        )
        self.backbone = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.acceptance_head = nn.Linear(hidden_dim, 1)
        self.hallucination_head = nn.Linear(hidden_dim, 1)
        self.delta_clip_head = nn.Linear(hidden_dim, 1)
        self.delta_psnr_head = nn.Linear(hidden_dim, 1)
        self.delta_ssim_head = nn.Linear(hidden_dim, 1)
        self.semantic_gain_head = nn.Linear(hidden_dim, 1)
        self.psnr_safe_head = nn.Linear(hidden_dim, 1)
        self.ssim_safe_head = nn.Linear(hidden_dim, 1)

    def forward(self, route_features: Tensor, verifier_features: Tensor) -> Dict[str, Tensor]:
        route = self.route_encoder(route_features.float())
        evidence = self.verifier_encoder(verifier_features.float())
        hidden = self.backbone(torch.cat([route, evidence], dim=1))
        return {
            "acceptance_probability": torch.sigmoid(
                self.acceptance_head(hidden).squeeze(-1)
            ),
            "hallucination_probability": torch.sigmoid(
                self.hallucination_head(hidden).squeeze(-1)
            ),
            "predicted_delta_clip": 0.10 * torch.tanh(
                self.delta_clip_head(hidden).squeeze(-1)
            ),
            "predicted_delta_psnr": 5.0 * torch.tanh(
                self.delta_psnr_head(hidden).squeeze(-1)
            ),
            "predicted_delta_ssim": 0.10 * torch.tanh(
                self.delta_ssim_head(hidden).squeeze(-1)
            ),
            "semantic_gain_probability": torch.sigmoid(
                self.semantic_gain_head(hidden).squeeze(-1)
            ),
            "psnr_safe_probability": torch.sigmoid(
                self.psnr_safe_head(hidden).squeeze(-1)
            ),
            "ssim_safe_probability": torch.sigmoid(
                self.ssim_safe_head(hidden).squeeze(-1)
            ),
        }


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def verifier_loss(
    predictions: Mapping[str, Tensor],
    accept_target: Tensor,
    hallucination_target: Tensor,
    delta_clip_target: Tensor,
    delta_psnr_target: Tensor,
    delta_ssim_target: Tensor,
    positive_weight: Union[float, Tensor] = 1.0,
    deployment_weight: Optional[Tensor] = None,
    min_clip_gain: float = 0.003,
    max_psnr_drop: float = 0.05,
    max_ssim_drop: float = 0.004,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    accept_target = accept_target.float()
    hallucination_target = hallucination_target.float()
    weight = torch.as_tensor(
        positive_weight,
        device=accept_target.device,
        dtype=accept_target.dtype,
    )
    class_weight = torch.where(
        accept_target > 0.5, weight, torch.ones_like(accept_target)
    )
    if deployment_weight is None:
        deployment_weight = torch.ones_like(accept_target)
    deployment_weight = deployment_weight.float().clamp_min(0.0)
    sample_weight = class_weight * deployment_weight
    acceptance_values = F.binary_cross_entropy(
        predictions["acceptance_probability"],
        accept_target,
        reduction="none",
    )
    acceptance = _weighted_mean(acceptance_values, sample_weight)
    hallucination_values = F.binary_cross_entropy(
        predictions["hallucination_probability"],
        hallucination_target,
        reduction="none",
    )
    hallucination = _weighted_mean(hallucination_values, deployment_weight)
    delta_clip_values = F.smooth_l1_loss(
        predictions["predicted_delta_clip"] / 0.10,
        delta_clip_target.float().clamp(-0.10, 0.10) / 0.10,
        reduction="none",
    )
    delta_clip = _weighted_mean(delta_clip_values, deployment_weight)
    delta_psnr_values = F.smooth_l1_loss(
        predictions["predicted_delta_psnr"] / 5.0,
        delta_psnr_target.float().clamp(-5.0, 5.0) / 5.0,
        reduction="none",
    )
    delta_psnr = _weighted_mean(delta_psnr_values, deployment_weight)
    delta_ssim_values = F.smooth_l1_loss(
        predictions["predicted_delta_ssim"] / 0.10,
        delta_ssim_target.float().clamp(-0.10, 0.10) / 0.10,
        reduction="none",
    )
    delta_ssim = _weighted_mean(delta_ssim_values, deployment_weight)
    semantic_target = (delta_clip_target.float() >= float(min_clip_gain)).float()
    psnr_safe_target = (delta_psnr_target.float() >= -float(max_psnr_drop)).float()
    ssim_safe_target = (delta_ssim_target.float() >= -float(max_ssim_drop)).float()
    semantic = _weighted_mean(
        F.binary_cross_entropy(
            predictions["semantic_gain_probability"], semantic_target, reduction="none"
        ),
        deployment_weight,
    )
    psnr_safe = _weighted_mean(
        F.binary_cross_entropy(
            predictions["psnr_safe_probability"], psnr_safe_target, reduction="none"
        ),
        deployment_weight,
    )
    ssim_safe = _weighted_mean(
        F.binary_cross_entropy(
            predictions["ssim_safe_probability"], ssim_safe_target, reduction="none"
        ),
        deployment_weight,
    )
    total = (
        1.5 * acceptance
        + hallucination
        + semantic
        + psnr_safe
        + ssim_safe
        + 1.5 * delta_clip
        + delta_psnr
        + delta_ssim
    )
    return total, {
        "acceptance": acceptance,
        "hallucination": hallucination,
        "delta_clip": delta_clip,
        "delta_psnr": delta_psnr,
        "delta_ssim": delta_ssim,
        "semantic": semantic,
        "psnr_safe": psnr_safe,
        "ssim_safe": ssim_safe,
    }


def load_generation_verifier(path: str, device: torch.device) -> SelectiveGenerationVerifier:
    checkpoint = torch.load(path, map_location=device)
    model = SelectiveGenerationVerifier(**checkpoint.get("verifier_config", {})).to(device)
    model.load_state_dict(checkpoint["verifier"])
    model.verifier_data_image_ids = {
        int(value) for value in checkpoint.get("verifier_data_image_ids", [])
    }
    model.verifier_train_image_ids = {
        int(value) for value in checkpoint.get("train_image_ids", [])
    }
    model.verifier_val_image_ids = {
        int(value) for value in checkpoint.get("val_image_ids", [])
    }
    model.all_training_image_ids = {
        int(value) for value in checkpoint.get(
            "all_training_image_ids", checkpoint.get("verifier_data_image_ids", [])
        )
    }
    return model.eval()
