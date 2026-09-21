"""Common-guided receiver-only candidate verifier for V9-B."""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .quality_adaptive_recovery import FEATURE_NAMES
from utils.v65_selective import VERIFIER_FEATURE_NAMES
from utils.v9b_selective import PROMPT_FEATURE_NAMES

MODE_FEATURE_NAMES = ["generation", "hybrid"]


class V9BCommonGuidedVerifier(nn.Module):
    """Predict whether a diffusion candidate should replace the regression.

    Inputs are available at the receiver. Common semantics are projected into
    the CLIP comparison space inside the verifier; source images/captions never
    enter this forward pass.
    """

    def __init__(
        self,
        common_dim: int = 256,
        clip_dim: int = 512,
        hidden_dim: int = 96,
    ):
        super().__init__()
        self.common_dim = int(common_dim)
        self.clip_dim = int(clip_dim)
        self.hidden_dim = int(hidden_dim)
        self.route_encoder = self._encoder(len(FEATURE_NAMES), hidden_dim)
        self.evidence_encoder = self._encoder(len(VERIFIER_FEATURE_NAMES), hidden_dim)
        self.prompt_quality_encoder = self._encoder(len(PROMPT_FEATURE_NAMES), hidden_dim)
        self.mode_encoder = self._encoder(len(MODE_FEATURE_NAMES), hidden_dim)
        self.common_encoder = self._encoder(common_dim, hidden_dim)
        self.common_clip_projector = nn.Sequential(
            nn.LayerNorm(common_dim),
            nn.Linear(common_dim, clip_dim),
        )
        self.common_relation_encoder = self._encoder(3, hidden_dim)
        self.regression_encoder = self._encoder(clip_dim, hidden_dim)
        self.candidate_encoder = self._encoder(clip_dim, hidden_dim)
        self.prompt_encoder = self._encoder(clip_dim, hidden_dim)
        # Eight base embeddings, four interactions and explicit Common/CLIP
        # similarities.  The latter receive direct CLIP-teacher supervision.
        self.backbone = nn.Sequential(
            nn.Linear(13 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
        )
        # Generation and Hybrid have materially different candidate
        # distributions.  Dedicated experts make the separation real (V9-B.1
        # only had separate final logits on top of a shared representation).
        self.mode_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(0.10),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
                for _ in MODE_FEATURE_NAMES
            ]
        )
        # Separate logits prevent the Generation-heavy operating region from
        # forcing the Hybrid candidates onto the same probability scale.
        self.acceptance_heads = nn.Linear(hidden_dim, len(MODE_FEATURE_NAMES))
        self.hallucination_heads = nn.Linear(hidden_dim, len(MODE_FEATURE_NAMES))
        self.semantic_gain_head = nn.Linear(hidden_dim, 1)
        self.psnr_safe_head = nn.Linear(hidden_dim, 1)
        self.ssim_safe_head = nn.Linear(hidden_dim, 1)
        self.delta_semantic_head = nn.Linear(hidden_dim, 1)
        self.delta_psnr_head = nn.Linear(hidden_dim, 1)
        self.delta_ssim_head = nn.Linear(hidden_dim, 1)
        self.utility_heads = nn.Linear(hidden_dim, len(MODE_FEATURE_NAMES))

    @staticmethod
    def _encoder(input_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.SiLU())

    def forward(
        self,
        route_features: Tensor,
        verifier_features: Tensor,
        prompt_quality: Tensor,
        mode_features: Tensor,
        common_semantic: Tensor,
        regression_clip: Tensor,
        candidate_clip: Tensor,
        prompt_clip: Tensor,
        common_ablation: bool = False,
        common_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        route = self.route_encoder(route_features.float())
        evidence = self.evidence_encoder(verifier_features.float())
        quality = self.prompt_quality_encoder(prompt_quality.float())
        mode_weights = mode_features.float().clamp_min(0.0)
        mode_weights = mode_weights / mode_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        mode = self.mode_encoder(mode_weights)
        common_input = common_semantic.float()
        if common_mask is None:
            common_mask = common_input.new_ones((common_input.size(0), 1))
        else:
            common_mask = common_mask.to(common_input).reshape(-1, 1)
        if common_ablation:
            common_mask = torch.zeros_like(common_mask)
        common_input = common_input * common_mask
        common = self.common_encoder(common_input) * common_mask
        common_clip = F.normalize(
            self.common_clip_projector(common_input), dim=-1
        ) * common_mask
        regression = self.regression_encoder(regression_clip.float())
        candidate = self.candidate_encoder(candidate_clip.float())
        prompt = self.prompt_encoder(prompt_clip.float())
        common_relations = torch.stack(
            [
                (common_clip * regression_clip.float()).sum(dim=-1),
                (common_clip * candidate_clip.float()).sum(dim=-1),
                (common_clip * prompt_clip.float()).sum(dim=-1),
            ],
            dim=1,
        )
        relation = self.common_relation_encoder(
            0.5 * (common_relations.clamp(-1.0, 1.0) + 1.0)
        ) * common_mask
        interactions = [
            common * regression,
            common * candidate,
            regression * prompt,
            candidate * prompt,
        ]
        shared = self.backbone(
            torch.cat(
                [
                    route, evidence, quality, mode, common, regression,
                    candidate, prompt, relation,
                ] + interactions,
                dim=1,
            )
        )
        expert_hidden = torch.stack(
            [expert(shared) for expert in self.mode_experts], dim=1
        )
        hidden = (expert_hidden * mode_weights.unsqueeze(-1)).sum(dim=1)
        acceptance_by_mode = torch.sigmoid(self.acceptance_heads(hidden))
        hallucination_by_mode = torch.sigmoid(self.hallucination_heads(hidden))
        utility_by_mode = 0.25 * torch.tanh(self.utility_heads(hidden))
        return {
            "acceptance_probability": (acceptance_by_mode * mode_weights).sum(dim=1),
            "hallucination_probability": (hallucination_by_mode * mode_weights).sum(dim=1),
            "acceptance_probability_by_mode": acceptance_by_mode,
            "hallucination_probability_by_mode": hallucination_by_mode,
            "predicted_utility": (utility_by_mode * mode_weights).sum(dim=1),
            "predicted_utility_by_mode": utility_by_mode,
            "semantic_gain_probability": torch.sigmoid(self.semantic_gain_head(hidden).squeeze(-1)),
            "psnr_safe_probability": torch.sigmoid(self.psnr_safe_head(hidden).squeeze(-1)),
            "ssim_safe_probability": torch.sigmoid(self.ssim_safe_head(hidden).squeeze(-1)),
            "predicted_delta_semantic": 0.20 * torch.tanh(self.delta_semantic_head(hidden).squeeze(-1)),
            "predicted_delta_psnr": 3.0 * torch.tanh(self.delta_psnr_head(hidden).squeeze(-1)),
            "predicted_delta_ssim": 0.20 * torch.tanh(self.delta_ssim_head(hidden).squeeze(-1)),
            "projected_common_clip": common_clip,
            "common_regression_similarity": common_relations[:, 0],
            "common_candidate_similarity": common_relations[:, 1],
            "common_prompt_similarity": common_relations[:, 2],
            "common_present": common_mask.squeeze(-1),
        }


def v9b_verifier_loss(
    prediction: Mapping[str, Tensor],
    accept: Tensor,
    hallucination: Tensor,
    semantic_gain: Tensor,
    delta_psnr: Tensor,
    delta_ssim: Tensor,
    common_target_clip: Tensor,
    mode_features: Tensor,
    positive_weight: float = 1.0,
    common_alignment_weight: float = 1.0,
    generation_max_psnr_drop: float = 1.0,
    hybrid_max_psnr_drop: float = 0.75,
    generation_max_ssim_drop: float = 0.10,
    hybrid_max_ssim_drop: float = 0.08,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    accept = accept.float()
    hallucination = hallucination.float()
    semantic_positive = (semantic_gain.float() > 0.0).float()
    mode_features = mode_features.float()
    psnr_limit = (
        mode_features[:, 0] * float(generation_max_psnr_drop)
        + mode_features[:, 1] * float(hybrid_max_psnr_drop)
    ).clamp_min(1e-6)
    ssim_limit = (
        mode_features[:, 0] * float(generation_max_ssim_drop)
        + mode_features[:, 1] * float(hybrid_max_ssim_drop)
    ).clamp_min(1e-6)
    psnr_safe = (delta_psnr.float() >= -psnr_limit).float()
    ssim_safe = (delta_ssim.float() >= -ssim_limit).float()
    weights = torch.where(
        accept > 0.5,
        accept.new_full(accept.shape, float(positive_weight)),
        torch.ones_like(accept),
    )
    acceptance = F.binary_cross_entropy(
        prediction["acceptance_probability"], accept, weight=weights
    ) / weights.mean().clamp_min(1e-6)
    hall = F.binary_cross_entropy(prediction["hallucination_probability"], hallucination)
    semantic_class = F.binary_cross_entropy(
        prediction["semantic_gain_probability"], semantic_positive
    )
    psnr_class = F.binary_cross_entropy(prediction["psnr_safe_probability"], psnr_safe)
    ssim_class = F.binary_cross_entropy(prediction["ssim_safe_probability"], ssim_safe)
    semantic_reg = F.smooth_l1_loss(
        prediction["predicted_delta_semantic"] / 0.20,
        semantic_gain.float().clamp(-0.20, 0.20) / 0.20,
    )
    psnr_reg = F.smooth_l1_loss(
        prediction["predicted_delta_psnr"] / 3.0,
        delta_psnr.float().clamp(-3.0, 3.0) / 3.0,
    )
    ssim_reg = F.smooth_l1_loss(
        prediction["predicted_delta_ssim"] / 0.20,
        delta_ssim.float().clamp(-0.20, 0.20) / 0.20,
    )
    common_distance = 1.0 - F.cosine_similarity(
        prediction["projected_common_clip"], common_target_clip.float(), dim=-1
    )
    common_present = prediction["common_present"].float()
    common_alignment = (common_distance * common_present).sum() / common_present.sum().clamp_min(1.0)
    # Smooth decision utility supplies substantially more information than a
    # single binary label.  Pixel penalties only start beyond each mode's own
    # safety budget, matching the collector and oracle definitions.
    utility_target = semantic_gain.float()
    utility_target = utility_target - 0.05 * (
        (-delta_psnr.float() - psnr_limit).clamp_min(0.0) / psnr_limit
    )
    utility_target = utility_target - 0.05 * (
        (-delta_ssim.float() - ssim_limit).clamp_min(0.0) / ssim_limit
    )
    utility_target = utility_target.clamp(-0.25, 0.25)
    utility_reg = F.smooth_l1_loss(
        prediction["predicted_utility"] / 0.25, utility_target / 0.25
    )
    total = (
        2.0 * acceptance + hall + semantic_class + 0.5 * psnr_class
        + 0.5 * ssim_class + semantic_reg + 0.5 * psnr_reg + 0.5 * ssim_reg
        + 2.0 * utility_reg
        + float(common_alignment_weight) * common_alignment
    )
    return total, {
        "acceptance": acceptance,
        "hallucination": hall,
        "semantic_class": semantic_class,
        "semantic_reg": semantic_reg,
        "utility_reg": utility_reg,
        "utility_target_mean": utility_target.mean(),
        "common_alignment": common_alignment,
    }


def load_v9b_verifier(path: str, device: torch.device) -> tuple[V9BCommonGuidedVerifier, dict]:
    checkpoint = torch.load(path, map_location=device)
    model = V9BCommonGuidedVerifier(**checkpoint.get("verifier_config", {})).to(device)
    model.load_state_dict(checkpoint["verifier"])
    model.eval()
    return model, checkpoint
