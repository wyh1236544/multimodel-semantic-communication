"""Quality-aware control modules for V6.1/V6.2 semantic communication.

The module deliberately has no dependency on diffusers.  It can therefore be
loaded together with the existing V6.1 checkpoint and trained separately.  A
receiver first extracts no-reference indicators from the regression output,
then predicts three decisions:

* how much of the fixed modality-token budget should be assigned to text;
* whether generative recovery is likely to beat regression recovery;
* the img2img strength to use when generation is selected.

Ground-truth images are used only by the data-collection script to supervise
these decisions.  ``receiver_features`` never reads the ground-truth image.
"""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


FEATURE_NAMES: List[str] = [
    "snr_normalized",
    "text_top1_probability",
    "text_entropy_confidence",
    "text_logit_margin",
    "global_image_caption_agreement",
    "cross_modal_semantic_cosine",
    "laplacian_sharpness",
    "total_variation",
    "color_variance",
    "dark_or_bright_fraction",
    "image_private_reliability",
    "text_private_reliability",
    "allocated_text_fraction",
    "received_latent_rms",
    "decoded_common_rms",
    "residual_gate_mean",
    "valid_text_fraction",
    "text_repetition_gain",
    "received_image_text_clip_alignment",
]


def _as_batch(value: Union[float, int, Tensor], reference: Tensor) -> Tensor:
    result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if result.ndim == 0:
        result = result.expand(reference.size(0))
    return result.reshape(reference.size(0))


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _safe_scalar_batch(value: Optional[Tensor], reference: Tensor, default: float) -> Tensor:
    if value is None:
        return reference.new_full((reference.size(0),), default)
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        return value.expand(reference.size(0))
    if value.size(0) != reference.size(0):
        return value.mean().expand(reference.size(0))
    return value.reshape(value.size(0), -1).mean(dim=1)


@torch.no_grad()
def receiver_features(
    outputs: Mapping[str, Tensor],
    snr_db: Union[float, int, Tensor],
    max_text_tokens: int = 64,
    image_text_alignment: Optional[Tensor] = None,
) -> Tensor:
    """Extract per-image no-reference features available at the receiver.

    The feature vector combines decoded-caption uncertainty, cross-modal
    agreement, simple natural-image statistics and semantic-channel state.
    This is intentionally lightweight: it does not require another IQA model.
    """
    image = outputs["image"].detach().float().clamp(0.0, 1.0)
    logits = outputs["text_logits"].detach().float()
    # Infer the received caption length from its own EOS prediction.  Do not
    # use ``text_attention_mask`` here: it belongs to the source caption and
    # would leak ground-truth length into the online receiver policy.
    predicted_ids = logits.argmax(dim=-1)
    eos_value = outputs.get("text_eos_token_id")
    if eos_value is not None:
        eos_id = int(torch.as_tensor(eos_value).reshape(-1)[0].item())
        positions = torch.arange(logits.size(1), device=logits.device)[None, :]
        eos_positions = torch.where(
            predicted_ids.eq(eos_id),
            positions.expand_as(predicted_ids),
            positions.new_full(predicted_ids.shape, logits.size(1)),
        )
        first_eos = eos_positions.min(dim=1).values
        mask = positions <= first_eos[:, None]
    else:
        mask = torch.ones(logits.shape[:2], device=logits.device, dtype=torch.bool)

    probabilities = logits.softmax(dim=-1)
    top2 = probabilities.topk(k=2, dim=-1).values
    top1_probability = _masked_mean(top2[..., 0], mask)
    margin = _masked_mean(top2[..., 0] - top2[..., 1], mask)
    entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1)
    entropy_confidence = 1.0 - _masked_mean(entropy, mask) / math.log(
        max(logits.size(-1), 2)
    )

    global_logits = outputs.get("text_logits_global")
    image_logits = outputs.get("image_conditioned_text_logits")
    if global_logits is not None and image_logits is not None:
        length = min(global_logits.size(1), image_logits.size(1), mask.size(1))
        agreement_values = (
            global_logits[:, :length].argmax(-1)
            == image_logits[:, :length].argmax(-1)
        ).float()
        agreement = _masked_mean(agreement_values, mask[:, :length])
    else:
        agreement = image.new_zeros(image.size(0))

    receiver_semantic = outputs.get("receiver_text_semantic")
    image_semantic = outputs.get("image_conditioned_text_semantic")
    if receiver_semantic is not None and image_semantic is not None:
        semantic_cosine = F.cosine_similarity(
            receiver_semantic.float(), image_semantic.float(), dim=-1
        ).clamp(-1.0, 1.0)
        semantic_cosine = 0.5 * (semantic_cosine + 1.0)
    else:
        semantic_cosine = image.new_zeros(image.size(0))

    gray = image.mean(dim=1, keepdim=True)
    laplace_kernel = image.new_tensor(
        [[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]]
    )
    laplace = F.conv2d(gray, laplace_kernel, padding=1)
    # Smooth bounded scalings are easier for the small policy MLP to learn.
    sharpness = torch.tanh(20.0 * laplace.flatten(1).var(dim=1))
    horizontal = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean((1, 2, 3))
    vertical = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean((1, 2, 3))
    total_variation = torch.tanh(5.0 * (horizontal + vertical))
    color_variance = torch.tanh(10.0 * image.flatten(2).var(dim=2).mean(dim=1))
    clipped_fraction = ((image < 0.02) | (image > 0.98)).float().mean((1, 2, 3))

    image_reliability = _safe_scalar_batch(
        outputs.get("image_private_reliability"), image, 1.0
    )
    text_reliability = _safe_scalar_batch(
        outputs.get("text_private_reliability"), image, 1.0
    )
    text_count = _safe_scalar_batch(
        outputs.get("allocated_text_tokens"), image, float(max_text_tokens)
    )
    text_fraction = (text_count / max(float(max_text_tokens), 1.0)).clamp(0.0, 1.0)
    valid_text_count = mask.sum(dim=1).to(dtype=image.dtype)
    valid_text_fraction = (
        valid_text_count / max(float(max_text_tokens), 1.0)
    ).clamp(0.0, 1.0)
    represented_text = torch.minimum(text_count, valid_text_count).clamp_min(1.0)
    repetition_factor = text_count / represented_text
    repetition_gain = torch.tanh((repetition_factor - 1.0).clamp_min(0.0) / 3.0)
    if image_text_alignment is None:
        clip_alignment = image.new_full((image.size(0),), 0.5)
    else:
        clip_alignment = image_text_alignment.to(
            device=image.device, dtype=image.dtype
        ).reshape(image.size(0))
        clip_alignment = 0.5 * (clip_alignment.clamp(-1.0, 1.0) + 1.0)
    received_rms = outputs["z_shared_noisy"].detach().float().square().mean((1, 2)).sqrt()
    received_rms = torch.tanh(received_rms)
    common_rms = outputs["decoded_common_tokens"].detach().float().square().mean((1, 2)).sqrt()
    common_rms = torch.tanh(common_rms / 4.0)
    residual_gate = _safe_scalar_batch(outputs.get("residual_gate"), image, 0.0)

    snr = ((_as_batch(snr_db, image) + 10.0) / 40.0).clamp(0.0, 1.0)
    return torch.stack(
        [
            snr,
            top1_probability,
            entropy_confidence,
            margin,
            agreement,
            semantic_cosine,
            sharpness,
            total_variation,
            color_variance,
            clipped_fraction,
            image_reliability,
            text_reliability,
            text_fraction,
            received_rms,
            common_rms,
            residual_gate,
            valid_text_fraction,
            repetition_gain,
            clip_alignment,
        ],
        dim=1,
    ).clamp(0.0, 1.0)


class QualityAdaptivePolicy(nn.Module):
    """Small receiver policy with a safe, useful heuristic initialization."""

    def __init__(self, hidden_dim: int = 64, min_strength: float = 0.10, max_strength: float = 0.35):
        super().__init__()
        self.min_strength = float(min_strength)
        self.max_strength = float(max_strength)
        feature_dim = len(FEATURE_NAMES)
        self.normalizer = nn.LayerNorm(feature_dim)
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mode_residual = nn.Linear(hidden_dim, 1)
        self.text_residual = nn.Linear(hidden_dim, 1)
        self.strength_residual = nn.Linear(hidden_dim, 1)
        self.quality_head = nn.Linear(hidden_dim, 1)
        for layer in (self.mode_residual, self.text_residual, self.strength_residual):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        hidden = self.backbone(self.normalizer(features))
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

        # Generation is useful only when the regression image is weak *and*
        # the recovered language is trustworthy enough to constrain diffusion.
        heuristic_mode_logit = (
            7.0 * (0.52 - image_quality)
            + 3.0 * (text_confidence - 0.48)
            + 1.5 * (0.45 - snr)
        )
        generation_probability = torch.sigmoid(
            heuristic_mode_logit + self.mode_residual(hidden).squeeze(-1)
        )

        # Low caption confidence increases text quota; high image damage adds a
        # smaller increment because generation needs reliable language.
        heuristic_text_logit = (
            5.0 * (0.62 - text_confidence)
            + 1.5 * (0.50 - image_quality)
            + 0.8 * (0.40 - snr)
        )
        text_utility = torch.sigmoid(
            heuristic_text_logit + self.text_residual(hidden).squeeze(-1)
        )
        strength_fraction = torch.sigmoid(
            2.5 * (0.45 - image_quality)
            + 1.0 * (text_confidence - 0.5)
            + self.strength_residual(hidden).squeeze(-1)
        )
        strength = self.min_strength + (
            self.max_strength - self.min_strength
        ) * strength_fraction
        predicted_quality = torch.sigmoid(self.quality_head(hidden).squeeze(-1))
        return {
            "generation_probability": generation_probability,
            "text_utility": text_utility,
            "diffusion_strength": strength,
            "predicted_regression_quality": predicted_quality,
        }


class SemanticConditionAdapter(nn.Module):
    """Project received V6.1 semantics into Stable-Diffusion context tokens.

    The adapter is the only module that needs denoising training.  Stable
    Diffusion, V6.1, CLIP and the VAE can all remain frozen.
    """

    def __init__(
        self,
        semantic_dim: int = 256,
        cross_attention_dim: int = 768,
        num_condition_tokens: int = 8,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.num_condition_tokens = int(num_condition_tokens)
        self.cross_attention_dim = int(cross_attention_dim)
        self.input_norms = nn.ModuleList([nn.LayerNorm(semantic_dim) for _ in range(3)])
        self.projector = nn.Sequential(
            nn.Linear(semantic_dim * 3 + len(FEATURE_NAMES), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_condition_tokens * self.cross_attention_dim),
        )
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, outputs: Mapping[str, Tensor], features: Tensor) -> Tensor:
        pooled = []
        for norm, name in zip(
            self.input_norms,
            ("decoded_common_tokens", "decoded_image_tokens", "decoded_text_tokens"),
        ):
            pooled.append(norm(outputs[name].float()).mean(dim=1))
        condition = self.projector(torch.cat(pooled + [features.float()], dim=1))
        condition = condition.reshape(
            features.size(0), self.num_condition_tokens, self.cross_attention_dim
        )
        return torch.sigmoid(self.gate_logit) * condition


def policy_loss(
    predictions: Mapping[str, Tensor],
    generation_target: Tensor,
    text_utility_target: Tensor,
    strength_target: Tensor,
    regression_quality_target: Optional[Tensor] = None,
    mode_positive_weight: Union[float, Tensor] = 1.0,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    generation_target = generation_target.float()
    positive_weight = torch.as_tensor(
        mode_positive_weight,
        device=generation_target.device,
        dtype=generation_target.dtype,
    )
    mode_weights = torch.where(
        generation_target > 0.5,
        positive_weight,
        torch.ones_like(generation_target),
    )
    mode = F.binary_cross_entropy(
        predictions["generation_probability"],
        generation_target,
        weight=mode_weights,
    )
    mode = mode / mode_weights.mean().clamp_min(1e-6)
    text = F.smooth_l1_loss(predictions["text_utility"], text_utility_target.float())
    strength = F.smooth_l1_loss(
        predictions["diffusion_strength"], strength_target.float()
    )
    if regression_quality_target is None:
        quality = mode.new_zeros(())
    else:
        quality = F.smooth_l1_loss(
            predictions["predicted_regression_quality"],
            regression_quality_target.float(),
        )
    total = mode + 2.0 * text + strength + 0.5 * quality
    return total, {"mode": mode, "text": text, "strength": strength, "quality": quality}


def load_policy_checkpoint(path: str, device: torch.device) -> QualityAdaptivePolicy:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("policy_config", {}) if isinstance(checkpoint, dict) else {}
    policy = QualityAdaptivePolicy(**config).to(device)
    state = checkpoint.get("policy", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    policy.load_state_dict(state)
    if isinstance(checkpoint, dict):
        policy.policy_data_image_ids = {
            int(image_id)
            for image_id in checkpoint.get("policy_data_image_ids", [])
        }
        policy.policy_train_image_ids = {
            int(image_id)
            for image_id in checkpoint.get("train_image_ids", [])
        }
        policy.policy_val_image_ids = {
            int(image_id)
            for image_id in checkpoint.get("val_image_ids", [])
        }
    policy.eval()
    return policy
