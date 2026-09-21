"""Receiver-only safety helpers for V6.5 selective generative recovery.

Ground-truth images are deliberately absent from this module.  The functions
operate on the regression reconstruction, the generated candidate, received
caption alignment and channel/decoder features that are available online.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


VERIFIER_FEATURE_NAMES = [
    "candidate_regression_clip_similarity",
    "raw_regression_clip_similarity",
    "regression_text_alignment",
    "candidate_text_alignment",
    "text_alignment_gain",
    "candidate_regression_l1",
    "candidate_regression_rmse",
    "low_frequency_l1",
    "edge_cosine",
    "edge_l1",
    "mean_color_shift",
    "std_color_shift",
    "clipped_fraction_shift",
    "total_variation_shift",
    "raw_candidate_l1",
    "diffusion_strength",
    "blend_alpha",
    "detail_probability",
    "primary_generation_probability",
]


DEFAULT_VERIFIER_THRESHOLDS = {
    "acceptance_threshold": 0.65,
    "max_hallucination_probability": 0.40,
    "min_text_alignment_gain": 0.0,
    "min_edge_cosine": 0.80,
    "max_low_frequency_l1": 0.08,
    "min_semantic_gain_probability": 0.50,
    "min_psnr_safe_probability": 0.50,
    "min_ssim_safe_probability": 0.50,
}


def _batch_scalar(value: Union[float, Tensor], reference: Tensor) -> Tensor:
    result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if result.ndim == 0:
        result = result.expand(reference.size(0))
    return result.reshape(reference.size(0))


def _text_confidence(features: Tensor) -> Tensor:
    return (
        0.45 * features[:, 1]
        + 0.35 * features[:, 2]
        + 0.20 * features[:, 4]
    ).clamp(0.0, 1.0)


def received_text_quality(features: Tensor) -> Tensor:
    """No-reference confidence used to select probe or protected caption."""
    return (
        0.28 * features[:, 1]
        + 0.22 * features[:, 2]
        + 0.12 * features[:, 3]
        + 0.12 * features[:, 4]
        + 0.12 * features[:, 5]
        + 0.08 * features[:, 10]
        + 0.06 * features[:, 18]
    ).clamp(0.0, 1.0)


@torch.no_grad()
def select_received_text(
    probe_features: Tensor,
    protected_features: Tensor,
    minimum_gain: float = 0.005,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Choose protected text only when its receiver confidence is higher."""
    probe_quality = received_text_quality(probe_features)
    protected_quality = received_text_quality(protected_features)
    choose_protected = protected_quality >= probe_quality + float(minimum_gain)
    return choose_protected, probe_quality, protected_quality


@torch.no_grad()
def hybrid_text_utility(
    features: Tensor,
    learned_utility: Tensor,
    heuristic_utility: Tensor,
    snr_db: Union[float, Tensor],
    probe_utility: float = 0.5,
    generation_probability: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Conservative text UEP with confidence, channel and generation floors.

    The recommendation never falls below the probe allocation.  Low received
    text confidence and very noisy channels increase protection.  A candidate
    generation decision also raises the floor because diffusion relies on the
    received caption and semantic tokens.
    """
    learned = learned_utility.float().clamp(0.0, 1.0)
    heuristic = heuristic_utility.float().clamp(0.0, 1.0)
    snr = _batch_scalar(snr_db, learned)
    confidence = _text_confidence(features)

    channel_floor = torch.where(
        snr <= -2.5,
        learned.new_full(learned.shape, 0.80),
        torch.where(
            snr <= 2.5,
            learned.new_full(learned.shape, 0.70),
            torch.where(
                snr <= 7.5,
                learned.new_full(learned.shape, 0.60),
                torch.where(
                    snr <= 12.5,
                    learned.new_full(learned.shape, 0.50),
                    learned.new_full(learned.shape, float(probe_utility)),
                ),
            ),
        ),
    )
    confidence_floor = (
        float(probe_utility) + 0.75 * (0.58 - confidence).clamp_min(0.0)
    ).clamp(0.0, 0.90)
    if generation_probability is None:
        generation_floor = learned.new_full(learned.shape, float(probe_utility))
    else:
        generation_probability = generation_probability.float().clamp(0.0, 1.0)
        generation_floor = torch.where(
            generation_probability >= 0.65,
            learned.new_full(learned.shape, 0.75),
            learned.new_full(learned.shape, float(probe_utility)),
        )
    recommended = torch.stack(
        [
            learned,
            heuristic,
            channel_floor,
            confidence_floor,
            generation_floor,
            learned.new_full(learned.shape, float(probe_utility)),
        ],
        dim=0,
    ).max(dim=0).values.clamp(0.0, 1.0)
    return recommended, {
        "text_confidence": confidence,
        "channel_floor": channel_floor,
        "confidence_floor": confidence_floor,
        "generation_floor": generation_floor,
    }


def _gradient_magnitude(image: Tensor) -> Tensor:
    gray = image.mean(dim=1, keepdim=True)
    dx = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
    return (dx.square() + dy.square() + 1e-8).sqrt()


def _total_variation(image: Tensor) -> Tensor:
    horizontal = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean((1, 2, 3))
    vertical = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean((1, 2, 3))
    return horizontal + vertical


@torch.no_grad()
def verification_features(
    regression: Tensor,
    candidate: Tensor,
    raw_generation: Tensor,
    candidate_regression_clip: Tensor,
    raw_regression_clip: Tensor,
    regression_text_alignment: Tensor,
    candidate_text_alignment: Tensor,
    diffusion_strength: Union[float, Tensor],
    blend_alpha: Union[float, Tensor],
    detail_probability: Union[float, Tensor],
    primary_generation_probability: Union[float, Tensor],
) -> Tensor:
    """Extract post-generation evidence without reading the source image."""
    regression = regression.float().clamp(0.0, 1.0)
    candidate = candidate.float().clamp(0.0, 1.0)
    raw_generation = raw_generation.float().clamp(0.0, 1.0)
    if regression.ndim == 3:
        regression = regression.unsqueeze(0)
    if candidate.ndim == 3:
        candidate = candidate.unsqueeze(0)
    if raw_generation.ndim == 3:
        raw_generation = raw_generation.unsqueeze(0)
    difference = candidate - regression
    l1 = difference.abs().mean((1, 2, 3))
    rmse = difference.square().mean((1, 2, 3)).sqrt()
    low_regression = F.avg_pool2d(regression, kernel_size=9, stride=1, padding=4)
    low_candidate = F.avg_pool2d(candidate, kernel_size=9, stride=1, padding=4)
    low_frequency_l1 = (low_candidate - low_regression).abs().mean((1, 2, 3))
    regression_edges = _gradient_magnitude(regression)
    candidate_edges = _gradient_magnitude(candidate)
    edge_cosine = F.cosine_similarity(
        regression_edges.flatten(1), candidate_edges.flatten(1), dim=1
    ).clamp(-1.0, 1.0)
    edge_l1 = (candidate_edges - regression_edges).abs().mean((1, 2, 3))
    mean_shift = (
        candidate.mean((2, 3)) - regression.mean((2, 3))
    ).abs().mean(dim=1)
    std_shift = (
        candidate.flatten(2).std(dim=2) - regression.flatten(2).std(dim=2)
    ).abs().mean(dim=1)
    clipped_regression = ((regression < 0.02) | (regression > 0.98)).float().mean((1, 2, 3))
    clipped_candidate = ((candidate < 0.02) | (candidate > 0.98)).float().mean((1, 2, 3))
    clipped_shift = (clipped_candidate - clipped_regression).abs()
    tv_shift = (_total_variation(candidate) - _total_variation(regression)).abs()
    raw_candidate_l1 = (raw_generation - candidate).abs().mean((1, 2, 3))

    clip_pair = 0.5 * (_batch_scalar(candidate_regression_clip, l1).clamp(-1.0, 1.0) + 1.0)
    raw_pair = 0.5 * (_batch_scalar(raw_regression_clip, l1).clamp(-1.0, 1.0) + 1.0)
    regression_alignment = _batch_scalar(regression_text_alignment, l1).clamp(-1.0, 1.0)
    candidate_alignment = _batch_scalar(candidate_text_alignment, l1).clamp(-1.0, 1.0)
    alignment_gain = candidate_alignment - regression_alignment
    values = [
        clip_pair,
        raw_pair,
        0.5 * (regression_alignment + 1.0),
        0.5 * (candidate_alignment + 1.0),
        (0.5 + 5.0 * alignment_gain).clamp(0.0, 1.0),
        torch.tanh(5.0 * l1),
        torch.tanh(5.0 * rmse),
        torch.tanh(5.0 * low_frequency_l1),
        0.5 * (edge_cosine + 1.0),
        torch.tanh(10.0 * edge_l1),
        torch.tanh(10.0 * mean_shift),
        torch.tanh(10.0 * std_shift),
        torch.tanh(10.0 * clipped_shift),
        torch.tanh(10.0 * tv_shift),
        torch.tanh(5.0 * raw_candidate_l1),
        _batch_scalar(diffusion_strength, l1).clamp(0.0, 1.0),
        _batch_scalar(blend_alpha, l1).clamp(0.0, 1.0),
        _batch_scalar(detail_probability, l1).clamp(0.0, 1.0),
        _batch_scalar(primary_generation_probability, l1).clamp(0.0, 1.0),
    ]
    return torch.stack(values, dim=1).clamp(0.0, 1.0)


def raw_verification_statistics(features: Tensor) -> Dict[str, Tensor]:
    index = {name: i for i, name in enumerate(VERIFIER_FEATURE_NAMES)}
    return {
        "text_alignment_gain": (features[:, index["text_alignment_gain"]] - 0.5) / 5.0,
        "edge_cosine": 2.0 * features[:, index["edge_cosine"]] - 1.0,
        "low_frequency_l1": torch.atanh(
            features[:, index["low_frequency_l1"]].clamp(0.0, 1.0 - 1e-6)
        ) / 5.0,
        "candidate_regression_clip_similarity": (
            2.0 * features[:, index["candidate_regression_clip_similarity"]] - 1.0
        ),
    }


def load_verifier_thresholds(path: Optional[str]) -> Dict[str, float]:
    thresholds = dict(DEFAULT_VERIFIER_THRESHOLDS)
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        thresholds.update(payload.get("thresholds", payload))
    return {key: float(value) for key, value in thresholds.items()}


@torch.no_grad()
def verifier_acceptance(
    predictions: Mapping[str, Tensor],
    features: Tensor,
    thresholds: Mapping[str, float],
) -> Tuple[Tensor, Dict[str, Tensor]]:
    statistics = raw_verification_statistics(features)
    checks = {
        "acceptance_probability": predictions["acceptance_probability"]
        >= float(thresholds["acceptance_threshold"]),
        "hallucination_risk": predictions["hallucination_probability"]
        <= float(thresholds["max_hallucination_probability"]),
        "text_gain": statistics["text_alignment_gain"]
        >= float(thresholds["min_text_alignment_gain"]),
        "edge_consistency": statistics["edge_cosine"]
        >= float(thresholds["min_edge_cosine"]),
        "low_frequency_consistency": statistics["low_frequency_l1"]
        <= float(thresholds["max_low_frequency_l1"]),
        "semantic_gain_probability": predictions["semantic_gain_probability"]
        >= float(thresholds["min_semantic_gain_probability"]),
        "psnr_safe_probability": predictions["psnr_safe_probability"]
        >= float(thresholds["min_psnr_safe_probability"]),
        "ssim_safe_probability": predictions["ssim_safe_probability"]
        >= float(thresholds["min_ssim_safe_probability"]),
    }
    accepted = torch.ones_like(predictions["acceptance_probability"], dtype=torch.bool)
    for value in checks.values():
        accepted = accepted & value
    return accepted, checks
