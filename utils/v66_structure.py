"""Low-rate structural side information for V6.6 semantic recovery.

The transmitter extracts a compact binary Sobel sketch.  ``oracle`` mode
delivers it without errors and establishes an upper bound; ``noisy`` mode
uses repeated BPSK over an AWGN side channel.  None of the functions require
OpenCV, which keeps the server patch compatible with the existing Python 3.8
environment.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from PIL import Image
import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def _as_batch(image: Tensor) -> Tuple[Tensor, bool]:
    if image.ndim == 3:
        return image.unsqueeze(0), True
    if image.ndim != 4:
        raise ValueError("Expected CHW or BCHW image tensor")
    return image, False


@torch.no_grad()
def binary_structure_map(
    image: Tensor,
    side: int = 64,
    edge_quantile: float = 0.88,
) -> Tensor:
    """Return a compact one-channel binary Sobel structure map.

    Adaptive max pooling preserves thin boundaries before the per-image
    quantile threshold is applied.  The result has shape ``1xSxS`` for CHW
    input and ``Bx1xSxS`` for BCHW input.
    """
    if side < 8:
        raise ValueError("structure side must be at least 8")
    if not 0.50 <= edge_quantile < 1.0:
        raise ValueError("edge_quantile must be in [0.50, 1.0)")
    batch, squeezed = _as_batch(image.float().clamp(0.0, 1.0))
    if batch.size(1) == 1:
        gray = batch
    else:
        weights = batch.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        gray = (batch[:, :3] * weights).sum(dim=1, keepdim=True)
    kernel_x = gray.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(2, 3)
    dx = F.conv2d(gray, kernel_x, padding=1)
    dy = F.conv2d(gray, kernel_y, padding=1)
    magnitude = (dx.square() + dy.square() + 1e-12).sqrt()
    pooled = F.adaptive_max_pool2d(magnitude, (int(side), int(side)))
    thresholds = torch.quantile(
        pooled.flatten(1), float(edge_quantile), dim=1, keepdim=True
    ).view(-1, 1, 1, 1)
    structure = (pooled >= thresholds).to(batch.dtype)
    return structure[0] if squeezed else structure


def repetition_for_snr(snr_db: float) -> int:
    """Conservative odd repetition schedule for the development pilot."""
    if snr_db <= -5.0:
        return 7
    if snr_db <= 0.0:
        return 5
    if snr_db <= 5.0:
        return 3
    return 1


@torch.no_grad()
def transmit_binary_structure(
    structure: Tensor,
    snr_db: float,
    repetition: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Dict[str, float]]:
    """Transmit binary structure using repeated uncoded BPSK over AWGN."""
    if repetition < 1 or repetition % 2 == 0:
        raise ValueError("repetition must be a positive odd integer")
    original_device = structure.device
    source = (structure.detach().float().cpu() >= 0.5).float()
    symbols = 2.0 * source - 1.0
    snr_linear = max(10.0 ** (float(snr_db) / 10.0), 1e-8)
    sigma = math.sqrt(1.0 / (2.0 * snr_linear))
    votes = torch.zeros_like(symbols)
    for _ in range(int(repetition)):
        noise = torch.randn(
            symbols.shape,
            dtype=symbols.dtype,
            device=symbols.device,
            generator=generator,
        )
        votes += (symbols + sigma * noise >= 0.0).float()
    decoded = (votes > repetition // 2).float()
    bit_errors = float((decoded != source).float().mean())
    raw_bits = int(source.numel())
    return decoded.to(original_device), {
        "structure_ber": bit_errors,
        "structure_source_bits": float(raw_bits),
        "structure_channel_bits": float(raw_bits * int(repetition)),
        "structure_repetition": float(repetition),
    }


def structure_to_pil(structure: Tensor, size: int = 512) -> Image.Image:
    """Convert a binary structure tensor to a three-channel ControlNet image."""
    value = structure.detach().float().cpu()
    if value.ndim == 4:
        if value.size(0) != 1:
            raise ValueError("structure_to_pil expects one sample")
        value = value[0]
    if value.ndim == 3:
        value = value[0]
    if value.ndim != 2:
        raise ValueError("Expected HW, 1HW, or 11HW structure tensor")
    array = (value.clamp(0.0, 1.0) * 255.0).round().byte().numpy()
    return Image.fromarray(array, mode="L").resize(
        (int(size), int(size)), Image.Resampling.NEAREST
    ).convert("RGB")


def _dilate(structure: Tensor, radius: int = 1) -> Tensor:
    batch, squeezed = _as_batch(structure.float())
    if batch.size(1) != 1:
        batch = batch[:, :1]
    kernel = 2 * int(radius) + 1
    output = F.max_pool2d(batch, kernel_size=kernel, stride=1, padding=radius)
    return output[0] if squeezed else output


@torch.no_grad()
def tolerant_structure_f1(
    prediction: Tensor,
    target: Tensor,
    tolerance: int = 1,
) -> Tensor:
    """Boundary F1 allowing a small spatial tolerance at structure resolution."""
    pred, _ = _as_batch((prediction >= 0.5).float())
    truth, _ = _as_batch((target >= 0.5).float())
    if pred.shape != truth.shape:
        raise ValueError("prediction and target structure shapes must match")
    pred_dilated = _dilate(pred, tolerance)
    truth_dilated = _dilate(truth, tolerance)
    precision = (pred * truth_dilated).sum((1, 2, 3)) / pred.sum(
        (1, 2, 3)
    ).clamp_min(1.0)
    recall = (truth * pred_dilated).sum((1, 2, 3)) / truth.sum(
        (1, 2, 3)
    ).clamp_min(1.0)
    return 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)


@torch.no_grad()
def candidate_structure_statistics(
    candidate: Tensor,
    transmitted_structure: Tensor,
    source_structure: Tensor,
    side: int,
    edge_quantile: float,
) -> Dict[str, float]:
    candidate_structure = binary_structure_map(candidate, side, edge_quantile)
    received_f1 = float(
        tolerant_structure_f1(candidate_structure, transmitted_structure)[0]
    )
    source_f1 = float(tolerant_structure_f1(candidate_structure, source_structure)[0])
    return {
        "received_structure_f1": received_f1,
        "source_structure_f1": source_f1,
    }
