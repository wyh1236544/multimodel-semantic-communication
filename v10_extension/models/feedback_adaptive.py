from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
from torch import Tensor, nn


PAYLOAD_MODES = ("reconstruction", "hybrid", "generation")
FEEDBACK_STATES = (*PAYLOAD_MODES, "retransmit")


def _hadamard(order: int) -> Tensor:
    if order <= 0 or order & (order - 1):
        raise ValueError("Walsh code length must be a positive power of two")
    matrix = torch.ones(1, 1, dtype=torch.float32)
    while matrix.size(0) < order:
        matrix = torch.cat(
            [
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1),
            ],
            dim=0,
        )
    return matrix


def _as_batch(
    value: float | int | Tensor,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    result = torch.as_tensor(value, device=device, dtype=torch.float32)
    if result.ndim == 0:
        result = result.expand(batch_size)
    return result.reshape(batch_size, -1)[:, 0]


def _cpu_noise(
    shape: Sequence[int],
    device: torch.device,
    generator: torch.Generator | None,
) -> Tensor:
    return torch.randn(
        tuple(int(value) for value in shape),
        generator=generator,
        dtype=torch.float32,
    ).to(device)


@dataclass(frozen=True)
class ModeAllocation:
    common_tokens: int
    image_tokens: int
    text_tokens: int

    @property
    def payload_tokens(self) -> int:
        return self.common_tokens + self.image_tokens + self.text_tokens


class ThresholdModeRouter:
    """Conservative receiver policy based on a pilot SNR estimate."""

    def __init__(
        self,
        generation_threshold_db: float = 0.0,
        reconstruction_threshold_db: float = 10.0,
    ) -> None:
        if generation_threshold_db >= reconstruction_threshold_db:
            raise ValueError(
                "generation threshold must be below reconstruction threshold"
            )
        self.generation_threshold_db = float(generation_threshold_db)
        self.reconstruction_threshold_db = float(
            reconstruction_threshold_db
        )

    def mode_indices(self, estimated_snr_db: Tensor) -> Tensor:
        estimated_snr_db = estimated_snr_db.float()
        result = torch.full_like(estimated_snr_db, 1, dtype=torch.long)
        result = torch.where(
            estimated_snr_db <= self.generation_threshold_db,
            torch.full_like(result, 2),
            result,
        )
        result = torch.where(
            estimated_snr_db >= self.reconstruction_threshold_db,
            torch.zeros_like(result),
            result,
        )
        return result


class RobustModeFeedback(nn.Module):
    """Four-state orthogonal feedback with confidence rejection.

    The receiver transmits one Walsh codeword instead of two uncoded bits.
    If the sender cannot separate the two best correlations confidently, it
    uses the configured safe fallback rather than risking reconstruction mode.
    """

    def __init__(
        self,
        code_length: int = 16,
        confidence_threshold: float = 0.20,
        max_attempts: int = 2,
        safe_fallback: str = "generation",
    ) -> None:
        super().__init__()
        if code_length < len(FEEDBACK_STATES):
            raise ValueError("feedback code length must be at least four")
        if safe_fallback not in PAYLOAD_MODES:
            raise ValueError(f"Unsupported safe fallback: {safe_fallback}")
        self.code_length = int(code_length)
        self.confidence_threshold = float(confidence_threshold)
        self.max_attempts = max(int(max_attempts), 1)
        self.safe_fallback_index = PAYLOAD_MODES.index(safe_fallback)
        self.register_buffer(
            "codebook",
            _hadamard(self.code_length)[: len(FEEDBACK_STATES)],
            persistent=False,
        )

    def forward(
        self,
        requested_mode_indices: Tensor,
        feedback_snr_db: float | Tensor,
        ideal: bool = False,
        generator: torch.Generator | None = None,
    ) -> Dict[str, Tensor]:
        requested = requested_mode_indices.long().reshape(-1)
        if bool(((requested < 0) | (requested >= len(PAYLOAD_MODES))).any()):
            raise ValueError("requested payload mode index must be 0, 1, or 2")
        device = requested.device
        codebook = self.codebook.to(device=device)
        transmitted = codebook[requested]
        batch_size = requested.size(0)
        feedback_snr = _as_batch(feedback_snr_db, batch_size, device)
        noise_std = torch.pow(10.0, -feedback_snr / 20.0)[:, None]
        accumulated = torch.zeros_like(transmitted)
        attempts = 1 if ideal else self.max_attempts
        for _ in range(attempts):
            if ideal:
                received = transmitted
            else:
                received = transmitted + noise_std * _cpu_noise(
                    transmitted.shape,
                    device,
                    generator,
                )
            accumulated = accumulated + received
        averaged = accumulated / float(attempts)
        correlations = averaged @ codebook.transpose(0, 1)
        correlations = correlations / float(self.code_length)
        top_values, top_indices = correlations.topk(k=2, dim=1)
        raw_decoded = top_indices[:, 0]
        confidence_margin = top_values[:, 0] - top_values[:, 1]
        accepted = confidence_margin >= self.confidence_threshold
        payload_mode = torch.where(
            accepted & (raw_decoded < len(PAYLOAD_MODES)),
            raw_decoded,
            torch.full_like(raw_decoded, self.safe_fallback_index),
        )
        return {
            "requested_mode": requested,
            "raw_decoded_state": raw_decoded,
            "payload_mode": payload_mode,
            "feedback_accepted": accepted,
            "feedback_error": raw_decoded != requested,
            "confidence_margin": confidence_margin,
            "correlations": correlations,
            "feedback_channel_uses": requested.new_full(
                (batch_size,), self.code_length * attempts
            ),
        }


class PilotFeedbackController(nn.Module):
    """Pilot SNR estimation followed by threshold routing and coded feedback."""

    def __init__(
        self,
        pilot_length: int = 32,
        feedback_code_length: int = 16,
        feedback_confidence_threshold: float = 0.20,
        feedback_attempts: int = 2,
        generation_threshold_db: float = 0.0,
        reconstruction_threshold_db: float = 10.0,
        safe_fallback: str = "generation",
    ) -> None:
        super().__init__()
        self.pilot_length = max(int(pilot_length), 4)
        self.router = ThresholdModeRouter(
            generation_threshold_db=generation_threshold_db,
            reconstruction_threshold_db=reconstruction_threshold_db,
        )
        self.feedback = RobustModeFeedback(
            code_length=feedback_code_length,
            confidence_threshold=feedback_confidence_threshold,
            max_attempts=feedback_attempts,
            safe_fallback=safe_fallback,
        )

    def estimate_snr(
        self,
        true_snr_db: float | Tensor,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        true_snr = _as_batch(true_snr_db, batch_size, device)
        noise_std = torch.pow(10.0, -true_snr / 20.0)[:, None]
        noise = noise_std * _cpu_noise(
            (batch_size, self.pilot_length),
            device,
            generator,
        )
        estimated_noise_power = noise.square().mean(dim=1).clamp_min(1e-8)
        return -10.0 * torch.log10(estimated_noise_power)

    def forward(
        self,
        true_forward_snr_db: float | Tensor,
        feedback_snr_db: float | Tensor,
        batch_size: int,
        device: torch.device,
        ideal_feedback: bool = False,
        generator: torch.Generator | None = None,
    ) -> Dict[str, Tensor]:
        estimated_snr = self.estimate_snr(
            true_forward_snr_db,
            batch_size,
            device,
            generator=generator,
        )
        requested = self.router.mode_indices(estimated_snr)
        result = self.feedback(
            requested,
            feedback_snr_db=feedback_snr_db,
            ideal=ideal_feedback,
            generator=generator,
        )
        result["estimated_forward_snr_db"] = estimated_snr
        result["pilot_channel_uses"] = requested.new_full(
            (batch_size,), self.pilot_length
        )
        return result


def mode_names(indices: Tensor) -> list[str]:
    values = indices.detach().cpu().tolist()
    return [PAYLOAD_MODES[int(index)] for index in values]

