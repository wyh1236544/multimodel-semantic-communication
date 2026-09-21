"""Runtime-selectable nested channel widths for one shared model.

The encoder always predicts the maximum latent width. Only a prefix of that
latent is transmitted, and the receiver pads the missing suffix with zeros
before the normal decoder. Training the same network with several prefixes
therefore produces one checkpoint that can operate at several channel sizes.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class AdaptiveBandwidthController(nn.Module):
    """Choose a channel width from a validated set of nested widths."""

    supported_policies = {"fixed", "snr"}
    supported_train_sampling = {"policy", "random"}

    def __init__(
        self,
        candidate_widths: Sequence[int],
        policy: str = "snr",
        snr_thresholds: Optional[Sequence[float]] = None,
        default_width: Optional[int] = None,
        train_sampling: str = "random",
        channel_width_offsets: Optional[Mapping[str, int]] = None,
    ):
        super().__init__()
        candidates = sorted({int(width) for width in candidate_widths})
        if not candidates or candidates[0] <= 0:
            raise ValueError("candidate_widths must contain positive integers")
        self.candidate_widths = tuple(candidates)
        self.max_width = candidates[-1]
        self.policy = str(policy).lower()
        if self.policy not in self.supported_policies:
            raise ValueError(
                f"policy must be one of {sorted(self.supported_policies)}"
            )
        self.train_sampling = str(train_sampling).lower()
        if self.train_sampling not in self.supported_train_sampling:
            raise ValueError("train_sampling must be 'policy' or 'random'")
        self.default_width = int(default_width or self.max_width)
        self._validate_requested_width(self.default_width)
        thresholds = [float(value) for value in (snr_thresholds or [])]
        if self.policy == "snr" and len(thresholds) != len(candidates) - 1:
            raise ValueError(
                "snr policy requires len(candidate_widths)-1 thresholds"
            )
        if thresholds != sorted(thresholds):
            raise ValueError("snr_thresholds must be sorted in ascending order")
        self.snr_thresholds = tuple(thresholds)
        self.channel_width_offsets = {
            str(name).lower(): int(offset)
            for name, offset in (channel_width_offsets or {}).items()
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "AdaptiveBandwidthController":
        return cls(
            candidate_widths=config["candidate_widths"],
            policy=str(config.get("policy", "snr")),
            snr_thresholds=config.get("snr_thresholds"),
            default_width=config.get("default_width"),
            train_sampling=str(config.get("train_sampling", "random")),
            channel_width_offsets=config.get("channel_width_offsets", {}),
        )

    def _validate_requested_width(self, width: int) -> int:
        width = int(width)
        if width not in self.candidate_widths:
            raise ValueError(
                f"channel_width={width} is not in {list(self.candidate_widths)}"
            )
        return width

    @staticmethod
    def _batch_snr(snr_db: Union[float, int, Tensor]) -> float:
        if isinstance(snr_db, Tensor):
            if snr_db.numel() == 0:
                raise ValueError("snr_db tensor cannot be empty")
            return float(snr_db.detach().float().mean().item())
        return float(snr_db)

    def select_width(
        self,
        snr_db: Union[float, int, Tensor],
        requested_width: Optional[int] = None,
        training: bool = False,
        channel_type: str = "awgn",
    ) -> int:
        """Return one width for the current mini-batch."""
        if requested_width is not None:
            return self._validate_requested_width(requested_width)
        if training and self.train_sampling == "random":
            index = int(torch.randint(len(self.candidate_widths), (1,)).item())
            return self.candidate_widths[index]
        if self.policy == "fixed":
            return self.default_width

        snr = self._batch_snr(snr_db)
        threshold_bin = sum(snr >= threshold for threshold in self.snr_thresholds)
        index = len(self.candidate_widths) - 1 - threshold_bin
        offset = self.channel_width_offsets.get(str(channel_type).lower(), 0)
        index = min(max(index + offset, 0), len(self.candidate_widths) - 1)
        return self.candidate_widths[index]

    def prepare_transmission(
        self,
        latent: Tensor,
        width: int,
        standard_noise: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Slice the transmitted prefix and pair an optional noise tensor."""
        width = self._validate_requested_width(width)
        if latent.size(-1) != self.max_width:
            raise ValueError(
                f"Expected latent width {self.max_width}, got {latent.size(-1)}"
            )
        transmitted = latent[..., :width]
        if standard_noise is None:
            return transmitted, None
        if standard_noise.shape == latent.shape:
            return transmitted, standard_noise[..., :width]
        if standard_noise.shape == transmitted.shape:
            return transmitted, standard_noise
        raise ValueError(
            "channel_noise must match either the maximum latent shape or the "
            "active transmitted shape"
        )

    def restore_receiver_width(self, received: Tensor) -> Tensor:
        """Pad an active-width signal back to the decoder's maximum width."""
        if received.size(-1) > self.max_width:
            raise ValueError("Received width exceeds configured maximum")
        return F.pad(received, (0, self.max_width - received.size(-1)))

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "policy": self.policy,
            "train_sampling": self.train_sampling,
            "candidate_widths": list(self.candidate_widths),
            "max_symbols_per_token": self.max_width,
            "default_width": self.default_width,
            "snr_thresholds": list(self.snr_thresholds),
            "channel_width_offsets": dict(self.channel_width_offsets),
        }
