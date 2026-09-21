from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from utils.metrics import ssim_per_image


def _edge_map(images: torch.Tensor) -> torch.Tensor:
    gray = images.float().mean(dim=1, keepdim=True)
    kernel_x = images.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0)
    kernel_y = kernel_x.transpose(-1, -2)
    grad_x = F.conv2d(gray, kernel_x, padding=1)
    grad_y = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)


def _cosine_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a.float().mean(dim=1), dim=-1)
    b = F.normalize(b.float().mean(dim=1), dim=-1)
    return (a * b).sum(dim=-1).mean()


def _pooled_normalized(tokens: torch.Tensor) -> torch.Tensor:
    return F.normalize(tokens.float().mean(dim=1), dim=-1)


def _symmetric_infonce(
    a: torch.Tensor,
    b: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Batch-wise symmetric InfoNCE; unlike pair cosine, it resists collapse."""
    a = _pooled_normalized(a)
    b = _pooled_normalized(b)
    logits = a @ b.transpose(0, 1) / max(float(temperature), 1e-4)
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(0, 1), labels)
    )


def _symmetric_infonce_vectors(
    a: torch.Tensor,
    b: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    logits = a @ b.transpose(0, 1) / max(float(temperature), 1e-4)
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(0, 1), labels)
    )


class MultimodalCommunicationLoss(nn.Module):
    """Stable image-focused objective using only fields returned by the V6.1 model."""

    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.temperature = float(config.get("common_contrastive_temperature", 0.07))
        self.base_weights = {
            "img_latent": float(config.get("lambda_img_latent", 2.0)),
            "img_pixel": float(config.get("lambda_img_pixel", 1.0)),
            "img_ssim": float(config.get("lambda_img_ssim", 0.5)),
            "img_edge": float(config.get("lambda_img_edge", 0.1)),
            "txt": float(config.get("lambda_txt", 0.3)),
            "allocation": float(config.get("lambda_allocation_supervision", 0.1)),
            "common_alignment": float(config.get("lambda_common_alignment", 0.03)),
            "common_consistency": float(
                config.get("lambda_common_consistency", 0.0)
            ),
            "common_contrastive": float(
                config.get("lambda_common_contrastive", 0.0)
            ),
            "common_coarse": float(
                config.get("lambda_common_coarse", 0.0)
            ),
            "common_text": float(config.get("lambda_common_text", 0.0)),
            "common_semantic": float(
                config.get("lambda_common_semantic", 0.0)
            ),
            "private_orthogonality": float(
                config.get("lambda_private_orthogonality", 0.03)
            ),
        }
        self.weights = dict(self.base_weights)

    def set_weight_overrides(self, overrides: Mapping[str, Any] | None = None) -> None:
        """Reset to config weights, then apply one training-stage profile."""
        self.weights = dict(self.base_weights)
        for name, value in (overrides or {}).items():
            if name not in self.weights:
                raise ValueError(f"Unknown loss weight in training stage: {name}")
            self.weights[name] = float(value)

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        reconstructed = outputs["image"].float()
        target = images.float()
        pixel = F.mse_loss(reconstructed, target)
        latent = F.mse_loss(
            outputs["recovered_vae_latents"].float(),
            outputs["vae_latents"].detach().float(),
        )
        ssim = 1.0 - ssim_per_image(reconstructed, target).mean()
        edge = F.l1_loss(_edge_map(reconstructed), _edge_map(target))

        logits = outputs["text_logits"]
        ids = outputs["text_input_ids"]
        mask = outputs["text_attention_mask"].bool()
        length = min(logits.size(1), ids.size(1))
        ce = F.cross_entropy(
            logits[:, :length].reshape(-1, logits.size(-1)),
            ids[:, :length].reshape(-1),
            reduction="none",
        ).reshape(ids.size(0), length)
        text = (ce * mask[:, :length]).sum() / mask[:, :length].sum().clamp_min(1)

        allocation = F.mse_loss(
            outputs["predicted_text_quota"].float(),
            outputs["target_text_quota"].detach().float(),
        )
        common_alignment = 1.0 - _cosine_mean(
            outputs["common_image_tokens"], outputs["common_text_tokens"]
        )
        # The fused common stream must agree with both unimodal views before
        # transmission and after the noisy channel. This makes Z_C useful at
        # the receiver instead of merely making two source encoders similar.
        common_consistency = common_alignment.new_zeros(())
        if self.weights["common_consistency"] != 0.0:
            common_consistency = 0.25 * (
                1.0
                - _cosine_mean(
                    outputs["encoded_common_tokens"], outputs["common_image_tokens"]
                )
                + 1.0
                - _cosine_mean(
                    outputs["encoded_common_tokens"], outputs["common_text_tokens"]
                )
                + 1.0
                - _cosine_mean(
                    outputs["decoded_common_tokens"], outputs["common_image_tokens"]
                )
                + 1.0
                - _cosine_mean(
                    outputs["decoded_common_tokens"], outputs["common_text_tokens"]
                )
            )
        common_contrastive = common_alignment.new_zeros(())
        if self.weights["common_contrastive"] != 0.0:
            common_contrastive = (
                _symmetric_infonce(
                    outputs["common_image_tokens"],
                    outputs["common_text_tokens"],
                    self.temperature,
                )
                + _symmetric_infonce(
                    outputs["decoded_common_tokens"],
                    outputs["common_image_tokens"],
                    self.temperature,
                )
                + _symmetric_infonce(
                    outputs["decoded_common_tokens"],
                    outputs["common_text_tokens"],
                    self.temperature,
                )
            ) / 3.0
        common_coarse = common_alignment.new_zeros(())
        predicted_coarse = outputs.get("common_coarse_latents")
        if predicted_coarse is not None and self.weights["common_coarse"] != 0.0:
            kernel_value = outputs.get("common_coarse_blur_kernel", 5)
            kernel = int(
                kernel_value.item()
                if isinstance(kernel_value, torch.Tensor)
                else kernel_value
            )
            target_coarse = F.avg_pool2d(
                outputs["vae_latents"].detach().float(),
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
            )
            common_coarse = F.mse_loss(
                predicted_coarse.float(), target_coarse
            )
        common_text = common_alignment.new_zeros(())
        common_text_logits = outputs.get("common_base_text_logits")
        if common_text_logits is not None and self.weights["common_text"] != 0.0:
            common_length_value = outputs.get(
                "common_caption_max_tokens", common_text_logits.size(1)
            )
            common_length = min(
                int(
                    common_length_value.item()
                    if isinstance(common_length_value, torch.Tensor)
                    else common_length_value
                ),
                common_text_logits.size(1),
                ids.size(1),
            )
            common_ce = F.cross_entropy(
                common_text_logits[:, :common_length].reshape(
                    -1, common_text_logits.size(-1)
                ),
                ids[:, :common_length].reshape(-1),
                reduction="none",
            ).reshape(ids.size(0), common_length)
            common_mask = mask[:, :common_length]
            common_text = (
                common_ce * common_mask
            ).sum() / common_mask.sum().clamp_min(1)
        common_semantic = common_alignment.new_zeros(())
        receiver_semantic = outputs.get("common_receiver_semantic")
        semantic_target = outputs.get("common_text_semantic_target")
        if (
            receiver_semantic is not None
            and semantic_target is not None
            and self.weights["common_semantic"] != 0.0
        ):
            pair_cosine = 1.0 - (
                F.normalize(receiver_semantic.float(), dim=-1)
                * F.normalize(semantic_target.detach().float(), dim=-1)
            ).sum(dim=-1).mean()
            contrastive = _symmetric_infonce_vectors(
                receiver_semantic,
                semantic_target.detach(),
                self.temperature,
            )
            common_semantic = 0.5 * (pair_cosine + contrastive)
        common = F.normalize(outputs["encoded_common_tokens"].float().mean(1), dim=-1)
        image_private = F.normalize(
            outputs["encoded_image_private_tokens"].float().mean(1), dim=-1
        )
        text_private = F.normalize(
            outputs["encoded_text_private_tokens"].float().mean(1), dim=-1
        )
        orthogonality = 0.5 * (
            (common * image_private).sum(-1).abs().mean()
            + (common * text_private).sum(-1).abs().mean()
        )

        terms = {
            "img_latent": latent,
            "img_pixel": pixel,
            "img_ssim": ssim,
            "img_edge": edge,
            "txt": text,
            "allocation": allocation,
            "common_alignment": common_alignment,
            "common_consistency": common_consistency,
            "common_contrastive": common_contrastive,
            "common_coarse": common_coarse,
            "common_text": common_text,
            "common_semantic": common_semantic,
            "private_orthogonality": orthogonality,
        }
        total = sum(self.weights[name] * value for name, value in terms.items())
        terms["total"] = total
        return total, terms
