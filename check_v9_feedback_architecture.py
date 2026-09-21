from __future__ import annotations

import torch

from models.feedback_adaptive import PilotFeedbackController, mode_names
from models.shared_bottleneck import FixedBudgetTokenPacker


def check_adaptive_packer() -> None:
    torch.manual_seed(42)
    batch = 3
    hidden = 8
    latent = 4
    common_count = 32
    image_count = 256
    text_count = 77
    packer = FixedBudgetTokenPacker(
        hidden_dim=hidden,
        modality_budget=256,
        min_text_tokens=32,
        max_text_tokens=192,
        fixed_text_tokens=32,
        strategy="utility",
        text_selection_strategy="valid_prefix_repeat",
    )
    z_common = torch.randn(batch, common_count, latent)
    z_image = torch.randn(batch, image_count, latent)
    z_text = torch.randn(batch, text_count, latent)
    image_features = torch.randn(batch, image_count, hidden)
    text_features = torch.randn(batch, text_count, hidden)
    attention_mask = torch.zeros(batch, text_count, dtype=torch.long)
    attention_mask[:, :60] = 1
    quotas = torch.tensor([32.0, 128.0, 192.0])
    packed, metadata = packer.pack(
        z_common,
        z_image,
        z_text,
        image_features,
        text_features,
        valid_text_lengths=attention_mask.sum(dim=1),
        text_max_length=text_count,
        snr_db=torch.tensor([20.0, 5.0, -5.0]),
        text_attention_mask=attention_mask,
        text_token_quota=quotas,
    )
    if tuple(packed.shape) != (batch, 288, latent):
        raise AssertionError(f"unexpected packed shape: {tuple(packed.shape)}")
    expected_images = torch.tensor([224, 128, 64])
    if not torch.equal(metadata["text_counts"].cpu(), quotas.long()):
        raise AssertionError(metadata["text_counts"])
    if not torch.equal(metadata["image_counts"].cpu(), expected_images):
        raise AssertionError(metadata["image_counts"])
    common, image, text = packer.unpack(packed, metadata)
    if common.shape != z_common.shape or image.shape != z_image.shape:
        raise AssertionError("unpacked common/image shapes are wrong")
    if text.shape != z_text.shape:
        raise AssertionError("unpacked text shape is wrong")
    print(
        "packer OK: C32 + (I224,T32)/(I128,T128)/(I64,T192) = 288",
        flush=True,
    )
    print(
        "text repetition factors:",
        metadata["text_repetition_factor"].tolist(),
        flush=True,
    )


def check_feedback() -> None:
    controller = PilotFeedbackController(
        pilot_length=32,
        feedback_code_length=16,
        feedback_confidence_threshold=0.20,
        feedback_attempts=2,
        generation_threshold_db=2.5,
        reconstruction_threshold_db=8.0,
        safe_fallback="generation",
    )
    batch = 4096
    for snr_db in (-5.0, 0.0, 5.0, 10.0, 20.0):
        generator = torch.Generator().manual_seed(842 + int(snr_db * 10))
        result = controller(
            true_forward_snr_db=snr_db,
            feedback_snr_db=snr_db,
            batch_size=batch,
            device=torch.device("cpu"),
            ideal_feedback=False,
            generator=generator,
        )
        requested = result["requested_mode"]
        payload = result["payload_mode"]
        accuracy = (payload == requested).float().mean().item()
        raw_error = result["feedback_error"].float().mean().item()
        rejected = (~result["feedback_accepted"]).float().mean().item()
        names = mode_names(payload[:8])
        print(
            f"feedback snr={snr_db:g} accuracy={accuracy:.4f} "
            f"raw_error={raw_error:.4f} rejected={rejected:.4f} "
            f"examples={names}",
            flush=True,
        )


if __name__ == "__main__":
    check_adaptive_packer()
    check_feedback()
    print("V9 feedback architecture checks passed", flush=True)
