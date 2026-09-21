"""Fast CPU smoke test for the V6.2 allocation and policy modules."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile

import torch


def _load_local_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parent / relative_path
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError("Cannot load module from %s" % path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


quality_module = _load_local_module(
    "v62_quality_adaptive_recovery",
    "models/quality_adaptive_recovery.py",
)
packer_module = _load_local_module(
    "v62_shared_bottleneck",
    "models/shared_bottleneck.py",
)
FEATURE_NAMES = quality_module.FEATURE_NAMES
QualityAdaptivePolicy = quality_module.QualityAdaptivePolicy
SemanticConditionAdapter = quality_module.SemanticConditionAdapter
receiver_features = quality_module.receiver_features
policy_loss = quality_module.policy_loss
load_policy_checkpoint = quality_module.load_policy_checkpoint
FixedBudgetTokenPacker = packer_module.FixedBudgetTokenPacker


def main() -> None:
    torch.manual_seed(42)
    batch = 2
    packer = FixedBudgetTokenPacker(
        hidden_dim=256,
        modality_budget=272,
        min_text_tokens=16,
        max_text_tokens=64,
        fixed_text_tokens=16,
        strategy="utility",
        text_selection_strategy="valid_prefix_repeat",
    )
    z_common = torch.randn(batch, 16, 16, requires_grad=True)
    z_image = torch.randn(batch, 256, 16, requires_grad=True)
    z_text = torch.randn(batch, 64, 16, requires_grad=True)
    image_features = torch.randn(batch, 256, 256)
    text_features = torch.randn(batch, 64, 256)
    valid_lengths = torch.tensor([8, 20])
    attention_mask = torch.zeros(batch, 64, dtype=torch.long)
    attention_mask[0, :8] = 1
    attention_mask[1, :20] = 1
    packed, metadata = packer.pack(
        z_common,
        z_image,
        z_text,
        image_features,
        text_features,
        valid_lengths,
        64,
        snr_db=torch.tensor([0.0, 10.0]),
        utility_weight=torch.tensor([1.0, 0.5]),
        text_attention_mask=attention_mask,
    )
    assert packed.shape == (2, 288, 16)
    assert metadata["text_counts"].tolist() == [64, 40]
    assert metadata["image_counts"].tolist() == [208, 232]
    assert metadata["text_redundant_slots"].tolist() == [56, 20]
    expected_repetition = torch.tensor([8.0, 2.0])
    assert torch.allclose(
        metadata["text_repetition_factor"].cpu(), expected_repetition
    )

    common, image, text = packer.unpack(packed, metadata)
    assert common.shape == z_common.shape
    assert image.shape == z_image.shape
    assert text.shape == z_text.shape
    assert torch.count_nonzero(text[0, 8:]) == 0
    assert torch.count_nonzero(text[1, 20:]) == 0
    (common.square().mean() + image.square().mean() + text.square().mean()).backward()
    assert z_common.grad is not None and torch.isfinite(z_common.grad).all()
    assert z_image.grad is not None and torch.isfinite(z_image.grad).all()
    assert z_text.grad is not None and torch.isfinite(z_text.grad).all()

    features = torch.rand(batch, len(FEATURE_NAMES))
    policy = QualityAdaptivePolicy()
    decision = policy(features)
    assert set(decision) == {
        "generation_probability",
        "text_utility",
        "diffusion_strength",
        "predicted_regression_quality",
    }
    assert all(value.shape == (batch,) for value in decision.values())
    assert torch.all((decision["text_utility"] >= 0) & (decision["text_utility"] <= 1))
    balanced_loss, _ = policy_loss(
        decision,
        generation_target=torch.tensor([0.0, 1.0]),
        text_utility_target=torch.tensor([0.25, 0.75]),
        strength_target=torch.tensor([0.10, 0.30]),
        regression_quality_target=torch.tensor([0.8, 0.3]),
        mode_positive_weight=3.0,
    )
    balanced_loss.backward()
    assert torch.isfinite(balanced_loss)
    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "policy.pth"
        torch.save(
            {
                "policy": policy.state_dict(),
                "policy_config": {
                    "hidden_dim": 64,
                    "min_strength": 0.10,
                    "max_strength": 0.35,
                },
                "train_image_ids": [1, 2],
                "val_image_ids": [3],
                "policy_data_image_ids": [1, 2, 3],
            },
            str(checkpoint_path),
        )
        restored_policy = load_policy_checkpoint(
            str(checkpoint_path), torch.device("cpu")
        )
        assert restored_policy.policy_data_image_ids == {1, 2, 3}

    # Verify the safe initialization follows the intended causal rules.
    clear = torch.ones(1, len(FEATURE_NAMES))
    unclear = clear.clone()
    unclear[:, [1, 2, 4]] = 0.0
    assert policy(unclear)["text_utility"].item() > policy(clear)["text_utility"].item()
    bad_image = clear.clone()
    bad_image[:, [5, 6, 7, 8, 10, 18]] = 0.0
    assert (
        policy(bad_image)["generation_probability"].item()
        > policy(clear)["generation_probability"].item()
    )

    # The online feature extractor must ignore the source attention mask and
    # estimate received length from its own EOS prediction.
    logits = torch.randn(batch, 6, 10)
    logits[:, :, 1] = 5.0
    logits[0, 3, 9] = 10.0
    logits[1, 4, 9] = 10.0
    feature_outputs = {
        "image": torch.rand(batch, 3, 16, 16),
        "text_logits": logits,
        "text_logits_global": logits.clone(),
        "image_conditioned_text_logits": logits.clone(),
        "receiver_text_semantic": torch.randn(batch, 8),
        "image_conditioned_text_semantic": torch.randn(batch, 8),
        "image_private_reliability": torch.ones(batch),
        "text_private_reliability": torch.ones(batch),
        "allocated_text_tokens": torch.tensor([64.0, 40.0]),
        "z_shared_noisy": torch.randn(batch, 8, 4),
        "decoded_common_tokens": torch.randn(batch, 4, 8),
        "text_eos_token_id": torch.tensor(9),
        "text_attention_mask": torch.ones(batch, 6),
    }
    receiver_a = receiver_features(feature_outputs, torch.tensor([0.0, 10.0]))
    feature_outputs["text_attention_mask"] = torch.zeros(batch, 6)
    receiver_b = receiver_features(feature_outputs, torch.tensor([0.0, 10.0]))
    assert receiver_a.shape == (batch, len(FEATURE_NAMES))
    assert torch.allclose(receiver_a, receiver_b)

    adapter = SemanticConditionAdapter()
    semantic_outputs = {
        "decoded_common_tokens": torch.randn(batch, 16, 256),
        "decoded_image_tokens": torch.randn(batch, 256, 256),
        "decoded_text_tokens": torch.randn(batch, 64, 256),
    }
    condition = adapter(semantic_outputs, features)
    assert condition.shape == (batch, 8, 768)

    # Monte-Carlo evidence that the extra text slots reduce received semantic
    # feature noise, rather than merely changing an allocation counter.
    trials = 128
    monte_carlo_packer = FixedBudgetTokenPacker(
        hidden_dim=256,
        modality_budget=272,
        min_text_tokens=16,
        max_text_tokens=64,
        fixed_text_tokens=16,
        strategy="utility",
        text_selection_strategy="valid_prefix_repeat",
    )
    with torch.no_grad():
        for importance in (
            monte_carlo_packer.image_importance,
            monte_carlo_packer.text_importance,
        ):
            importance[-1].weight.zero_()
            importance[-1].bias.zero_()
    base_common = torch.randn(1, 16, 16).expand(trials, -1, -1).clone()
    base_image = torch.randn(1, 256, 16).expand(trials, -1, -1).clone()
    base_text = torch.randn(1, 64, 16).expand(trials, -1, -1).clone()
    base_image_features = torch.randn(1, 256, 256).expand(trials, -1, -1)
    base_text_features = torch.randn(1, 64, 256).expand(trials, -1, -1)
    short_lengths = torch.full((trials,), 8, dtype=torch.long)
    short_mask = torch.zeros(trials, 64, dtype=torch.long)
    short_mask[:, :8] = 1
    text_mse = {}
    image_token_counts = {}
    for utility in (0.0, 0.5, 1.0):
        clean_packed, mc_metadata = monte_carlo_packer.pack(
            base_common,
            base_image,
            base_text,
            base_image_features,
            base_text_features,
            short_lengths,
            64,
            snr_db=0.0,
            utility_weight=utility,
            text_attention_mask=short_mask,
        )
        mean_power = clean_packed.square().mean((1, 2), keepdim=True)
        normalized = clean_packed / torch.sqrt(mean_power + 1e-8)
        _, _, clean_text = monte_carlo_packer.unpack(normalized, mc_metadata)
        received = normalized + torch.randn_like(normalized)  # 0 dB AWGN
        _, _, noisy_text = monte_carlo_packer.unpack(received, mc_metadata)
        text_mse[utility] = float(
            (noisy_text[:, :8] - clean_text[:, :8]).square().mean()
        )
        image_token_counts[utility] = int(mc_metadata["image_counts"][0])
    assert text_mse[1.0] < 0.50 * text_mse[0.0]
    assert image_token_counts[1.0] < image_token_counts[0.0]

    print("V6.2 architecture smoke test: PASS")
    print("feature_dim=%d" % len(FEATURE_NAMES))
    print("text_counts=%s" % metadata["text_counts"].tolist())
    print("text_repetition_factor=%s" % metadata["text_repetition_factor"].tolist())
    print("0dB_text_mse=%s" % text_mse)
    print("image_token_counts=%s" % image_token_counts)


if __name__ == "__main__":
    main()
