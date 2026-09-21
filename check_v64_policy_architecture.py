"""Fast CPU smoke test for the V6.4 receiver-only policy."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from models.controlnet_fusion_policy import (
    ControlNetFusionPolicy,
    controlnet_policy_loss,
    load_controlnet_policy,
)
from models.quality_adaptive_recovery import FEATURE_NAMES


def main() -> None:
    torch.manual_seed(42)
    policy = ControlNetFusionPolicy()
    features = torch.rand(8, len(FEATURE_NAMES))
    probe = policy(features)
    route = policy(features.flip(0))
    expected = {
        "generation_probability",
        "text_utility",
        "diffusion_strength",
        "blend_alpha",
        "detail_probability",
        "predicted_regression_quality",
    }
    if set(route) != expected:
        raise RuntimeError("Unexpected V6.4 policy outputs: %s" % sorted(route))
    for name in ("generation_probability", "text_utility", "detail_probability"):
        if not bool(((route[name] >= 0.0) & (route[name] <= 1.0)).all()):
            raise RuntimeError("Out-of-range policy output: %s" % name)
    generation = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.float32)
    loss, terms = controlnet_policy_loss(
        probe,
        route,
        generation,
        torch.rand(8),
        torch.full((8,), 0.45),
        torch.full((8,), 0.25),
        torch.tensor([0, 1, 0, 1, 0, 0, 1, 0], dtype=torch.float32),
        torch.rand(8),
        mode_positive_weight=1.5,
    )
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("V6.4 loss is not finite")
    loss.backward()
    if not any(parameter.grad is not None for parameter in policy.parameters()):
        raise RuntimeError("V6.4 policy did not receive gradients")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "policy.pth"
        torch.save(
            {
                "policy": policy.state_dict(),
                "policy_config": {},
                "policy_data_image_ids": [11, 22],
                "train_image_ids": [11],
                "val_image_ids": [22],
            },
            path,
        )
        restored = load_controlnet_policy(str(path), torch.device("cpu"))
        if restored.policy_data_image_ids != {11, 22}:
            raise RuntimeError("Policy image-id audit metadata was not restored")
    print("V6.4 policy architecture: PASS")
    print("feature_dim=%d" % len(FEATURE_NAMES))
    print("outputs=%s" % sorted(expected))
    print("loss_terms=%s" % sorted(terms))


if __name__ == "__main__":
    main()
