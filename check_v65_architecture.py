"""CPU smoke test for V6.5 policy, UEP and verifier interfaces."""
from __future__ import annotations

import torch

from models.quality_adaptive_recovery import FEATURE_NAMES
from models.selective_generation_verifier import (
    SelectiveGenerationVerifier,
    verifier_loss,
)
from utils.v65_selective import (
    DEFAULT_VERIFIER_THRESHOLDS,
    VERIFIER_FEATURE_NAMES,
    hybrid_text_utility,
    verification_features,
    verifier_acceptance,
)


def main() -> None:
    torch.manual_seed(7)
    batch = 3
    route = torch.rand(batch, len(FEATURE_NAMES))
    regression = torch.rand(batch, 3, 64, 64)
    raw = torch.rand(batch, 3, 64, 64)
    candidate = (0.7 * regression + 0.3 * raw).clamp(0.0, 1.0)
    evidence = verification_features(
        regression,
        candidate,
        raw,
        torch.tensor([0.91, 0.88, 0.94]),
        torch.tensor([0.80, 0.77, 0.83]),
        torch.tensor([0.20, 0.30, 0.40]),
        torch.tensor([0.22, 0.29, 0.44]),
        torch.tensor([0.45, 0.50, 0.55]),
        torch.tensor([0.25, 0.30, 0.35]),
        torch.tensor([0.4, 0.5, 0.6]),
        torch.tensor([0.7, 0.8, 0.9]),
    )
    assert evidence.shape == (batch, len(VERIFIER_FEATURE_NAMES))
    assert bool(((evidence >= 0.0) & (evidence <= 1.0)).all())

    model = SelectiveGenerationVerifier(hidden_dim=32)
    prediction = model(route, evidence)
    required = {
        "acceptance_probability",
        "hallucination_probability",
        "predicted_delta_clip",
        "predicted_delta_psnr",
        "predicted_delta_ssim",
        "semantic_gain_probability",
        "psnr_safe_probability",
        "ssim_safe_probability",
    }
    assert set(prediction) == required
    for value in prediction.values():
        assert value.shape == (batch,)
    loss, terms = verifier_loss(
        prediction,
        torch.tensor([1.0, 0.0, 1.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.01, -0.02, 0.03]),
        torch.tensor([0.02, -0.10, 0.00]),
        torch.tensor([0.001, -0.005, 0.002]),
    )
    loss.backward()
    assert all(torch.isfinite(value) for value in terms.values())

    learned = torch.tensor([0.1, 0.2, 0.3])
    heuristic = torch.tensor([0.4, 0.5, 0.6])
    utility, safety = hybrid_text_utility(
        route,
        learned,
        heuristic,
        torch.tensor([-5.0, 5.0, 20.0]),
        probe_utility=0.5,
        generation_probability=torch.tensor([0.8, 0.4, 0.2]),
    )
    assert bool((utility >= 0.5).all())
    assert float(utility[0]) >= 0.80
    assert set(safety) == {
        "text_confidence",
        "channel_floor",
        "confidence_floor",
        "generation_floor",
    }
    accepted, checks = verifier_acceptance(
        prediction, evidence, DEFAULT_VERIFIER_THRESHOLDS
    )
    assert accepted.shape == (batch,)
    assert len(checks) == 8

    print("V6.5.1 architecture smoke test: PASS")
    print("route_feature_dim=%d" % len(FEATURE_NAMES))
    print("verifier_feature_dim=%d" % len(VERIFIER_FEATURE_NAMES))
    print("outputs=%s" % sorted(prediction.keys()))
    print("loss_terms=%s" % sorted(terms.keys()))
    print("hybrid_text_utility=%s" % [round(float(value), 3) for value in utility])


if __name__ == "__main__":
    main()
