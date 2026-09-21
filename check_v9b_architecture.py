"""CPU-only shape and loss smoke test for V9-B."""
import torch

from models.quality_adaptive_recovery import FEATURE_NAMES
from models.v9b_common_verifier import V9BCommonGuidedVerifier, v9b_verifier_loss
from utils.v65_selective import VERIFIER_FEATURE_NAMES
from utils.v9b_selective import PROMPT_FEATURE_NAMES, prompt_quality_features, sanitize_received_caption


def main():
    batch = 4
    model = V9BCommonGuidedVerifier(common_dim=256, clip_dim=512, hidden_dim=32)
    prediction = model(
        torch.rand(batch, len(FEATURE_NAMES)),
        torch.rand(batch, len(VERIFIER_FEATURE_NAMES)),
        torch.rand(batch, len(PROMPT_FEATURE_NAMES)),
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        torch.rand(batch, 256),
        torch.rand(batch, 512),
        torch.rand(batch, 512),
        torch.rand(batch, 512),
    )
    loss, _ = v9b_verifier_loss(
        prediction,
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        torch.tensor([-0.02, 0.03, -0.01, 0.04]),
        torch.tensor([-1.2, -0.4, -0.8, -0.3]),
        torch.tensor([-0.12, -0.04, -0.09, -0.03]),
        torch.nn.functional.normalize(torch.rand(batch, 512), dim=-1),
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
    )
    loss.backward()
    collapsed = "the the the people people people people on the the road"
    sanitized = sanitize_received_caption(collapsed)
    quality = prompt_quality_features([collapsed])
    assert prediction["acceptance_probability"].shape == (batch,)
    assert prediction["predicted_utility"].shape == (batch,)
    assert torch.isfinite(loss)
    assert quality.shape == (1, len(PROMPT_FEATURE_NAMES))
    assert sanitized.count("people") <= 3 and sanitized.count("the") <= 2
    print(f"V9-B architecture OK loss={float(loss):.4f}")
    print(f"sanitized prompt: {sanitized}")


if __name__ == "__main__":
    main()
