from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import torch


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """Load both old and new PyTorch checkpoints, including PyTorch 2.6+."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a state-dict-like object")
    for key in ("model", "model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    if all(isinstance(key, str) for key in checkpoint):
        return checkpoint
    raise ValueError("Cannot find model weights in checkpoint")


def _remove_module_prefix(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def load_state_dict_flexible(
    model: torch.nn.Module,
    checkpoint: Any,
    allow_shape_expansion: bool = False,
) -> Dict[str, Any]:
    """Load exact weights or copy overlapping slices into a wider bandwidth model.

    Shape expansion is intended for changing ``latent_dim``. Existing dimensions
    are copied exactly and newly added rows/columns keep the model initializer.
    Every adapted tensor is reported so a silent partial load is impossible.
    """
    source = _remove_module_prefix(extract_state_dict(checkpoint))
    target = model.state_dict()
    prepared: Dict[str, torch.Tensor] = {}
    exact = []
    expanded = []
    skipped = []
    unexpected = []
    for key, value in source.items():
        if key not in target:
            unexpected.append(key)
            continue
        if not isinstance(value, torch.Tensor):
            skipped.append(key)
            continue
        wanted = target[key]
        if tuple(value.shape) == tuple(wanted.shape):
            prepared[key] = value
            exact.append(key)
        elif allow_shape_expansion and value.ndim == wanted.ndim:
            adapted = wanted.detach().clone()
            slices = tuple(slice(0, min(a, b)) for a, b in zip(value.shape, wanted.shape))
            adapted[slices] = value[slices].to(
                device=adapted.device,
                dtype=adapted.dtype,
            )
            prepared[key] = adapted
            expanded.append(
                {"name": key, "from": list(value.shape), "to": list(wanted.shape)}
            )
        else:
            skipped.append(key)
    incompatible = model.load_state_dict(prepared, strict=False)
    missing = sorted(set(incompatible.missing_keys))
    report = {
        "exact_count": len(exact),
        "expanded": expanded,
        "skipped": skipped,
        "unexpected": unexpected,
        "missing": missing,
    }
    if not allow_shape_expansion and (skipped or missing or unexpected):
        raise RuntimeError(
            "Strict-compatible checkpoint load failed. "
            f"skipped={skipped[:10]}, missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )
    return report


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: Dict[str, Any],
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    metrics: Dict[str, Any] | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "config": config,
        "epoch": int(epoch),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if metrics is not None:
        payload["metrics"] = metrics
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
