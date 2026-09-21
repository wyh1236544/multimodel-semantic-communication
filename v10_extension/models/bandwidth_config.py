"""Load and validate the optional channel-bandwidth preset file.

The original V6.1 model uses ``latent_dim`` real channel symbols for every
transmitted semantic token.  This helper changes only that projection width;
the common/image/text token allocation and the rest of the model stay intact.
If the JSON file is absent or disabled, the caller's original config is
returned unchanged.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union


DEFAULT_BANDWIDTH_CONFIG = Path(__file__).with_name("bandwidth_config.json")


def transmitted_token_count(config: Mapping[str, Any]) -> int:
    """Return the number of semantic tokens actually sent through the channel."""
    common = int(config["num_common_tokens"])
    allocation = config.get("dynamic_token_allocation", {})
    if bool(allocation.get("enable", False)):
        modality = int(
            allocation.get(
                "modality_token_budget",
                int(config["num_img_tokens"]) + int(config["num_txt_tokens"]),
            )
        )
    else:
        modality = int(config["num_img_tokens"]) + int(config["num_txt_tokens"])
    total = common + modality
    if total <= 0:
        raise ValueError("The transmitted semantic-token count must be positive")
    return total


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Bandwidth config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in bandwidth config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Bandwidth config root must be a JSON object")
    return data


def resolve_bandwidth_config(
    model_config: Mapping[str, Any],
    config_path: Optional[Union[str, Path]] = None,
    preset: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a copied model config with one validated bandwidth preset applied.

    Selection order:
      1. explicit ``preset`` argument;
      2. ``model_config['bandwidth']['preset']``;
      3. JSON ``active_preset``.

    Put ``bandwidth: {enable: false}`` in the normal model config to bypass the
    adjacent JSON file and preserve the caller's ``latent_dim`` verbatim.
    """
    resolved = copy.deepcopy(dict(model_config))
    inline = resolved.get("bandwidth", {})
    if inline is False:
        return resolved
    if inline is None:
        inline = {}
    if not isinstance(inline, Mapping):
        raise ValueError("model_config['bandwidth'] must be an object or false")
    if not bool(inline.get("enable", True)):
        return resolved

    selected_path = Path(
        config_path or inline.get("config_path") or DEFAULT_BANDWIDTH_CONFIG
    ).expanduser()
    # Backward compatibility: an old models directory has no preset file.
    if not selected_path.exists() and config_path is None and "config_path" not in inline:
        return resolved

    bandwidth_file = _load_json(selected_path)
    if not bool(bandwidth_file.get("enabled", True)):
        return resolved
    presets = bandwidth_file.get("presets")
    if not isinstance(presets, Mapping) or not presets:
        raise ValueError("bandwidth_config.json must contain a non-empty 'presets' object")

    selected_name = str(
        preset or inline.get("preset") or bandwidth_file.get("active_preset", "")
    )
    if selected_name not in presets:
        choices = ", ".join(sorted(str(name) for name in presets))
        raise ValueError(
            f"Unknown bandwidth preset '{selected_name}'. Available presets: {choices}"
        )
    profile = presets[selected_name]
    if not isinstance(profile, Mapping):
        raise ValueError(f"Bandwidth preset '{selected_name}' must be an object")

    # Optional token overrides are supported for research sweeps, but the
    # provided presets deliberately omit them to preserve the original split.
    for key in ("num_common_tokens", "num_img_tokens", "num_txt_tokens"):
        if key in profile:
            value = int(profile[key])
            if value <= 0:
                raise ValueError(f"{key} must be positive")
            resolved[key] = value

    token_count = transmitted_token_count(resolved)
    has_per_token = "symbols_per_token" in profile
    has_total = "target_channel_symbols" in profile
    if has_per_token == has_total:
        raise ValueError(
            f"Preset '{selected_name}' must define exactly one of "
            "'symbols_per_token' and 'target_channel_symbols'"
        )
    if has_per_token:
        symbols_per_token = int(profile["symbols_per_token"])
        if symbols_per_token <= 0:
            raise ValueError("symbols_per_token must be positive")
        total_symbols = token_count * symbols_per_token
    else:
        total_symbols = int(profile["target_channel_symbols"])
        if total_symbols <= 0 or total_symbols % token_count != 0:
            raise ValueError(
                f"target_channel_symbols={total_symbols} is not a positive multiple "
                f"of the current transmitted token count ({token_count})"
            )
        symbols_per_token = total_symbols // token_count

    resolved["latent_dim"] = symbols_per_token
    resolved["_resolved_bandwidth"] = {
        "preset": selected_name,
        "config_path": str(selected_path.resolve()),
        "channel_tokens": token_count,
        "symbols_per_token": symbols_per_token,
        "channel_uses": total_symbols,
        "description": str(profile.get("description", "")),
    }
    return resolved


def bandwidth_summary(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Summarize the effective bandwidth of an already resolved model config."""
    token_count = transmitted_token_count(config)
    symbols_per_token = int(config["latent_dim"])
    resolved = config.get("_resolved_bandwidth", {})
    return {
        "preset": resolved.get("preset", "model_config"),
        "channel_tokens": token_count,
        "symbols_per_token": symbols_per_token,
        "channel_uses": token_count * symbols_per_token,
    }
