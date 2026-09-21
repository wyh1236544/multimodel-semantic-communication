from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be an object: {path}")
    return data


def prepare_config(
    config: Dict[str, Any],
    preset: Optional[str] = None,
    vae_path: Optional[str] = None,
    text_model_path: Optional[str] = None,
    data_root: Optional[str] = None,
    modality_token_budget: Optional[int] = None,
    utility_weight_choices: Optional[list[float]] = None,
) -> Dict[str, Any]:
    """Copy a config and apply command-line path/bandwidth overrides."""
    result = copy.deepcopy(config)
    model = result.setdefault("model", {})
    if preset:
        model["bandwidth"] = {"enable": True, "preset": preset}
    if modality_token_budget is not None:
        allocation = model.setdefault("dynamic_token_allocation", {})
        if not bool(allocation.get("enable", False)):
            raise ValueError(
                "modality_token_budget requires model.dynamic_token_allocation.enable=true"
            )
        budget = int(modality_token_budget)
        num_image = int(model["num_img_tokens"])
        num_text = int(model["num_txt_tokens"])
        min_text = int(allocation.get("min_text_tokens", 0))
        max_text = int(allocation.get("max_text_tokens", num_text))
        low = max(min_text, budget - num_image, 0)
        high = min(max_text, num_text, budget)
        if budget <= 0 or low > high:
            raise ValueError(
                f"Invalid modality token budget {budget}: candidate pools are "
                f"image={num_image}, text={num_text}, feasible text quota=[{low}, {high}]"
            )
        allocation["modality_token_budget"] = budget
    if utility_weight_choices is not None:
        choices = [float(value) for value in utility_weight_choices]
        if not choices or any(value < 0.0 or value > 1.0 for value in choices):
            raise ValueError("utility_weight_choices must contain values in [0, 1]")
        result.setdefault("train", {})["utility_weight_choices"] = choices
    if vae_path:
        model["vae_model_name_or_path"] = vae_path
    if text_model_path:
        model["text_model_name"] = text_model_path
    if data_root:
        root = Path(data_root)
        data = result.setdefault("data", {})
        localized_root = root / "annotations" / "localized_narratives"
        data.update(
            {
                "train_image_root": str(root / "train2017"),
                "val_image_root": str(root / "val2017"),
                "train_annotation_file": str(
                    localized_root / "coco_train_captions.jsonl"
                ),
                "val_annotation_file": str(
                    localized_root / "coco_val_captions.jsonl"
                ),
            }
        )
    return result


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
