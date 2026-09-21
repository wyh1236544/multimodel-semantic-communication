"""Calibrate V6.5 verifier constraints on image-disjoint validation records."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from models.selective_generation_verifier import load_generation_verifier
from utils.experiment import select_device
from utils.v65_selective import raw_verification_statistics


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    dataset = torch.load(args.dataset, map_location="cpu")
    model = load_generation_verifier(args.verifier_checkpoint, device)
    val_ids = set(getattr(model, "verifier_val_image_ids", set()))
    if not val_ids:
        raise RuntimeError("Verifier checkpoint has no image-disjoint validation ids")
    records = [
        item for item in dataset["records"] if int(item["image_id"]) in val_ids
    ]
    if not records:
        raise RuntimeError("No checkpoint validation records found in dataset")
    route = torch.stack(
        [torch.as_tensor(item["route_features"]).float() for item in records]
    ).to(device)
    evidence = torch.stack(
        [torch.as_tensor(item["verifier_features"]).float() for item in records]
    ).to(device)
    target = torch.tensor(
        [float(item["accept_target"]) for item in records], device=device
    ).bool()
    prediction = model(route, evidence)
    statistics = raw_verification_statistics(evidence)

    grids = {
        "acceptance_threshold": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
        "max_hallucination_probability": [0.45, 0.55, 0.65],
        "min_semantic_gain_probability": [0.40, 0.50, 0.60],
        "min_psnr_safe_probability": [0.40, 0.50, 0.60],
        "min_ssim_safe_probability": [0.40, 0.50, 0.60],
        "min_text_alignment_gain": [-0.005, 0.0],
        "min_edge_cosine": [0.75, 0.85],
        "max_low_frequency_l1": [0.05, 0.10],
    }
    names = list(grids.keys())
    best = None
    target_rate = float(target.float().mean())
    evaluated = 0
    for values in itertools.product(*(grids[name] for name in names)):
        thresholds = dict(zip(names, values))
        accepted = (
            (prediction["acceptance_probability"] >= thresholds["acceptance_threshold"])
            & (prediction["hallucination_probability"] <= thresholds["max_hallucination_probability"])
            & (prediction["semantic_gain_probability"] >= thresholds["min_semantic_gain_probability"])
            & (prediction["psnr_safe_probability"] >= thresholds["min_psnr_safe_probability"])
            & (prediction["ssim_safe_probability"] >= thresholds["min_ssim_safe_probability"])
            & (statistics["text_alignment_gain"] >= thresholds["min_text_alignment_gain"])
            & (statistics["edge_cosine"] >= thresholds["min_edge_cosine"])
            & (statistics["low_frequency_l1"] <= thresholds["max_low_frequency_l1"])
        )
        tp = float((accepted & target).sum())
        tn = float((~accepted & ~target).sum())
        fp = float((accepted & ~target).sum())
        fn = float((~accepted & target).sum())
        tpr = tp / max(tp + fn, 1.0)
        tnr = tn / max(tn + fp, 1.0)
        precision = tp / max(tp + fp, 1.0)
        f1 = 2.0 * precision * tpr / max(precision + tpr, 1e-12)
        fpr = fp / max(fp + tn, 1.0)
        rate = float(accepted.float().mean())
        balanced = 0.5 * (tpr + tnr)
        score = balanced + 0.25 * f1 - 0.50 * fpr - 0.05 * abs(rate - target_rate)
        candidate = {
            "score": score,
            "balanced_accuracy": balanced,
            "f1": f1,
            "false_positive_rate": fpr,
            "acceptance_rate": rate,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "thresholds": thresholds,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
        evaluated += 1

    output = {
        "version": "V6.5.1",
        "validation_records": len(records),
        "validation_image_ids": sorted(val_ids),
        "validation_positive": int(target.sum()),
        "validation_negative": int((~target).sum()),
        "evaluated_combinations": evaluated,
        **best,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    print(
        "validation=%d positive=%d negative=%d combinations=%d"
        % (len(records), int(target.sum()), int((~target).sum()), evaluated)
    )
    print(
        "balanced=%.3f F1=%.3f FPR=%.3f acceptance_rate=%.3f score=%.3f"
        % (
            best["balanced_accuracy"],
            best["f1"],
            best["false_positive_rate"],
            best["acceptance_rate"],
            best["score"],
        )
    )
    print("thresholds=%s" % best["thresholds"])
    print("Saved: %s" % output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate V6.5 verifier constraints")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--verifier-checkpoint", required=True)
    parser.add_argument(
        "--output", default="V6.5.1_model_weights/verifier/verifier_thresholds.json"
    )
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
