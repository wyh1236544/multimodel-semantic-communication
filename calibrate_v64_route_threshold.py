"""Calibrate the V6.4 generation threshold on policy-validation image IDs."""
from __future__ import annotations

import argparse

import torch

from models.controlnet_fusion_policy import load_controlnet_policy
from utils.experiment import select_device


def _metrics(probability, target, threshold):
    predicted = probability >= threshold
    target = target.bool()
    tp = float((predicted & target).sum())
    tn = float((~predicted & ~target).sum())
    fp = float((predicted & ~target).sum())
    fn = float((~predicted & target).sum())
    tpr = tp / max(tp + fn, 1.0)
    tnr = tn / max(tn + fp, 1.0)
    precision = tp / max(tp + fp, 1.0)
    f1 = 2.0 * precision * tpr / max(precision + tpr, 1e-12)
    fpr = fp / max(fp + tn, 1.0)
    return {
        "balanced": 0.5 * (tpr + tnr),
        "f1": f1,
        "fpr": fpr,
        "generation_rate": float(predicted.float().mean()),
    }


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    raw = torch.load(args.policy_checkpoint, map_location="cpu")
    val_ids = {int(value) for value in raw.get("val_image_ids", [])}
    if not val_ids:
        raise RuntimeError("Policy checkpoint has no validation image IDs")
    dataset = torch.load(args.dataset, map_location="cpu")
    records = [
        item for item in dataset["records"] if int(item["image_id"]) in val_ids
    ]
    if not records:
        raise RuntimeError("No policy records match checkpoint validation IDs")
    policy = load_controlnet_policy(args.policy_checkpoint, device)
    features = torch.stack(
        [torch.as_tensor(item["route_features"]).float() for item in records]
    ).to(device)
    target = torch.tensor(
        [float(item["generation_target"]) for item in records], device=device
    )
    probability = policy(features)["generation_probability"]
    print(
        "validation_records=%d positive=%d negative=%d probability_min=%.4f max=%.4f"
        % (
            len(records), int(target.sum()), len(records) - int(target.sum()),
            float(probability.min()), float(probability.max()),
        )
    )
    best = None
    threshold = args.start
    while threshold <= args.stop + 1e-9:
        metrics = _metrics(probability, target, threshold)
        # False-positive generation incurs latency and typically loses
        # distortion fidelity, so ties favor a conservative threshold.
        score = metrics["balanced"] + 0.10 * metrics["f1"] - args.fp_penalty * metrics["fpr"]
        print(
            "threshold=%.2f balanced=%.3f F1=%.3f FPR=%.3f generation_rate=%.3f score=%.3f"
            % (
                threshold, metrics["balanced"], metrics["f1"], metrics["fpr"],
                metrics["generation_rate"], score,
            )
        )
        candidate = (score, metrics["balanced"], -metrics["fpr"], threshold)
        if best is None or candidate > best:
            best = candidate
        threshold += args.step
    print("recommended_route_threshold=%.2f" % best[3])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate V6.4 route threshold")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--start", type=float, default=0.30)
    parser.add_argument("--stop", type=float, default=0.90)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--fp-penalty", type=float, default=0.10)
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
