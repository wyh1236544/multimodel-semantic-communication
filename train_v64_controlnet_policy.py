"""Train the deployable receiver-only V6.4 ControlNet/fusion policy."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from models.controlnet_fusion_policy import (
    ControlNetFusionPolicy,
    controlnet_policy_loss,
)
from models.quality_adaptive_recovery import FEATURE_NAMES
from utils.experiment import seed_everything, select_device


class PolicyDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, object]]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        item = self.records[index]
        return {
            "probe_features": torch.as_tensor(item["probe_features"]).float(),
            "route_features": torch.as_tensor(item["route_features"]).float(),
            "generation": torch.tensor(float(item["generation_target"])),
            "text": torch.tensor(float(item["text_utility_target"])),
            "strength": torch.tensor(float(item["strength_target"])),
            "alpha": torch.tensor(float(item["blend_alpha_target"])),
            "detail": torch.tensor(float(item["detail_target"])),
            "quality": torch.tensor(float(item["regression_quality_target"])),
        }


def split_by_image(records: List[Dict[str, object]], val_fraction: float, seed: int):
    image_ids = sorted({int(item["image_id"]) for item in records})
    positive_by_image = {
        image_id: any(
            int(item["image_id"]) == image_id
            and float(item["generation_target"]) >= 0.5
            for item in records
        )
        for image_id in image_ids
    }
    positive = [value for value in image_ids if positive_by_image[value]]
    negative = [value for value in image_ids if not positive_by_image[value]]
    rng = random.Random(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)

    def validation_slice(values):
        if len(values) <= 1:
            return []
        count = min(max(int(round(len(values) * val_fraction)), 1), len(values) - 1)
        return values[:count]

    val_ids = set(validation_slice(positive) + validation_slice(negative))
    train = [item for item in records if int(item["image_id"]) not in val_ids]
    val = [item for item in records if int(item["image_id"]) in val_ids]
    return train, val


@torch.no_grad()
def evaluate(policy, loader, device):
    policy.eval()
    totals = {
        "count": 0.0,
        "correct": 0.0,
        "tp": 0.0,
        "tn": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "text_abs": 0.0,
        "strength_abs": 0.0,
        "alpha_abs": 0.0,
        "expert_correct": 0.0,
        "positive_count": 0.0,
    }
    for batch in loader:
        probe = policy(batch["probe_features"].to(device))
        route = policy(batch["route_features"].to(device))
        target = batch["generation"].to(device).bool()
        predicted = route["generation_probability"] >= 0.5
        totals["count"] += float(target.numel())
        totals["correct"] += float((predicted == target).sum())
        totals["tp"] += float((predicted & target).sum())
        totals["tn"] += float((~predicted & ~target).sum())
        totals["fp"] += float((predicted & ~target).sum())
        totals["fn"] += float((~predicted & target).sum())
        totals["text_abs"] += float(
            (probe["text_utility"] - batch["text"].to(device)).abs().sum()
        )
        mask = target.float()
        totals["positive_count"] += float(mask.sum())
        totals["strength_abs"] += float(
            ((route["diffusion_strength"] - batch["strength"].to(device)).abs() * mask).sum()
        )
        totals["alpha_abs"] += float(
            ((route["blend_alpha"] - batch["alpha"].to(device)).abs() * mask).sum()
        )
        expert = route["detail_probability"] >= 0.5
        totals["expert_correct"] += float(
            ((expert == batch["detail"].to(device).bool()).float() * mask).sum()
        )
    count = max(totals["count"], 1.0)
    positives = max(totals["positive_count"], 1.0)
    tpr = totals["tp"] / max(totals["tp"] + totals["fn"], 1.0)
    tnr = totals["tn"] / max(totals["tn"] + totals["fp"], 1.0)
    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1.0)
    return {
        "mode_accuracy": totals["correct"] / count,
        "mode_balanced_accuracy": 0.5 * (tpr + tnr),
        "mode_precision": precision,
        "mode_recall": tpr,
        "mode_f1": 2.0 * precision * tpr / max(precision + tpr, 1e-12),
        "text_mae": totals["text_abs"] / count,
        "strength_mae_positive": totals["strength_abs"] / positives,
        "alpha_mae_positive": totals["alpha_abs"] / positives,
        "expert_accuracy_positive": totals["expert_correct"] / positives,
    }


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_device(args.device)
    dataset = torch.load(args.dataset, map_location="cpu")
    records = list(dataset["records"])
    if not records:
        raise RuntimeError("V6.4 policy dataset is empty")
    if list(dataset.get("feature_names", [])) != list(FEATURE_NAMES):
        raise RuntimeError("V6.4 policy feature schema mismatch; recollect data")
    for item in records:
        if torch.as_tensor(item["probe_features"]).numel() != len(FEATURE_NAMES):
            raise RuntimeError("Wrong probe feature dimension")
        if torch.as_tensor(item["route_features"]).numel() != len(FEATURE_NAMES):
            raise RuntimeError("Wrong route feature dimension")
    positive_count = sum(float(item["generation_target"]) >= 0.5 for item in records)
    negative_count = len(records) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise RuntimeError(
            "Routing labels have one class only (positive=%d negative=%d)"
            % (positive_count, negative_count)
        )
    positive_weight = max(0.25, min(negative_count / float(positive_count), 10.0))
    train_records, val_records = split_by_image(
        records, args.val_fraction, args.seed
    )
    if not train_records or not val_records:
        raise RuntimeError("Image-disjoint train/validation split is empty")
    val_positive = sum(float(item["generation_target"]) >= 0.5 for item in val_records)
    if val_positive == 0 or val_positive == len(val_records):
        raise RuntimeError("Validation split contains only one routing class")
    train_ids = sorted({int(item["image_id"]) for item in train_records})
    val_ids = sorted({int(item["image_id"]) for item in val_records})
    all_ids = sorted(set(train_ids) | set(val_ids))
    train_loader = DataLoader(
        PolicyDataset(train_records), batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        PolicyDataset(val_records), batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    policy_config = {
        "hidden_dim": args.hidden_dim,
        "min_strength": args.min_strength,
        "max_strength": args.max_strength,
        "min_blend_alpha": args.min_blend_alpha,
        "max_blend_alpha": args.max_blend_alpha,
    }
    policy = ControlNetFusionPolicy(**policy_config).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "train=%d val=%d positive=%d/%d pos_weight=%.3f device=%s"
        % (len(train_records), len(val_records), positive_count, len(records), positive_weight, device)
    )
    best_score = -1e9
    for epoch in range(args.epochs):
        policy.train()
        running = 0.0
        for batch in train_loader:
            probe = policy(batch["probe_features"].to(device))
            route = policy(batch["route_features"].to(device))
            loss, _ = controlnet_policy_loss(
                probe,
                route,
                batch["generation"].to(device),
                batch["text"].to(device),
                batch["strength"].to(device),
                batch["alpha"].to(device),
                batch["detail"].to(device),
                batch["quality"].to(device),
                mode_positive_weight=positive_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * batch["generation"].numel()
        metrics = evaluate(policy, val_loader, device)
        score = (
            metrics["mode_balanced_accuracy"]
            - metrics["text_mae"]
            - 0.25 * metrics["alpha_mae_positive"]
        )
        print(
            "epoch=%d loss=%.5f balanced=%.3f F1=%.3f text_mae=%.3f strength_mae=%.3f alpha_mae=%.3f expert_acc=%.3f"
            % (
                epoch + 1,
                running / len(train_records),
                metrics["mode_balanced_accuracy"],
                metrics["mode_f1"],
                metrics["text_mae"],
                metrics["strength_mae_positive"],
                metrics["alpha_mae_positive"],
                metrics["expert_accuracy_positive"],
            )
        )
        payload = {
            "policy": policy.state_dict(),
            "policy_config": policy_config,
            "epoch": epoch + 1,
            "metrics": metrics,
            "feature_names": FEATURE_NAMES,
            "train_image_ids": train_ids,
            "val_image_ids": val_ids,
            "policy_data_image_ids": all_ids,
            "policy_dataset_metadata": dataset.get("metadata", {}),
        }
        torch.save(payload, output_dir / "controlnet_policy_latest.pth")
        if score > best_score:
            best_score = score
            torch.save(payload, output_dir / "controlnet_policy_best.pth")
    print("Best policy: %s" % (output_dir / "controlnet_policy_best.pth"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train V6.4 receiver-only policy")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="V6.4_model_weights/policy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--min-strength", type=float, default=0.20)
    parser.add_argument("--max-strength", type=float, default=0.70)
    parser.add_argument("--min-blend-alpha", type=float, default=0.10)
    parser.add_argument("--max-blend-alpha", type=float, default=0.55)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
