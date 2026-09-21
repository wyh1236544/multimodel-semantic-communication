"""Train the lightweight V6.2 bandwidth and regression/generation policy."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from models.quality_adaptive_recovery import (
    FEATURE_NAMES,
    QualityAdaptivePolicy,
    policy_loss,
)
from utils.experiment import seed_everything, select_device


class PolicyDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, object]]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        item = self.records[index]
        return {
            "features": torch.as_tensor(item["features"]).float(),
            "generation": torch.tensor(float(item["generation_target"])),
            "text": torch.tensor(float(item["text_utility_target"])),
            "strength": torch.tensor(float(item["strength_target"])),
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
    positive_ids = [image_id for image_id in image_ids if positive_by_image[image_id]]
    negative_ids = [image_id for image_id in image_ids if not positive_by_image[image_id]]
    generator = random.Random(seed)
    generator.shuffle(positive_ids)
    generator.shuffle(negative_ids)

    def validation_slice(group):
        if len(group) <= 1:
            return []
        count = int(round(len(group) * val_fraction))
        count = min(max(count, 1), len(group) - 1)
        return group[:count]

    val_ids = set(validation_slice(positive_ids) + validation_slice(negative_ids))
    train = [item for item in records if int(item["image_id"]) not in val_ids]
    val = [item for item in records if int(item["image_id"]) in val_ids]
    return train, val


@torch.no_grad()
def evaluate(policy, loader, device):
    policy.eval()
    totals = {
        "count": 0.0,
        "correct": 0.0,
        "text_abs": 0.0,
        "strength_abs": 0.0,
        "tp": 0.0,
        "tn": 0.0,
        "fp": 0.0,
        "fn": 0.0,
    }
    for batch in loader:
        features = batch["features"].to(device)
        output = policy(features)
        generation = batch["generation"].to(device)
        predicted = output["generation_probability"] >= 0.5
        target = generation.bool()
        totals["count"] += float(features.size(0))
        totals["correct"] += float(
            (predicted == target).sum()
        )
        totals["tp"] += float((predicted & target).sum())
        totals["tn"] += float((~predicted & ~target).sum())
        totals["fp"] += float((predicted & ~target).sum())
        totals["fn"] += float((~predicted & target).sum())
        totals["text_abs"] += float(
            (output["text_utility"] - batch["text"].to(device)).abs().sum()
        )
        totals["strength_abs"] += float(
            (output["diffusion_strength"] - batch["strength"].to(device)).abs().sum()
        )
    count = max(totals["count"], 1.0)
    true_positive_rate = totals["tp"] / max(totals["tp"] + totals["fn"], 1.0)
    true_negative_rate = totals["tn"] / max(totals["tn"] + totals["fp"], 1.0)
    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1.0)
    recall = true_positive_rate
    return {
        "mode_accuracy": totals["correct"] / count,
        "mode_balanced_accuracy": 0.5 * (true_positive_rate + true_negative_rate),
        "mode_precision": precision,
        "mode_recall": recall,
        "mode_f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "text_mae": totals["text_abs"] / count,
        "strength_mae": totals["strength_abs"] / count,
    }


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_device(args.device)
    dataset = torch.load(args.dataset, map_location="cpu")
    records = list(dataset["records"])
    if not records:
        raise RuntimeError("Policy dataset has no records")
    if list(dataset.get("feature_names", [])) != list(FEATURE_NAMES):
        raise RuntimeError(
            "Policy feature schema does not match this code version. "
            "Re-run collect_v62_policy_data.py."
        )
    if any(torch.as_tensor(item["features"]).numel() != len(FEATURE_NAMES) for item in records):
        raise RuntimeError("At least one policy record has the wrong feature dimension")
    positive_count = sum(float(item["generation_target"]) >= 0.5 for item in records)
    negative_count = len(records) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise RuntimeError(
            "Generation oracle labels contain only one class "
            "(positive=%d, negative=%d). A meaningful router cannot be trained; "
            "improve/retrain the semantic adapter or adjust the oracle utility, "
            "then recollect data."
            % (positive_count, negative_count)
        )
    mode_positive_weight = max(
        0.25, min(float(negative_count) / float(positive_count), 10.0)
    )
    train_records, val_records = split_by_image(records, args.val_fraction, args.seed)
    if not train_records or not val_records:
        raise RuntimeError("Image-disjoint train/validation split is empty; collect more images")
    val_positive = sum(float(item["generation_target"]) >= 0.5 for item in val_records)
    if val_positive == 0 or val_positive == len(val_records):
        raise RuntimeError(
            "Image-disjoint validation has only one routing class. Collect more "
            "policy images so balanced accuracy is meaningful."
        )
    train_image_ids = sorted({int(item["image_id"]) for item in train_records})
    val_image_ids = sorted({int(item["image_id"]) for item in val_records})
    policy_data_image_ids = sorted(set(train_image_ids) | set(val_image_ids))
    train_loader = DataLoader(
        PolicyDataset(train_records), batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(
        PolicyDataset(val_records), batch_size=args.batch_size, shuffle=False,
        num_workers=0, drop_last=False,
    )
    policy_config = {
        "hidden_dim": args.hidden_dim,
        "min_strength": args.min_strength,
        "max_strength": args.max_strength,
    }
    policy = QualityAdaptivePolicy(**policy_config).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "train=%d val=%d device=%s generation_positive=%d/%d pos_weight=%.3f"
        % (
            len(train_records), len(val_records), device,
            positive_count, len(records), mode_positive_weight,
        )
    )
    best_score = -1e9
    for epoch in range(args.epochs):
        policy.train()
        running = 0.0
        for batch in train_loader:
            features = batch["features"].to(device)
            predictions = policy(features)
            loss, terms = policy_loss(
                predictions,
                batch["generation"].to(device),
                batch["text"].to(device),
                batch["strength"].to(device),
                batch["quality"].to(device),
                mode_positive_weight=mode_positive_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * features.size(0)
        metrics = evaluate(policy, val_loader, device)
        selection_score = metrics["mode_balanced_accuracy"] - metrics["text_mae"]
        print(
            "epoch=%d loss=%.5f mode_acc=%.3f balanced=%.3f F1=%.3f text_mae=%.3f strength_mae=%.3f"
            % (
                epoch + 1,
                running / max(len(train_records), 1),
                metrics["mode_accuracy"],
                metrics["mode_balanced_accuracy"],
                metrics["mode_f1"],
                metrics["text_mae"],
                metrics["strength_mae"],
            )
        )
        payload = {
            "policy": policy.state_dict(),
            "policy_config": policy_config,
            "epoch": epoch + 1,
            "metrics": metrics,
            "feature_names": dataset.get("feature_names"),
            "train_image_ids": train_image_ids,
            "val_image_ids": val_image_ids,
            "policy_data_image_ids": policy_data_image_ids,
            "policy_dataset_metadata": dataset.get("metadata", {}),
        }
        torch.save(payload, output_dir / "quality_policy_latest.pth")
        if selection_score > best_score:
            best_score = selection_score
            torch.save(payload, output_dir / "quality_policy_best.pth")
    print("Best policy: %s" % (output_dir / "quality_policy_best.pth"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train V6.2 quality-adaptive policy")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="V6.2_model_weights/policy")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--min-strength", type=float, default=0.10)
    parser.add_argument("--max-strength", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
