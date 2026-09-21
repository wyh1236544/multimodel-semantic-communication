"""Train the V6.5 receiver-only post-generation verifier."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from models.quality_adaptive_recovery import FEATURE_NAMES
from models.selective_generation_verifier import (
    SelectiveGenerationVerifier,
    verifier_loss,
)
from utils.experiment import seed_everything, select_device
from utils.v65_selective import VERIFIER_FEATURE_NAMES


PRIMARY_PROBABILITY_INDEX = VERIFIER_FEATURE_NAMES.index(
    "primary_generation_probability"
)


class VerifierDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, object]]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        item = self.records[index]
        verifier_features = torch.as_tensor(item["verifier_features"]).float()
        return {
            "route_features": torch.as_tensor(item["route_features"]).float(),
            "verifier_features": verifier_features,
            "primary_probability": verifier_features[PRIMARY_PROBABILITY_INDEX],
            "accept": torch.tensor(float(item["accept_target"])),
            "hallucination": torch.tensor(float(item["hallucination_target"])),
            "delta_clip": torch.tensor(float(item["delta_clip_target"])),
            "delta_psnr": torch.tensor(float(item["delta_psnr_target"])),
            "delta_ssim": torch.tensor(float(item["delta_ssim_target"])),
        }


def split_by_image(records: List[Dict[str, object]], val_fraction: float, seed: int):
    image_ids = sorted({int(item["image_id"]) for item in records})
    positive_by_image = {
        image_id: any(
            int(item["image_id"]) == image_id
            and float(item["accept_target"]) >= 0.5
            for item in records
        )
        for image_id in image_ids
    }
    groups = [
        [value for value in image_ids if positive_by_image[value]],
        [value for value in image_ids if not positive_by_image[value]],
    ]
    rng = random.Random(seed)
    val_ids = set()
    for values in groups:
        rng.shuffle(values)
        if len(values) > 1:
            count = min(max(int(round(len(values) * val_fraction)), 1), len(values) - 1)
            val_ids.update(values[:count])
    train = [item for item in records if int(item["image_id"]) not in val_ids]
    val = [item for item in records if int(item["image_id"]) in val_ids]
    return train, val


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = {
        "count": 0.0,
        "tp": 0.0,
        "tn": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "clip_abs": 0.0,
        "psnr_abs": 0.0,
        "ssim_abs": 0.0,
        "hallucination_correct": 0.0,
        "semantic_correct": 0.0,
        "psnr_safe_correct": 0.0,
        "ssim_safe_correct": 0.0,
    }
    for batch in loader:
        prediction = model(
            batch["route_features"].to(device),
            batch["verifier_features"].to(device),
        )
        target = batch["accept"].to(device).bool()
        predicted = prediction["acceptance_probability"] >= 0.5
        hallucination = batch["hallucination"].to(device).bool()
        predicted_hallucination = prediction["hallucination_probability"] >= 0.5
        semantic_target = batch["delta_clip"].to(device) >= 0.003
        psnr_safe_target = batch["delta_psnr"].to(device) >= -0.05
        ssim_safe_target = batch["delta_ssim"].to(device) >= -0.004
        totals["count"] += float(target.numel())
        totals["tp"] += float((predicted & target).sum())
        totals["tn"] += float((~predicted & ~target).sum())
        totals["fp"] += float((predicted & ~target).sum())
        totals["fn"] += float((~predicted & target).sum())
        totals["hallucination_correct"] += float(
            (predicted_hallucination == hallucination).sum()
        )
        totals["semantic_correct"] += float(
            ((prediction["semantic_gain_probability"] >= 0.5) == semantic_target).sum()
        )
        totals["psnr_safe_correct"] += float(
            ((prediction["psnr_safe_probability"] >= 0.5) == psnr_safe_target).sum()
        )
        totals["ssim_safe_correct"] += float(
            ((prediction["ssim_safe_probability"] >= 0.5) == ssim_safe_target).sum()
        )
        totals["clip_abs"] += float(
            (
                prediction["predicted_delta_clip"]
                - batch["delta_clip"].to(device)
            ).abs().sum()
        )
        totals["psnr_abs"] += float(
            (
                prediction["predicted_delta_psnr"]
                - batch["delta_psnr"].to(device)
            ).abs().sum()
        )
        totals["ssim_abs"] += float(
            (
                prediction["predicted_delta_ssim"]
                - batch["delta_ssim"].to(device)
            ).abs().sum()
        )
    count = max(totals["count"], 1.0)
    tpr = totals["tp"] / max(totals["tp"] + totals["fn"], 1.0)
    tnr = totals["tn"] / max(totals["tn"] + totals["fp"], 1.0)
    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1.0)
    return {
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "precision": precision,
        "recall": tpr,
        "f1": 2.0 * precision * tpr / max(precision + tpr, 1e-12),
        "false_positive_rate": totals["fp"] / max(totals["fp"] + totals["tn"], 1.0),
        "hallucination_accuracy": totals["hallucination_correct"] / count,
        "semantic_gain_accuracy": totals["semantic_correct"] / count,
        "psnr_safe_accuracy": totals["psnr_safe_correct"] / count,
        "ssim_safe_accuracy": totals["ssim_safe_correct"] / count,
        "delta_clip_mae": totals["clip_abs"] / count,
        "delta_psnr_mae": totals["psnr_abs"] / count,
        "delta_ssim_mae": totals["ssim_abs"] / count,
    }


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_device(args.device)
    dataset = torch.load(args.dataset, map_location="cpu")
    records = list(dataset["records"])
    if not records:
        raise RuntimeError("V6.5 verifier dataset is empty")
    if list(dataset.get("route_feature_names", [])) != list(FEATURE_NAMES):
        raise RuntimeError("V6.5 route feature schema mismatch")
    if list(dataset.get("verifier_feature_names", [])) != list(VERIFIER_FEATURE_NAMES):
        raise RuntimeError("V6.5 verifier feature schema mismatch")
    for item in records:
        if torch.as_tensor(item["route_features"]).numel() != len(FEATURE_NAMES):
            raise RuntimeError("Wrong route feature dimension")
        if torch.as_tensor(item["verifier_features"]).numel() != len(VERIFIER_FEATURE_NAMES):
            raise RuntimeError("Wrong verifier feature dimension")
    positive_count = sum(float(item["accept_target"]) >= 0.5 for item in records)
    negative_count = len(records) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise RuntimeError(
            "Verifier labels have one class only (positive=%d negative=%d)"
            % (positive_count, negative_count)
        )
    positive_weight = max(0.25, min(negative_count / float(positive_count), 10.0))
    train_records, val_records = split_by_image(records, args.val_fraction, args.seed)
    if not train_records or not val_records:
        raise RuntimeError("Image-disjoint train/validation split is empty")
    val_positive = sum(float(item["accept_target"]) >= 0.5 for item in val_records)
    if val_positive == 0 or val_positive == len(val_records):
        raise RuntimeError("Validation split contains only one acceptance class")

    train_ids = sorted({int(item["image_id"]) for item in train_records})
    val_ids = sorted({int(item["image_id"]) for item in val_records})
    all_ids = sorted(set(train_ids) | set(val_ids))
    train_loader = DataLoader(
        VerifierDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        VerifierDataset(val_records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    verifier_config = {"hidden_dim": args.hidden_dim}
    model = SelectiveGenerationVerifier(**verifier_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "train=%d val=%d positive=%d/%d pos_weight=%.3f device=%s"
        % (
            len(train_records),
            len(val_records),
            positive_count,
            len(records),
            positive_weight,
            device,
        )
    )
    best_score = -1e9
    metadata = dataset.get("metadata", {})
    upstream_ids = {
        int(value) for value in metadata.get("v64_policy_data_image_ids", [])
    }
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            prediction = model(
                batch["route_features"].to(device),
                batch["verifier_features"].to(device),
            )
            loss, _ = verifier_loss(
                prediction,
                batch["accept"].to(device),
                batch["hallucination"].to(device),
                batch["delta_clip"].to(device),
                batch["delta_psnr"].to(device),
                batch["delta_ssim"].to(device),
                positive_weight=positive_weight,
                deployment_weight=(
                    1.0
                    + args.deployment_weight
                    * (
                        (batch["primary_probability"].to(device) - 0.50) / 0.50
                    ).clamp(0.0, 1.0)
                ),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * batch["accept"].numel()
        metrics = evaluate(model, val_loader, device)
        score = (
            metrics["balanced_accuracy"]
            + 0.25 * metrics["f1"]
            - 0.35 * metrics["false_positive_rate"]
            - metrics["delta_clip_mae"]
            - 0.05 * metrics["delta_psnr_mae"]
            - metrics["delta_ssim_mae"]
            + 0.10 * metrics["semantic_gain_accuracy"]
            + 0.10 * metrics["psnr_safe_accuracy"]
            + 0.10 * metrics["ssim_safe_accuracy"]
        )
        print(
            "epoch=%d loss=%.5f balanced=%.3f F1=%.3f FPR=%.3f hall=%.3f sem=%.3f psnr_safe=%.3f ssim_safe=%.3f dCLIP_mae=%.4f dPSNR_mae=%.3f dSSIM_mae=%.4f"
            % (
                epoch + 1,
                running / len(train_records),
                metrics["balanced_accuracy"],
                metrics["f1"],
                metrics["false_positive_rate"],
                metrics["hallucination_accuracy"],
                metrics["semantic_gain_accuracy"],
                metrics["psnr_safe_accuracy"],
                metrics["ssim_safe_accuracy"],
                metrics["delta_clip_mae"],
                metrics["delta_psnr_mae"],
                metrics["delta_ssim_mae"],
            )
        )
        payload = {
            "verifier": model.state_dict(),
            "verifier_config": verifier_config,
            "epoch": epoch + 1,
            "metrics": metrics,
            "route_feature_names": FEATURE_NAMES,
            "verifier_feature_names": VERIFIER_FEATURE_NAMES,
            "train_image_ids": train_ids,
            "val_image_ids": val_ids,
            "verifier_data_image_ids": all_ids,
            "all_training_image_ids": sorted(set(all_ids) | upstream_ids),
            "verifier_dataset_metadata": metadata,
        }
        torch.save(payload, output_dir / "generation_verifier_latest.pth")
        if score > best_score:
            best_score = score
            torch.save(payload, output_dir / "generation_verifier_best.pth")
    print("Best verifier: %s" % (output_dir / "generation_verifier_best.pth"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train V6.5 generation verifier")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="V6.5.1_model_weights/verifier")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--deployment-weight",
        type=float,
        default=2.0,
        help="Extra weight for candidates close to the deployed route region",
    )
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
