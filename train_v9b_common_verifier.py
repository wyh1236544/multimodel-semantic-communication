"""Train and calibrate the V9-B Common-guided selective verifier."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from models.quality_adaptive_recovery import FEATURE_NAMES
from models.v9b_common_verifier import (
    MODE_FEATURE_NAMES,
    V9BCommonGuidedVerifier,
    v9b_verifier_loss,
)
from utils.experiment import seed_everything, select_device
from utils.v65_selective import VERIFIER_FEATURE_NAMES
from utils.v9b_selective import PROMPT_FEATURE_NAMES


FIELDS = (
    "route_features", "verifier_features", "prompt_quality", "mode_features", "common_semantic",
    "regression_clip", "candidate_clip", "prompt_clip",
)


class V9BDataset(Dataset):
    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        item = self.records[index]
        result = {name: torch.as_tensor(item[name]).float() for name in FIELDS}
        for name in (
            "accept_target", "hallucination_target", "semantic_gain_target",
            "delta_psnr_target", "delta_ssim_target",
        ):
            result[name] = torch.tensor(float(item[name]))
        result["common_target_clip"] = torch.as_tensor(
            item["common_target_clip"]
        ).float()
        result["mode_name"] = str(item["mode"])
        return result


def _group_counts(records):
    counts = {}
    for item in records:
        key = (str(item["mode"]), int(float(item["accept_target"]) >= 0.5))
        counts[key] = counts.get(key, 0) + 1
    return counts


def split_by_image(
    records, val_fraction, seed, min_train_per_class=20, min_val_per_class=5
):
    """Group split with class coverage in every operating mode.

    A random image split can leave the small Hybrid validation set with one
    class, making threshold calibration select the trivial all-reject policy.
    We search deterministic grouped shuffles and fail loudly if the collected
    dataset is too small for a meaningful experiment.
    """
    image_ids = sorted({int(item["image_id"]) for item in records})
    if len(image_ids) < 4:
        raise RuntimeError("Verifier data needs at least four unique images")
    count = min(max(int(round(len(image_ids) * val_fraction)), 1), len(image_ids) - 1)
    best = None
    for attempt in range(512):
        shuffled = list(image_ids)
        random.Random(seed + 104729 * attempt).shuffle(shuffled)
        val_ids = set(shuffled[:count])
        train = [item for item in records if int(item["image_id"]) not in val_ids]
        val = [item for item in records if int(item["image_id"]) in val_ids]
        train_counts, val_counts = _group_counts(train), _group_counts(val)
        valid = all(
            train_counts.get((mode, label), 0) >= min_train_per_class
            and val_counts.get((mode, label), 0) >= min_val_per_class
            for mode in MODE_FEATURE_NAMES for label in (0, 1)
        )
        if not valid:
            continue
        # Prefer a validation set whose positive rates resemble the full data.
        full_counts = _group_counts(records)
        imbalance = 0.0
        for mode in MODE_FEATURE_NAMES:
            full_n = sum(full_counts.get((mode, label), 0) for label in (0, 1))
            val_n = sum(val_counts.get((mode, label), 0) for label in (0, 1))
            full_rate = full_counts.get((mode, 1), 0) / max(full_n, 1)
            val_rate = val_counts.get((mode, 1), 0) / max(val_n, 1)
            imbalance += abs(full_rate - val_rate)
        if best is None or imbalance < best[0]:
            best = (imbalance, train, val)
    if best is None:
        raise RuntimeError(
            "Could not create a leakage-free train/validation split with both "
            "classes in Generation and Hybrid. Recollect with more --samples "
            "(recommended >=300), or use --allow-small-dataset only for a smoke test. "
            f"Full counts={_group_counts(records)}"
        )
    return best[1], best[2]


def _forward(model, batch, device, common_ablation=False, common_dropout=0.0):
    values = [batch[name].to(device) for name in FIELDS]
    common_mask = None
    if model.training and common_dropout > 0.0 and not common_ablation:
        common_mask = (
            torch.rand(values[0].size(0), 1, device=device)
            >= float(common_dropout)
        ).to(values[0].dtype)
    return model(
        *values, common_ablation=common_ablation, common_mask=common_mask
    )


def _decision_utility(mode, semantic_gain, delta_psnr, delta_ssim):
    psnr_limit = 1.0 if mode == "generation" else 0.75
    ssim_limit = 0.10 if mode == "generation" else 0.08
    return (
        float(semantic_gain)
        - 0.05 * max(-float(delta_psnr) - psnr_limit, 0.0) / psnr_limit
        - 0.05 * max(-float(delta_ssim) - ssim_limit, 0.0) / ssim_limit
    )


def _calibrate(subset):
    best = None
    for utility_alpha in (0.0, 0.5, 1.0, 2.0, 4.0):
        scores = [
            float(item["probability"]) + utility_alpha * float(item["predicted_utility"])
            for item in subset
        ]
        unique_scores = sorted(set(scores))
        thresholds = [unique_scores[0] - 1e-6, unique_scores[-1] + 1e-6]
        thresholds.extend(
            0.5 * (left + right)
            for left, right in zip(unique_scores[:-1], unique_scores[1:])
        )
        for threshold in sorted(thresholds):
            candidate = _score_calibration_candidate(
                subset, scores, threshold, utility_alpha
            )
            rank = (
                candidate["score"], candidate["f1"],
                candidate["accepted_decision_utility"], -utility_alpha,
            )
            if best is None or rank > best[0]:
                best = (rank, candidate)
    return best[1]


def _score_calibration_candidate(subset, scores, threshold, utility_alpha):
    predicted = [score >= threshold for score in scores]
    targets = [item["target"] for item in subset]
    tp = sum(p and t for p, t in zip(predicted, targets))
    tn = sum((not p) and (not t) for p, t in zip(predicted, targets))
    fp = sum(p and (not t) for p, t in zip(predicted, targets))
    fn = sum((not p) and t for p, t in zip(predicted, targets))
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)
    semantic_utility = sum(
        item["gain"] if decision else 0.0
        for item, decision in zip(subset, predicted)
    ) / max(len(subset), 1)
    decision_utility = sum(
        item["utility"] if decision else 0.0
        for item, decision in zip(subset, predicted)
    ) / max(len(subset), 1)
    # Accuracy prevents reckless acceptance, while continuous utility breaks
    # near-ties in favour of useful rather than trivial decisions.
    score = (
        0.5 * (recall + specificity) + 0.25 * f1 - 0.25 * fpr
        + 2.0 * decision_utility
    )
    return {
        "threshold": threshold, "score": score,
        "utility_alpha": utility_alpha,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": fpr,
        "accept_rate": sum(predicted) / max(len(subset), 1),
        "accepted_semantic_utility": semantic_utility,
        "accepted_decision_utility": decision_utility,
        "count": len(subset),
    }


@torch.no_grad()
def evaluate(model, loader, device, common_ablation=False):
    model.eval()
    records = []
    hallucination_correct = semantic_mae = psnr_mae = ssim_mae = utility_mae = 0.0
    count = 0
    for batch in loader:
        prediction = _forward(model, batch, device, common_ablation=common_ablation)
        probabilities = prediction["acceptance_probability"].cpu().tolist()
        targets = batch["accept_target"].bool().tolist()
        gains = batch["semantic_gain_target"].float().tolist()
        delta_psnr = batch["delta_psnr_target"].float().tolist()
        delta_ssim = batch["delta_ssim_target"].float().tolist()
        predicted_utility = prediction["predicted_utility"].cpu().tolist()
        for index, mode_name in enumerate(batch["mode_name"]):
            utility = _decision_utility(
                str(mode_name), gains[index], delta_psnr[index], delta_ssim[index]
            )
            records.append(
                {
                    "mode": str(mode_name),
                    "probability": float(probabilities[index]),
                    "target": bool(targets[index]),
                    "gain": float(gains[index]),
                    "utility": utility,
                    "predicted_utility": float(predicted_utility[index]),
                }
            )
            utility_mae += abs(float(predicted_utility[index]) - utility)
        hallucination_correct += float(
            ((prediction["hallucination_probability"] >= 0.5).cpu() == batch["hallucination_target"].bool()).sum()
        )
        semantic_mae += float((prediction["predicted_delta_semantic"].cpu() - batch["semantic_gain_target"]).abs().sum())
        psnr_mae += float((prediction["predicted_delta_psnr"].cpu() - batch["delta_psnr_target"]).abs().sum())
        ssim_mae += float((prediction["predicted_delta_ssim"].cpu() - batch["delta_ssim_target"]).abs().sum())
        count += len(batch["accept_target"])
    by_mode = {}
    for mode in MODE_FEATURE_NAMES:
        subset = [item for item in records if item["mode"] == mode]
        if not subset:
            continue
        by_mode[mode] = _calibrate(subset)
    macro = lambda name: sum(value[name] for value in by_mode.values()) / max(len(by_mode), 1)
    result = {
        "by_mode": by_mode,
        "score": macro("score"),
        "balanced_accuracy": macro("balanced_accuracy"),
        "f1": macro("f1"),
        "false_positive_rate": macro("false_positive_rate"),
        "accepted_semantic_utility": macro("accepted_semantic_utility"),
        "accepted_decision_utility": macro("accepted_decision_utility"),
    }
    result.update(
        {
            "hallucination_accuracy": hallucination_correct / max(count, 1),
            "semantic_gain_mae": semantic_mae / max(count, 1),
            "delta_psnr_mae": psnr_mae / max(count, 1),
            "delta_ssim_mae": ssim_mae / max(count, 1),
            "utility_mae": utility_mae / max(count, 1),
            "count": count,
        }
    )
    return result


def main(args):
    seed_everything(args.seed)
    device = select_device(args.device)
    payload = torch.load(args.dataset, map_location="cpu")
    if list(payload.get("route_feature_names", [])) != list(FEATURE_NAMES):
        raise RuntimeError("route feature schema mismatch")
    if list(payload.get("verifier_feature_names", [])) != list(VERIFIER_FEATURE_NAMES):
        raise RuntimeError("verifier feature schema mismatch")
    if list(payload.get("prompt_feature_names", [])) != list(PROMPT_FEATURE_NAMES):
        raise RuntimeError("prompt feature schema mismatch")
    if list(payload.get("mode_feature_names", [])) != list(MODE_FEATURE_NAMES):
        raise RuntimeError("mode feature schema mismatch; recollect V9-B.1 data")
    records = list(payload["records"])
    min_train = 1 if args.allow_small_dataset else args.min_train_per_class
    min_val = 1 if args.allow_small_dataset else args.min_val_per_class
    train_records, val_records = split_by_image(
        records, args.val_fraction, args.seed,
        min_train_per_class=min_train, min_val_per_class=min_val,
    )
    group_counts = _group_counts(train_records)
    val_group_counts = _group_counts(val_records)
    print(
        f"dataset records={len(records)} unique_images={len(set(int(x['image_id']) for x in records))} "
        f"train_counts={group_counts} val_counts={val_group_counts}", flush=True,
    )
    for mode in MODE_FEATURE_NAMES:
        if group_counts.get((mode, 0), 0) == 0 or group_counts.get((mode, 1), 0) == 0:
            raise RuntimeError(f"{mode} training split must contain both classes")
    sample_weights = [
        1.0 / group_counts[(str(item["mode"]), int(float(item["accept_target"]) >= 0.5))]
        for item in train_records
    ]
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=max(len(train_records), max(group_counts.values()) * len(group_counts)),
        replacement=True,
    )
    train_loader = DataLoader(
        V9BDataset(train_records), batch_size=args.batch_size, sampler=sampler
    )
    val_loader = DataLoader(V9BDataset(val_records), batch_size=args.batch_size, shuffle=False)
    sample = records[0]
    verifier_config = {
        "common_dim": int(torch.as_tensor(sample["common_semantic"]).numel()),
        "clip_dim": int(torch.as_tensor(sample["candidate_clip"]).numel()),
        "hidden_dim": args.hidden_dim,
    }
    model = V9BCommonGuidedVerifier(**verifier_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ids = sorted({int(item["image_id"]) for item in train_records})
    val_ids = sorted({int(item["image_id"]) for item in val_records})
    best_score = -1e9
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            prediction = _forward(
                model, batch, device, common_dropout=args.common_dropout
            )
            loss, _ = v9b_verifier_loss(
                prediction,
                batch["accept_target"].to(device),
                batch["hallucination_target"].to(device),
                batch["semantic_gain_target"].to(device),
                batch["delta_psnr_target"].to(device),
                batch["delta_ssim_target"].to(device),
                batch["common_target_clip"].to(device),
                batch["mode_features"].to(device),
                positive_weight=1.0,
                common_alignment_weight=args.common_alignment_weight,
                generation_max_psnr_drop=args.generation_max_psnr_drop,
                hybrid_max_psnr_drop=args.hybrid_max_psnr_drop,
                generation_max_ssim_drop=args.generation_max_ssim_drop,
                hybrid_max_ssim_drop=args.hybrid_max_ssim_drop,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(batch["accept_target"])
        metrics = evaluate(model, val_loader, device)
        no_common = evaluate(model, val_loader, device, common_ablation=True)
        common_gain = metrics["balanced_accuracy"] - no_common["balanced_accuracy"]
        worst_mode_score = min(value["score"] for value in metrics["by_mode"].values())
        score = metrics["score"] + 0.20 * worst_mode_score + 0.10 * common_gain
        threshold_text = ", ".join(
            "%s:%.4f" % (name, value["threshold"])
            for name, value in metrics["by_mode"].items()
        )
        print(
            f"epoch={epoch+1}/{args.epochs} loss={running/max(len(train_records),1):.5f} "
            f"macro_bal={metrics['balanced_accuracy']:.3f} macro_F1={metrics['f1']:.3f} "
            f"macro_FPR={metrics['false_positive_rate']:.3f} "
            f"utility={metrics['accepted_semantic_utility']:+.4f} "
            f"decision_utility={metrics['accepted_decision_utility']:+.4f} "
            f"thresholds={{{threshold_text}}} "
            f"common_dBal={common_gain:+.3f}", flush=True,
        )
        checkpoint = {
            "verifier": model.state_dict(), "verifier_config": verifier_config,
            "metrics": metrics, "no_common_metrics": no_common,
            "acceptance_thresholds": {
                mode: value["threshold"] for mode, value in metrics["by_mode"].items()
            },
            "decision_score_alphas": {
                mode: value["utility_alpha"] for mode, value in metrics["by_mode"].items()
            },
            "no_common_acceptance_thresholds": {
                mode: value["threshold"] for mode, value in no_common["by_mode"].items()
            },
            "no_common_decision_score_alphas": {
                mode: value["utility_alpha"] for mode, value in no_common["by_mode"].items()
            },
            "epoch": epoch + 1,
            "train_image_ids": train_ids, "val_image_ids": val_ids,
            "all_training_image_ids": sorted(set(train_ids) | set(val_ids)),
            "dataset_metadata": payload.get("metadata", {}),
        }
        torch.save(checkpoint, output_dir / "v9b_verifier_latest.pth")
        if score > best_score:
            best_score = score
            torch.save(checkpoint, output_dir / "v9b_verifier_best.pth")
    print(f"saved best verifier: {output_dir / 'v9b_verifier_best.pth'}")


def build_parser():
    parser = argparse.ArgumentParser(description="Train V9-B Common-guided verifier")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--min-train-per-class", type=int, default=20)
    parser.add_argument("--min-val-per-class", type=int, default=5)
    parser.add_argument(
        "--allow-small-dataset", action="store_true",
        help="Smoke tests only; permits one image per mode/class in each split.",
    )
    parser.add_argument(
        "--common-dropout",
        type=float,
        default=0.25,
        help="Makes the no-Common ablation calibrated instead of out-of-distribution.",
    )
    parser.add_argument("--common-alignment-weight", type=float, default=1.0)
    parser.add_argument("--generation-max-psnr-drop", type=float, default=1.0)
    parser.add_argument("--hybrid-max-psnr-drop", type=float, default=0.75)
    parser.add_argument("--generation-max-ssim-drop", type=float, default=0.10)
    parser.add_argument("--hybrid-max-ssim-drop", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
