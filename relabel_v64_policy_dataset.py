"""Relabel V6.4 routing targets from saved oracle scores without regeneration."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import torch


def _key(image_id, snr):
    return int(image_id), round(float(snr), 6)


def main(args: argparse.Namespace) -> None:
    dataset = torch.load(args.dataset, map_location="cpu")
    records = list(dataset.get("records", []))
    if not records:
        raise RuntimeError("Input policy dataset has no records")
    with Path(args.oracle_csv).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(records):
        raise RuntimeError(
            "Dataset/CSV length mismatch: %d versus %d" % (len(records), len(rows))
        )
    oracle = {}
    for row in rows:
        key = _key(row["image_id"], row["snr_db"])
        if key in oracle:
            raise RuntimeError("Duplicate oracle key: %s" % (key,))
        gain = float(row["best_generation_utility"]) - float(row["regression_utility"])
        oracle[key] = (gain, row)

    counts = Counter()
    by_snr = defaultdict(Counter)
    expert_positive = Counter()
    for record in records:
        key = _key(record["image_id"], record["snr"])
        if key not in oracle:
            raise RuntimeError("Missing oracle label for record: %s" % (key,))
        gain, row = oracle[key]
        target = float(gain > args.margin)
        record["generation_target"] = target
        record["oracle_utility_gain"] = float(gain)
        label = "positive" if target >= 0.5 else "negative"
        counts[label] += 1
        by_snr[key[1]][label] += 1
        if target >= 0.5:
            expert_positive[str(row["best_expert"])] += 1

    metadata = dict(dataset.get("metadata", {}))
    metadata["original_generation_margin"] = metadata.get("generation_margin")
    metadata["generation_margin"] = float(args.margin)
    metadata["relabelled_from_oracle_csv"] = str(args.oracle_csv)
    payload = dict(dataset)
    payload["records"] = records
    payload["metadata"] = metadata
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.resolve() == Path(args.dataset).resolve():
        raise RuntimeError("Refusing to overwrite the original dataset")
    torch.save(payload, output)
    print("Saved: %s" % output)
    print("margin=%.6f records=%d positive=%d negative=%d" % (
        args.margin, len(records), counts["positive"], counts["negative"]
    ))
    print("positive_experts=%s" % dict(sorted(expert_positive.items())))
    for snr in sorted(by_snr):
        print(
            "snr=%g positive=%d negative=%d"
            % (snr, by_snr[snr]["positive"], by_snr[snr]["negative"])
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Relabel V6.4 oracle routing targets")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--oracle-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--margin", type=float, default=0.006)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
