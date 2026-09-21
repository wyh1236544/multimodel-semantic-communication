"""Print label balance and oracle gain statistics for a V6.5 dataset."""
from __future__ import annotations

import argparse
import collections
import statistics

import torch


def main(args: argparse.Namespace) -> None:
    dataset = torch.load(args.dataset, map_location="cpu")
    records = list(dataset.get("records", []))
    if not records:
        raise RuntimeError("Dataset is empty")
    accept = [int(float(item["accept_target"]) >= 0.5) for item in records]
    hallucination = [
        int(float(item["hallucination_target"]) >= 0.5) for item in records
    ]
    print("records=%d unique_images=%d" % (
        len(records), len({int(item["image_id"]) for item in records})
    ))
    print("accept=%s" % dict(collections.Counter(accept)))
    print("hallucination=%s" % dict(collections.Counter(hallucination)))
    for name in ("delta_clip_target", "delta_psnr_target", "delta_ssim_target"):
        values = sorted(float(item[name]) for item in records)
        print(
            "%s min/q1/median/q3/max/mean=%+.5f %+.5f %+.5f %+.5f %+.5f %+.5f"
            % (
                name,
                values[0],
                values[len(values) // 4],
                values[len(values) // 2],
                values[3 * len(values) // 4],
                values[-1],
                statistics.mean(values),
            )
        )
    for snr in sorted({float(item["snr"]) for item in records}):
        subset = [item for item in records if float(item["snr"]) == snr]
        positive = sum(float(item["accept_target"]) >= 0.5 for item in subset)
        print(
            "snr=%g count=%d accept=%d reject=%d"
            % (snr, len(subset), positive, len(subset) - positive)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect V6.5 verifier labels")
    parser.add_argument(
        "--dataset",
        default="experiment_results/v65_verifier_data/verifier_dataset.pth",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
