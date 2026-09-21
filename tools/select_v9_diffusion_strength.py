from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_candidate_key(key: str) -> tuple[str, float]:
    marker = "_strength_"
    if marker not in key:
        raise ValueError(f"Invalid candidate key: {key}")
    mode, strength = key.rsplit(marker, 1)
    return mode, float(strength)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank paired V9 diffusion strengths without per-sample oracle selection"
    )
    parser.add_argument("report")
    parser.add_argument("--max-mean-psnr-drop", type=float, default=1.5)
    args = parser.parse_args()
    report_path = Path(args.report)
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    aggregate = defaultdict(
        lambda: {
            "weighted_semantic_gain": 0.0,
            "weighted_clip_image_gain": 0.0,
            "weighted_clip_text_gain": 0.0,
            "weighted_psnr_delta": 0.0,
            "weighted_ssim_delta": 0.0,
            "samples": 0,
        }
    )
    for _, candidates in report["candidate_summary"].items():
        for key, metrics in candidates.items():
            mode, strength = parse_candidate_key(key)
            if mode == "reconstruction":
                continue
            count = int(metrics["samples"])
            target = aggregate[(mode, strength)]
            target["samples"] += count
            for source, destination in (
                ("semantic_utility_gain", "weighted_semantic_gain"),
                ("clip_image_gain", "weighted_clip_image_gain"),
                ("clip_source_text_gain", "weighted_clip_text_gain"),
                ("psnr_delta", "weighted_psnr_delta"),
                ("ssim_delta", "weighted_ssim_delta"),
            ):
                target[destination] += float(metrics[source]) * count

    for mode in ("generation", "hybrid"):
        rows = []
        for (candidate_mode, strength), totals in aggregate.items():
            if candidate_mode != mode:
                continue
            count = max(int(totals["samples"]), 1)
            row = {
                "strength": strength,
                "samples": count,
                "semantic_gain": totals["weighted_semantic_gain"] / count,
                "clip_image_gain": totals["weighted_clip_image_gain"] / count,
                "clip_text_gain": totals["weighted_clip_text_gain"] / count,
                "psnr_delta": totals["weighted_psnr_delta"] / count,
                "ssim_delta": totals["weighted_ssim_delta"] / count,
            }
            row["eligible"] = (
                row["psnr_delta"] >= -abs(args.max_mean_psnr_drop)
                and row["semantic_gain"] > 0.0
            )
            rows.append(row)
        rows.sort(
            key=lambda row: (
                bool(row["eligible"]),
                float(row["semantic_gain"]),
                float(row["psnr_delta"]),
            ),
            reverse=True,
        )
        print(f"\n{mode.upper()} strength ranking")
        for row in rows:
            print(
                "strength={strength:.2f} eligible={eligible} n={samples} "
                "semantic={semantic_gain:+.4f} clipI={clip_image_gain:+.4f} "
                "clipT={clip_text_gain:+.4f} dPSNR={psnr_delta:+.3f} "
                "dSSIM={ssim_delta:+.4f}".format(**row)
            )
        eligible = [row for row in rows if row["eligible"]]
        if eligible:
            print(f"recommended fixed {mode} strength={eligible[0]['strength']:.2f}")
        else:
            print(
                f"no {mode} strength passed the semantic-gain/PSNR constraint; "
                "do not claim a diffusion benefit"
            )


if __name__ == "__main__":
    main()
