from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def main():
    parser = argparse.ArgumentParser(description="Merge V6.1 bandwidth evaluation summaries")
    parser.add_argument("--inputs", nargs="+", required=True, help="preset=path/to/summary.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for item in args.inputs:
        preset, path = item.split("=", 1)
        with Path(path).open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        by_snr = {}
        for seed, seed_result in summary.items():
            channel_uses = seed_result["channel_uses"]
            for snr, ablations in seed_result["snr"].items():
                metrics = ablations["full"]
                by_snr.setdefault(snr, []).append(metrics)
        for snr, metrics_list in sorted(by_snr.items(), key=lambda pair: float(pair[0])):
            for metric in ("psnr", "ssim", "token_accuracy", "bleu2"):
                values = [float(entry[metric]) for entry in metrics_list]
                rows.append(
                    {
                        "preset": preset,
                        "channel_uses": channel_uses,
                        "snr_db": float(snr),
                        "metric": metric,
                        "mean": mean(values),
                        "std": stdev(values) if len(values) > 1 else 0.0,
                        "num_seeds": len(values),
                    }
                )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {output}")


if __name__ == "__main__":
    main()

