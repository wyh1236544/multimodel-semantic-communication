from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = ("psnr", "ssim", "token_accuracy", "bleu2")


def main():
    parser = argparse.ArgumentParser(description="Summarize token-budget JSON results")
    parser.add_argument("--input", required=True, help="all_token_budgets.json")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as handle:
        all_results = json.load(handle)

    rows = []
    for budget_text, seed_results in sorted(
        all_results.items(), key=lambda item: int(item[0])
    ):
        budget = int(budget_text)
        first_result = next(iter(seed_results.values()))
        channel_uses = int(first_result["channel_uses"])
        for snr_text in first_result["snr"]:
            for ablation in first_result["snr"][snr_text]:
                records = [
                    result["snr"][snr_text][ablation]
                    for result in seed_results.values()
                ]
                for metric in METRICS:
                    values = [float(record[metric]) for record in records]
                    rows.append(
                        {
                            "modality_budget": budget,
                            "total_tokens": budget
                            + int(records[0]["allocated_common_tokens"]),
                            "channel_uses": channel_uses,
                            "snr_db": float(snr_text),
                            "ablation": ablation,
                            "allocated_common_tokens": int(
                                records[0]["allocated_common_tokens"]
                            ),
                            "allocated_image_tokens": statistics.mean(
                                float(record["allocated_image_tokens"])
                                for record in records
                            ),
                            "allocated_text_tokens": statistics.mean(
                                float(record["allocated_text_tokens"])
                                for record in records
                            ),
                            "metric": metric,
                            "mean": statistics.mean(values),
                            "std": statistics.stdev(values)
                            if len(values) > 1
                            else 0.0,
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
