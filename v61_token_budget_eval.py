from __future__ import annotations

import argparse
import json
from pathlib import Path

from test import evaluate_checkpoint
from utils.config import load_config, prepare_config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints from the V6.1 token-budget sweep"
    )
    parser.add_argument("--config", default="configs/v61_bandwidth.yaml")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[208, 240, 272, 304, 320]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--utility-weight", type=float, default=0.5)
    parser.add_argument("--snr-list", type=float, nargs="+", default=[0, 5, 10, 15, 20])
    parser.add_argument("--ablations", nargs="+", default=["full"])
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root")
    parser.add_argument("--vae-path")
    parser.add_argument("--text-model-path")
    args = parser.parse_args()

    if not 0.0 <= args.utility_weight <= 1.0:
        parser.error("--utility-weight must be in [0, 1]")

    checkpoint_root = Path(args.checkpoint_root)
    output_root = Path(args.output_dir)
    combined = {}
    for budget in args.budgets:
        config = prepare_config(
            load_config(args.config),
            preset="token_width_16",
            vae_path=args.vae_path,
            text_model_path=args.text_model_path,
            data_root=args.data_root,
            modality_token_budget=budget,
        )
        if args.batch_size:
            config["test"]["batch_size"] = args.batch_size
        if args.num_workers is not None:
            config["test"]["num_workers"] = args.num_workers

        budget_results = {}
        for seed in args.seeds:
            checkpoint = checkpoint_root / f"budget_{budget}" / f"seed{seed}.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
            result = evaluate_checkpoint(
                checkpoint_path=str(checkpoint),
                config=config,
                output_path=output_root / f"budget_{budget}" / f"seed{seed}.json",
                snr_list=args.snr_list,
                ablations=args.ablations,
                samples=args.samples,
                seed=seed,
                device_name=args.device,
                utility_weight=args.utility_weight,
            )
            budget_results[str(seed)] = result
        combined[str(budget)] = budget_results

    output_root.mkdir(parents=True, exist_ok=True)
    combined_path = output_root / "all_token_budgets.json"
    with combined_path.open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, ensure_ascii=False, indent=2)
    print(f"saved combined token-budget results: {combined_path}")


if __name__ == "__main__":
    main()
