from __future__ import annotations

import argparse
import json
from pathlib import Path

from test import evaluate_checkpoint
from utils.config import load_config, prepare_config


def resolve_checkpoint(directory: Path, seed: int) -> Path:
    candidates = [
        directory / f"seed{seed}.pth",
        directory / f"best_seed{seed}.pth",
        directory / f"seed{seed}" / "best.pth",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot find seed {seed} checkpoint in {directory}; tried {candidates}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate three V6.1 model seeds")
    parser.add_argument("--config", default="configs/v61_bandwidth.yaml")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--stage-a-dir", dest="checkpoint_dir")
    parser.add_argument("--checkpoint", help="Evaluate one checkpoint instead of a directory")
    parser.add_argument("--preset", default="original_4608")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--snr-list", type=float, nargs="+", default=[0, 5, 10, 15, 20])
    parser.add_argument("--ablations", nargs="+", default=["full"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root")
    parser.add_argument("--vae-path")
    parser.add_argument("--text-model-path")
    parser.add_argument("--modality-token-budget", type=int)
    parser.add_argument("--utility-weight", type=float, default=0.5)
    parser.add_argument("--base-config", help="Accepted for compatibility; not used")
    args = parser.parse_args()
    if not args.checkpoint and not args.checkpoint_dir:
        parser.error("Provide --checkpoint or --checkpoint-dir")

    config = prepare_config(
        load_config(args.config),
        args.preset,
        args.vae_path,
        args.text_model_path,
        args.data_root,
        modality_token_budget=args.modality_token_budget,
    )
    if args.batch_size:
        config["test"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["test"]["num_workers"] = args.num_workers
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    seeds = [args.seeds[0]] if args.checkpoint else args.seeds
    for seed in seeds:
        checkpoint = Path(args.checkpoint) if args.checkpoint else resolve_checkpoint(
            Path(args.checkpoint_dir), seed
        )
        result = evaluate_checkpoint(
            checkpoint_path=str(checkpoint),
            config=config,
            output_path=output_dir / f"seed{seed}.json",
            snr_list=args.snr_list,
            ablations=args.ablations,
            samples=args.samples,
            seed=seed,
            device_name=args.device,
            utility_weight=args.utility_weight,
        )
        all_results[str(seed)] = result
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, ensure_ascii=False, indent=2)
    print(f"saved combined summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
