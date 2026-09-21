from __future__ import annotations

import argparse
from pathlib import Path

from utils.config import load_config, prepare_config
from utils.experiment import train_one_seed


def resolve_seed_checkpoint(value: str | None, seed: int) -> str | None:
    if not value:
        return None
    value = value.format(seed=seed)
    path = Path(value)
    if path.is_dir():
        path = path / f"seed{seed}.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Initial checkpoint not found: {path}")
    return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a multimodal semantic model")
    parser.add_argument("--config", default="configs/v61_bandwidth.yaml")
    parser.add_argument("--preset", default="high_6912")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--allow-shape-expansion", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root")
    parser.add_argument("--vae-path")
    parser.add_argument("--text-model-path")
    parser.add_argument("--modality-token-budget", type=int)
    parser.add_argument(
        "--utility-weight-choices",
        type=float,
        nargs="+",
        help="Training allocation weights; use 0.5 for a strict token-count control.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = prepare_config(
        load_config(args.config),
        preset=args.preset,
        vae_path=args.vae_path,
        text_model_path=args.text_model_path,
        data_root=args.data_root,
        modality_token_budget=args.modality_token_budget,
        utility_weight_choices=args.utility_weight_choices,
    )
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
        config["test"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["train"]["num_workers"] = args.num_workers
        config["test"]["num_workers"] = args.num_workers
    if args.smoke_test:
        args.epochs = 1
        args.train_samples = min(args.train_samples or 64, 64)
        args.val_samples = min(args.val_samples or 32, 32)
    if args.init_checkpoint and args.preset != "original_4608" and not args.allow_shape_expansion:
        raise ValueError(
            "A non-4608 model cannot exactly load the old checkpoint. "
            "Add --allow-shape-expansion to copy overlapping channel weights."
        )

    summaries = []
    for seed in args.seeds:
        initial = resolve_seed_checkpoint(args.init_checkpoint, seed)
        summaries.append(
            train_one_seed(
                config=config,
                output_dir=args.output_dir,
                seed=seed,
                init_checkpoint=initial,
                allow_shape_expansion=args.allow_shape_expansion,
                train_samples=args.train_samples,
                val_samples=args.val_samples,
                epochs=args.epochs,
                device_name=args.device,
                lr_override=args.lr,
            )
        )
    return summaries


if __name__ == "__main__":
    main()
