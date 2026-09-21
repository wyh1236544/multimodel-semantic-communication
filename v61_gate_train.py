from __future__ import annotations

import argparse

from train import resolve_seed_checkpoint
from utils.config import load_config, prepare_config
from utils.experiment import train_one_seed


def main():
    parser = argparse.ArgumentParser(
        description="V6.1 bandwidth training with optional old-weight expansion"
    )
    parser.add_argument("--config", default="configs/v61_bandwidth.yaml")
    parser.add_argument("--preset", default="high_6912")
    parser.add_argument("--checkpoint", help="File, directory, or path containing {seed}")
    parser.add_argument("--residual-init", help="Accepted for compatibility; normally unnecessary")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root")
    parser.add_argument("--vae-path")
    parser.add_argument("--text-model-path")
    parser.add_argument("--modality-token-budget", type=int)
    parser.add_argument("--utility-weight-choices", type=float, nargs="+")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = prepare_config(
        load_config(args.config),
        args.preset,
        args.vae_path,
        args.text_model_path,
        args.data_root,
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
    if args.residual_init:
        print("Note: --residual-init is not needed because the supplied seed weights already contain the residual gate.")

    for seed in args.seeds:
        initial = None if args.from_scratch else resolve_seed_checkpoint(args.checkpoint, seed)
        train_one_seed(
            config=config,
            output_dir=args.output_dir,
            seed=seed,
            init_checkpoint=initial,
            allow_shape_expansion=bool(initial and args.preset != "original_4608"),
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            epochs=args.epochs,
            device_name=args.device,
            lr_override=args.lr,
        )


if __name__ == "__main__":
    main()
