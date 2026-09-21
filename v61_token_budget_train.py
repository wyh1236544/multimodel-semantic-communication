from __future__ import annotations

import argparse
from pathlib import Path

from train import resolve_seed_checkpoint
from utils.config import load_config, prepare_config
from utils.experiment import train_one_seed


def expected_allocation(config, budget: int, utility_weight: float) -> tuple[int, int]:
    model = config["model"]
    allocation = model["dynamic_token_allocation"]
    num_image = int(model["num_img_tokens"])
    num_text = int(model["num_txt_tokens"])
    low = max(int(allocation.get("min_text_tokens", 0)), budget - num_image, 0)
    high = min(
        int(allocation.get("max_text_tokens", num_text)),
        num_text,
        budget,
    )
    text_count = round(low + utility_weight * (high - low))
    return budget - text_count, text_count


def main():
    parser = argparse.ArgumentParser(
        description="Train a controlled V6.1 transmitted-token-budget sweep"
    )
    parser.add_argument("--config", default="configs/v61_bandwidth.yaml")
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[208, 240, 272, 304, 320],
        help="Image+text token budgets; 16 common tokens are added automatically.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Original checkpoint file/directory, or a path containing {seed}.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--utility-weight", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root")
    parser.add_argument("--vae-path")
    parser.add_argument("--text-model-path")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.utility_weight <= 1.0:
        parser.error("--utility-weight must be in [0, 1]")
    if args.smoke_test:
        args.epochs = 1
        args.train_samples = min(args.train_samples, 64)
        args.val_samples = min(args.val_samples, 32)

    root = Path(args.output_dir)
    for budget in args.budgets:
        config = prepare_config(
            load_config(args.config),
            preset="token_width_16",
            vae_path=args.vae_path,
            text_model_path=args.text_model_path,
            data_root=args.data_root,
            modality_token_budget=budget,
            utility_weight_choices=[args.utility_weight],
        )
        if args.batch_size:
            config["train"]["batch_size"] = args.batch_size
            config["test"]["batch_size"] = args.batch_size
        if args.num_workers is not None:
            config["train"]["num_workers"] = args.num_workers
            config["test"]["num_workers"] = args.num_workers

        image_count, text_count = expected_allocation(
            config, budget, args.utility_weight
        )
        common_count = int(config["model"]["num_common_tokens"])
        channel_uses = (common_count + budget) * 16
        print(
            f"\n[token budget {budget}] expected allocation "
            f"C={common_count}, I={image_count}, T={text_count}; "
            f"channel_uses={channel_uses}"
        )
        budget_dir = root / f"budget_{budget}"
        for seed in args.seeds:
            initial = resolve_seed_checkpoint(args.checkpoint, seed)
            train_one_seed(
                config=config,
                output_dir=budget_dir,
                seed=seed,
                init_checkpoint=initial,
                allow_shape_expansion=False,
                train_samples=args.train_samples,
                val_samples=args.val_samples,
                epochs=args.epochs,
                device_name=args.device,
                lr_override=args.lr,
            )


if __name__ == "__main__":
    main()
