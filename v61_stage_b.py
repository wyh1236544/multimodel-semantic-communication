from __future__ import annotations

import argparse
from pathlib import Path

from utils.config import load_config, prepare_config
from utils.experiment import train_one_seed
from v61_gate_eval import resolve_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Conservative low-learning-rate fine-tuning of Stage A checkpoints"
    )
    parser.add_argument("--config", default="configs/v61_bandwidth.yaml")
    parser.add_argument("--preset", default="high_6912")
    parser.add_argument("--stage-a-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--lr", type=float, default=0.000004)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root")
    parser.add_argument("--vae-path")
    parser.add_argument("--text-model-path")
    parser.add_argument("--teacher-config", help="Accepted for old-command compatibility")
    parser.add_argument("--teacher-checkpoint", help="Accepted but no teacher distillation is performed")
    args = parser.parse_args()
    if args.teacher_checkpoint:
        print(
            "Note: this replacement Stage B performs conservative fine-tuning only; "
            "it does not claim to reproduce the unavailable original distillation loss."
        )
    config = prepare_config(
        load_config(args.config), args.preset, args.vae_path, args.text_model_path, args.data_root
    )
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
        config["test"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["train"]["num_workers"] = args.num_workers
        config["test"]["num_workers"] = args.num_workers
    for seed in args.seeds:
        checkpoint = resolve_checkpoint(Path(args.stage_a_dir), seed)
        train_one_seed(
            config=config,
            output_dir=args.output_dir,
            seed=seed,
            init_checkpoint=str(checkpoint),
            allow_shape_expansion=False,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            epochs=args.epochs,
            device_name=args.device,
            lr_override=args.lr,
        )


if __name__ == "__main__":
    main()

