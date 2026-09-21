from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from data import build_dataloader
from utils.checkpoint import load_checkpoint, load_state_dict_flexible
from utils.config import load_config, prepare_config
from utils.experiment import create_model, evaluate_model, seed_everything, select_device


def evaluate_checkpoint(
    checkpoint_path: str,
    config: Dict[str, Any],
    output_path: str | Path,
    snr_list: Sequence[float],
    ablations: Sequence[str],
    samples: int,
    seed: int,
    device_name: str = "auto",
    utility_weight: float = 0.5,
) -> Dict[str, Any]:
    seed_everything(seed)
    device = select_device(device_name)
    model = create_model(config, device)
    checkpoint = load_checkpoint(checkpoint_path)
    load_report = load_state_dict_flexible(model, checkpoint, allow_shape_expansion=False)
    print(
        f"loaded {checkpoint_path}: exact={load_report['exact_count']} "
        f"channel_uses={model.num_channel_uses()}"
    )
    data = config["data"]
    loader = build_dataloader(
        image_root=data["val_image_root"],
        annotation_file=data["val_annotation_file"],
        image_size=int(config["model"]["image_size"]),
        batch_size=int(config["test"].get("batch_size", 4)),
        num_workers=int(config["test"].get("num_workers", 4)),
        train=False,
        caption_mode=data.get("val_caption_mode", "shortest"),
        max_samples=samples,
        seed=seed,
    )
    results: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "bandwidth": model.get_bandwidth_info(),
        "channel_uses": model.num_channel_uses(),
        "snr": {},
    }
    amp = bool(config.get("train", {}).get("amp", True))
    for snr in snr_list:
        results["snr"][str(float(snr))] = {}
        for ablation in ablations:
            # Reset before every ablation so VAE sampling and channel noise are
            # paired; differences then come from the semantic stream removal.
            evaluation_seed = seed * 10000 + 42
            seed_everything(evaluation_seed)
            metrics = evaluate_model(
                model,
                loader,
                device=device,
                snr_db=float(snr),
                amp=amp,
                semantic_ablation=ablation,
                utility_weight=utility_weight,
            )
            results["snr"][str(float(snr))][ablation] = metrics
            print(
                f"snr={snr:g} ablation={ablation} "
                f"psnr={metrics['psnr']:.4f} ssim={metrics['ssim']:.4f} "
                f"token_acc={metrics['token_accuracy']:.4f} bleu2={metrics['bleu2']:.4f} "
                f"common_i2t={metrics['common_i2t_top1']:.4f} "
                f"rx_common_t={metrics['receiver_common_text_top1']:.4f} "
                f"base_r1={metrics.get('common_semantic_text_r1', float('nan')):.4f} "
                f"base_r5={metrics.get('common_semantic_text_r5', float('nan')):.4f} "
                f"base_txt={metrics.get('common_caption_token_accuracy', float('nan')):.4f} "
                f"text_rel={metrics.get('text_channel_reliability', 1.0):.3f} "
                f"coarse_mse={metrics.get('common_coarse_latent_mse', float('nan')):.5f} "
                f"uses={metrics.get('channel_uses', model.num_channel_uses())} "
                f"tokens=C{metrics['allocated_common_tokens']}/"
                f"I{metrics['allocated_image_tokens']:.1f}/"
                f"T{metrics['allocated_text_tokens']:.1f}"
            )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"saved {output_path}")
    return results


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate one V6.1 checkpoint")
    parser.add_argument("--config", default="configs/v61_bandwidth.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preset", default="original_4608")
    parser.add_argument("--output", required=True)
    parser.add_argument("--snr-list", type=float, nargs="+", default=[0, 5, 10, 15, 20])
    parser.add_argument("--ablations", nargs="+", default=["full"])
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root")
    parser.add_argument("--vae-path")
    parser.add_argument("--text-model-path")
    parser.add_argument("--modality-token-budget", type=int)
    parser.add_argument("--utility-weight", type=float, default=0.5)
    parser.add_argument(
        "--disable-text-snr-gate",
        action="store_true",
        help="Evaluate the same checkpoint without receiver text-to-image SNR gating.",
    )
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
    )
    if args.batch_size:
        config["test"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["test"]["num_workers"] = args.num_workers
    if args.disable_text_snr_gate:
        config["model"].setdefault("decoder_text_snr_gate", {})["enable"] = False
    return evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        config=config,
        output_path=args.output,
        snr_list=args.snr_list,
        ablations=args.ablations,
        samples=args.samples,
        seed=args.seed,
        device_name=args.device,
        utility_weight=args.utility_weight,
    )


if __name__ == "__main__":
    main()
