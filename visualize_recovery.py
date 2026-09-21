"""Export qualitative V6.1 image/text recovery examples.

The script does not change model weights or training.  For every selected COCO
pair it saves one panel per SNR:

    original | regression-only reconstruction | gated reconstruction | errors

When the loaded model has a V6.1 residual gate, the regression-only image is
obtained by removing the gated residual from the final latent.  Thus the two
middle columns isolate the visible contribution of the gate using exactly the
same channel-noise realization.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from data import build_dataloader
from utils.checkpoint import load_checkpoint, load_state_dict_flexible
from utils.config import load_config, prepare_config
from utils.experiment import create_model, seed_everything, select_device
from utils.metrics import batch_image_metrics, ssim_per_image, token_accuracy


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a CHW image in [0, 1] to RGB PIL."""
    array = (
        image.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255
    ).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def error_to_pil(reconstruction: torch.Tensor, target: torch.Tensor) -> Image.Image:
    """Absolute RGB error, amplified for visual inspection."""
    error = (reconstruction.detach().float() - target.detach().float()).abs()
    error = (error.clamp(0, 0.25) / 0.25).cpu()
    return tensor_to_pil(error)


def add_caption(image: Image.Image, title: str, font: ImageFont.ImageFont) -> Image.Image:
    height = 30
    result = Image.new("RGB", (image.width, image.height + height), "white")
    result.paste(image, (0, height))
    ImageDraw.Draw(result).text((5, 7), title, fill="black", font=font)
    return result


def make_panel(
    original: torch.Tensor,
    regression: torch.Tensor,
    gated: torch.Tensor,
    common_base: torch.Tensor | None,
    snr_db: float,
    has_gate: bool,
) -> Image.Image:
    font = ImageFont.load_default()
    entries = [
        ("Original", tensor_to_pil(original)),
        *(
            [("Common C-only", tensor_to_pil(common_base))]
            if common_base is not None
            else []
        ),
        ("Regression", tensor_to_pil(regression)),
        ("V6.1 gated" if has_gate else "Model output", tensor_to_pil(gated)),
        ("|Regression - original|", error_to_pil(regression, original)),
        ("|Gated - original|", error_to_pil(gated, original)),
    ]
    tiles = [add_caption(image, title, font) for title, image in entries]
    panel = Image.new("RGB", (sum(tile.width for tile in tiles), tiles[0].height), "white")
    x = 0
    for tile in tiles:
        panel.paste(tile, (x, 0))
        x += tile.width
    return panel


def per_sample_token_accuracy(outputs: Dict[str, torch.Tensor], index: int) -> float:
    logits = outputs["text_logits"][index : index + 1]
    ids = outputs["text_input_ids"][index : index + 1]
    mask = outputs["text_attention_mask"][index : index + 1]
    correct, count = token_accuracy(logits, ids, mask)
    return correct / max(count, 1)


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = select_device(args.device)

    config = prepare_config(
        load_config(args.config),
        preset=args.preset,
        vae_path=args.vae_path,
        text_model_path=args.text_model_path,
        data_root=args.data_root,
    )
    # Keep compatibility with the earlier server-side utils/config.py, whose
    # prepare_config() does not yet accept a channel_type keyword.
    if args.channel_type:
        config.setdefault("channel", {})["type"] = args.channel_type
    config["test"]["batch_size"] = min(args.samples, args.batch_size)
    config["test"]["num_workers"] = args.num_workers
    model = create_model(config, device)
    checkpoint = load_checkpoint(args.checkpoint)
    report = load_state_dict_flexible(model, checkpoint, allow_shape_expansion=False)
    del checkpoint
    model.eval()
    print(f"Loaded {args.checkpoint}: exact tensors={report['exact_count']}")

    data = config["data"]
    loader = build_dataloader(
        image_root=data["val_image_root"],
        annotation_file=data["val_annotation_file"],
        image_size=int(config["model"]["image_size"]),
        batch_size=min(args.samples, args.batch_size),
        num_workers=args.num_workers,
        train=False,
        caption_mode=data.get("val_caption_mode", "shortest"),
        max_samples=args.samples,
        seed=args.seed,
    )
    batch = next(iter(loader))
    images = batch["images"].to(device, non_blocking=True)
    texts = batch["texts"]
    image_ids = batch["image_ids"]

    has_gate = getattr(model, "residual_gate", None) is not None
    rows: list[dict[str, Any]] = []
    autocast = bool(config.get("train", {}).get("amp", True) and device.type == "cuda")
    # The project has several historical model revisions.  Filter optional
    # visualization-only arguments against the actual installed forward()
    # signature so this script also works with the earlier V6.1 model.
    supported_forward_args = set(inspect.signature(model.forward).parameters)
    for snr_db in args.snr_list:
        # Pair the channel noise for all comparisons at this SNR.
        seed_everything(args.seed * 10000 + int((snr_db + 100) * 100))
        forward_kwargs = {
            "snr_db": snr_db,
            "semantic_ablation": args.semantic_ablation,
            "disable_modality_dropout": True,
            "image_token_drop_rate": args.image_damage,
            "utility_weight": args.utility_weight,
            "channel_width": args.channel_width,
        }
        forward_kwargs = {
            name: value for name, value in forward_kwargs.items()
            if name in supported_forward_args
        }
        with torch.cuda.amp.autocast(enabled=autocast):
            outputs = model(images, texts, **forward_kwargs)
            gated = outputs["image"]
            common_coarse_latents = outputs.get("common_coarse_latents")
            common_base = (
                model.vae_wrapper.decode_latents(common_coarse_latents)
                if common_coarse_latents is not None
                else None
            )
            residual = outputs.get("residual_latent")
            if has_gate and residual is not None:
                regression_latent = outputs["recovered_vae_latents"] - residual
                regression = model.vae_wrapper.decode_latents(regression_latent)
            else:
                regression = gated
        gated_metrics = batch_image_metrics(gated, images)
        regression_metrics = batch_image_metrics(regression, images)
        predictions = model.decode_text_logits(outputs["text_logits"])
        gate_mean = outputs.get("residual_gate")
        reported_channel_uses = outputs.get("channel_uses")
        if reported_channel_uses is None:
            reported_channel_uses = model.num_channel_uses()
        elif isinstance(reported_channel_uses, torch.Tensor):
            reported_channel_uses = int(reported_channel_uses.item())

        for index, image_id in enumerate(image_ids):
            stem = f"id{image_id}_snr{snr_db:g}".replace(".", "p")
            panel = make_panel(
                images[index],
                regression[index],
                gated[index],
                common_base[index] if common_base is not None else None,
                snr_db,
                has_gate,
            )
            panel.save(output_dir / f"{stem}.png")
            rows.append(
                {
                    "image_id": image_id,
                    "snr_db": snr_db,
                    "source_caption": texts[index],
                    "recovered_caption": predictions[index],
                    "regression_psnr": float(regression_metrics["psnr"][index]),
                    "gated_psnr": float(gated_metrics["psnr"][index]),
                    "regression_ssim": float(ssim_per_image(regression[index:index + 1], images[index:index + 1])[0]),
                    "gated_ssim": float(gated_metrics["ssim"][index]),
                    "common_base_psnr": (
                        float(batch_image_metrics(
                            common_base[index:index + 1],
                            images[index:index + 1],
                        )["psnr"][0])
                        if common_base is not None else float("nan")
                    ),
                    "text_token_accuracy": per_sample_token_accuracy(outputs, index),
                    "common_caption": (
                        model.decode_text_logits(
                            outputs["common_base_text_logits"][index:index + 1]
                        )[0]
                        if outputs.get("common_base_text_logits") is not None
                        else ""
                    ),
                    "residual_gate_mean": (
                        float(gate_mean[index].float().mean())
                        if gate_mean is not None else 0.0
                    ),
                    "image_token_damage": args.image_damage,
                    "semantic_ablation": args.semantic_ablation,
                    "channel_type": model.channel.channel_type,
                    "channel_uses": int(reported_channel_uses),
                    "panel": f"{stem}.png",
                }
            )
            print(
                f"saved {stem}: reg/gated PSNR="
                f"{rows[-1]['regression_psnr']:.2f}/{rows[-1]['gated_psnr']:.2f}, "
                f"gate={rows[-1]['residual_gate_mean']:.3f}"
            )

    with (output_dir / "samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)
    print(f"\nDone. Open the PNG files and {output_dir / 'samples.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export V6.1 recovery panels")
    parser.add_argument("--checkpoint", required=True)
    # The downloaded seed42/43/44 checkpoints are the original 4608-symbol
    # V6.1 checkpoints, so this must stay the default unless a matching
    # adaptive-bandwidth checkpoint is supplied.
    parser.add_argument("--config", default="configs/v61_bandwidth.yaml")
    parser.add_argument("--output-dir", default="qualitative_recovery")
    parser.add_argument("--data-root", required=True, help="COCO root containing val2017/ and annotations/")
    parser.add_argument("--vae-path", help="Optional local Stable Diffusion VAE directory")
    parser.add_argument("--text-model-path", help="Optional local CLIP text-model directory")
    parser.add_argument("--snr-list", type=float, nargs="+", default=[20, 10, 0])
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--preset", default="original_4608")
    parser.add_argument("--channel-type", choices=["awgn", "rayleigh", "gamma_gamma", "fso"], default="awgn")
    parser.add_argument("--channel-width", type=int)
    parser.add_argument("--utility-weight", type=float, default=0.5)
    parser.add_argument("--semantic-ablation", default="full")
    parser.add_argument(
        "--image-damage", type=float, default=0.5,
        help="Image-token erasure rate in [0,1]. Use 0.5 to expose the V6.1 gate; 0.0 disables it by design.",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
