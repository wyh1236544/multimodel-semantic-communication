"""Selective regression/diffusion receiver for the existing V6.1 checkpoint.

This is an inference-only feasibility experiment.  Above --switch-snr the
receiver returns the V6.1 regression output.  At or below the threshold it
uses that reconstruction as the img2img condition for a frozen Stable
Diffusion v1.5 pipeline.  The default prompt is the *received* caption, not
the ground-truth caption, to avoid leaking source information to the receiver.
"""
from __future__ import annotations

import argparse
import csv
import inspect
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import StableDiffusionImg2ImgPipeline

from data import build_dataloader
from utils.checkpoint import load_checkpoint, load_state_dict_flexible
from utils.config import load_config, prepare_config
from utils.experiment import create_model, seed_everything, select_device
from utils.metrics import batch_image_metrics


def to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array, "RGB")


def to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def error_image(reconstruction: torch.Tensor, target: torch.Tensor) -> Image.Image:
    return to_pil((reconstruction.float() - target.float()).abs().clamp(0, 0.25) / 0.25)


def tile(image: Image.Image, title: str) -> Image.Image:
    header = 30
    canvas = Image.new("RGB", (image.width, image.height + header), "white")
    canvas.paste(image, (0, header))
    ImageDraw.Draw(canvas).text((5, 7), title, fill="black", font=ImageFont.load_default())
    return canvas


def panel(original: torch.Tensor, regression: torch.Tensor, diffusion: torch.Tensor, selected: torch.Tensor, used_diffusion: bool) -> Image.Image:
    tiles = [
        tile(to_pil(original), "Original"),
        tile(to_pil(regression), "Regression"),
        tile(to_pil(diffusion), "Diffusion img2img"),
        tile(to_pil(selected), "Selected: diffusion" if used_diffusion else "Selected: regression"),
        tile(error_image(selected, original), "|Selected - original|"),
    ]
    result = Image.new("RGB", (sum(item.width for item in tiles), tiles[0].height), "white")
    x = 0
    for item in tiles:
        result.paste(item, (x, 0))
        x += item.width
    return result


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = select_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This first diffusion experiment requires CUDA.")

    config = prepare_config(
        load_config(args.config), preset=args.preset, vae_path=args.vae_path,
        text_model_path=args.text_model_path, data_root=args.data_root,
    )
    config.setdefault("channel", {})["type"] = args.channel_type
    config["test"]["batch_size"] = min(args.batch_size, args.samples)
    config["test"]["num_workers"] = args.num_workers
    v6 = create_model(config, device)
    report = load_state_dict_flexible(v6, load_checkpoint(args.checkpoint), allow_shape_expansion=False)
    v6.eval()
    print(f"Loaded V6.1: exact tensors={report['exact_count']}")

    data = config["data"]
    loader = build_dataloader(
        image_root=data["val_image_root"], annotation_file=data["val_annotation_file"],
        image_size=int(config["model"]["image_size"]), batch_size=min(args.batch_size, args.samples),
        num_workers=args.num_workers, train=False, caption_mode=data.get("val_caption_mode", "shortest"),
        max_samples=args.samples, seed=args.seed,
    )
    supported = set(inspect.signature(v6.forward).parameters)
    cached = []
    remaining = args.samples
    for batch in loader:
        if remaining <= 0:
            break
        images = batch["images"].to(device)
        forward_kwargs = {"snr_db": args.snr, "disable_modality_dropout": True, "semantic_ablation": "full", "utility_weight": args.utility_weight}
        forward_kwargs = {key: value for key, value in forward_kwargs.items() if key in supported}
        output = v6(images, batch["texts"], **forward_kwargs)
        received_captions = v6.decode_text_logits(output["text_logits"])
        for index, image_id in enumerate(batch["image_ids"]):
            if remaining <= 0:
                break
            cached.append((
                int(image_id), images[index].detach().cpu(), output["image"][index].detach().cpu(),
                batch["texts"][index], received_captions[index],
            ))
            remaining -= 1
    # Release V6.1 GPU memory before loading the 0.9B diffusion prior.
    v6.to("cpu")
    del v6
    torch.cuda.empty_cache()

    rows = []
    use_diffusion = args.snr <= args.switch_snr
    pipe = None
    if use_diffusion:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            args.diffusion_model_path, torch_dtype=torch.float16,
            local_files_only=True, safety_checker=None, requires_safety_checker=False,
        ).to(device)
        pipe.set_progress_bar_config(disable=False)
    for index, (image_id, original, regression, source_caption, received_caption) in enumerate(cached):
        prompt = source_caption if args.prompt_source == "source" else received_caption
        if not prompt.strip():
            prompt = "a natural photograph"
        generator = torch.Generator(device=device).manual_seed(args.seed * 1000 + index)
        if use_diffusion:
            generated_pil = pipe(
                prompt=prompt, image=to_pil(regression).resize((512, 512), Image.Resampling.BICUBIC),
                strength=args.strength, guidance_scale=args.guidance_scale,
                num_inference_steps=args.steps, generator=generator,
            ).images[0].resize((original.shape[2], original.shape[1]), Image.Resampling.LANCZOS)
            diffusion = to_tensor(generated_pil)
        else:
            # A true selective receiver does not pay diffusion latency at high SNR.
            diffusion = regression.clone()
        selected = diffusion if use_diffusion else regression
        reg_metrics = batch_image_metrics(regression.unsqueeze(0), original.unsqueeze(0))
        dif_metrics = batch_image_metrics(diffusion.unsqueeze(0), original.unsqueeze(0))
        selected_metrics = batch_image_metrics(selected.unsqueeze(0), original.unsqueeze(0))
        stem = f"id{image_id}_snr{args.snr:g}".replace(".", "p")
        panel(original, regression, diffusion, selected, use_diffusion).save(output_dir / f"{stem}.png")
        rows.append({
            "image_id": image_id, "snr_db": args.snr, "selected_mode": "diffusion" if use_diffusion else "regression",
            "source_caption_ORACLE_ONLY": source_caption, "received_caption_used": received_caption,
            "prompt_used": prompt, "regression_psnr": float(reg_metrics["psnr"][0]),
            "diffusion_psnr": float(dif_metrics["psnr"][0]), "selected_psnr": float(selected_metrics["psnr"][0]),
            "regression_ssim": float(reg_metrics["ssim"][0]), "diffusion_ssim": float(dif_metrics["ssim"][0]),
            "selected_ssim": float(selected_metrics["ssim"][0]), "panel": f"{stem}.png",
        })
        print(f"saved {stem}: regression/diffusion PSNR={rows[-1]['regression_psnr']:.2f}/{rows[-1]['diffusion_psnr']:.2f}")
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Done: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Selective V6.1 regression/diffusion recovery")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--diffusion-model-path", required=True)
    parser.add_argument("--config", default="configs/v61_bandwidth.yaml")
    parser.add_argument("--output-dir", default="experiment_results/selective_diffusion")
    parser.add_argument("--vae-path")
    parser.add_argument("--text-model-path")
    parser.add_argument("--snr", type=float, default=0.0)
    parser.add_argument("--switch-snr", type=float, default=10.0)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preset", default="original_4608")
    parser.add_argument("--channel-type", default="awgn", choices=["awgn", "rayleigh", "gamma_gamma", "fso"])
    parser.add_argument("--utility-weight", type=float, default=0.5)
    parser.add_argument("--prompt-source", choices=["received", "source"], default="received")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
