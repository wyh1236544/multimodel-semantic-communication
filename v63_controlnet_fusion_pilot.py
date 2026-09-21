"""V6.3 feasibility pilot: spatially constrained generative recovery.

This script is intentionally an oracle *diagnostic*, not the final deployable
receiver.  It asks whether SD1.5 + ControlNet Tile can produce at least one
candidate that beats the V6.2 regression result, and whether conservative
fusion can retain fidelity while adding useful generated detail.

The source image is used only after reception to score candidates.  No source
pixels or source-side masks are passed to the recovery pipeline.
"""
from __future__ import annotations

import argparse
import csv
import inspect
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

from data import build_dataloader
from models.clip_semantics import CLIPSemanticSimilarity
from models.quality_adaptive_recovery import (
    QualityAdaptivePolicy,
    SemanticConditionAdapter,
    receiver_features,
)
from utils.checkpoint import load_checkpoint, load_state_dict_flexible
from utils.config import load_config, prepare_config
from utils.experiment import create_model, seed_everything, select_device
from utils.metrics import batch_image_metrics
from utils.v62_diffusion import (
    add_semantic_context,
    pil_to_tensor,
    prompt_embeddings,
    tensor_to_pil,
)


def _load_adapter(path: str, device: torch.device) -> SemanticConditionAdapter:
    checkpoint = torch.load(path, map_location=device)
    adapter = SemanticConditionAdapter(**checkpoint["adapter_config"]).to(device)
    adapter.load_state_dict(checkpoint["adapter"])
    return adapter.eval()


def _metric_bundle(
    scorer: CLIPSemanticSimilarity,
    candidate: torch.Tensor,
    target: torch.Tensor,
    prompt: str,
) -> Dict[str, float]:
    image_metrics = batch_image_metrics(
        candidate.unsqueeze(0), target.unsqueeze(0)
    )
    clip_image = scorer.image_similarity(
        candidate.unsqueeze(0), target.unsqueeze(0)
    )
    clip_text = scorer.similarity(
        candidate.unsqueeze(0), [prompt or "a natural photograph"]
    )
    psnr = float(image_metrics["psnr"][0])
    ssim = float(image_metrics["ssim"][0])
    clip_image_value = float(clip_image[0])
    clip_text_value = float(clip_text[0])
    psnr_scaled = max(0.0, min(1.0, (psnr - 10.0) / 30.0))
    utility = (
        0.25 * ssim
        + 0.15 * psnr_scaled
        + 0.40 * (0.5 * (clip_image_value + 1.0))
        + 0.20 * (0.5 * (clip_text_value + 1.0))
    )
    return {
        "utility": utility,
        "psnr": psnr,
        "ssim": ssim,
        "clip_image": clip_image_value,
        "clip_text": clip_text_value,
    }


def _blur(image: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    pad = kernel_size // 2
    value = image.unsqueeze(0)
    value = F.pad(value, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(value, kernel_size, stride=1).squeeze(0)


def _fuse(
    regression: torch.Tensor,
    generated: torch.Tensor,
    expert: str,
    alpha: float,
) -> torch.Tensor:
    alpha = float(alpha)
    if expert == "raw":
        return generated
    if expert == "rgb":
        return ((1.0 - alpha) * regression + alpha * generated).clamp(0.0, 1.0)
    if expert == "detail":
        regression_detail = regression - _blur(regression)
        generated_detail = generated - _blur(generated)
        return (
            regression + alpha * (generated_detail - regression_detail)
        ).clamp(0.0, 1.0)
    raise ValueError("Unsupported fusion expert: %s" % expert)


def _generate_controlnet(
    pipe: StableDiffusionControlNetImg2ImgPipeline,
    regression: torch.Tensor,
    prompt: str,
    semantic_tokens: Optional[torch.Tensor],
    strength: float,
    steps: int,
    guidance_scale: float,
    control_scale: float,
    generator: torch.Generator,
) -> torch.Tensor:
    height, width = int(regression.size(1)), int(regression.size(2))
    condition = tensor_to_pil(regression, size=512)
    kwargs = {
        "image": condition,
        "control_image": condition,
        "height": 512,
        "width": 512,
        "strength": float(strength),
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "controlnet_conditioning_scale": float(control_scale),
        "generator": generator,
    }
    if semantic_tokens is None:
        kwargs["prompt"] = prompt if prompt.strip() else "a natural photograph"
    else:
        conditional, negative = prompt_embeddings(pipe, [prompt], regression.device)
        conditional, negative = add_semantic_context(
            conditional, negative, semantic_tokens
        )
        kwargs["prompt_embeds"] = conditional
        kwargs["negative_prompt_embeds"] = negative
    result = pipe(**kwargs).images[0]
    result = result.resize((width, height), Image.Resampling.LANCZOS)
    return pil_to_tensor(result).to(regression.device)


def _tile(image: Image.Image, title: str) -> Image.Image:
    header = 34
    canvas = Image.new("RGB", (image.width, image.height + header), "white")
    canvas.paste(image, (0, header))
    ImageDraw.Draw(canvas).text(
        (4, 8), title, fill="black", font=ImageFont.load_default()
    )
    return canvas


def _panel(
    original: torch.Tensor,
    regression: torch.Tensor,
    raw: torch.Tensor,
    best_generation: torch.Tensor,
    selected: torch.Tensor,
    best_expert: str,
    best_alpha: float,
    positive: bool,
) -> Image.Image:
    tiles = [
        _tile(tensor_to_pil(original), "Original (score only)"),
        _tile(tensor_to_pil(regression), "Adaptive regression"),
        _tile(tensor_to_pil(raw), "Best raw ControlNet"),
        _tile(
            tensor_to_pil(best_generation),
            "%s fusion alpha=%.2f" % (best_expert, best_alpha),
        ),
        _tile(
            tensor_to_pil(selected),
            "Oracle selected: %s" % ("generation" if positive else "regression"),
        ),
        _tile(
            tensor_to_pil((selected - original).abs().clamp(0, 0.25) / 0.25),
            "Selected error",
        ),
    ]
    canvas = Image.new(
        "RGB", (sum(tile.width for tile in tiles), tiles[0].height), "white"
    )
    offset = 0
    for tile in tiles:
        canvas.paste(tile, (offset, 0))
        offset += tile.width
    return canvas


def _write_summary(rows: List[Dict[str, object]], path: Path) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[float(row["snr_db"])].append(row)
        groups["ALL"].append(row)
    records = []
    for name, items in groups.items():
        def mean(field: str) -> float:
            return sum(float(item[field]) for item in items) / len(items)

        records.append(
            {
                "snr_db": name,
                "samples": len(items),
                "oracle_generation_positive_rate": mean("generation_positive"),
                "regression_utility": mean("regression_utility"),
                "best_generation_utility": mean("best_generation_utility"),
                "oracle_selected_utility": mean("selected_utility"),
                "oracle_utility_gain": mean("selected_utility")
                - mean("regression_utility"),
                "regression_psnr": mean("regression_psnr"),
                "selected_psnr": mean("selected_psnr"),
                "oracle_psnr_gain": mean("selected_psnr")
                - mean("regression_psnr"),
                "regression_ssim": mean("regression_ssim"),
                "selected_ssim": mean("selected_ssim"),
                "oracle_ssim_gain": mean("selected_ssim")
                - mean("regression_ssim"),
                "regression_clip_image": mean("regression_clip_image"),
                "selected_clip_image": mean("selected_clip_image"),
                "oracle_clip_gain": mean("selected_clip_image")
                - mean("regression_clip_image"),
                "mean_best_strength": mean("best_strength"),
                "mean_best_alpha": mean("best_alpha"),
            }
        )
    records.sort(
        key=lambda row: (
            row["snr_db"] == "ALL",
            float(row["snr_db"]) if row["snr_db"] != "ALL" else 1e9,
        )
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("V6.3 ControlNet pilot requires CUDA")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = prepare_config(
        load_config(args.config),
        preset=args.preset,
        vae_path=args.vae_path,
        text_model_path=args.text_model_path,
        data_root=args.data_root,
    )
    config["model"].setdefault("dynamic_token_allocation", {})[
        "text_selection_strategy"
    ] = "valid_prefix_repeat"
    config.setdefault("channel", {})["type"] = args.channel_type
    v61 = create_model(config, device)
    report = load_state_dict_flexible(
        v61, load_checkpoint(args.checkpoint), allow_shape_expansion=False
    )
    v61.eval()
    policy = QualityAdaptivePolicy().to(device).eval()
    scorer = CLIPSemanticSimilarity(args.clip_model_path).to(device).eval()
    adapter = _load_adapter(args.adapter_checkpoint, device)
    print("Loaded V6.1 exact tensors=%d" % report["exact_count"])

    data = config["data"]
    loader = build_dataloader(
        image_root=data["val_image_root"],
        annotation_file=data["val_annotation_file"],
        image_size=int(config["model"]["image_size"]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=False,
        caption_mode=data.get("val_caption_mode", "shortest"),
        max_samples=args.samples,
        seed=args.seed,
    )
    supported = set(inspect.signature(v61.forward).parameters)
    total_tokens = int(v61.num_common_tokens + v61.channel_token_budget)
    cached: List[Dict[str, object]] = []
    for snr in args.snr_list:
        for batch in loader:
            images = batch["images"].to(device)
            shared_noise = torch.randn(
                images.size(0), total_tokens, int(v61.latent_dim), device=device
            )
            shared_gain = None
            if v61.channel.channel_type != "awgn":
                shared_gain = v61.channel.channel.sample_gain(shared_noise)
            probe_kwargs = {
                "snr_db": snr,
                "disable_modality_dropout": True,
                "utility_weight": args.probe_utility,
                "channel_noise": shared_noise,
                "channel_gain": shared_gain,
            }
            probe_kwargs = {
                key: value for key, value in probe_kwargs.items() if key in supported
            }
            probe = v61(images, batch["texts"], **probe_kwargs)
            probe_captions = v61.decode_text_logits(probe["text_logits"])
            probe_alignment = scorer.similarity(probe["image"], probe_captions)
            probe_features = receiver_features(
                probe, snr, image_text_alignment=probe_alignment
            )
            decision = policy(probe_features)
            utility = decision["text_utility"].clamp(0.0, 1.0)
            adaptive_kwargs = {
                "snr_db": snr,
                "disable_modality_dropout": True,
                "utility_weight": utility,
                "channel_noise": shared_noise,
                "channel_gain": shared_gain,
            }
            adaptive_kwargs = {
                key: value
                for key, value in adaptive_kwargs.items()
                if key in supported
            }
            adaptive = v61(images, batch["texts"], **adaptive_kwargs)
            captions = v61.decode_text_logits(adaptive["text_logits"])
            alignment = scorer.similarity(adaptive["image"], captions)
            features = receiver_features(
                adaptive, snr, image_text_alignment=alignment
            )
            for index, image_id in enumerate(batch["image_ids"]):
                cached.append(
                    {
                        "image_id": int(image_id),
                        "snr": float(snr),
                        "original": images[index].detach().cpu(),
                        "regression": adaptive["image"][index].detach().cpu(),
                        "prompt": captions[index],
                        "features": features[index].detach().cpu(),
                        "semantic_output": {
                            name: adaptive[name][index : index + 1].detach().cpu()
                            for name in (
                                "decoded_common_tokens",
                                "decoded_image_tokens",
                                "decoded_text_tokens",
                            )
                        },
                    }
                )
            print("Cached SNR=%g total=%d" % (snr, len(cached)))

    v61.to("cpu")
    scorer.to("cpu")
    del v61, scorer
    torch.cuda.empty_cache()

    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_path,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.diffusion_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=False)
    scorer = CLIPSemanticSimilarity(args.clip_model_path).to(device).eval()

    rows: List[Dict[str, object]] = []
    for index, item in enumerate(cached):
        original = item["original"].to(device)
        regression = item["regression"].to(device)
        prompt = str(item["prompt"])
        semantic_output = {
            key: value.to(device)
            for key, value in item["semantic_output"].items()
        }
        semantic_tokens = adapter(
            semantic_output, item["features"].to(device).unsqueeze(0)
        )
        regression_metrics = _metric_bundle(
            scorer, regression, original, prompt
        )
        best_raw = None
        best_raw_metrics = None
        best_generation = None
        best_metrics = None
        best_strength = float(args.strengths[0])
        best_alpha = 1.0
        best_expert = "raw"
        for strength in args.strengths:
            generator = torch.Generator(device=device).manual_seed(
                args.seed * 100000 + index * 100 + int(round(strength * 100))
            )
            generated = _generate_controlnet(
                pipe,
                regression,
                prompt,
                semantic_tokens,
                strength,
                args.steps,
                args.guidance_scale,
                args.control_scale,
                generator,
            )
            raw_metrics = _metric_bundle(scorer, generated, original, prompt)
            if best_raw_metrics is None or raw_metrics["utility"] > best_raw_metrics["utility"]:
                best_raw = generated
                best_raw_metrics = raw_metrics
            candidates = [("raw", 1.0, generated, raw_metrics)]
            for expert in ("rgb", "detail"):
                for alpha in args.blend_alphas:
                    fused = _fuse(regression, generated, expert, alpha)
                    metrics = _metric_bundle(scorer, fused, original, prompt)
                    candidates.append((expert, float(alpha), fused, metrics))
            for expert, alpha, candidate, metrics in candidates:
                if best_metrics is None or metrics["utility"] > best_metrics["utility"]:
                    best_generation = candidate
                    best_metrics = metrics
                    best_strength = float(strength)
                    best_alpha = float(alpha)
                    best_expert = expert

        positive = bool(
            best_metrics["utility"]
            > regression_metrics["utility"] + args.generation_margin
        )
        selected = best_generation if positive else regression
        selected_metrics = best_metrics if positive else regression_metrics
        row = {
            "image_id": item["image_id"],
            "snr_db": item["snr"],
            "received_caption": prompt,
            "generation_positive": int(positive),
            "best_strength": best_strength,
            "best_expert": best_expert,
            "best_alpha": best_alpha,
            "regression_utility": regression_metrics["utility"],
            "best_generation_utility": best_metrics["utility"],
            "selected_utility": selected_metrics["utility"],
            "regression_psnr": regression_metrics["psnr"],
            "best_generation_psnr": best_metrics["psnr"],
            "selected_psnr": selected_metrics["psnr"],
            "regression_ssim": regression_metrics["ssim"],
            "best_generation_ssim": best_metrics["ssim"],
            "selected_ssim": selected_metrics["ssim"],
            "regression_clip_image": regression_metrics["clip_image"],
            "best_generation_clip_image": best_metrics["clip_image"],
            "selected_clip_image": selected_metrics["clip_image"],
        }
        stem = "id%s_snr%s" % (
            item["image_id"], str(item["snr"]).replace(".", "p")
        )
        _panel(
            original.cpu(),
            regression.cpu(),
            best_raw.cpu(),
            best_generation.cpu(),
            selected.cpu(),
            best_expert,
            best_alpha,
            positive,
        ).save(output_dir / (stem + ".png"))
        row["panel"] = stem + ".png"
        rows.append(row)
        print(
            "[%d/%d] %s positive=%d expert=%s alpha=%.2f strength=%.2f "
            "utility=%.4f->%.4f PSNR=%.2f->%.2f"
            % (
                index + 1,
                len(cached),
                stem,
                int(positive),
                best_expert,
                best_alpha,
                best_strength,
                regression_metrics["utility"],
                best_metrics["utility"],
                regression_metrics["psnr"],
                best_metrics["psnr"],
            )
        )

    with (output_dir / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(rows, output_dir / "summary.csv")
    print("Done: %s" % output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V6.3 ControlNet Tile + conservative fusion feasibility pilot"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--diffusion-model-path", required=True)
    parser.add_argument("--controlnet-path", required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--vae-path")
    parser.add_argument("--text-model-path")
    parser.add_argument("--output-dir", default="experiment_results/v63_controlnet_pilot")
    parser.add_argument("--config", default="configs/v61_bandwidth.yaml")
    parser.add_argument("--preset", default="original_4608")
    parser.add_argument(
        "--channel-type",
        default="awgn",
        choices=["awgn", "rayleigh", "gamma_gamma", "fso"],
    )
    parser.add_argument("--snr-list", type=float, nargs="+", default=[0, 10])
    parser.add_argument("--probe-utility", type=float, default=0.5)
    parser.add_argument(
        "--strengths", type=float, nargs="+", default=[0.25, 0.45, 0.65]
    )
    parser.add_argument(
        "--blend-alphas", type=float, nargs="+", default=[0.25, 0.50, 0.75]
    )
    parser.add_argument("--control-scale", type=float, default=0.8)
    parser.add_argument("--generation-margin", type=float, default=0.002)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
