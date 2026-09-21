"""End-to-end V6.2 quality-adaptive transmission and recovery evaluation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import inspect
from pathlib import Path
import time
from typing import Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import StableDiffusionImg2ImgPipeline

from data import build_dataloader
from models.clip_semantics import CLIPSemanticSimilarity
from models.quality_adaptive_recovery import (
    QualityAdaptivePolicy,
    SemanticConditionAdapter,
    load_policy_checkpoint,
    receiver_features,
)
from utils.checkpoint import load_checkpoint, load_state_dict_flexible
from utils.config import load_config, prepare_config
from utils.experiment import create_model, seed_everything, select_device
from utils.metrics import batch_image_metrics, sentence_bleu2
from utils.v62_diffusion import generate_img2img, tensor_to_pil


def _load_adapter(path: Optional[str], device: torch.device):
    if not path:
        return None
    checkpoint = torch.load(path, map_location=device)
    adapter = SemanticConditionAdapter(**checkpoint["adapter_config"]).to(device)
    adapter.load_state_dict(checkpoint["adapter"])
    adapter.eval()
    return adapter


def _tile(image: Image.Image, title: str) -> Image.Image:
    header = 30
    canvas = Image.new("RGB", (image.width, image.height + header), "white")
    canvas.paste(image, (0, header))
    ImageDraw.Draw(canvas).text((5, 7), title, fill="black", font=ImageFont.load_default())
    return canvas


def _panel(original, first, adaptive, selected, mode, probability, utility):
    tiles = [
        _tile(tensor_to_pil(original), "Original"),
        _tile(tensor_to_pil(first), "Probe regression"),
        _tile(tensor_to_pil(adaptive), "Adaptive regression"),
        _tile(tensor_to_pil(selected), "Selected %s p=%.2f" % (mode, probability)),
        _tile(
            tensor_to_pil((selected - original).abs().clamp(0, 0.25) / 0.25),
            "Error |selected-original|, text u=%.2f" % utility,
        ),
    ]
    result = Image.new("RGB", (sum(tile.width for tile in tiles), tiles[0].height), "white")
    position = 0
    for tile in tiles:
        result.paste(tile, (position, 0))
        position += tile.width
    return result


def _ablation_panel(
    original,
    probe,
    snr_baseline,
    adaptive,
    quality_prompt,
    full_selected,
    mode,
    probability,
):
    tiles = [
        _tile(tensor_to_pil(original), "Original"),
        _tile(tensor_to_pil(probe), "Fixed regression"),
        _tile(tensor_to_pil(snr_baseline), "SNR-only baseline"),
        _tile(tensor_to_pil(adaptive), "Text-adaptive regression"),
        _tile(tensor_to_pil(quality_prompt), "Quality + prompt-only"),
        _tile(
            tensor_to_pil(full_selected),
            "Full %s p=%.2f" % (mode, probability),
        ),
        _tile(
            tensor_to_pil(
                (full_selected - original).abs().clamp(0, 0.25) / 0.25
            ),
            "Full error",
        ),
    ]
    result = Image.new(
        "RGB", (sum(tile.width for tile in tiles), tiles[0].height), "white"
    )
    position = 0
    for tile in tiles:
        result.paste(tile, (position, 0))
        position += tile.width
    return result


def _text_accuracy_per_sample(outputs):
    logits = outputs["text_logits"]
    ids = outputs["text_input_ids"]
    mask = outputs["text_attention_mask"].bool()
    length = min(logits.size(1), ids.size(1), mask.size(1))
    correct = (logits[:, :length].argmax(-1) == ids[:, :length]).float()
    return (correct * mask[:, :length]).sum(1) / mask[:, :length].sum(1).clamp_min(1)


def _write_summary(rows, path):
    groups = defaultdict(list)
    for row in rows:
        groups[float(row["snr_db"])].append(row)
        groups["ALL"].append(row)
    summary = []
    for group_name, items in groups.items():
        def mean(name):
            return sum(float(item[name]) for item in items) / max(len(items), 1)

        record = {
            "snr_db": group_name,
            "samples": len(items),
            "generation_rate": mean("is_generation"),
            "feedback_rate": mean("feedback_used"),
            "mean_text_tokens": mean("allocated_text_tokens"),
            "mean_text_repetition_factor": mean("text_repetition_factor"),
            "probe_text_accuracy": mean("probe_text_accuracy"),
            "adaptive_text_accuracy": mean("adaptive_text_accuracy"),
            "text_accuracy_gain": mean("adaptive_text_accuracy") - mean("probe_text_accuracy"),
            "probe_bleu2": mean("probe_bleu2"),
            "adaptive_bleu2": mean("adaptive_bleu2"),
            "probe_psnr": mean("probe_psnr"),
            "adaptive_psnr": mean("adaptive_psnr"),
            "selected_psnr": mean("selected_psnr"),
            "feedback_psnr_gain": mean("adaptive_psnr") - mean("probe_psnr"),
            "routing_psnr_gain": mean("selected_psnr") - mean("adaptive_psnr"),
            "probe_ssim": mean("probe_ssim"),
            "adaptive_ssim": mean("adaptive_ssim"),
            "selected_ssim": mean("selected_ssim"),
            "feedback_ssim_gain": mean("adaptive_ssim") - mean("probe_ssim"),
            "routing_ssim_gain": mean("selected_ssim") - mean("adaptive_ssim"),
            "policy_data_overlap_count": sum(
                int(item["policy_data_overlap"]) for item in items
            ),
            "selected_diffusion_latency_ms": mean("selected_diffusion_latency_ms"),
        }
        if "adaptive_clip_image" in items[0]:
            record.update(
                {
                    "adaptive_clip_image": mean("adaptive_clip_image"),
                    "selected_clip_image": mean("selected_clip_image"),
                    "routing_clip_gain": mean("selected_clip_image")
                    - mean("adaptive_clip_image"),
                }
            )
        if "snr_baseline_psnr" in items[0]:
            record.update(
                {
                    "snr_baseline_generation_rate": mean("is_snr_generation"),
                    "snr_baseline_psnr": mean("snr_baseline_psnr"),
                    "snr_baseline_ssim": mean("snr_baseline_ssim"),
                    "quality_prompt_psnr": mean("quality_prompt_psnr"),
                    "quality_prompt_ssim": mean("quality_prompt_ssim"),
                    "quality_vs_snr_psnr_gain": mean("selected_psnr")
                    - mean("snr_baseline_psnr"),
                    "quality_vs_snr_ssim_gain": mean("selected_ssim")
                    - mean("snr_baseline_ssim"),
                    "semantic_adapter_psnr_gain": mean("selected_psnr")
                    - mean("quality_prompt_psnr"),
                    "semantic_adapter_ssim_gain": mean("selected_ssim")
                    - mean("quality_prompt_ssim"),
                    "snr_baseline_latency_ms": mean("snr_baseline_latency_ms"),
                    "quality_prompt_latency_ms": mean("quality_prompt_latency_ms"),
                }
            )
            if "snr_baseline_clip_image" in items[0]:
                record.update(
                    {
                        "snr_baseline_clip_image": mean("snr_baseline_clip_image"),
                        "quality_prompt_clip_image": mean("quality_prompt_clip_image"),
                        "quality_vs_snr_clip_gain": mean("selected_clip_image")
                        - mean("snr_baseline_clip_image"),
                        "semantic_adapter_clip_gain": mean("selected_clip_image")
                        - mean("quality_prompt_clip_image"),
                    }
                )
        summary.append(record)
    summary.sort(
        key=lambda item: (
            item["snr_db"] == "ALL",
            float(item["snr_db"]) if item["snr_db"] != "ALL" else 1e9,
        )
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def _timed_generate(pipe, device, **kwargs):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = generate_img2img(pipe, **kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, 1000.0 * (time.perf_counter() - started)


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = prepare_config(
        load_config(args.config), preset=args.preset, vae_path=args.vae_path,
        text_model_path=args.text_model_path, data_root=args.data_root,
    )
    config["model"].setdefault("dynamic_token_allocation", {})[
        "text_selection_strategy"
    ] = "valid_prefix_repeat"
    config.setdefault("channel", {})["type"] = args.channel_type
    v61 = create_model(config, device)
    report = load_state_dict_flexible(v61, load_checkpoint(args.checkpoint), False)
    v61.eval()
    if args.policy_checkpoint:
        policy = load_policy_checkpoint(args.policy_checkpoint, device)
        policy_name = args.policy_checkpoint
    else:
        policy = QualityAdaptivePolicy().to(device).eval()
        policy_name = "safe heuristic initialization"
    excluded_policy_ids = set()
    if args.policy_checkpoint and not args.allow_policy_data_overlap:
        excluded_policy_ids = set(
            getattr(policy, "policy_data_image_ids", set())
        )
        if not excluded_policy_ids:
            raise RuntimeError(
                "The policy checkpoint does not record policy_data_image_ids. "
                "Retrain it with the updated train_v62_quality_policy.py, or "
                "pass --allow-policy-data-overlap only for debugging."
            )
        print(
            "Strict holdout enabled: excluding %d policy-data image ids"
            % len(excluded_policy_ids)
        )
    scorer = None
    if args.clip_model_path:
        scorer = CLIPSemanticSimilarity(args.clip_model_path).to(device).eval()
    print("Loaded V6.1 tensors=%d; policy=%s" % (report["exact_count"], policy_name))
    data = config["data"]
    loader = build_dataloader(
        image_root=data["val_image_root"], annotation_file=data["val_annotation_file"],
        image_size=int(config["model"]["image_size"]), batch_size=args.batch_size,
        num_workers=args.num_workers, train=False,
        caption_mode=data.get("val_caption_mode", "shortest"),
        max_samples=args.samples, seed=args.seed,
        exclude_image_ids=excluded_policy_ids,
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
            base_kwargs = {
                "snr_db": snr,
                "disable_modality_dropout": True,
                "utility_weight": args.probe_utility,
                "channel_noise": shared_noise,
                "channel_gain": shared_gain,
            }
            base_kwargs = {key: value for key, value in base_kwargs.items() if key in supported}
            first = v61(images, batch["texts"], **base_kwargs)
            first_captions = v61.decode_text_logits(first["text_logits"])
            first_text_accuracy = _text_accuracy_per_sample(first)
            first_alignment = (
                scorer.similarity(first["image"], first_captions)
                if scorer is not None
                else None
            )
            first_features = receiver_features(
                first, snr, image_text_alignment=first_alignment
            )
            first_decision = policy(first_features)
            recommended_utility = first_decision["text_utility"].clamp(0.0, 1.0)
            use_feedback = (
                (recommended_utility - args.probe_utility).abs() >= args.feedback_deadband
            )
            actual_utility = torch.where(
                use_feedback,
                recommended_utility,
                recommended_utility.new_full(recommended_utility.shape, args.probe_utility),
            )
            adaptive_kwargs = {
                "snr_db": snr,
                "disable_modality_dropout": True,
                "utility_weight": actual_utility,
                "channel_noise": shared_noise,
                "channel_gain": shared_gain,
            }
            adaptive_kwargs = {
                key: value for key, value in adaptive_kwargs.items() if key in supported
            }
            adaptive = v61(images, batch["texts"], **adaptive_kwargs)
            received_captions = v61.decode_text_logits(adaptive["text_logits"])
            adaptive_text_accuracy = _text_accuracy_per_sample(adaptive)
            final_alignment = (
                scorer.similarity(adaptive["image"], received_captions)
                if scorer is not None
                else None
            )
            final_features = receiver_features(
                adaptive, snr, image_text_alignment=final_alignment
            )
            final_decision = policy(final_features)
            for index, image_id in enumerate(batch["image_ids"]):
                cached.append(
                    {
                        "image_id": int(image_id),
                        "snr": float(snr),
                        "original": images[index].detach().cpu(),
                        "first": first["image"][index].detach().cpu(),
                        "adaptive": adaptive["image"][index].detach().cpu(),
                        "source_caption": batch["texts"][index],
                        "received_caption": received_captions[index],
                        "first_caption": first_captions[index],
                        "probe_text_accuracy": float(first_text_accuracy[index]),
                        "adaptive_text_accuracy": float(adaptive_text_accuracy[index]),
                        "features": final_features[index].detach().cpu(),
                        "generation_probability": float(final_decision["generation_probability"][index]),
                        "strength": float(final_decision["diffusion_strength"][index]),
                        "text_utility": float(actual_utility[index]),
                        "text_tokens": float(adaptive["allocated_text_tokens"][index]),
                        "valid_text_tokens": float(adaptive["valid_text_tokens"][index]),
                        "text_redundant_slots": float(adaptive["text_redundant_slots"][index]),
                        "text_repetition_factor": float(adaptive["text_repetition_factor"][index]),
                        "feedback_used": bool(use_feedback[index]),
                        "semantic_output": {
                            name: adaptive[name][index : index + 1].detach().cpu()
                            for name in (
                                "decoded_common_tokens", "decoded_image_tokens", "decoded_text_tokens"
                            )
                        },
                    }
                )
            print("Receiver decisions SNR=%g total_cached=%d" % (snr, len(cached)))
    v61.to("cpu")
    del v61
    torch.cuda.empty_cache()

    any_quality_generation = any(
        float(item["generation_probability"]) >= args.route_threshold for item in cached
    )
    any_snr_generation = args.full_ablation and any(
        float(item["snr"]) <= args.snr_baseline_threshold for item in cached
    )
    if args.full_ablation and not args.adapter_checkpoint:
        raise RuntimeError(
            "--full-ablation requires --adapter-checkpoint so prompt-only and "
            "semantic-conditioned generation can be compared explicitly."
        )
    pipe = None
    adapter = None
    if any_quality_generation or any_snr_generation:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            args.diffusion_model_path, torch_dtype=torch.float16,
            local_files_only=True, safety_checker=None, requires_safety_checker=False,
        ).to(device)
        pipe.set_progress_bar_config(disable=False)
        if args.adapter_checkpoint:
            adapter = _load_adapter(args.adapter_checkpoint, device)
    rows = []
    for index, item in enumerate(cached):
        original = item["original"].to(device)
        adaptive = item["adaptive"].to(device)
        probe = item["first"].to(device)
        probability = float(item["generation_probability"])
        use_generation = probability >= args.route_threshold
        selected_latency_ms = 0.0
        if use_generation:
            semantic_tokens = None
            if adapter is not None:
                semantic_output = {
                    key: value.to(device) for key, value in item["semantic_output"].items()
                }
                semantic_tokens = adapter(
                    semantic_output, item["features"].to(device).unsqueeze(0)
                )
            generator = torch.Generator(device=device).manual_seed(
                args.seed * 100000 + index * 10 + 1
            )
            selected, selected_latency_ms = _timed_generate(
                pipe,
                device,
                regression=adaptive,
                prompt=str(item["received_caption"]),
                strength=float(item["strength"]),
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                output_height=int(original.size(1)), output_width=int(original.size(2)),
                semantic_tokens=semantic_tokens,
            )
            selected = selected.to(device)
            mode = "generation"
        else:
            selected = adaptive
            mode = "regression"

        snr_baseline = None
        quality_prompt = None
        snr_baseline_latency_ms = 0.0
        quality_prompt_latency_ms = 0.0
        is_snr_generation = False
        if args.full_ablation:
            is_snr_generation = float(item["snr"]) <= args.snr_baseline_threshold
            if is_snr_generation:
                generator = torch.Generator(device=device).manual_seed(
                    args.seed * 100000 + index * 10 + 3
                )
                snr_baseline, snr_baseline_latency_ms = _timed_generate(
                    pipe,
                    device,
                    regression=probe,
                    prompt=str(item["first_caption"]),
                    strength=args.snr_baseline_strength,
                    steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    output_height=int(original.size(1)),
                    output_width=int(original.size(2)),
                    semantic_tokens=None,
                )
                snr_baseline = snr_baseline.to(device)
            else:
                snr_baseline = probe
            if use_generation:
                generator = torch.Generator(device=device).manual_seed(
                    # Match the semantic-adapter candidate's diffusion noise;
                    # conditioning is the only ablated variable.
                    args.seed * 100000 + index * 10 + 1
                )
                quality_prompt, quality_prompt_latency_ms = _timed_generate(
                    pipe,
                    device,
                    regression=adaptive,
                    prompt=str(item["received_caption"]),
                    strength=float(item["strength"]),
                    steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    output_height=int(original.size(1)),
                    output_width=int(original.size(2)),
                    semantic_tokens=None,
                )
                quality_prompt = quality_prompt.to(device)
            else:
                quality_prompt = adaptive
        adaptive_metrics = batch_image_metrics(adaptive.unsqueeze(0), original.unsqueeze(0))
        probe_metrics = batch_image_metrics(probe.unsqueeze(0), original.unsqueeze(0))
        selected_metrics = batch_image_metrics(selected.unsqueeze(0), original.unsqueeze(0))
        row = {
            "image_id": item["image_id"],
            "snr_db": item["snr"],
            "selected_mode": mode,
            "is_generation": int(use_generation),
            "generation_probability": probability,
            "diffusion_strength": item["strength"],
            "text_utility": item["text_utility"],
            "allocated_text_tokens": item["text_tokens"],
            "valid_text_tokens": item["valid_text_tokens"],
            "text_redundant_slots": item["text_redundant_slots"],
            "text_repetition_factor": item["text_repetition_factor"],
            "feedback_used": int(item["feedback_used"]),
            "policy_data_overlap": int(
                int(item["image_id"]) in excluded_policy_ids
            ),
            "selected_diffusion_latency_ms": selected_latency_ms,
            "probe_text_accuracy": item["probe_text_accuracy"],
            "adaptive_text_accuracy": item["adaptive_text_accuracy"],
            "probe_bleu2": sentence_bleu2(
                str(item["source_caption"]), str(item["first_caption"])
            ),
            "adaptive_bleu2": sentence_bleu2(
                str(item["source_caption"]), str(item["received_caption"])
            ),
            "probe_psnr": float(probe_metrics["psnr"][0]),
            "adaptive_psnr": float(adaptive_metrics["psnr"][0]),
            "selected_psnr": float(selected_metrics["psnr"][0]),
            "probe_ssim": float(probe_metrics["ssim"][0]),
            "adaptive_ssim": float(adaptive_metrics["ssim"][0]),
            "selected_ssim": float(selected_metrics["ssim"][0]),
            "received_caption": item["received_caption"],
        }
        if args.full_ablation:
            snr_metrics = batch_image_metrics(
                snr_baseline.unsqueeze(0), original.unsqueeze(0)
            )
            quality_prompt_metrics = batch_image_metrics(
                quality_prompt.unsqueeze(0), original.unsqueeze(0)
            )
            row.update(
                {
                    "is_snr_generation": int(is_snr_generation),
                    "snr_baseline_psnr": float(snr_metrics["psnr"][0]),
                    "snr_baseline_ssim": float(snr_metrics["ssim"][0]),
                    "quality_prompt_psnr": float(quality_prompt_metrics["psnr"][0]),
                    "quality_prompt_ssim": float(quality_prompt_metrics["ssim"][0]),
                    "snr_baseline_latency_ms": snr_baseline_latency_ms,
                    "quality_prompt_latency_ms": quality_prompt_latency_ms,
                }
            )
        if scorer is not None:
            row["adaptive_clip_image"] = float(
                scorer.image_similarity(adaptive.unsqueeze(0), original.unsqueeze(0))[0]
            )
            row["selected_clip_image"] = float(
                scorer.image_similarity(selected.unsqueeze(0), original.unsqueeze(0))[0]
            )
            if args.full_ablation:
                row["snr_baseline_clip_image"] = float(
                    scorer.image_similarity(
                        snr_baseline.unsqueeze(0), original.unsqueeze(0)
                    )[0]
                )
                row["quality_prompt_clip_image"] = float(
                    scorer.image_similarity(
                        quality_prompt.unsqueeze(0), original.unsqueeze(0)
                    )[0]
                )
        stem = "id%s_snr%s" % (item["image_id"], str(item["snr"]).replace(".", "p"))
        if args.full_ablation:
            comparison = _ablation_panel(
                original.cpu(),
                probe.cpu(),
                snr_baseline.cpu(),
                adaptive.cpu(),
                quality_prompt.cpu(),
                selected.cpu(),
                mode,
                probability,
            )
        else:
            comparison = _panel(
                original.cpu(), item["first"], adaptive.cpu(), selected.cpu(), mode,
                probability, float(item["text_utility"]),
            )
        comparison.save(output_dir / (stem + ".png"))
        row["panel"] = stem + ".png"
        rows.append(row)
        print(
            "[%d/%d] %s mode=%s p=%.2f text_tokens=%.0f repeat=%.2fx PSNR=%.2f->%.2f"
            % (
                index + 1, len(cached), stem, mode, probability,
                item["text_tokens"], item["text_repetition_factor"],
                row["adaptive_psnr"], row["selected_psnr"],
            )
        )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(rows, output_dir / "summary.csv")
    print("Done: %s" % output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate quality-adaptive V6.2")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--diffusion-model-path", required=True)
    parser.add_argument("--policy-checkpoint")
    parser.add_argument("--allow-policy-data-overlap", action="store_true")
    parser.add_argument("--adapter-checkpoint")
    parser.add_argument("--clip-model-path")
    parser.add_argument("--output-dir", default="experiment_results/v62_quality_adaptive")
    parser.add_argument("--config", default="configs/v61_bandwidth.yaml")
    parser.add_argument("--vae-path")
    parser.add_argument("--text-model-path")
    parser.add_argument("--preset", default="original_4608")
    parser.add_argument("--channel-type", default="awgn", choices=["awgn", "rayleigh", "gamma_gamma", "fso"])
    parser.add_argument("--snr-list", type=float, nargs="+", default=[0, 5, 10, 15, 20])
    parser.add_argument("--probe-utility", type=float, default=0.5)
    parser.add_argument("--feedback-deadband", type=float, default=0.10)
    parser.add_argument("--route-threshold", type=float, default=0.5)
    parser.add_argument("--full-ablation", action="store_true")
    parser.add_argument("--snr-baseline-threshold", type=float, default=10.0)
    parser.add_argument("--snr-baseline-strength", type=float, default=0.26)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
