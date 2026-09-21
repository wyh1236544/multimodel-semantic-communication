"""Strict holdout evaluation of the deployable V6.4 receiver policy."""
from __future__ import annotations

import argparse
import csv
import inspect
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

from data import build_dataloader
from models.clip_semantics import CLIPSemanticSimilarity
from models.controlnet_fusion_policy import load_controlnet_policy
from models.quality_adaptive_recovery import QualityAdaptivePolicy, receiver_features
from utils.checkpoint import load_checkpoint, load_state_dict_flexible
from utils.config import load_config, prepare_config
from utils.experiment import create_model, seed_everything, select_device
from utils.metrics import batch_image_metrics, sentence_bleu2
from utils.v62_diffusion import tensor_to_pil
from v63_controlnet_fusion_pilot import _fuse, _generate_controlnet, _load_adapter


def _text_accuracy(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    logits = outputs["text_logits"]
    ids = outputs["text_input_ids"]
    mask = outputs["text_attention_mask"].bool()
    length = min(logits.size(1), ids.size(1), mask.size(1))
    correct = (logits[:, :length].argmax(-1) == ids[:, :length]).float()
    return (correct * mask[:, :length]).sum(1) / mask[:, :length].sum(1).clamp_min(1)


def _tile(image: Image.Image, title: str) -> Image.Image:
    header = 34
    canvas = Image.new("RGB", (image.width, image.height + header), "white")
    canvas.paste(image, (0, header))
    ImageDraw.Draw(canvas).text((4, 8), title, fill="black", font=ImageFont.load_default())
    return canvas


def _panel(
    original, regression, raw, hard, soft, selected,
    mode, probability, expert, alpha, detail_probability, selection,
):
    tiles = [
        _tile(tensor_to_pil(original), "Original (metric only)"),
        _tile(tensor_to_pil(regression), "Adaptive regression"),
        _tile(tensor_to_pil(raw), "ControlNet raw"),
        _tile(
            tensor_to_pil(hard),
            "Hard %s alpha=%.2f" % (expert, alpha),
        ),
        _tile(
            tensor_to_pil(soft),
            "Soft detail-p=%.2f" % detail_probability,
        ),
        _tile(
            tensor_to_pil(selected),
            "%s selected=%s p=%.2f" % (mode, selection, probability),
        ),
        _tile(
            tensor_to_pil((selected - original).abs().clamp(0, 0.25) / 0.25),
            "Selected error",
        ),
    ]
    canvas = Image.new("RGB", (sum(tile.width for tile in tiles), tiles[0].height), "white")
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
        def mean(field):
            return sum(float(item[field]) for item in items) / len(items)

        record = {
            "snr_db": name,
            "samples": len(items),
            "generation_rate": mean("is_generation"),
            "detail_expert_rate": mean("is_detail_expert"),
            "feedback_rate": mean("feedback_used"),
            "mean_text_tokens": mean("allocated_text_tokens"),
            "mean_text_repetition_factor": mean("text_repetition_factor"),
            "mean_learned_text_utility": mean("learned_text_utility"),
            "mean_guarded_text_utility": mean("guarded_text_utility"),
            "probe_text_accuracy": mean("probe_text_accuracy"),
            "adaptive_text_accuracy": mean("adaptive_text_accuracy"),
            "text_accuracy_gain": mean("adaptive_text_accuracy") - mean("probe_text_accuracy"),
            "probe_bleu2": mean("probe_bleu2"),
            "adaptive_bleu2": mean("adaptive_bleu2"),
            "regression_psnr": mean("regression_psnr"),
            "hard_psnr": mean("hard_psnr"),
            "soft_psnr": mean("soft_psnr"),
            "selected_psnr": mean("selected_psnr"),
            "routing_psnr_gain": mean("selected_psnr") - mean("regression_psnr"),
            "soft_vs_hard_psnr_gain": mean("soft_psnr") - mean("hard_psnr"),
            "regression_ssim": mean("regression_ssim"),
            "hard_ssim": mean("hard_ssim"),
            "soft_ssim": mean("soft_ssim"),
            "selected_ssim": mean("selected_ssim"),
            "routing_ssim_gain": mean("selected_ssim") - mean("regression_ssim"),
            "soft_vs_hard_ssim_gain": mean("soft_ssim") - mean("hard_ssim"),
            "regression_clip_image": mean("regression_clip_image"),
            "hard_clip_image": mean("hard_clip_image"),
            "soft_clip_image": mean("soft_clip_image"),
            "selected_clip_image": mean("selected_clip_image"),
            "routing_clip_gain": mean("selected_clip_image") - mean("regression_clip_image"),
            "soft_vs_hard_clip_gain": mean("soft_clip_image") - mean("hard_clip_image"),
            "mean_diffusion_strength": mean("diffusion_strength"),
            "mean_blend_alpha": mean("blend_alpha"),
            "mean_generation_latency_ms": mean("generation_latency_ms"),
            "policy_data_overlap_count": sum(int(item["policy_data_overlap"]) for item in items),
        }
        records.append(record)
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = load_controlnet_policy(args.policy_checkpoint, device)
    text_guard_policy = QualityAdaptivePolicy().to(device).eval()
    policy_ids = set(getattr(policy, "policy_data_image_ids", set()))
    if not policy_ids and not args.allow_policy_data_overlap:
        raise RuntimeError(
            "Policy checkpoint lacks policy_data_image_ids; strict holdout cannot be verified"
        )
    if args.allow_policy_data_overlap:
        excluded_ids = set()
    else:
        excluded_ids = set(policy_ids)
        print("Strict holdout: excluding %d policy-data image ids" % len(excluded_ids))

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
    adapter = _load_adapter(args.adapter_checkpoint, device)
    scorer = CLIPSemanticSimilarity(args.clip_model_path).to(device).eval()
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
        exclude_image_ids=excluded_ids,
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
            probe_kwargs = {key: value for key, value in probe_kwargs.items() if key in supported}
            probe = v61(images, batch["texts"], **probe_kwargs)
            probe_captions = v61.decode_text_logits(probe["text_logits"])
            probe_accuracy = _text_accuracy(probe)
            probe_alignment = scorer.similarity(probe["image"], probe_captions)
            probe_features = receiver_features(probe, snr, image_text_alignment=probe_alignment)
            text_decision = policy(probe_features)
            learned_utility = text_decision["text_utility"].clamp(0.0, 1.0)
            if args.text_safety == "heuristic_max":
                guard_utility = text_guard_policy(probe_features)[
                    "text_utility"
                ].clamp(0.0, 1.0)
                recommended = torch.maximum(learned_utility, guard_utility)
            elif args.text_safety == "probe_floor":
                guard_utility = learned_utility.new_full(
                    learned_utility.shape, args.probe_utility
                )
                recommended = torch.maximum(learned_utility, guard_utility)
            else:
                guard_utility = learned_utility
                recommended = learned_utility
            feedback = (recommended - args.probe_utility).abs() >= args.feedback_deadband
            actual_utility = torch.where(
                feedback,
                recommended,
                recommended.new_full(recommended.shape, args.probe_utility),
            )
            adaptive_kwargs = {
                "snr_db": snr,
                "disable_modality_dropout": True,
                "utility_weight": actual_utility,
                "channel_noise": shared_noise,
                "channel_gain": shared_gain,
            }
            adaptive_kwargs = {key: value for key, value in adaptive_kwargs.items() if key in supported}
            adaptive = v61(images, batch["texts"], **adaptive_kwargs)
            captions = v61.decode_text_logits(adaptive["text_logits"])
            adaptive_accuracy = _text_accuracy(adaptive)
            alignment = scorer.similarity(adaptive["image"], captions)
            route_features = receiver_features(adaptive, snr, image_text_alignment=alignment)
            route = policy(route_features)
            for index, image_id in enumerate(batch["image_ids"]):
                cached.append(
                    {
                        "image_id": int(image_id),
                        "snr": float(snr),
                        "original": images[index].detach().cpu(),
                        "regression": adaptive["image"][index].detach().cpu(),
                        "source_caption": batch["texts"][index],
                        "probe_caption": probe_captions[index],
                        "received_caption": captions[index],
                        "probe_text_accuracy": float(probe_accuracy[index]),
                        "adaptive_text_accuracy": float(adaptive_accuracy[index]),
                        "generation_probability": float(route["generation_probability"][index]),
                        "strength": float(route["diffusion_strength"][index]),
                        "alpha": float(route["blend_alpha"][index]),
                        "detail_probability": float(route["detail_probability"][index]),
                        "feedback": bool(feedback[index]),
                        "learned_text_utility": float(learned_utility[index]),
                        "guarded_text_utility": float(recommended[index]),
                        "text_tokens": float(adaptive["allocated_text_tokens"][index]),
                        "text_repeat": float(adaptive["text_repetition_factor"][index]),
                        "route_features": route_features[index].detach().cpu(),
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
            print("Receiver decisions SNR=%g total=%d" % (snr, len(cached)))

    v61.to("cpu")
    scorer.to("cpu")
    del v61, scorer
    torch.cuda.empty_cache()
    any_generation = any(
        float(item["generation_probability"]) >= args.route_threshold for item in cached
    )
    pipe = None
    if any_generation:
        controlnet = ControlNetModel.from_pretrained(
            args.controlnet_path, torch_dtype=torch.float16, local_files_only=True
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

    rows = []
    for index, item in enumerate(cached):
        original = item["original"].to(device)
        regression = item["regression"].to(device)
        probability = float(item["generation_probability"])
        use_generation = probability >= args.route_threshold
        expert = "detail" if float(item["detail_probability"]) >= 0.5 else "rgb"
        latency_ms = 0.0
        raw = regression
        hard = regression
        soft = regression
        if use_generation:
            semantic_output = {key: value.to(device) for key, value in item["semantic_output"].items()}
            semantic_tokens = adapter(
                semantic_output, item["route_features"].to(device).unsqueeze(0)
            )
            generator = torch.Generator(device=device).manual_seed(
                args.seed * 100000 + index * 10 + 1
            )
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            raw = _generate_controlnet(
                pipe,
                regression,
                str(item["received_caption"]),
                semantic_tokens,
                float(item["strength"]),
                args.steps,
                args.guidance_scale,
                args.control_scale,
                generator,
            )
            torch.cuda.synchronize(device)
            latency_ms = 1000.0 * (time.perf_counter() - started)
            rgb = _fuse(regression, raw, "rgb", float(item["alpha"]))
            detail = _fuse(regression, raw, "detail", float(item["alpha"]))
            hard = detail if expert == "detail" else rgb
            detail_probability = float(item["detail_probability"])
            soft = (
                (1.0 - detail_probability) * rgb
                + detail_probability * detail
            ).clamp(0.0, 1.0)
            selected = soft if args.fusion_selection == "soft" else hard
            mode = "generation"
        else:
            selected = regression
            mode = "regression"
        regression_metrics = batch_image_metrics(regression.unsqueeze(0), original.unsqueeze(0))
        hard_metrics = batch_image_metrics(hard.unsqueeze(0), original.unsqueeze(0))
        soft_metrics = batch_image_metrics(soft.unsqueeze(0), original.unsqueeze(0))
        selected_metrics = batch_image_metrics(selected.unsqueeze(0), original.unsqueeze(0))
        regression_clip = scorer.image_similarity(regression.unsqueeze(0), original.unsqueeze(0))
        hard_clip = scorer.image_similarity(hard.unsqueeze(0), original.unsqueeze(0))
        soft_clip = scorer.image_similarity(soft.unsqueeze(0), original.unsqueeze(0))
        selected_clip = scorer.image_similarity(selected.unsqueeze(0), original.unsqueeze(0))
        row = {
            "image_id": item["image_id"],
            "snr_db": item["snr"],
            "selected_mode": mode,
            "is_generation": int(use_generation),
            "is_detail_expert": int(use_generation and expert == "detail"),
            "generation_probability": probability,
            "diffusion_strength": item["strength"],
            "blend_alpha": item["alpha"],
            "detail_probability": item["detail_probability"],
            "feedback_used": int(item["feedback"]),
            "learned_text_utility": item["learned_text_utility"],
            "guarded_text_utility": item["guarded_text_utility"],
            "allocated_text_tokens": item["text_tokens"],
            "text_repetition_factor": item["text_repeat"],
            "probe_text_accuracy": item["probe_text_accuracy"],
            "adaptive_text_accuracy": item["adaptive_text_accuracy"],
            "probe_bleu2": sentence_bleu2(str(item["source_caption"]), str(item["probe_caption"])),
            "adaptive_bleu2": sentence_bleu2(str(item["source_caption"]), str(item["received_caption"])),
            "regression_psnr": float(regression_metrics["psnr"][0]),
            "hard_psnr": float(hard_metrics["psnr"][0]),
            "soft_psnr": float(soft_metrics["psnr"][0]),
            "selected_psnr": float(selected_metrics["psnr"][0]),
            "regression_ssim": float(regression_metrics["ssim"][0]),
            "hard_ssim": float(hard_metrics["ssim"][0]),
            "soft_ssim": float(soft_metrics["ssim"][0]),
            "selected_ssim": float(selected_metrics["ssim"][0]),
            "regression_clip_image": float(regression_clip[0]),
            "hard_clip_image": float(hard_clip[0]),
            "soft_clip_image": float(soft_clip[0]),
            "selected_clip_image": float(selected_clip[0]),
            "fusion_selection": args.fusion_selection,
            "generation_latency_ms": latency_ms,
            "policy_data_overlap": int(int(item["image_id"]) in policy_ids),
            "received_caption": item["received_caption"],
        }
        stem = "id%s_snr%s" % (item["image_id"], str(item["snr"]).replace(".", "p"))
        _panel(
            original.cpu(), regression.cpu(), raw.cpu(), hard.cpu(), soft.cpu(),
            selected.cpu(), mode, probability, expert, float(item["alpha"]),
            float(item["detail_probability"]), args.fusion_selection,
        ).save(output_dir / (stem + ".png"))
        row["panel"] = stem + ".png"
        rows.append(row)
        print(
            "[%d/%d] %s mode=%s p=%.2f expert=%s alpha=%.2f PSNR=%.2f->%.2f CLIP=%.3f->%.3f"
            % (
                index + 1, len(cached), stem, mode, probability, expert,
                float(item["alpha"]), row["regression_psnr"], row["selected_psnr"],
                row["regression_clip_image"], row["selected_clip_image"],
            )
        )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(rows, output_dir / "summary.csv")
    print("Done: %s" % output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate deployable V6.4 ControlNet policy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--diffusion-model-path", required=True)
    parser.add_argument("--controlnet-path", required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--vae-path")
    parser.add_argument("--text-model-path")
    parser.add_argument("--output-dir", default="experiment_results/v64_holdout")
    parser.add_argument("--config", default="configs/v61_bandwidth.yaml")
    parser.add_argument("--preset", default="original_4608")
    parser.add_argument("--channel-type", default="awgn", choices=["awgn", "rayleigh", "gamma_gamma", "fso"])
    parser.add_argument("--snr-list", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 20])
    parser.add_argument("--probe-utility", type=float, default=0.5)
    parser.add_argument("--feedback-deadband", type=float, default=0.10)
    parser.add_argument("--route-threshold", type=float, default=0.5)
    parser.add_argument("--fusion-selection", choices=["soft", "hard"], default="soft")
    parser.add_argument(
        "--text-safety",
        choices=["heuristic_max", "probe_floor", "none"],
        default="heuristic_max",
    )
    parser.add_argument("--control-scale", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-policy-data-overlap", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
