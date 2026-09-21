"""Collect image-disjoint oracle supervision for the deployable V6.4 policy.

The source image is used only to label the best receiver action.  Every policy
input is computed from received tensors, decoded text and no-reference image
statistics.  Both probe and post-feedback features are stored because text
allocation and final recovery routing occur at different receiver stages.
"""
from __future__ import annotations

import argparse
import csv
import inspect
from pathlib import Path
from typing import Dict, List

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

from data import build_dataloader
from models.clip_semantics import CLIPSemanticSimilarity
from models.quality_adaptive_recovery import FEATURE_NAMES, receiver_features
from utils.checkpoint import load_checkpoint, load_state_dict_flexible
from utils.config import load_config, prepare_config
from utils.experiment import create_model, seed_everything, select_device
from utils.metrics import batch_image_metrics
from v63_controlnet_fusion_pilot import (
    _fuse,
    _generate_controlnet,
    _load_adapter,
    _metric_bundle,
)


def _text_accuracy(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    logits = outputs["text_logits"]
    ids = outputs["text_input_ids"]
    mask = outputs["text_attention_mask"].bool()
    length = min(logits.size(1), ids.size(1), mask.size(1))
    correct = (logits[:, :length].argmax(-1) == ids[:, :length]).float()
    return (correct * mask[:, :length]).sum(1) / mask[:, :length].sum(1).clamp_min(1)


def _regression_utility(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    text_accuracy: torch.Tensor,
) -> torch.Tensor:
    metrics = batch_image_metrics(reconstruction, target)
    psnr_scaled = ((metrics["psnr"].float() - 10.0) / 30.0).clamp(0.0, 1.0)
    return 0.45 * metrics["ssim"].float() + 0.30 * psnr_scaled + 0.25 * text_accuracy


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("V6.4 policy collection requires CUDA")
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
    scorer = CLIPSemanticSimilarity(args.clip_model_path).to(device).eval()
    print("Loaded V6.1 exact tensors=%d" % report["exact_count"])

    data = config["data"]
    loader = build_dataloader(
        image_root=data["train_image_root"],
        annotation_file=data["train_annotation_file"],
        image_size=int(config["model"]["image_size"]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=True,
        caption_mode=data.get("train_caption_mode", "random"),
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
            sweep_outputs = []
            sweep_scores = []
            sweep_features = []
            for utility in args.utility_choices:
                kwargs = {
                    "snr_db": snr,
                    "disable_modality_dropout": True,
                    "utility_weight": utility,
                    "channel_noise": shared_noise,
                    "channel_gain": shared_gain,
                }
                kwargs = {
                    key: value for key, value in kwargs.items() if key in supported
                }
                output = v61(images, batch["texts"], **kwargs)
                captions = v61.decode_text_logits(output["text_logits"])
                accuracy = _text_accuracy(output)
                score = _regression_utility(output["image"], images, accuracy)
                alignment = scorer.similarity(output["image"], captions)
                features = receiver_features(
                    output, snr, image_text_alignment=alignment
                )
                sweep_outputs.append((output, captions))
                sweep_scores.append(score)
                sweep_features.append(features)
            score_matrix = torch.stack(sweep_scores, dim=1)
            best_indices = score_matrix.argmax(dim=1)
            probe_index = min(
                range(len(args.utility_choices)),
                key=lambda index: abs(
                    float(args.utility_choices[index]) - args.probe_utility
                ),
            )
            for item_index, image_id in enumerate(batch["image_ids"]):
                best_index = int(best_indices[item_index])
                best_output, best_captions = sweep_outputs[best_index]
                cached.append(
                    {
                        "image_id": int(image_id),
                        "snr": float(snr),
                        "original": images[item_index].detach().cpu(),
                        "regression": best_output["image"][item_index].detach().cpu(),
                        "received_caption": best_captions[item_index],
                        "probe_features": sweep_features[probe_index][item_index].detach().cpu(),
                        "route_features": sweep_features[best_index][item_index].detach().cpu(),
                        "text_utility_target": float(args.utility_choices[best_index]),
                        "semantic_output": {
                            name: best_output[name][item_index : item_index + 1].detach().cpu()
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
    adapter = _load_adapter(args.adapter_checkpoint, device)
    scorer = CLIPSemanticSimilarity(args.clip_model_path).to(device).eval()

    records = []
    rows = []
    metadata = {
        "version": "V6.4",
        "snr_list": [float(value) for value in args.snr_list],
        "utility_choices": [float(value) for value in args.utility_choices],
        "strengths": [float(value) for value in args.strengths],
        "blend_alphas": [float(value) for value in args.blend_alphas],
        "control_scale": float(args.control_scale),
        "generation_margin": float(args.generation_margin),
        "channel_type": args.channel_type,
        "adapter_checkpoint": str(args.adapter_checkpoint),
        "controlnet_path": str(args.controlnet_path),
        "image_ids": sorted({int(item["image_id"]) for item in cached}),
    }
    for index, item in enumerate(cached):
        original = item["original"].to(device)
        regression = item["regression"].to(device)
        prompt = str(item["received_caption"])
        semantic_output = {
            key: value.to(device)
            for key, value in item["semantic_output"].items()
        }
        semantic_tokens = adapter(
            semantic_output, item["route_features"].to(device).unsqueeze(0)
        )
        regression_metrics = _metric_bundle(
            scorer, regression, original, prompt
        )
        best_metrics = None
        best_strength = float(args.strengths[0])
        best_alpha = float(args.blend_alphas[0])
        best_expert = "rgb"
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
            for expert in ("rgb", "detail"):
                for alpha in args.blend_alphas:
                    candidate = _fuse(regression, generated, expert, alpha)
                    metrics = _metric_bundle(scorer, candidate, original, prompt)
                    if best_metrics is None or metrics["utility"] > best_metrics["utility"]:
                        best_metrics = metrics
                        best_strength = float(strength)
                        best_alpha = float(alpha)
                        best_expert = expert
        generation_target = float(
            best_metrics["utility"]
            > regression_metrics["utility"] + args.generation_margin
        )
        records.append(
            {
                "probe_features": item["probe_features"],
                "route_features": item["route_features"],
                "generation_target": generation_target,
                "text_utility_target": item["text_utility_target"],
                "strength_target": best_strength,
                "blend_alpha_target": best_alpha,
                "detail_target": float(best_expert == "detail"),
                "regression_quality_target": max(
                    0.0, min(1.0, regression_metrics["utility"])
                ),
                "image_id": item["image_id"],
                "snr": item["snr"],
            }
        )
        rows.append(
            {
                "image_id": item["image_id"],
                "snr_db": item["snr"],
                "generation_target": int(generation_target),
                "text_utility_target": item["text_utility_target"],
                "best_strength": best_strength,
                "best_alpha": best_alpha,
                "best_expert": best_expert,
                "regression_utility": regression_metrics["utility"],
                "best_generation_utility": best_metrics["utility"],
                "regression_psnr": regression_metrics["psnr"],
                "generation_psnr": best_metrics["psnr"],
                "regression_ssim": regression_metrics["ssim"],
                "generation_ssim": best_metrics["ssim"],
                "regression_clip_image": regression_metrics["clip_image"],
                "generation_clip_image": best_metrics["clip_image"],
                "received_caption": prompt,
            }
        )
        print(
            "[%d/%d] id=%s snr=%g gen=%d expert=%s alpha=%.2f strength=%.2f utility=%.4f/%.4f"
            % (
                index + 1,
                len(cached),
                item["image_id"],
                item["snr"],
                int(generation_target),
                best_expert,
                best_alpha,
                best_strength,
                regression_metrics["utility"],
                best_metrics["utility"],
            )
        )
        if (index + 1) % args.save_interval == 0:
            torch.save(
                {"records": records, "feature_names": FEATURE_NAMES, "metadata": metadata},
                output_dir / "policy_dataset.partial.pth",
            )

    torch.save(
        {"records": records, "feature_names": FEATURE_NAMES, "metadata": metadata},
        output_dir / "policy_dataset.pth",
    )
    with (output_dir / "oracle_labels.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Done: %s (%d records)" % (output_dir / "policy_dataset.pth", len(records)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect V6.4 ControlNet policy labels")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--diffusion-model-path", required=True)
    parser.add_argument("--controlnet-path", required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--vae-path")
    parser.add_argument("--text-model-path")
    parser.add_argument("--output-dir", default="experiment_results/v64_policy_data")
    parser.add_argument("--config", default="configs/v61_bandwidth.yaml")
    parser.add_argument("--preset", default="original_4608")
    parser.add_argument("--channel-type", default="awgn", choices=["awgn", "rayleigh", "gamma_gamma", "fso"])
    parser.add_argument("--snr-list", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 20])
    parser.add_argument("--utility-choices", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--probe-utility", type=float, default=0.5)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.25, 0.45, 0.65])
    parser.add_argument("--blend-alphas", type=float, nargs="+", default=[0.15, 0.25, 0.40])
    parser.add_argument("--control-scale", type=float, default=0.8)
    parser.add_argument("--generation-margin", type=float, default=0.002)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
