"""Collect oracle labels for the V6.2 quality/bandwidth policy.

For every source/SNR pair this script:
  1. probes several text-token allocations under the same AWGN realization;
  2. records no-reference features from the neutral allocation;
  3. generates several diffusion candidates from the best regression result;
  4. uses the original image only to label the best allocation/mode/strength.

The resulting ``policy_dataset.pth`` contains feature vectors and scalar
targets, never source pixels.  It is therefore safe to use for receiver-policy
training without leaking the reference image into inference.
"""
from __future__ import annotations

import argparse
import csv
import inspect
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import torch
from diffusers import StableDiffusionImg2ImgPipeline

from data import build_dataloader
from models.clip_semantics import CLIPSemanticSimilarity
from models.quality_adaptive_recovery import (
    FEATURE_NAMES,
    SemanticConditionAdapter,
    receiver_features,
)
from utils.checkpoint import load_checkpoint, load_state_dict_flexible
from utils.config import load_config, prepare_config
from utils.experiment import create_model, seed_everything, select_device
from utils.metrics import batch_image_metrics
from utils.v62_diffusion import generate_img2img


def _text_accuracy_per_sample(outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    logits = outputs["text_logits"]
    ids = outputs["text_input_ids"]
    mask = outputs["text_attention_mask"].bool()
    length = min(logits.size(1), ids.size(1), mask.size(1))
    correct = (logits[:, :length].argmax(-1) == ids[:, :length]).float()
    return (correct * mask[:, :length]).sum(1) / mask[:, :length].sum(1).clamp_min(1)


def _fidelity_values(reconstruction: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    metrics = batch_image_metrics(reconstruction, target)
    psnr = metrics["psnr"].float()
    ssim = metrics["ssim"].float()
    return psnr, ssim


def _regression_utility(reconstruction: torch.Tensor, target: torch.Tensor, text_accuracy: torch.Tensor) -> torch.Tensor:
    psnr, ssim = _fidelity_values(reconstruction, target)
    psnr_scaled = ((psnr - 10.0) / 30.0).clamp(0.0, 1.0)
    return 0.45 * ssim + 0.30 * psnr_scaled + 0.25 * text_accuracy


def _semantic_utility(
    scorer: CLIPSemanticSimilarity,
    candidate: torch.Tensor,
    target: torch.Tensor,
    received_caption: str,
) -> Tuple[float, Dict[str, float]]:
    candidate = candidate.to(target.device).unsqueeze(0)
    target = target.unsqueeze(0)
    psnr, ssim = _fidelity_values(candidate, target)
    clip_image = scorer.image_similarity(candidate, target)
    clip_text = scorer.similarity(candidate, [received_caption or "a natural photograph"])
    psnr_scaled = ((psnr - 10.0) / 30.0).clamp(0.0, 1.0)
    score = 0.25 * ssim + 0.15 * psnr_scaled + 0.40 * (
        0.5 * (clip_image + 1.0)
    ) + 0.20 * (0.5 * (clip_text + 1.0))
    details = {
        "psnr": float(psnr[0]),
        "ssim": float(ssim[0]),
        "clip_image": float(clip_image[0]),
        "clip_text": float(clip_text[0]),
    }
    return float(score[0]), details


def _load_adapter(path: Optional[str], device: torch.device) -> Optional[SemanticConditionAdapter]:
    if not path:
        return None
    checkpoint = torch.load(path, map_location=device)
    adapter = SemanticConditionAdapter(**checkpoint["adapter_config"]).to(device)
    adapter.load_state_dict(checkpoint["adapter"])
    adapter.eval()
    print("Loaded semantic adapter: %s" % path)
    return adapter


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Policy collection with diffusion requires CUDA")
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
    print("Loaded V6.1 exact tensors=%d" % report["exact_count"])
    scorer = CLIPSemanticSimilarity(args.clip_model_path).to(device).eval()
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
    cached: List[Dict[str, object]] = []
    total_tokens = int(v61.num_common_tokens + v61.channel_token_budget)
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
                kwargs = {key: value for key, value in kwargs.items() if key in supported}
                outputs = v61(images, batch["texts"], **kwargs)
                score = _regression_utility(
                    outputs["image"], images, _text_accuracy_per_sample(outputs)
                )
                decoded_captions = v61.decode_text_logits(outputs["text_logits"])
                received_alignment = scorer.similarity(
                    outputs["image"], decoded_captions
                )
                sweep_outputs.append(outputs)
                sweep_scores.append(score)
                sweep_features.append(
                    receiver_features(
                        outputs,
                        snr,
                        image_text_alignment=received_alignment,
                    )
                )
            score_matrix = torch.stack(sweep_scores, dim=1)
            best_indices = score_matrix.argmax(dim=1)
            neutral_index = min(
                range(len(args.utility_choices)),
                key=lambda index: abs(args.utility_choices[index] - args.probe_utility),
            )
            probe_features = sweep_features[neutral_index]
            for item_index, image_id in enumerate(batch["image_ids"]):
                best_index = int(best_indices[item_index])
                best_output = sweep_outputs[best_index]
                received_caption = v61.decode_text_logits(
                    best_output["text_logits"][item_index : item_index + 1]
                )[0]
                semantic_output = {
                    name: best_output[name][item_index : item_index + 1].detach().cpu()
                    for name in (
                        "decoded_common_tokens",
                        "decoded_image_tokens",
                        "decoded_text_tokens",
                    )
                }
                cached.append(
                    {
                        "image_id": int(image_id),
                        "snr": float(snr),
                        "original": images[item_index].detach().cpu(),
                        "regression": best_output["image"][item_index].detach().cpu(),
                        "received_caption": received_caption,
                        "source_caption": batch["texts"][item_index],
                        "features": probe_features[item_index].detach().cpu(),
                        "best_features": sweep_features[best_index][item_index].detach().cpu(),
                        "text_utility_target": float(args.utility_choices[best_index]),
                        "semantic_output": semantic_output,
                    }
                )
            print("V6.1 cached SNR=%g samples=%d" % (snr, len(cached)))
    v61.to("cpu")
    del v61
    torch.cuda.empty_cache()

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.diffusion_model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=False)
    adapter = _load_adapter(args.adapter_checkpoint, device)
    records = []
    rows = []
    dataset_metadata = {
        "snr_list": [float(value) for value in args.snr_list],
        "utility_choices": [float(value) for value in args.utility_choices],
        "strengths": [float(value) for value in args.strengths],
        "probe_utility": float(args.probe_utility),
        "generation_margin": float(args.generation_margin),
        "channel_type": str(args.channel_type),
        "seed": int(args.seed),
        "adapter_checkpoint": str(args.adapter_checkpoint or "prompt_only"),
        "image_ids": sorted({int(item["image_id"]) for item in cached}),
    }
    for index, item in enumerate(cached):
        original = item["original"].to(device)
        regression = item["regression"].to(device)
        prompt = str(item["received_caption"])
        semantic_tokens = None
        if adapter is not None:
            semantic_output = {
                key: value.to(device) for key, value in item["semantic_output"].items()
            }
            semantic_tokens = adapter(
                semantic_output, item["best_features"].to(device).unsqueeze(0)
            )
        reg_score, reg_details = _semantic_utility(
            scorer, regression, original, prompt
        )
        best_score = -1.0
        best_strength = float(args.strengths[0])
        best_details = None
        for strength in args.strengths:
            generator = torch.Generator(device=device).manual_seed(
                args.seed * 100000 + index * 100 + int(round(strength * 100))
            )
            generated = generate_img2img(
                pipe,
                regression,
                prompt,
                strength,
                args.steps,
                args.guidance_scale,
                generator,
                output_height=int(original.size(1)),
                output_width=int(original.size(2)),
                semantic_tokens=semantic_tokens,
            ).to(device)
            score, details = _semantic_utility(scorer, generated, original, prompt)
            if score > best_score:
                best_score = score
                best_strength = float(strength)
                best_details = details
        generation_target = float(best_score > reg_score + args.generation_margin)
        regression_quality = max(0.0, min(1.0, reg_score))
        records.append(
            {
                "features": item["features"],
                "generation_target": generation_target,
                "text_utility_target": item["text_utility_target"],
                "strength_target": best_strength,
                "regression_quality_target": regression_quality,
                "image_id": item["image_id"],
                "snr": item["snr"],
            }
        )
        rows.append(
            {
                "image_id": item["image_id"],
                "snr_db": item["snr"],
                "text_utility_target": item["text_utility_target"],
                "generation_target": int(generation_target),
                "best_strength": best_strength,
                "regression_utility": reg_score,
                "best_generation_utility": best_score,
                "regression_psnr": reg_details["psnr"],
                "generation_psnr": best_details["psnr"],
                "regression_ssim": reg_details["ssim"],
                "generation_ssim": best_details["ssim"],
                "regression_clip_image": reg_details["clip_image"],
                "generation_clip_image": best_details["clip_image"],
                "received_caption": prompt,
            }
        )
        print(
            "[%d/%d] id=%s snr=%g text=%.2f gen=%d utility=%.3f/%.3f"
            % (
                index + 1, len(cached), item["image_id"], item["snr"],
                item["text_utility_target"], int(generation_target), reg_score, best_score,
            )
        )
        if (index + 1) % args.save_interval == 0:
            torch.save(
                {
                    "records": records,
                    "feature_names": FEATURE_NAMES,
                    "metadata": dataset_metadata,
                },
                output_dir / "policy_dataset.partial.pth",
            )
    torch.save(
        {
            "records": records,
            "feature_names": FEATURE_NAMES,
            "metadata": dataset_metadata,
        },
        output_dir / "policy_dataset.pth",
    )
    with (output_dir / "oracle_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Done: %s (%d records)" % (output_dir / "policy_dataset.pth", len(records)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect V6.2 adaptive-policy supervision")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--diffusion-model-path", required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--adapter-checkpoint")
    parser.add_argument("--output-dir", default="experiment_results/v62_policy_data")
    parser.add_argument("--config", default="configs/v61_bandwidth.yaml")
    parser.add_argument("--vae-path")
    parser.add_argument("--text-model-path")
    parser.add_argument("--preset", default="original_4608")
    parser.add_argument("--channel-type", default="awgn", choices=["awgn", "rayleigh", "gamma_gamma", "fso"])
    parser.add_argument("--snr-list", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 20])
    parser.add_argument("--utility-choices", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--probe-utility", type=float, default=0.5)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.10, 0.18, 0.26, 0.34])
    parser.add_argument("--generation-margin", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
