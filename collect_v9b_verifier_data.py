"""Collect source-supervised, receiver-only V9-B verifier training records."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from diffusers import StableDiffusionImg2ImgPipeline

from models.clip_semantics import CLIPSemanticSimilarity
from utils.metrics import batch_image_metrics
from utils.v65_selective import VERIFIER_FEATURE_NAMES, verification_features
from utils.v9b_selective import PROMPT_FEATURE_NAMES, prompt_quality_features, sanitize_received_caption
from models.quality_adaptive_recovery import FEATURE_NAMES
from models.v9b_common_verifier import MODE_FEATURE_NAMES
from v9_feedback_task_adaptive_eval import collect_receiver_outputs, pil_to_tensor, tensor_to_pil


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    if args.data_split != "train":
        raise ValueError("V9-B verifier data must be collected from --data-split train")
    _, device, cached = collect_receiver_outputs(args)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.diffusion_model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=False)
    scorer = CLIPSemanticSimilarity(args.clip_model_path).to(device).eval()
    records = []
    csv_rows = []
    for index, item in enumerate(cached):
        mode = str(item["payload_mode"])
        if mode == "reconstruction":
            continue
        strength = args.generation_strength if mode == "generation" else args.hybrid_strength
        original = item["original"].to(device)
        regression = item["regression"].to(device)
        received = str(item["received_caption"])
        prompt = sanitize_received_caption(received, max_tokens=args.prompt_max_tokens)
        generator = torch.Generator(device=device).manual_seed(args.seed * 100000 + index)
        generated = pipe(
            prompt=prompt,
            image=tensor_to_pil(regression, size=512),
            strength=float(strength),
            guidance_scale=float(args.guidance_scale),
            num_inference_steps=int(args.steps),
            generator=generator,
        ).images[0]
        generated = generated.resize(
            (original.size(2), original.size(1)),
            resample=3,
        )
        candidate = pil_to_tensor(generated).to(device)

        regression_batch = regression.unsqueeze(0)
        candidate_batch = candidate.unsqueeze(0)
        original_batch = original.unsqueeze(0)
        regression_clip = scorer.image_features(regression_batch)
        candidate_clip = scorer.image_features(candidate_batch)
        prompt_clip = scorer.text_features([prompt], device)
        regression_prompt = (regression_clip * prompt_clip).sum(dim=-1)
        candidate_prompt = (candidate_clip * prompt_clip).sum(dim=-1)
        pair_clip = (candidate_clip * regression_clip).sum(dim=-1)
        evidence = verification_features(
            regression_batch,
            candidate_batch,
            candidate_batch,
            pair_clip,
            pair_clip,
            regression_prompt,
            candidate_prompt,
            strength,
            1.0,
            1.0,
            1.0,
        )
        regression_metrics = batch_image_metrics(regression_batch, original_batch)
        candidate_metrics = batch_image_metrics(candidate_batch, original_batch)
        original_clip = scorer.image_features(original_batch)
        source_clip = scorer.text_features([str(item["source_caption"])], device)
        common_target_clip = torch.nn.functional.normalize(
            original_clip + source_clip, dim=-1
        )
        clip_image_gain = float(
            (candidate_clip * original_clip).sum() - (regression_clip * original_clip).sum()
        )
        clip_source_gain = float(
            (candidate_clip * source_clip).sum() - (regression_clip * source_clip).sum()
        )
        semantic_gain = 0.5 * (clip_image_gain + clip_source_gain)
        delta_psnr = float(candidate_metrics["psnr"][0] - regression_metrics["psnr"][0])
        delta_ssim = float(candidate_metrics["ssim"][0] - regression_metrics["ssim"][0])
        max_psnr_drop = (
            args.generation_max_psnr_drop if mode == "generation" else args.hybrid_max_psnr_drop
        )
        max_ssim_drop = (
            args.generation_max_ssim_drop if mode == "generation" else args.hybrid_max_ssim_drop
        )
        pixel_safe = delta_psnr >= -max_psnr_drop and delta_ssim >= -max_ssim_drop
        accept = semantic_gain >= args.min_semantic_gain and pixel_safe
        hallucination = (semantic_gain < 0.0) or not pixel_safe
        prompt_quality = prompt_quality_features([received], device=device)
        record = {
            "image_id": int(item["image_id"]),
            "snr_db": float(item["forward_snr_db"]),
            "mode": mode,
            "route_features": item["route_features"].float(),
            "verifier_features": evidence[0].detach().cpu().float(),
            "prompt_quality": prompt_quality[0].detach().cpu().float(),
            "mode_features": torch.tensor(
                [float(mode == "generation"), float(mode == "hybrid")],
                dtype=torch.float32,
            ),
            "common_semantic": item["common_receiver_semantic"].float(),
            # Offline CLIP teacher for the Common projector. This field is a
            # training target only and is never read by the online verifier.
            "common_target_clip": common_target_clip[0].detach().cpu().float(),
            "regression_clip": regression_clip[0].detach().cpu().float(),
            "candidate_clip": candidate_clip[0].detach().cpu().float(),
            "prompt_clip": prompt_clip[0].detach().cpu().float(),
            "accept_target": float(accept),
            "hallucination_target": float(hallucination),
            "semantic_gain_target": semantic_gain,
            "delta_psnr_target": delta_psnr,
            "delta_ssim_target": delta_ssim,
        }
        records.append(record)
        csv_rows.append(
            {
                "image_id": record["image_id"],
                "snr_db": record["snr_db"],
                "mode": mode,
                "accept": int(accept),
                "hallucination": int(hallucination),
                "semantic_gain": semantic_gain,
                "clip_image_gain": clip_image_gain,
                "clip_source_gain": clip_source_gain,
                "delta_psnr": delta_psnr,
                "delta_ssim": delta_ssim,
                "received_caption": received,
                "sanitized_prompt": prompt,
            }
        )
        print(
            f"[{index + 1}/{len(cached)}] id={item['image_id']} mode={mode} "
            f"accept={int(accept)} sem={semantic_gain:+.4f} "
            f"dPSNR={delta_psnr:+.3f} dSSIM={delta_ssim:+.4f}",
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_counts = {}
    for record in records:
        key = f"{record['mode']}_accept_{int(record['accept_target'] >= 0.5)}"
        class_counts[key] = class_counts.get(key, 0) + 1
    metadata = {
        "version": "V9-B.2",
        "source_split": args.data_split,
        "source_caption_is_label_only": True,
        "samples_requested": int(args.samples),
        "snr_list": [float(value) for value in args.snr_list],
        "min_semantic_gain": float(args.min_semantic_gain),
        "generation_max_psnr_drop": float(args.generation_max_psnr_drop),
        "hybrid_max_psnr_drop": float(args.hybrid_max_psnr_drop),
        "generation_max_ssim_drop": float(args.generation_max_ssim_drop),
        "hybrid_max_ssim_drop": float(args.hybrid_max_ssim_drop),
        "records": len(records),
        "unique_images": len({int(record["image_id"]) for record in records}),
        "mode_class_counts": class_counts,
    }
    torch.save(
        {
            "records": records,
            "route_feature_names": FEATURE_NAMES,
            "verifier_feature_names": VERIFIER_FEATURE_NAMES,
            "prompt_feature_names": PROMPT_FEATURE_NAMES,
            "mode_feature_names": MODE_FEATURE_NAMES,
            "metadata": metadata,
        },
        output_dir / "v9b_verifier_dataset.pt",
    )
    with (output_dir / "v9b_verifier_labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(
        f"saved {len(records)} records to {output_dir}; "
        f"unique_images={metadata['unique_images']} class_counts={class_counts}",
        flush=True,
    )
    if any(value < 25 for value in class_counts.values()) or len(class_counts) < 4:
        print(
            "WARNING: at least one mode/class group is small. Collect more images "
            "before treating verifier results as research evidence.",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect V9-B verifier data")
    parser.add_argument("--config", default="configs/v9a_feedback_generation.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--text-model-path", required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--diffusion-model-path", required=True)
    parser.add_argument("--preset", default="original_4608")
    parser.add_argument("--snr-list", type=float, nargs="+", default=[-5, 0, 5])
    parser.add_argument("--feedback-snr-db", type=float, default=None)
    parser.add_argument("--ideal-feedback", action="store_true")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--hybrid-strength", type=float, default=0.35)
    parser.add_argument("--generation-strength", type=float, default=0.55)
    parser.add_argument("--prompt-max-tokens", type=int, default=60)
    parser.add_argument("--min-semantic-gain", type=float, default=0.005)
    parser.add_argument("--generation-max-psnr-drop", type=float, default=1.0)
    parser.add_argument("--hybrid-max-psnr-drop", type=float, default=0.75)
    parser.add_argument("--generation-max-ssim-drop", type=float, default=0.10)
    parser.add_argument("--hybrid-max-ssim-drop", type=float, default=0.08)
    parser.add_argument("--skip-diffusion", action="store_false", dest="skip_diffusion")
    parser.set_defaults(skip_diffusion=False)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=342)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-split", choices=("train",), default="train")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
