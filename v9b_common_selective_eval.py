"""Final V9-B evaluation: forced, selective, oracle and no-Common decisions."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image, ImageDraw, ImageFont

from models.clip_semantics import CLIPSemanticSimilarity
from models.v9b_common_verifier import load_v9b_verifier
from utils.metrics import batch_image_metrics
from utils.v65_selective import verification_features
from utils.v9b_selective import (
    pre_generation_reliability,
    prompt_quality_features,
    sanitize_received_caption,
)
from v9_feedback_task_adaptive_eval import collect_receiver_outputs, pil_to_tensor, tensor_to_pil


def _tile(image: Image.Image, title: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 34), "white")
    canvas.paste(image, (0, 34))
    ImageDraw.Draw(canvas).text((5, 9), title, fill="black", font=ImageFont.load_default())
    return canvas


def save_panel(original, regression, candidate, selected, title, path):
    tiles = [
        _tile(tensor_to_pil(original), "Original (evaluation only)"),
        _tile(tensor_to_pil(regression), "Channel regression"),
        _tile(tensor_to_pil(candidate), "Diffusion candidate"),
        _tile(tensor_to_pil(selected), title),
    ]
    canvas = Image.new("RGB", (sum(tile.width for tile in tiles), tiles[0].height), "white")
    offset = 0
    for tile in tiles:
        canvas.paste(tile, (offset, 0))
        offset += tile.width
    canvas.save(path)


def _mean(rows, name):
    values = [float(row[name]) for row in rows]
    return sum(values) / max(len(values), 1)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(str(row["forward_snr_db"]), str(row["payload_mode"]))].append(row)
    result = {}
    for (snr, mode), items in groups.items():
        entry = {
            "samples": len(items),
            "diffusion_fraction": _mean(items, "diffusion_used"),
            "pre_gate_pass_rate": _mean(items, "pre_gate_pass"),
            "verifier_accept_rate": _mean(items, "verifier_accepted"),
            "no_common_accept_rate": _mean(items, "no_common_accepted"),
            "oracle_accept_rate": _mean(items, "oracle_accepted"),
            "forced_semantic_gain": _mean(items, "forced_semantic_gain"),
            "selective_semantic_gain": _mean(items, "selective_semantic_gain"),
            "no_common_semantic_gain": _mean(items, "no_common_semantic_gain"),
            "oracle_semantic_gain": _mean(items, "oracle_semantic_gain"),
            "forced_clip_image_gain": _mean(items, "forced_clip_image_gain"),
            "selective_clip_image_gain": _mean(items, "selective_clip_image_gain"),
            "forced_clip_source_gain": _mean(items, "forced_clip_source_gain"),
            "selective_clip_source_gain": _mean(items, "selective_clip_source_gain"),
            "forced_clip_received_gain": _mean(items, "forced_clip_received_gain"),
            "selective_clip_received_gain": _mean(items, "selective_clip_received_gain"),
            "forced_psnr_delta": _mean(items, "forced_psnr_delta"),
            "selective_psnr_delta": _mean(items, "selective_psnr_delta"),
            "forced_ssim_delta": _mean(items, "forced_ssim_delta"),
            "selective_ssim_delta": _mean(items, "selective_ssim_delta"),
            "feedback_error": _mean(items, "feedback_error"),
            "feedback_rejected": _mean(items, "feedback_rejected"),
            "common_candidate_advantage": _mean(items, "common_candidate_advantage"),
            "accepted_common_candidate_advantage": _mean(
                items, "accepted_common_candidate_advantage"
            ),
        }
        if mode != "reconstruction":
            tp = sum(int(row["verifier_accepted"]) and int(row["oracle_accepted"]) for row in items)
            fp = sum(int(row["verifier_accepted"]) and not int(row["oracle_accepted"]) for row in items)
            fn = sum(not int(row["verifier_accepted"]) and int(row["oracle_accepted"]) for row in items)
            tn = len(items) - tp - fp - fn
            entry.update(
                {
                    "verifier_precision": tp / max(tp + fp, 1),
                    "verifier_recall": tp / max(tp + fn, 1),
                    "verifier_false_positive_rate": fp / max(fp + tn, 1),
                }
            )
            no_tp = sum(int(row["no_common_accepted"]) and int(row["oracle_accepted"]) for row in items)
            no_fp = sum(int(row["no_common_accepted"]) and not int(row["oracle_accepted"]) for row in items)
            no_fn = sum(not int(row["no_common_accepted"]) and int(row["oracle_accepted"]) for row in items)
            no_tn = len(items) - no_tp - no_fp - no_fn
            no_precision = no_tp / max(no_tp + no_fp, 1)
            no_recall = no_tp / max(no_tp + no_fn, 1)
            entry.update(
                {
                    "no_common_precision": no_precision,
                    "no_common_recall": no_recall,
                    "no_common_f1": 2 * no_precision * no_recall
                    / max(no_precision + no_recall, 1e-12),
                    "no_common_false_positive_rate": no_fp / max(no_fp + no_tn, 1),
                }
            )
        result.setdefault(snr, {})[mode] = entry
    return result


@torch.no_grad()
def main(args):
    if args.data_split != "val":
        raise ValueError("Final V9-B evaluation must use --data-split val")
    _, device, cached = collect_receiver_outputs(args)
    verifier, checkpoint = load_v9b_verifier(args.verifier_checkpoint, device)
    current_ids = {int(item["image_id"]) for item in cached}
    trained_ids = {int(value) for value in checkpoint.get("all_training_image_ids", [])}
    overlap = current_ids & trained_ids
    if overlap and not args.allow_image_overlap:
        raise RuntimeError(
            f"Verifier/evaluation image leakage detected ({len(overlap)} ids). "
            "Collect verifier data with --data-split train."
        )
    checkpoint_thresholds = checkpoint.get("acceptance_thresholds")
    if not isinstance(checkpoint_thresholds, dict):
        raise RuntimeError(
            "V9-B.1 requires per-mode thresholds. Recollect data and retrain the verifier."
        )
    acceptance_thresholds = {
        mode: (
            float(args.acceptance_threshold)
            if args.acceptance_threshold is not None
            else float(checkpoint_thresholds[mode])
        )
        for mode in ("generation", "hybrid")
    }
    no_common_checkpoint_thresholds = checkpoint.get(
        "no_common_acceptance_thresholds", checkpoint_thresholds
    )
    no_common_acceptance_thresholds = {
        mode: float(no_common_checkpoint_thresholds[mode])
        for mode in ("generation", "hybrid")
    }
    decision_score_alphas = {
        mode: float(checkpoint.get("decision_score_alphas", {}).get(mode, 0.0))
        for mode in ("generation", "hybrid")
    }
    no_common_decision_score_alphas = {
        mode: float(
            checkpoint.get("no_common_decision_score_alphas", {}).get(mode, 0.0)
        )
        for mode in ("generation", "hybrid")
    }
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.diffusion_model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=False)
    scorer = CLIPSemanticSimilarity(args.clip_model_path).to(device).eval()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, item in enumerate(cached):
        mode = str(item["payload_mode"])
        original = item["original"].to(device)
        regression = item["regression"].to(device)
        received = str(item["received_caption"])
        prompt = sanitize_received_caption(received, max_tokens=args.prompt_max_tokens)
        route = item["route_features"].to(device).unsqueeze(0)
        prompt_quality = prompt_quality_features([received], device=device)
        pre_score = float(pre_generation_reliability(route, prompt_quality)[0])
        pre_pass = pre_score >= args.pre_gate_threshold
        diffusion_used = mode in {"hybrid", "generation"}
        strength = args.generation_strength if mode == "generation" else args.hybrid_strength
        if diffusion_used:
            generator = torch.Generator(device=device).manual_seed(args.seed * 100000 + index)
            generated = pipe(
                prompt=prompt,
                image=tensor_to_pil(regression, size=512),
                strength=float(strength),
                guidance_scale=float(args.guidance_scale),
                num_inference_steps=int(args.steps),
                generator=generator,
            ).images[0]
            generated = generated.resize((original.size(2), original.size(1)), Image.Resampling.BICUBIC)
            candidate = pil_to_tensor(generated).to(device)
        else:
            strength = 0.0
            candidate = regression

        regression_batch = regression.unsqueeze(0)
        candidate_batch = candidate.unsqueeze(0)
        original_batch = original.unsqueeze(0)
        regression_clip = scorer.image_features(regression_batch)
        candidate_clip = scorer.image_features(candidate_batch)
        original_clip = scorer.image_features(original_batch)
        prompt_clip = scorer.text_features([prompt], device)
        source_clip = scorer.text_features([str(item["source_caption"])], device)
        regression_prompt = (regression_clip * prompt_clip).sum(dim=-1)
        candidate_prompt = (candidate_clip * prompt_clip).sum(dim=-1)
        pair_clip = (candidate_clip * regression_clip).sum(dim=-1)
        evidence = verification_features(
            regression_batch, candidate_batch, candidate_batch,
            pair_clip, pair_clip, regression_prompt, candidate_prompt,
            strength, 1.0, 1.0, 1.0,
        )
        common = item["common_receiver_semantic"].to(device).unsqueeze(0)
        mode_features = torch.tensor(
            [[float(mode == "generation"), float(mode == "hybrid")]],
            device=device,
        )
        prediction = verifier(
            route, evidence, prompt_quality, mode_features, common,
            regression_clip, candidate_clip, prompt_clip,
        )
        no_common_prediction = verifier(
            route, evidence, prompt_quality, mode_features, common,
            regression_clip, candidate_clip, prompt_clip,
            common_ablation=True,
        )
        mode_threshold = acceptance_thresholds.get(mode, 1.0)
        no_common_mode_threshold = no_common_acceptance_thresholds.get(mode, 1.0)
        decision_score = float(prediction["acceptance_probability"][0]) + (
            decision_score_alphas.get(mode, 0.0)
            * float(prediction["predicted_utility"][0])
        )
        no_common_decision_score = float(
            no_common_prediction["acceptance_probability"][0]
        ) + (
            no_common_decision_score_alphas.get(mode, 0.0)
            * float(no_common_prediction["predicted_utility"][0])
        )
        pre_condition = pre_pass if args.enforce_pre_gate else True
        accepted = bool(
            diffusion_used and pre_condition
            and decision_score >= mode_threshold
            and float(prediction["hallucination_probability"][0]) <= args.max_hallucination_probability
        )
        no_common_accepted = bool(
            diffusion_used and pre_condition
            and no_common_decision_score >= no_common_mode_threshold
            and float(no_common_prediction["hallucination_probability"][0]) <= args.max_hallucination_probability
        )
        selected = candidate if accepted else regression

        regression_metrics = batch_image_metrics(regression_batch, original_batch)
        candidate_metrics = batch_image_metrics(candidate_batch, original_batch)
        selected_metrics = batch_image_metrics(selected.unsqueeze(0), original_batch)
        forced_clip_image_gain = float(
            (candidate_clip * original_clip).sum() - (regression_clip * original_clip).sum()
        )
        forced_clip_source_gain = float(
            (candidate_clip * source_clip).sum() - (regression_clip * source_clip).sum()
        )
        forced_clip_received_gain = float(candidate_prompt.sum() - regression_prompt.sum())
        forced_semantic_gain = 0.5 * (forced_clip_image_gain + forced_clip_source_gain)
        forced_psnr_delta = float(candidate_metrics["psnr"][0] - regression_metrics["psnr"][0])
        forced_ssim_delta = float(candidate_metrics["ssim"][0] - regression_metrics["ssim"][0])
        max_psnr_drop = args.generation_max_psnr_drop if mode == "generation" else args.hybrid_max_psnr_drop
        max_ssim_drop = args.generation_max_ssim_drop if mode == "generation" else args.hybrid_max_ssim_drop
        oracle_accepted = bool(
            diffusion_used and forced_semantic_gain >= args.min_semantic_gain
            and forced_psnr_delta >= -max_psnr_drop and forced_ssim_delta >= -max_ssim_drop
        )
        stem = f"id{item['image_id']}_snr{str(item['forward_snr_db']).replace('-', 'm').replace('.', 'p')}_{mode}"
        panel_path = output_dir / f"{stem}.png"
        save_panel(
            original.cpu(), regression.cpu(), candidate.cpu(), selected.cpu(),
            f"V9-B {'ACCEPT' if accepted else 'REJECT'} p={float(prediction['acceptance_probability'][0]):.2f}",
            panel_path,
        )
        rows.append(
            {
                "image_id": int(item["image_id"]),
                "forward_snr_db": float(item["forward_snr_db"]),
                "payload_mode": mode,
                "feedback_error": int(item["feedback_error"]),
                "feedback_rejected": int(item["feedback_rejected"]),
                "received_caption": received,
                "sanitized_prompt": prompt,
                "diffusion_used": int(diffusion_used),
                "pre_gate_score": pre_score,
                "pre_gate_pass": int(pre_pass),
                "acceptance_probability": float(prediction["acceptance_probability"][0]),
                "no_common_acceptance_probability": float(
                    no_common_prediction["acceptance_probability"][0]
                ),
                "acceptance_threshold": float(mode_threshold),
                "no_common_acceptance_threshold": float(no_common_mode_threshold),
                "decision_score": decision_score,
                "decision_score_utility_alpha": decision_score_alphas.get(mode, 0.0),
                "no_common_decision_score": no_common_decision_score,
                "no_common_decision_score_utility_alpha": no_common_decision_score_alphas.get(mode, 0.0),
                "hallucination_probability": float(prediction["hallucination_probability"][0]),
                "predicted_semantic_gain": float(prediction["predicted_delta_semantic"][0]),
                "predicted_decision_utility": float(prediction["predicted_utility"][0]),
                "common_regression_similarity": float(prediction["common_regression_similarity"][0]),
                "common_candidate_similarity": float(prediction["common_candidate_similarity"][0]),
                "common_prompt_similarity": float(prediction["common_prompt_similarity"][0]),
                "common_candidate_advantage": float(
                    prediction["common_candidate_similarity"][0]
                    - prediction["common_regression_similarity"][0]
                ),
                "accepted_common_candidate_advantage": (
                    float(
                        prediction["common_candidate_similarity"][0]
                        - prediction["common_regression_similarity"][0]
                    )
                    if accepted else 0.0
                ),
                "verifier_accepted": int(accepted),
                "no_common_accepted": int(no_common_accepted),
                "oracle_accepted": int(oracle_accepted),
                "forced_semantic_gain": forced_semantic_gain if diffusion_used else 0.0,
                "selective_semantic_gain": forced_semantic_gain if accepted else 0.0,
                "no_common_semantic_gain": forced_semantic_gain if no_common_accepted else 0.0,
                "oracle_semantic_gain": forced_semantic_gain if oracle_accepted else 0.0,
                "forced_clip_image_gain": forced_clip_image_gain if diffusion_used else 0.0,
                "selective_clip_image_gain": forced_clip_image_gain if accepted else 0.0,
                "forced_clip_source_gain": forced_clip_source_gain if diffusion_used else 0.0,
                "selective_clip_source_gain": forced_clip_source_gain if accepted else 0.0,
                "forced_clip_received_gain": forced_clip_received_gain if diffusion_used else 0.0,
                "selective_clip_received_gain": forced_clip_received_gain if accepted else 0.0,
                "forced_psnr_delta": forced_psnr_delta if diffusion_used else 0.0,
                "selective_psnr_delta": float(selected_metrics["psnr"][0] - regression_metrics["psnr"][0]),
                "forced_ssim_delta": forced_ssim_delta if diffusion_used else 0.0,
                "selective_ssim_delta": float(selected_metrics["ssim"][0] - regression_metrics["ssim"][0]),
                "panel": str(panel_path),
            }
        )
        print(
            f"[{index+1}/{len(cached)}] id={item['image_id']} snr={item['forward_snr_db']:g} "
            f"mode={mode} pre={pre_score:.3f} p={float(prediction['acceptance_probability'][0]):.3f} "
            f"select={'G' if accepted else 'R'} oracle={'G' if oracle_accepted else 'R'} "
            f"sem={forced_semantic_gain:+.4f}", flush=True,
        )

    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "version": "V9-B.2 utility-calibrated mode-expert selective generation",
        "receiver_uses_source_caption": False,
        "source_caption_is_evaluation_only": True,
        "acceptance_thresholds": acceptance_thresholds,
        "no_common_acceptance_thresholds": no_common_acceptance_thresholds,
        "decision_score_alphas": decision_score_alphas,
        "no_common_decision_score_alphas": no_common_decision_score_alphas,
        "evaluation_unique_images": len(current_ids),
        "pre_gate_threshold": float(args.pre_gate_threshold),
        "pre_gate_enforced": bool(args.enforce_pre_gate),
        "max_hallucination_probability": float(args.max_hallucination_probability),
        "verifier_checkpoint_metrics": checkpoint.get("metrics", {}),
        "verifier_checkpoint_no_common_metrics": checkpoint.get("no_common_metrics", {}),
        "summary": summarize(rows),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"saved V9-B evaluation to {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate V9-B Common-guided selection")
    parser.add_argument("--config", default="configs/v9a_feedback_generation.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--verifier-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--text-model-path", required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--diffusion-model-path", required=True)
    parser.add_argument("--preset", default="original_4608")
    parser.add_argument("--snr-list", type=float, nargs="+", default=[-5, 0, 5, 10, 20])
    parser.add_argument("--feedback-snr-db", type=float, default=None)
    parser.add_argument("--ideal-feedback", action="store_true")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--hybrid-strength", type=float, default=0.35)
    parser.add_argument("--generation-strength", type=float, default=0.55)
    parser.add_argument("--prompt-max-tokens", type=int, default=60)
    parser.add_argument("--pre-gate-threshold", type=float, default=0.20)
    parser.add_argument(
        "--enforce-pre-gate",
        action="store_true",
        help="Disabled by default: V9-B pre-gate was not discriminative.",
    )
    parser.add_argument("--acceptance-threshold", type=float, default=None)
    parser.add_argument(
        "--max-hallucination-probability",
        type=float,
        default=1.0,
        help="Auxiliary risk is reported; default leaves the calibrated acceptance head in control.",
    )
    parser.add_argument("--min-semantic-gain", type=float, default=0.005)
    parser.add_argument("--generation-max-psnr-drop", type=float, default=1.0)
    parser.add_argument("--hybrid-max-psnr-drop", type=float, default=0.75)
    parser.add_argument("--generation-max-ssim-drop", type=float, default=0.10)
    parser.add_argument("--hybrid-max-ssim-drop", type=float, default=0.08)
    parser.add_argument("--allow-image-overlap", action="store_true")
    parser.add_argument("--skip-diffusion", action="store_false", dest="skip_diffusion")
    parser.set_defaults(skip_diffusion=False)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-split", choices=("val",), default="val")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
