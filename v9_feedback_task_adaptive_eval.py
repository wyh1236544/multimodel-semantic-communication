from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import StableDiffusionImg2ImgPipeline

from data import build_dataloader
from models.clip_semantics import CLIPSemanticSimilarity
from models.feedback_adaptive import (
    PAYLOAD_MODES,
    PilotFeedbackController,
    mode_names,
)
from models.quality_adaptive_recovery import receiver_features
from utils.checkpoint import load_checkpoint, load_state_dict_flexible
from utils.config import load_config, prepare_config
from utils.experiment import create_model, seed_everything, select_device
from utils.metrics import batch_image_metrics, sentence_bleu2


MODE_TEXT_QUOTAS = {
    "reconstruction": 32,
    "hybrid": 128,
    "generation": 192,
}


def tensor_to_pil(image: torch.Tensor, size: int | None = None) -> Image.Image:
    array = (
        image.detach()
        .float()
        .clamp(0, 1)
        .cpu()
        .permute(1, 2, 0)
        .numpy()
        * 255.0
    ).round().astype(np.uint8)
    result = Image.fromarray(array, "RGB")
    if size is not None:
        result = result.resize((size, size), Image.Resampling.BICUBIC)
    return result


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def _tile(image: Image.Image, title: str) -> Image.Image:
    header = 34
    canvas = Image.new("RGB", (image.width, image.height + header), "white")
    canvas.paste(image, (0, header))
    ImageDraw.Draw(canvas).text(
        (5, 9), title, fill="black", font=ImageFont.load_default()
    )
    return canvas


def save_panel(
    original: torch.Tensor,
    regression: torch.Tensor,
    selected: torch.Tensor,
    mode: str,
    path: Path,
) -> None:
    error = (selected.float() - original.float()).abs().clamp(0, 0.25) / 0.25
    tiles = [
        _tile(tensor_to_pil(original), "Original"),
        _tile(tensor_to_pil(regression), "Channel regression"),
        _tile(tensor_to_pil(selected), f"Selected: {mode}"),
        _tile(tensor_to_pil(error), "|Selected-original|"),
    ]
    canvas = Image.new(
        "RGB",
        (sum(tile.width for tile in tiles), tiles[0].height),
        "white",
    )
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0))
        x += tile.width
    canvas.save(path)


def _feedback_controller(config: Dict[str, Any]) -> PilotFeedbackController:
    feedback = config.get("feedback", {})
    return PilotFeedbackController(
        pilot_length=int(feedback.get("pilot_length", 32)),
        feedback_code_length=int(feedback.get("code_length", 16)),
        feedback_confidence_threshold=float(
            feedback.get("confidence_threshold", 0.20)
        ),
        feedback_attempts=int(feedback.get("max_attempts", 2)),
        generation_threshold_db=float(
            feedback.get("generation_threshold_db", 2.5)
        ),
        reconstruction_threshold_db=float(
            feedback.get("reconstruction_threshold_db", 8.0)
        ),
        safe_fallback=str(feedback.get("safe_fallback", "generation")),
    )


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        snr_key = str(row["forward_snr_db"])
        candidate_key = "%s_strength_%g" % (
            row["payload_mode"],
            float(row["diffusion_strength"]),
        )
        grouped[snr_key][candidate_key].append(row)
    summary: Dict[str, Any] = {}
    metric_names = (
        "feedback_error",
        "feedback_rejected",
        "caption_bleu2",
        "regression_psnr",
        "selected_psnr",
        "regression_ssim",
        "selected_ssim",
        "regression_clip_image",
        "selected_clip_image",
        "clip_image_gain",
        "regression_clip_source_text",
        "selected_clip_source_text",
        "clip_source_text_gain",
        "regression_clip_received_text",
        "selected_clip_received_text",
        "clip_received_text_gain",
        "psnr_delta",
        "ssim_delta",
        "semantic_utility_gain",
    )
    for snr, candidates in grouped.items():
        summary[snr] = {}
        for candidate_key, items in candidates.items():
            entry: Dict[str, Any] = {"samples": len(items)}
            for name in metric_names:
                values = [
                    float(item[name]) for item in items if item[name] != ""
                ]
                entry[name] = sum(values) / max(len(values), 1)
            entry["mean_text_tokens"] = sum(
                float(item["allocated_text_tokens"]) for item in items
            ) / max(len(items), 1)
            entry["mean_image_tokens"] = sum(
                float(item["allocated_image_tokens"]) for item in items
            ) / max(len(items), 1)
            summary[snr][candidate_key] = entry
    return summary


def _routing_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    unique: Dict[tuple[int, float], Dict[str, Any]] = {}
    for row in rows:
        key = (int(row["image_id"]), float(row["forward_snr_db"]))
        unique.setdefault(key, row)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in unique.values():
        grouped[str(row["forward_snr_db"])].append(row)
    result: Dict[str, Any] = {}
    for snr, items in grouped.items():
        count = max(len(items), 1)
        result[snr] = {
            "samples": len(items),
            "feedback_error": sum(
                float(item["feedback_error"]) for item in items
            ) / count,
            "feedback_rejected": sum(
                float(item["feedback_rejected"]) for item in items
            ) / count,
            "requested_mode_fraction": {
                mode: sum(item["requested_mode"] == mode for item in items)
                / count
                for mode in PAYLOAD_MODES
            },
            "payload_mode_fraction": {
                mode: sum(item["payload_mode"] == mode for item in items)
                / count
                for mode in PAYLOAD_MODES
            },
        }
    return result


@torch.no_grad()
def collect_receiver_outputs(args: argparse.Namespace):
    seed_everything(args.seed)
    device = select_device(args.device)
    if device.type != "cuda" and not args.skip_diffusion:
        raise RuntimeError("Diffusion evaluation requires CUDA")
    config = prepare_config(
        load_config(args.config),
        preset=args.preset,
        vae_path=args.vae_path,
        text_model_path=args.text_model_path,
        data_root=args.data_root,
    )
    model = create_model(config, device)
    report = load_state_dict_flexible(
        model,
        load_checkpoint(args.checkpoint, map_location="cpu"),
        allow_shape_expansion=False,
    )
    model.eval()
    controller = _feedback_controller(config).to(device).eval()
    print(
        f"loaded checkpoint exact={report['exact_count']} device={device}",
        flush=True,
    )

    data = config["data"]
    split = str(getattr(args, "data_split", "val"))
    if split not in {"train", "val"}:
        raise ValueError("data_split must be 'train' or 'val'")
    loader = build_dataloader(
        image_root=data[f"{split}_image_root"],
        annotation_file=data[f"{split}_annotation_file"],
        image_size=int(config["model"]["image_size"]),
        batch_size=min(args.batch_size, args.samples),
        num_workers=args.num_workers,
        train=False,
        caption_mode=data.get(f"{split}_caption_mode", "first"),
        max_samples=args.samples,
        seed=args.seed,
    )
    cached: List[Dict[str, Any]] = []
    amp = bool(not args.no_amp and device.type == "cuda")

    for snr_index, forward_snr in enumerate(args.snr_list):
        for batch_index, batch in enumerate(loader):
            images = batch["images"].to(device, non_blocking=True)
            feedback_snr = (
                float(forward_snr)
                if args.feedback_snr_db is None
                else float(args.feedback_snr_db)
            )
            feedback_generator = torch.Generator().manual_seed(
                args.seed * 1000000 + snr_index * 10000 + batch_index
            )
            feedback = controller(
                true_forward_snr_db=float(forward_snr),
                feedback_snr_db=feedback_snr,
                batch_size=images.size(0),
                device=device,
                ideal_feedback=args.ideal_feedback,
                generator=feedback_generator,
            )
            payload_modes = mode_names(feedback["payload_mode"])
            requested_modes = mode_names(feedback["requested_mode"])
            text_quotas = torch.tensor(
                [MODE_TEXT_QUOTAS[mode] for mode in payload_modes],
                device=device,
                dtype=torch.float32,
            )
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(
                    images,
                    batch["texts"],
                    snr_db=float(forward_snr),
                    disable_modality_dropout=True,
                    semantic_ablation="full",
                    text_token_quota=text_quotas,
                )
            received_captions = model.decode_text_logits(outputs["text_logits"])
            regression_metrics = batch_image_metrics(outputs["image"], images)
            route_features = receiver_features(
                outputs,
                snr_db=float(forward_snr),
                max_text_tokens=max(MODE_TEXT_QUOTAS.values()),
            )

            for index, image_id in enumerate(batch["image_ids"]):
                pilot_uses = int(feedback["pilot_channel_uses"][index].item())
                feedback_uses = int(
                    feedback["feedback_channel_uses"][index].item()
                )
                cached.append(
                    {
                        "image_id": int(image_id),
                        "forward_snr_db": float(forward_snr),
                        "feedback_snr_db": feedback_snr,
                        "estimated_snr_db": float(
                            feedback["estimated_forward_snr_db"][index].item()
                        ),
                        "requested_mode": requested_modes[index],
                        "payload_mode": payload_modes[index],
                        "raw_feedback_state": int(
                            feedback["raw_decoded_state"][index].item()
                        ),
                        "feedback_error": int(
                            feedback["feedback_error"][index].item()
                        ),
                        "feedback_rejected": int(
                            not feedback["feedback_accepted"][index].item()
                        ),
                        "feedback_margin": float(
                            feedback["confidence_margin"][index].item()
                        ),
                        "pilot_channel_uses": pilot_uses,
                        "feedback_channel_uses": feedback_uses,
                        "payload_channel_uses": float(
                            outputs["channel_uses"].item()
                        ),
                        "total_channel_uses": float(
                            outputs["channel_uses"].item()
                            + pilot_uses
                            + feedback_uses
                        ),
                        "allocated_common_tokens": 32.0,
                        "allocated_image_tokens": float(
                            outputs["allocated_image_tokens"][index].item()
                        ),
                        "allocated_text_tokens": float(
                            outputs["allocated_text_tokens"][index].item()
                        ),
                        "text_repetition_factor": float(
                            outputs["text_repetition_factor"][index].item()
                        ),
                        "source_caption": str(batch["texts"][index]),
                        "received_caption": str(received_captions[index]),
                        "caption_bleu2": sentence_bleu2(
                            str(batch["texts"][index]),
                            str(received_captions[index]),
                        ),
                        "regression_psnr": float(
                            regression_metrics["psnr"][index].item()
                        ),
                        "regression_ssim": float(
                            regression_metrics["ssim"][index].item()
                        ),
                        "original": images[index].detach().cpu(),
                        "regression": outputs["image"][index].detach().cpu(),
                        # Receiver-only features used by V9-B. The Common
                        # semantic is transmitted/decoded state; it is not a
                        # source-caption feature.
                        "route_features": route_features[index].detach().cpu(),
                        "common_receiver_semantic": outputs[
                            "common_receiver_semantic"
                        ][index].detach().cpu(),
                    }
                )
            print(
                f"cached snr={forward_snr:g} batch={batch_index + 1}/{len(loader)} "
                f"requested={requested_modes} payload={payload_modes} "
                f"T={text_quotas.tolist()}",
                flush=True,
            )
    model.to("cpu")
    controller.to("cpu")
    del model, controller
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return config, device, cached


@torch.no_grad()
def generate_and_score(
    args: argparse.Namespace,
    device: torch.device,
    cached: List[Dict[str, Any]],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    pipe = None
    if not args.skip_diffusion:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            args.diffusion_model_path,
            torch_dtype=torch.float16,
            local_files_only=True,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
        pipe.set_progress_bar_config(disable=False)
    scorer = CLIPSemanticSimilarity(args.clip_model_path).to(device).eval()

    rows: List[Dict[str, Any]] = []
    for item_index, item in enumerate(cached):
        original = item.pop("original").to(device)
        regression = item.pop("regression").to(device)
        mode = str(item["payload_mode"])
        prompt = str(item["received_caption"]).strip()
        if not prompt:
            prompt = "a natural photograph"

        regression_clip_image = float(
            scorer.image_similarity(
                regression.unsqueeze(0), original.unsqueeze(0)
            )[0].item()
        )
        regression_clip_source_text = float(
            scorer.similarity(
                regression.unsqueeze(0), [str(item["source_caption"])]
            )[0].item()
        )
        regression_clip_received_text = float(
            scorer.similarity(regression.unsqueeze(0), [prompt])[0].item()
        )
        if pipe is None or mode == "reconstruction":
            strengths = [0.0]
        elif mode == "hybrid":
            strengths = list(
                args.hybrid_strength_list
                if args.hybrid_strength_list is not None
                else [args.hybrid_strength]
            )
        else:
            strengths = list(
                args.generation_strength_list
                if args.generation_strength_list is not None
                else [args.generation_strength]
            )
        strengths = list(dict.fromkeys(float(value) for value in strengths))

        for strength in strengths:
            diffusion_used = bool(
                pipe is not None
                and mode in {"hybrid", "generation"}
                and strength > 0.0
            )
            if diffusion_used:
                # Reset the generator for every strength. Candidate differences
                # then come from strength, not a different diffusion noise draw.
                generator = torch.Generator(device=device).manual_seed(
                    args.seed * 100000 + item_index
                )
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
                    Image.Resampling.LANCZOS,
                )
                selected = pil_to_tensor(generated).to(device)
            else:
                selected = regression

            selected_metrics = batch_image_metrics(
                selected.unsqueeze(0), original.unsqueeze(0)
            )
            selected_psnr = float(selected_metrics["psnr"][0].item())
            selected_ssim = float(selected_metrics["ssim"][0].item())
            selected_clip_image = float(
                scorer.image_similarity(
                    selected.unsqueeze(0), original.unsqueeze(0)
                )[0].item()
            )
            selected_clip_source_text = float(
                scorer.similarity(
                    selected.unsqueeze(0), [str(item["source_caption"])]
                )[0].item()
            )
            selected_clip_received_text = float(
                scorer.similarity(selected.unsqueeze(0), [prompt])[0].item()
            )
            clip_image_gain = selected_clip_image - regression_clip_image
            clip_source_text_gain = (
                selected_clip_source_text - regression_clip_source_text
            )
            clip_received_text_gain = (
                selected_clip_received_text - regression_clip_received_text
            )
            psnr_delta = selected_psnr - float(item["regression_psnr"])
            ssim_delta = selected_ssim - float(item["regression_ssim"])
            semantic_utility_gain = 0.5 * (
                clip_image_gain + clip_source_text_gain
            )
            strength_name = str(strength).replace(".", "p")
            stem = "id%d_snr%s_%s_strength%s" % (
                int(item["image_id"]),
                str(item["forward_snr_db"])
                .replace(".", "p")
                .replace("-", "m"),
                mode,
                strength_name,
            )
            panel_path = output_dir / f"{stem}.png"
            save_panel(
                original.cpu(),
                regression.cpu(),
                selected.cpu(),
                f"{mode} s={strength:g}",
                panel_path,
            )
            row = dict(item)
            row.update(
                {
                    "prompt_used": prompt,
                    "diffusion_used": int(diffusion_used),
                    "diffusion_strength": float(strength),
                    "selected_psnr": selected_psnr,
                    "selected_ssim": selected_ssim,
                    "psnr_delta": psnr_delta,
                    "ssim_delta": ssim_delta,
                    "regression_clip_image": regression_clip_image,
                    "selected_clip_image": selected_clip_image,
                    "clip_image_gain": clip_image_gain,
                    "regression_clip_source_text": regression_clip_source_text,
                    "selected_clip_source_text": selected_clip_source_text,
                    "clip_source_text_gain": clip_source_text_gain,
                    "regression_clip_received_text": regression_clip_received_text,
                    "selected_clip_received_text": selected_clip_received_text,
                    "clip_received_text_gain": clip_received_text_gain,
                    "semantic_utility_gain": semantic_utility_gain,
                    "panel": str(panel_path),
                }
            )
            rows.append(row)
            print(
                f"[{item_index + 1}/{len(cached)}] id={item['image_id']} "
                f"snr={item['forward_snr_db']:g} mode={mode} "
                f"strength={strength:g} "
                f"dPSNR={psnr_delta:+.2f} "
                f"dCLIP-I={clip_image_gain:+.4f} "
                f"dCLIP-T={clip_source_text_gain:+.4f}",
                flush=True,
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V9 feedback-driven reconstruction/generation evaluation"
    )
    parser.add_argument("--config", default="configs/v9a_feedback_generation.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--text-model-path", required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--diffusion-model-path", default=None)
    parser.add_argument("--preset", default="original_4608")
    parser.add_argument(
        "--snr-list", type=float, nargs="+", default=[-5, 0, 5, 10, 20]
    )
    parser.add_argument("--feedback-snr-db", type=float, default=None)
    parser.add_argument("--ideal-feedback", action="store_true")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--hybrid-strength", type=float, default=0.25)
    parser.add_argument("--generation-strength", type=float, default=0.65)
    parser.add_argument(
        "--hybrid-strength-list",
        type=float,
        nargs="+",
        default=None,
        help="Paired hybrid strength sweep; overrides --hybrid-strength.",
    )
    parser.add_argument(
        "--generation-strength-list",
        type=float,
        nargs="+",
        default=None,
        help="Paired generation strength sweep; overrides --generation-strength.",
    )
    parser.add_argument("--skip-diffusion", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--data-split",
        choices=("train", "val"),
        default="val",
        help="Use train for verifier-data collection; final evaluation uses val.",
    )
    args = parser.parse_args()
    if not args.skip_diffusion and not args.diffusion_model_path:
        parser.error("--diffusion-model-path is required unless --skip-diffusion")
    for name, values in (
        ("hybrid", args.hybrid_strength_list or [args.hybrid_strength]),
        ("generation", args.generation_strength_list or [args.generation_strength]),
    ):
        if any(float(value) <= 0.0 or float(value) > 1.0 for value in values):
            parser.error(f"{name} diffusion strengths must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, device, cached = collect_receiver_outputs(args)
    rows = generate_and_score(args, device, cached, output_dir)

    csv_path = output_dir / "samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "checkpoint": args.checkpoint,
        "source_caption_used_by_receiver": False,
        "source_caption_used_for_evaluation_only": True,
        "diffusion_prompt_source": "received_caption",
        "mode_text_quotas": MODE_TEXT_QUOTAS,
        "ideal_feedback": bool(args.ideal_feedback),
        "fixed_feedback_snr_db": args.feedback_snr_db,
        "hybrid_strengths": (
            args.hybrid_strength_list
            if args.hybrid_strength_list is not None
            else [args.hybrid_strength]
        ),
        "generation_strengths": (
            args.generation_strength_list
            if args.generation_strength_list is not None
            else [args.generation_strength]
        ),
        "routing_summary": _routing_summary(rows),
        "candidate_summary": _summarize(rows),
    }
    report_path = output_dir / "summary.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"saved {csv_path}", flush=True)
    print(f"saved {report_path}", flush=True)


if __name__ == "__main__":
    main()
