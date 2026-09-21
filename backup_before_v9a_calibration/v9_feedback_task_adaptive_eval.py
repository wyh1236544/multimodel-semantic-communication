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
            feedback.get("generation_threshold_db", 0.0)
        ),
        reconstruction_threshold_db=float(
            feedback.get("reconstruction_threshold_db", 10.0)
        ),
        safe_fallback=str(feedback.get("safe_fallback", "generation")),
    )


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["forward_snr_db"])].append(row)
    summary: Dict[str, Any] = {}
    metric_names = (
        "feedback_error",
        "feedback_rejected",
        "caption_bleu2",
        "regression_psnr",
        "selected_psnr",
        "regression_ssim",
        "selected_ssim",
        "selected_clip_image",
        "selected_clip_source_text",
    )
    for snr, items in grouped.items():
        entry: Dict[str, Any] = {"samples": len(items)}
        for name in metric_names:
            values = [float(item[name]) for item in items if item[name] != ""]
            entry[name] = sum(values) / max(len(values), 1)
        entry["requested_mode_fraction"] = {
            mode: sum(item["requested_mode"] == mode for item in items)
            / max(len(items), 1)
            for mode in PAYLOAD_MODES
        }
        entry["payload_mode_fraction"] = {
            mode: sum(item["payload_mode"] == mode for item in items)
            / max(len(items), 1)
            for mode in PAYLOAD_MODES
        }
        entry["mean_text_tokens"] = sum(
            float(item["allocated_text_tokens"]) for item in items
        ) / max(len(items), 1)
        entry["mean_image_tokens"] = sum(
            float(item["allocated_image_tokens"]) for item in items
        ) / max(len(items), 1)
        summary[snr] = entry
    return summary


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
    loader = build_dataloader(
        image_root=data["val_image_root"],
        annotation_file=data["val_annotation_file"],
        image_size=int(config["model"]["image_size"]),
        batch_size=min(args.batch_size, args.samples),
        num_workers=args.num_workers,
        train=False,
        caption_mode=data.get("val_caption_mode", "first"),
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

        diffusion_used = bool(
            pipe is not None and mode in {"hybrid", "generation"}
        )
        if diffusion_used:
            strength = (
                args.hybrid_strength
                if mode == "hybrid"
                else args.generation_strength
            )
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
            strength = 0.0
            selected = regression

        selected_metrics = batch_image_metrics(
            selected.unsqueeze(0), original.unsqueeze(0)
        )
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
        stem = "id%d_snr%s_%s" % (
            int(item["image_id"]),
            str(item["forward_snr_db"]).replace(".", "p").replace("-", "m"),
            mode,
        )
        panel_path = output_dir / f"{stem}.png"
        save_panel(
            original.cpu(),
            regression.cpu(),
            selected.cpu(),
            mode,
            panel_path,
        )
        row = dict(item)
        row.update(
            {
                "prompt_used": prompt,
                "diffusion_used": int(diffusion_used),
                "diffusion_strength": float(strength),
                "selected_psnr": float(selected_metrics["psnr"][0].item()),
                "selected_ssim": float(selected_metrics["ssim"][0].item()),
                "selected_clip_image": selected_clip_image,
                "selected_clip_source_text": selected_clip_source_text,
                "panel": str(panel_path),
            }
        )
        rows.append(row)
        print(
            f"[{item_index + 1}/{len(cached)}] id={item['image_id']} "
            f"snr={item['forward_snr_db']:g} mode={mode} "
            f"feedback_error={item['feedback_error']} "
            f"PSNR={row['selected_psnr']:.2f} "
            f"CLIP-I={selected_clip_image:.3f}",
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
    parser.add_argument("--hybrid-strength", type=float, default=0.45)
    parser.add_argument("--generation-strength", type=float, default=0.85)
    parser.add_argument("--skip-diffusion", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.skip_diffusion and not args.diffusion_model_path:
        parser.error("--diffusion-model-path is required unless --skip-diffusion")
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
        "diffusion_prompt_source": "received_caption",
        "mode_text_quotas": MODE_TEXT_QUOTAS,
        "ideal_feedback": bool(args.ideal_feedback),
        "fixed_feedback_snr_db": args.feedback_snr_db,
        "summary": _summarize(rows),
    }
    report_path = output_dir / "summary.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"saved {csv_path}", flush=True)
    print(f"saved {report_path}", flush=True)


if __name__ == "__main__":
    main()
