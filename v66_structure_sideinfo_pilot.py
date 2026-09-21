"""V6.6 pilot: protected transmitter structure for generative recovery.

This is a development/oracle experiment.  The original image is never passed
to the diffusion model.  Only a counted, low-resolution binary structure map
derived at the transmitter may cross the side channel.  Source pixels remain
available solely for offline metrics and oracle candidate selection.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

from data import build_dataloader
from models.clip_semantics import CLIPSemanticSimilarity
from models.controlnet_fusion_policy import load_controlnet_policy
from models.quality_adaptive_recovery import QualityAdaptivePolicy, receiver_features
from models.selective_generation_verifier import load_generation_verifier
from utils.checkpoint import load_checkpoint, load_state_dict_flexible
from utils.config import load_config, prepare_config
from utils.experiment import create_model, seed_everything, select_device
from utils.metrics import batch_image_metrics, sentence_bleu2
from utils.v62_diffusion import (
    add_semantic_context,
    pil_to_tensor,
    prompt_embeddings,
    tensor_to_pil,
)
from utils.v65_selective import hybrid_text_utility, select_received_text
from utils.v66_structure import (
    binary_structure_map,
    candidate_structure_statistics,
    repetition_for_snr,
    structure_to_pil,
    tolerant_structure_f1,
    transmit_binary_structure,
)
from v63_controlnet_fusion_pilot import _fuse, _load_adapter


def _text_accuracy(outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    logits = outputs["text_logits"]
    ids = outputs["text_input_ids"]
    mask = outputs["text_attention_mask"].bool()
    length = min(logits.size(1), ids.size(1), mask.size(1))
    correct = (logits[:, :length].argmax(-1) == ids[:, :length]).float()
    return (correct * mask[:, :length]).sum(1) / mask[:, :length].sum(1).clamp_min(1)


def _read_excluded_ids(paths: List[str]) -> set:
    image_ids = set()
    for value in paths:
        path = Path(value)
        if not path.is_file():
            print("Optional exclusion CSV not found, skipping: %s" % path)
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("image_id"):
                    image_ids.add(int(row["image_id"]))
    return image_ids


def _generate_structure_controlnet(
    pipe: StableDiffusionControlNetImg2ImgPipeline,
    regression: torch.Tensor,
    control_structure: torch.Tensor,
    prompt: str,
    semantic_tokens: Optional[torch.Tensor],
    strength: float,
    steps: int,
    guidance_scale: float,
    control_scale: float,
    generator: torch.Generator,
) -> torch.Tensor:
    height, width = int(regression.size(1)), int(regression.size(2))
    initial_image = tensor_to_pil(regression, size=512)
    control_image = structure_to_pil(control_structure, size=512)
    kwargs = {
        "image": initial_image,
        "control_image": control_image,
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


def _metric_bundle(
    scorer: CLIPSemanticSimilarity,
    candidate: torch.Tensor,
    target: torch.Tensor,
    prompt: str,
    source_structure: torch.Tensor,
    received_structure: torch.Tensor,
    structure_side: int,
    edge_quantile: float,
) -> Dict[str, float]:
    image_metrics = batch_image_metrics(candidate.unsqueeze(0), target.unsqueeze(0))
    clip_image = float(
        scorer.image_similarity(candidate.unsqueeze(0), target.unsqueeze(0))[0]
    )
    clip_text = float(
        scorer.similarity(candidate.unsqueeze(0), [prompt or "a natural photograph"])[0]
    )
    psnr = float(image_metrics["psnr"][0])
    ssim = float(image_metrics["ssim"][0])
    structure = candidate_structure_statistics(
        candidate,
        received_structure,
        source_structure,
        side=structure_side,
        edge_quantile=edge_quantile,
    )
    psnr_scaled = max(0.0, min(1.0, (psnr - 10.0) / 30.0))
    utility = (
        0.35 * (0.5 * (clip_image + 1.0))
        + 0.20 * (0.5 * (clip_text + 1.0))
        + 0.20 * ssim
        + 0.10 * psnr_scaled
        + 0.15 * structure["source_structure_f1"]
    )
    return {
        "utility": utility,
        "psnr": psnr,
        "ssim": ssim,
        "clip_image": clip_image,
        "clip_text": clip_text,
        "received_structure_f1": structure["received_structure_f1"],
        "source_structure_f1": structure["source_structure_f1"],
    }


@torch.no_grad()
def _best_candidate(
    pipe: StableDiffusionControlNetImg2ImgPipeline,
    scorer: CLIPSemanticSimilarity,
    regression: torch.Tensor,
    target: torch.Tensor,
    prompt: str,
    semantic_tokens: torch.Tensor,
    control_structure: torch.Tensor,
    source_structure: torch.Tensor,
    args: argparse.Namespace,
    seed_offset: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, object]]:
    best_raw = None
    best_candidate = None
    best_metrics = None
    best_metadata: Dict[str, object] = {}
    total_latency_ms = 0.0
    for strength_index, strength in enumerate(args.strengths):
        generator = torch.Generator(device=regression.device).manual_seed(
            args.seed * 1000000 + seed_offset * 100 + strength_index
        )
        torch.cuda.synchronize(regression.device)
        started = time.perf_counter()
        raw = _generate_structure_controlnet(
            pipe,
            regression,
            control_structure,
            prompt,
            semantic_tokens,
            float(strength),
            args.steps,
            args.guidance_scale,
            args.control_scale,
            generator,
        )
        torch.cuda.synchronize(regression.device)
        total_latency_ms += 1000.0 * (time.perf_counter() - started)
        candidates = [("raw", 1.0, raw)]
        for expert in ("rgb", "detail"):
            for alpha in args.blend_alphas:
                candidates.append((expert, float(alpha), _fuse(regression, raw, expert, alpha)))
        for expert, alpha, candidate in candidates:
            metrics = _metric_bundle(
                scorer,
                candidate,
                target,
                prompt,
                source_structure,
                control_structure,
                args.structure_side,
                args.edge_quantile,
            )
            if best_metrics is None or metrics["utility"] > best_metrics["utility"]:
                best_raw = raw
                best_candidate = candidate
                best_metrics = metrics
                best_metadata = {
                    "strength": float(strength),
                    "expert": expert,
                    "alpha": float(alpha),
                }
    best_metadata["generation_latency_ms"] = total_latency_ms
    return best_raw, best_candidate, best_metrics, best_metadata


def _tile(image: Image.Image, title: str) -> Image.Image:
    header = 34
    canvas = Image.new("RGB", (image.width, image.height + header), "white")
    canvas.paste(image, (0, header))
    ImageDraw.Draw(canvas).text((4, 8), title, fill="black", font=ImageFont.load_default())
    return canvas


def _panel(
    original: torch.Tensor,
    regression: torch.Tensor,
    source_structure: torch.Tensor,
    noisy_structure: torch.Tensor,
    candidates: Mapping[str, torch.Tensor],
    positives: Mapping[str, bool],
) -> Image.Image:
    width = int(original.size(2))
    tiles = [
        _tile(tensor_to_pil(original), "Original (metrics only)"),
        _tile(tensor_to_pil(regression), "Regression"),
        _tile(structure_to_pil(source_structure, width), "Tx structure"),
        _tile(structure_to_pil(noisy_structure, width), "Decoded structure"),
    ]
    labels = {
        "receiver": "RX-edge candidate",
        "oracle": "TX clean candidate",
        "noisy": "TX noisy candidate",
    }
    for mode in ("receiver", "oracle", "noisy"):
        if mode in candidates:
            suffix = " +" if positives.get(mode, False) else " -"
            tiles.append(_tile(tensor_to_pil(candidates[mode]), labels[mode] + suffix))
    canvas = Image.new("RGB", (sum(tile.width for tile in tiles), tiles[0].height), "white")
    offset = 0
    for tile in tiles:
        canvas.paste(tile, (offset, 0))
        offset += tile.width
    return canvas


def _write_summary(rows: List[Dict[str, object]], path: Path) -> None:
    groups = defaultdict(list)
    for row in rows:
        key = (float(row["snr_db"]), str(row["structure_mode"]))
        groups[key].append(row)
        groups[("ALL", str(row["structure_mode"]))].append(row)
    records = []
    for (snr, mode), items in groups.items():
        def mean(field: str) -> float:
            return sum(float(item[field]) for item in items) / len(items)

        records.append(
            {
                "snr_db": snr,
                "structure_mode": mode,
                "samples": len(items),
                "generation_positive_rate": mean("generation_positive"),
                "metric_safe_positive_rate": mean("metric_safe_positive"),
                "regression_utility": mean("regression_utility"),
                "candidate_utility": mean("candidate_utility"),
                "utility_gain": mean("candidate_utility") - mean("regression_utility"),
                "regression_psnr": mean("regression_psnr"),
                "candidate_psnr": mean("candidate_psnr"),
                "psnr_gain": mean("candidate_psnr") - mean("regression_psnr"),
                "regression_ssim": mean("regression_ssim"),
                "candidate_ssim": mean("candidate_ssim"),
                "ssim_gain": mean("candidate_ssim") - mean("regression_ssim"),
                "regression_clip_image": mean("regression_clip_image"),
                "candidate_clip_image": mean("candidate_clip_image"),
                "clip_gain": mean("candidate_clip_image") - mean("regression_clip_image"),
                "regression_source_structure_f1": mean("regression_source_structure_f1"),
                "candidate_source_structure_f1": mean("candidate_source_structure_f1"),
                "source_structure_f1_gain": mean("candidate_source_structure_f1")
                - mean("regression_source_structure_f1"),
                "control_source_structure_f1": mean("control_source_structure_f1"),
                "structure_ber": mean("structure_ber"),
                "mean_structure_source_bits": mean("structure_source_bits"),
                "mean_structure_channel_bits": mean("structure_channel_bits"),
                "mean_structure_bpp": mean("structure_bpp"),
                "mean_generation_latency_ms": mean("generation_latency_ms"),
                "protected_text_selection_rate": mean("protected_text_selected"),
                "adaptive_text_accuracy": mean("adaptive_text_accuracy"),
                "adaptive_bleu2": mean("adaptive_bleu2"),
                "training_data_overlap_count": sum(int(item["training_data_overlap"]) for item in items),
            }
        )
    records.sort(
        key=lambda value: (
            value["snr_db"] == "ALL",
            float(value["snr_db"]) if value["snr_db"] != "ALL" else 1e9,
            value["structure_mode"],
        )
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _write_paired_comparison(rows: List[Dict[str, object]], path: Path) -> None:
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(int(row["image_id"]), float(row["snr_db"]))][str(row["structure_mode"])] = row
    records = []
    for (image_id, snr), modes in grouped.items():
        if "receiver" not in modes or "oracle" not in modes:
            continue
        receiver = modes["receiver"]
        oracle = modes["oracle"]
        records.append(
            {
                "image_id": image_id,
                "snr_db": snr,
                "oracle_minus_receiver_utility": float(oracle["candidate_utility"])
                - float(receiver["candidate_utility"]),
                "oracle_minus_receiver_psnr": float(oracle["candidate_psnr"])
                - float(receiver["candidate_psnr"]),
                "oracle_minus_receiver_ssim": float(oracle["candidate_ssim"])
                - float(receiver["candidate_ssim"]),
                "oracle_minus_receiver_clip": float(oracle["candidate_clip_image"])
                - float(receiver["candidate_clip_image"]),
                "oracle_minus_receiver_structure_f1": float(oracle["candidate_source_structure_f1"])
                - float(receiver["candidate_source_structure_f1"]),
                "panel": oracle["panel"],
            }
        )
    if records:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("V6.6 structure pilot requires CUDA")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = load_controlnet_policy(args.policy_checkpoint, device)
    verifier = load_generation_verifier(args.verifier_checkpoint, device)
    text_guard = QualityAdaptivePolicy().to(device).eval()
    adapter = _load_adapter(args.adapter_checkpoint, device)
    training_ids = set(getattr(policy, "policy_data_image_ids", set())) | set(
        getattr(verifier, "all_training_image_ids", set())
    )
    development_ids = _read_excluded_ids(args.exclude_results_csv)
    excluded_ids = training_ids | development_ids
    if not training_ids and not args.allow_training_data_overlap:
        raise RuntimeError("Checkpoints lack training image ids; strict split cannot be verified")
    if args.allow_training_data_overlap:
        excluded_ids = development_ids
    print(
        "V6.6 split: excluding training=%d previous-development=%d total=%d"
        % (len(training_ids), len(development_ids), len(excluded_ids))
    )

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
            probe = v61(
                images,
                batch["texts"],
                **{key: value for key, value in probe_kwargs.items() if key in supported}
            )
            probe_captions = v61.decode_text_logits(probe["text_logits"])
            probe_accuracy = _text_accuracy(probe)
            probe_alignment = scorer.similarity(probe["image"], probe_captions)
            probe_features = receiver_features(probe, snr, image_text_alignment=probe_alignment)
            probe_policy = policy(probe_features)
            learned = probe_policy["text_utility"].clamp(0.0, 1.0)
            heuristic = text_guard(probe_features)["text_utility"].clamp(0.0, 1.0)
            recommended, safety = hybrid_text_utility(
                probe_features,
                learned,
                heuristic,
                snr,
                probe_utility=args.probe_utility,
                generation_probability=probe_policy["generation_probability"],
            )
            feedback = (recommended - float(args.probe_utility)).abs() >= args.feedback_deadband
            actual_utility = torch.where(
                feedback,
                recommended,
                recommended.new_full(recommended.shape, float(args.probe_utility)),
            )
            adaptive_kwargs = {
                "snr_db": snr,
                "disable_modality_dropout": True,
                "utility_weight": actual_utility,
                "channel_noise": shared_noise,
                "channel_gain": shared_gain,
            }
            adaptive = v61(
                images,
                batch["texts"],
                **{key: value for key, value in adaptive_kwargs.items() if key in supported}
            )
            protected_captions = v61.decode_text_logits(adaptive["text_logits"])
            protected_accuracy = _text_accuracy(adaptive)
            protected_alignment = scorer.similarity(adaptive["image"], protected_captions)
            protected_features = receiver_features(
                adaptive, snr, image_text_alignment=protected_alignment
            )
            choose_protected, probe_quality, protected_quality = select_received_text(
                probe_features,
                protected_features,
                minimum_gain=args.text_selection_margin,
            )
            selected_captions = [
                protected_captions[index] if bool(choose_protected[index]) else probe_captions[index]
                for index in range(len(protected_captions))
            ]
            selected_alignment = scorer.similarity(adaptive["image"], selected_captions)
            route_features = receiver_features(
                adaptive, snr, image_text_alignment=selected_alignment
            )
            selected_accuracy = torch.where(
                choose_protected, protected_accuracy, probe_accuracy
            )
            for index, image_id in enumerate(batch["image_ids"]):
                semantic_source = adaptive if bool(choose_protected[index]) else probe
                original = images[index].detach().cpu()
                regression = adaptive["image"][index].detach().cpu()
                cached.append(
                    {
                        "image_id": int(image_id),
                        "snr": float(snr),
                        "original": original,
                        "regression": regression,
                        "source_caption": batch["texts"][index],
                        "received_caption": selected_captions[index],
                        "protected_text_selected": bool(choose_protected[index]),
                        "adaptive_text_accuracy": float(selected_accuracy[index]),
                        "adaptive_bleu2": sentence_bleu2(
                            str(batch["texts"][index]), str(selected_captions[index])
                        ),
                        "guarded_text_utility": float(recommended[index]),
                        "channel_floor": float(safety["channel_floor"][index]),
                        "text_tokens": float(adaptive["allocated_text_tokens"][index]),
                        "text_repeat": float(adaptive["text_repetition_factor"][index]),
                        "route_features": route_features[index].detach().cpu(),
                        "source_structure": binary_structure_map(
                            original, args.structure_side, args.edge_quantile
                        ),
                        "receiver_structure": binary_structure_map(
                            regression, args.structure_side, args.edge_quantile
                        ),
                        "semantic_output": {
                            name: semantic_source[name][index : index + 1].detach().cpu()
                            for name in (
                                "decoded_common_tokens",
                                "decoded_image_tokens",
                                "decoded_text_tokens",
                            )
                        },
                    }
                )
            print("Cached receiver outputs SNR=%g total=%d" % (snr, len(cached)))

    v61.to("cpu")
    scorer.to("cpu")
    policy.to("cpu")
    verifier.to("cpu")
    text_guard.to("cpu")
    del v61, scorer, policy, verifier, text_guard
    torch.cuda.empty_cache()

    controlnet = ControlNetModel.from_pretrained(
        args.canny_controlnet_path,
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
    adapter.to(device).eval()

    rows: List[Dict[str, object]] = []
    for item_index, item in enumerate(cached):
        original = item["original"].to(device)
        regression = item["regression"].to(device)
        source_structure = item["source_structure"].to(device)
        receiver_structure = item["receiver_structure"].to(device)
        prompt = str(item["received_caption"])
        semantic_output = {
            key: value.to(device) for key, value in item["semantic_output"].items()
        }
        semantic_tokens = adapter(
            semantic_output, item["route_features"].to(device).unsqueeze(0)
        )
        repetition = args.structure_repetition
        if repetition <= 0:
            repetition = repetition_for_snr(float(item["snr"]))
        channel_generator = torch.Generator(device="cpu").manual_seed(
            args.seed * 100000 + item_index
        )
        noisy_structure_cpu, noisy_stats = transmit_binary_structure(
            item["source_structure"],
            float(item["snr"]),
            repetition,
            generator=channel_generator,
        )
        noisy_structure = noisy_structure_cpu.to(device)
        structures = {
            "receiver": receiver_structure,
            "oracle": source_structure,
            "noisy": noisy_structure,
        }
        source_pixels = int(original.size(1) * original.size(2))
        regression_metrics = _metric_bundle(
            scorer,
            regression,
            original,
            prompt,
            source_structure,
            source_structure,
            args.structure_side,
            args.edge_quantile,
        )
        candidates: Dict[str, torch.Tensor] = {}
        positives: Dict[str, bool] = {}
        for mode in args.structure_modes:
            control_structure = structures[mode]
            raw, candidate, metrics, metadata = _best_candidate(
                pipe,
                scorer,
                regression,
                original,
                prompt,
                semantic_tokens,
                control_structure,
                source_structure,
                args,
                # Keep diffusion noise identical across structure modes so the
                # paired difference isolates the control map rather than RNG.
                seed_offset=item_index,
            )
            candidates[mode] = candidate.detach().cpu()
            positive = bool(
                metrics["utility"] > regression_metrics["utility"] + args.generation_margin
            )
            positives[mode] = positive
            metric_safe_positive = bool(
                metrics["clip_image"] - regression_metrics["clip_image"] >= args.min_clip_gain
                and metrics["psnr"] - regression_metrics["psnr"] >= -args.max_psnr_drop
                and metrics["ssim"] - regression_metrics["ssim"] >= -args.max_ssim_drop
            )
            if mode == "receiver":
                channel = {
                    "structure_ber": 0.0,
                    "structure_source_bits": 0.0,
                    "structure_channel_bits": 0.0,
                    "structure_repetition": 0.0,
                }
            elif mode == "oracle":
                bits = float(source_structure.numel())
                channel = {
                    "structure_ber": 0.0,
                    "structure_source_bits": bits,
                    "structure_channel_bits": bits,
                    "structure_repetition": 1.0,
                }
            else:
                channel = noisy_stats
            control_source_f1 = float(
                tolerant_structure_f1(control_structure, source_structure)[0]
            )
            row = {
                "image_id": item["image_id"],
                "snr_db": item["snr"],
                "structure_mode": mode,
                "generation_positive": int(positive),
                "metric_safe_positive": int(metric_safe_positive),
                "best_strength": metadata["strength"],
                "best_expert": metadata["expert"],
                "best_alpha": metadata["alpha"],
                "regression_utility": regression_metrics["utility"],
                "candidate_utility": metrics["utility"],
                "regression_psnr": regression_metrics["psnr"],
                "candidate_psnr": metrics["psnr"],
                "regression_ssim": regression_metrics["ssim"],
                "candidate_ssim": metrics["ssim"],
                "regression_clip_image": regression_metrics["clip_image"],
                "candidate_clip_image": metrics["clip_image"],
                "regression_clip_text": regression_metrics["clip_text"],
                "candidate_clip_text": metrics["clip_text"],
                "regression_source_structure_f1": regression_metrics["source_structure_f1"],
                "candidate_source_structure_f1": metrics["source_structure_f1"],
                "candidate_received_structure_f1": metrics["received_structure_f1"],
                "control_source_structure_f1": control_source_f1,
                "structure_ber": channel["structure_ber"],
                "structure_source_bits": channel["structure_source_bits"],
                "structure_channel_bits": channel["structure_channel_bits"],
                "structure_repetition": channel["structure_repetition"],
                "structure_bpp": float(channel["structure_channel_bits"]) / source_pixels,
                "generation_latency_ms": metadata["generation_latency_ms"],
                "protected_text_selected": int(item["protected_text_selected"]),
                "adaptive_text_accuracy": item["adaptive_text_accuracy"],
                "adaptive_bleu2": item["adaptive_bleu2"],
                "allocated_text_tokens": item["text_tokens"],
                "text_repetition_factor": item["text_repeat"],
                "training_data_overlap": int(int(item["image_id"]) in training_ids),
                "received_caption": prompt,
            }
            rows.append(row)
            print(
                "[%d/%d] id=%s snr=%g mode=%s positive=%d safe=%d "
                "CLIP=%+.4f PSNR=%+.3f SSIM=%+.4f structure=%+.4f"
                % (
                    item_index + 1,
                    len(cached),
                    item["image_id"],
                    item["snr"],
                    mode,
                    int(positive),
                    int(metric_safe_positive),
                    metrics["clip_image"] - regression_metrics["clip_image"],
                    metrics["psnr"] - regression_metrics["psnr"],
                    metrics["ssim"] - regression_metrics["ssim"],
                    metrics["source_structure_f1"] - regression_metrics["source_structure_f1"],
                )
            )
        stem = "id%s_snr%s" % (
            item["image_id"], str(item["snr"]).replace(".", "p")
        )
        panel_path = output_dir / (stem + ".png")
        _panel(
            original.cpu(),
            regression.cpu(),
            source_structure.cpu(),
            noisy_structure.cpu(),
            candidates,
            positives,
        ).save(panel_path)
        for row in rows[-len(args.structure_modes):]:
            row["panel"] = panel_path.name

    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(rows, output_dir / "summary.csv")
    _write_paired_comparison(rows, output_dir / "paired_comparison.csv")
    manifest = {
        "version": "V6.6",
        "pilot_type": "transmitter_structure_side_information",
        "seed": args.seed,
        "structure_modes": args.structure_modes,
        "structure_side": args.structure_side,
        "edge_quantile": args.edge_quantile,
        "strengths": args.strengths,
        "blend_alphas": args.blend_alphas,
        "snr_list": args.snr_list,
        "note": "Original pixels are metric-only; oracle mode transmits a counted error-free binary structure map.",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print("Done: %s" % output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V6.6 transmitter-structure ControlNet oracle/channel pilot"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--verifier-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--diffusion-model-path", required=True)
    parser.add_argument("--canny-controlnet-path", required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--vae-path")
    parser.add_argument("--text-model-path")
    parser.add_argument("--output-dir", default="experiment_results/v66_structure_oracle_seed842")
    parser.add_argument("--config", default="configs/v61_bandwidth.yaml")
    parser.add_argument("--preset", default="original_4608")
    parser.add_argument(
        "--channel-type",
        default="awgn",
        choices=["awgn", "rayleigh", "gamma_gamma", "fso"],
    )
    parser.add_argument("--snr-list", type=float, nargs="+", default=[-5, 0, 5])
    parser.add_argument(
        "--structure-modes",
        nargs="+",
        choices=["receiver", "oracle", "noisy"],
        default=["receiver", "oracle"],
    )
    parser.add_argument("--structure-side", type=int, default=64)
    parser.add_argument("--edge-quantile", type=float, default=0.88)
    parser.add_argument(
        "--structure-repetition",
        type=int,
        default=0,
        help="odd repetition; <=0 uses the SNR-adaptive schedule",
    )
    parser.add_argument("--probe-utility", type=float, default=0.5)
    parser.add_argument("--feedback-deadband", type=float, default=0.10)
    parser.add_argument("--text-selection-margin", type=float, default=0.005)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.30, 0.50])
    parser.add_argument("--blend-alphas", type=float, nargs="+", default=[0.25, 0.50])
    parser.add_argument("--control-scale", type=float, default=0.9)
    parser.add_argument("--generation-margin", type=float, default=0.002)
    parser.add_argument("--min-clip-gain", type=float, default=0.003)
    parser.add_argument("--max-psnr-drop", type=float, default=0.05)
    parser.add_argument("--max-ssim-drop", type=float, default=0.004)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=842)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exclude-results-csv", nargs="*", default=[])
    parser.add_argument("--allow-training-data-overlap", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
