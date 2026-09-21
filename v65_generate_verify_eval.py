"""Strict holdout evaluation of V6.5 generate--verify--reject recovery."""
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
from models.selective_generation_verifier import load_generation_verifier
from utils.checkpoint import load_checkpoint, load_state_dict_flexible
from utils.config import load_config, prepare_config
from utils.experiment import create_model, seed_everything, select_device
from utils.metrics import batch_image_metrics, sentence_bleu2
from utils.v62_diffusion import tensor_to_pil
from utils.v65_selective import (
    hybrid_text_utility,
    load_verifier_thresholds,
    raw_verification_statistics,
    select_received_text,
    verification_features,
    verifier_acceptance,
)
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


def _panel(original, regression, raw, candidate, selected, invoked, accepted, route_p, accept_p):
    if not invoked:
        decision = "REGRESSION route-p=%.2f" % route_p
    elif accepted:
        decision = "ACCEPT verifier-p=%.2f" % accept_p
    else:
        decision = "REJECT verifier-p=%.2f" % accept_p
    tiles = [
        _tile(tensor_to_pil(original), "Original (metric only)"),
        _tile(tensor_to_pil(regression), "Adaptive regression"),
        _tile(tensor_to_pil(raw), "ControlNet raw"),
        _tile(tensor_to_pil(candidate), "Soft generated candidate"),
        _tile(tensor_to_pil(selected), decision),
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

        invoked_count = sum(int(item["primary_invoked"]) for item in items)
        accepted_count = sum(int(item["verified_accept"]) for item in items)
        record = {
            "snr_db": name,
            "samples": len(items),
            "primary_generation_rate": mean("primary_invoked"),
            "verified_generation_rate": mean("verified_accept"),
            "conditional_acceptance_rate": accepted_count / max(invoked_count, 1),
            "verification_rejection_rate": (invoked_count - accepted_count) / max(invoked_count, 1),
            "feedback_rate": mean("feedback_used"),
            "mean_text_tokens": mean("allocated_text_tokens"),
            "mean_text_repetition_factor": mean("text_repetition_factor"),
            "mean_guarded_text_utility": mean("guarded_text_utility"),
            "protected_text_selection_rate": mean("protected_text_selected"),
            "probe_text_accuracy": mean("probe_text_accuracy"),
            "protected_text_accuracy": mean("protected_text_accuracy"),
            "adaptive_text_accuracy": mean("adaptive_text_accuracy"),
            "text_accuracy_gain": mean("adaptive_text_accuracy") - mean("probe_text_accuracy"),
            "probe_bleu2": mean("probe_bleu2"),
            "protected_bleu2": mean("protected_bleu2"),
            "adaptive_bleu2": mean("adaptive_bleu2"),
            "regression_psnr": mean("regression_psnr"),
            "candidate_psnr": mean("candidate_psnr"),
            "selected_psnr": mean("selected_psnr"),
            "candidate_psnr_gain": mean("candidate_psnr") - mean("regression_psnr"),
            "selected_psnr_gain": mean("selected_psnr") - mean("regression_psnr"),
            "verification_psnr_recovery": mean("selected_psnr") - mean("candidate_psnr"),
            "regression_ssim": mean("regression_ssim"),
            "candidate_ssim": mean("candidate_ssim"),
            "selected_ssim": mean("selected_ssim"),
            "candidate_ssim_gain": mean("candidate_ssim") - mean("regression_ssim"),
            "selected_ssim_gain": mean("selected_ssim") - mean("regression_ssim"),
            "verification_ssim_recovery": mean("selected_ssim") - mean("candidate_ssim"),
            "regression_clip_image": mean("regression_clip_image"),
            "candidate_clip_image": mean("candidate_clip_image"),
            "selected_clip_image": mean("selected_clip_image"),
            "candidate_clip_gain": mean("candidate_clip_image") - mean("regression_clip_image"),
            "selected_clip_gain": mean("selected_clip_image") - mean("regression_clip_image"),
            "semantic_gain_retention": (
                (mean("selected_clip_image") - mean("regression_clip_image"))
                / max(mean("candidate_clip_image") - mean("regression_clip_image"), 1e-12)
                if mean("candidate_clip_image") > mean("regression_clip_image")
                else 0.0
            ),
            "mean_acceptance_probability": mean("acceptance_probability"),
            "mean_hallucination_probability": mean("hallucination_probability"),
            "mean_semantic_gain_probability": mean("semantic_gain_probability"),
            "mean_psnr_safe_probability": mean("psnr_safe_probability"),
            "mean_ssim_safe_probability": mean("ssim_safe_probability"),
            "mean_predicted_delta_clip": mean("predicted_delta_clip"),
            "mean_text_alignment_gain": mean("text_alignment_gain"),
            "mean_edge_cosine": mean("edge_cosine"),
            "mean_low_frequency_l1": mean("low_frequency_l1"),
            "mean_generation_latency_ms": mean("generation_latency_ms"),
            "mean_verification_latency_ms": mean("verification_latency_ms"),
            "training_data_overlap_count": sum(int(item["training_data_overlap"]) for item in items),
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
    if device.type != "cuda":
        raise RuntimeError("V6.5 evaluation requires CUDA")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = load_controlnet_policy(args.policy_checkpoint, device)
    verifier = load_generation_verifier(args.verifier_checkpoint, device)
    thresholds = load_verifier_thresholds(args.verifier_thresholds)
    text_guard = QualityAdaptivePolicy().to(device).eval()
    adapter = _load_adapter(args.adapter_checkpoint, device)
    policy_ids = set(getattr(policy, "policy_data_image_ids", set()))
    verifier_ids = set(getattr(verifier, "all_training_image_ids", set()))
    training_ids = policy_ids | verifier_ids
    if not training_ids and not args.allow_training_data_overlap:
        raise RuntimeError("Checkpoints lack training image ids; strict holdout cannot be verified")
    excluded_ids = set() if args.allow_training_data_overlap else training_ids
    print("Strict holdout: excluding %d policy/verifier image ids" % len(excluded_ids))
    print("Verifier thresholds: %s" % thresholds)

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
            probe_kwargs = {
                key: value for key, value in probe_kwargs.items() if key in supported
            }
            probe = v61(images, batch["texts"], **probe_kwargs)
            probe_captions = v61.decode_text_logits(probe["text_logits"])
            probe_accuracy = _text_accuracy(probe)
            probe_alignment = scorer.similarity(probe["image"], probe_captions)
            probe_features = receiver_features(
                probe, snr, image_text_alignment=probe_alignment
            )
            probe_policy = policy(probe_features)
            learned = probe_policy["text_utility"].clamp(0.0, 1.0)
            heuristic = text_guard(probe_features)["text_utility"].clamp(0.0, 1.0)
            if args.text_safety == "hybrid":
                recommended, safety = hybrid_text_utility(
                    probe_features,
                    learned,
                    heuristic,
                    snr,
                    probe_utility=args.probe_utility,
                    generation_probability=probe_policy["generation_probability"],
                )
            elif args.text_safety == "heuristic_max":
                recommended = torch.maximum(learned, heuristic)
                safety = {"channel_floor": recommended.new_full(recommended.shape, args.probe_utility)}
            elif args.text_safety == "probe_floor":
                recommended = torch.maximum(
                    learned, learned.new_full(learned.shape, args.probe_utility)
                )
                safety = {"channel_floor": recommended.new_full(recommended.shape, args.probe_utility)}
            else:
                recommended = learned
                safety = {"channel_floor": recommended.new_zeros(recommended.shape)}
            feedback = (
                recommended - float(args.probe_utility)
            ).abs() >= args.feedback_deadband
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
            adaptive_kwargs = {
                key: value for key, value in adaptive_kwargs.items() if key in supported
            }
            adaptive = v61(images, batch["texts"], **adaptive_kwargs)
            captions = v61.decode_text_logits(adaptive["text_logits"])
            adaptive_accuracy = _text_accuracy(adaptive)
            alignment = scorer.similarity(adaptive["image"], captions)
            protected_features = receiver_features(
                adaptive, snr, image_text_alignment=alignment
            )
            choose_protected, probe_text_quality, protected_text_quality = select_received_text(
                probe_features,
                protected_features,
                minimum_gain=args.text_selection_margin,
            )
            selected_captions = [
                captions[index] if bool(choose_protected[index]) else probe_captions[index]
                for index in range(len(captions))
            ]
            selected_alignment = scorer.similarity(adaptive["image"], selected_captions)
            route_features = receiver_features(
                adaptive, snr, image_text_alignment=selected_alignment
            )
            route = policy(route_features)
            selected_accuracy = torch.where(
                choose_protected, adaptive_accuracy, probe_accuracy
            )
            for index, image_id in enumerate(batch["image_ids"]):
                semantic_source = adaptive if bool(choose_protected[index]) else probe
                cached.append(
                    {
                        "image_id": int(image_id),
                        "snr": float(snr),
                        "original": images[index].detach().cpu(),
                        "regression": adaptive["image"][index].detach().cpu(),
                        "source_caption": batch["texts"][index],
                        "probe_caption": probe_captions[index],
                        "received_caption": selected_captions[index],
                        "protected_caption": captions[index],
                        "probe_text_accuracy": float(probe_accuracy[index]),
                        "protected_text_accuracy": float(adaptive_accuracy[index]),
                        "adaptive_text_accuracy": float(selected_accuracy[index]),
                        "protected_text_selected": bool(choose_protected[index]),
                        "probe_text_quality": float(probe_text_quality[index]),
                        "protected_text_quality": float(protected_text_quality[index]),
                        "generation_probability": float(route["generation_probability"][index]),
                        "strength": float(route["diffusion_strength"][index]),
                        "alpha": float(route["blend_alpha"][index]),
                        "detail_probability": float(route["detail_probability"][index]),
                        "feedback": bool(feedback[index]),
                        "guarded_text_utility": float(recommended[index]),
                        "channel_floor": float(safety["channel_floor"][index]),
                        "text_tokens": float(adaptive["allocated_text_tokens"][index]),
                        "text_repeat": float(adaptive["text_repetition_factor"][index]),
                        "route_features": route_features[index].detach().cpu(),
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
            print("Receiver decisions SNR=%g total=%d" % (snr, len(cached)))

    v61.to("cpu")
    scorer.to("cpu")
    policy.to("cpu")
    text_guard.to("cpu")
    del v61, scorer, policy, text_guard
    torch.cuda.empty_cache()
    any_invocation = any(
        float(item["generation_probability"]) >= args.route_threshold for item in cached
    )
    pipe = None
    if any_invocation:
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
    verifier.to(device).eval()
    adapter.to(device).eval()

    rows = []
    for index, item in enumerate(cached):
        original = item["original"].to(device)
        regression = item["regression"].to(device)
        route_probability = float(item["generation_probability"])
        invoked = route_probability >= args.route_threshold
        accepted = False
        raw = regression
        candidate = regression
        generation_latency_ms = 0.0
        verification_latency_ms = 0.0
        acceptance_probability = 0.0
        hallucination_probability = 0.0
        predicted_delta_clip = 0.0
        predicted_delta_psnr = 0.0
        predicted_delta_ssim = 0.0
        semantic_gain_probability = 0.0
        psnr_safe_probability = 0.0
        ssim_safe_probability = 0.0
        statistics = {
            "text_alignment_gain": regression.new_zeros(1),
            "edge_cosine": regression.new_ones(1),
            "low_frequency_l1": regression.new_zeros(1),
            "candidate_regression_clip_similarity": regression.new_ones(1),
        }
        check_values = {
            "acceptance_probability": False,
            "hallucination_risk": False,
            "text_gain": False,
            "edge_consistency": False,
            "low_frequency_consistency": False,
            "semantic_gain_probability": False,
            "psnr_safe_probability": False,
            "ssim_safe_probability": False,
        }
        if invoked:
            semantic_output = {
                key: value.to(device) for key, value in item["semantic_output"].items()
            }
            route_features = item["route_features"].to(device).unsqueeze(0)
            semantic_tokens = adapter(semantic_output, route_features)
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
            generation_latency_ms = 1000.0 * (time.perf_counter() - started)
            rgb = _fuse(regression, raw, "rgb", float(item["alpha"]))
            detail = _fuse(regression, raw, "detail", float(item["alpha"]))
            detail_probability = float(item["detail_probability"])
            candidate = (
                (1.0 - detail_probability) * rgb + detail_probability * detail
            ).clamp(0.0, 1.0)
            torch.cuda.synchronize(device)
            verification_started = time.perf_counter()
            prompt = str(item["received_caption"])
            regression_text = scorer.similarity(regression.unsqueeze(0), [prompt])
            candidate_text = scorer.similarity(candidate.unsqueeze(0), [prompt])
            candidate_regression_clip = scorer.image_similarity(
                candidate.unsqueeze(0), regression.unsqueeze(0)
            )
            raw_regression_clip = scorer.image_similarity(
                raw.unsqueeze(0), regression.unsqueeze(0)
            )
            evidence = verification_features(
                regression.unsqueeze(0),
                candidate.unsqueeze(0),
                raw.unsqueeze(0),
                candidate_regression_clip,
                raw_regression_clip,
                regression_text,
                candidate_text,
                item["strength"],
                item["alpha"],
                item["detail_probability"],
                item["generation_probability"],
            )
            prediction = verifier(route_features, evidence)
            accept_tensor, checks = verifier_acceptance(
                prediction, evidence, thresholds
            )
            torch.cuda.synchronize(device)
            verification_latency_ms = 1000.0 * (
                time.perf_counter() - verification_started
            )
            accepted = bool(accept_tensor[0])
            statistics = raw_verification_statistics(evidence)
            check_values = {
                key: bool(value[0]) for key, value in checks.items()
            }
            acceptance_probability = float(prediction["acceptance_probability"][0])
            hallucination_probability = float(prediction["hallucination_probability"][0])
            predicted_delta_clip = float(prediction["predicted_delta_clip"][0])
            predicted_delta_psnr = float(prediction["predicted_delta_psnr"][0])
            predicted_delta_ssim = float(prediction["predicted_delta_ssim"][0])
            semantic_gain_probability = float(
                prediction["semantic_gain_probability"][0]
            )
            psnr_safe_probability = float(prediction["psnr_safe_probability"][0])
            ssim_safe_probability = float(prediction["ssim_safe_probability"][0])
        selected = candidate if accepted else regression

        regression_metrics = batch_image_metrics(regression.unsqueeze(0), original.unsqueeze(0))
        candidate_metrics = batch_image_metrics(candidate.unsqueeze(0), original.unsqueeze(0))
        selected_metrics = batch_image_metrics(selected.unsqueeze(0), original.unsqueeze(0))
        regression_clip = scorer.image_similarity(regression.unsqueeze(0), original.unsqueeze(0))
        candidate_clip = scorer.image_similarity(candidate.unsqueeze(0), original.unsqueeze(0))
        selected_clip = scorer.image_similarity(selected.unsqueeze(0), original.unsqueeze(0))
        row = {
            "image_id": item["image_id"],
            "snr_db": item["snr"],
            "primary_invoked": int(invoked),
            "verified_accept": int(accepted),
            "selected_mode": "generation" if accepted else "regression",
            "generation_probability": route_probability,
            "acceptance_probability": acceptance_probability,
            "hallucination_probability": hallucination_probability,
            "predicted_delta_clip": predicted_delta_clip,
            "predicted_delta_psnr": predicted_delta_psnr,
            "predicted_delta_ssim": predicted_delta_ssim,
            "semantic_gain_probability": semantic_gain_probability,
            "psnr_safe_probability": psnr_safe_probability,
            "ssim_safe_probability": ssim_safe_probability,
            "diffusion_strength": item["strength"],
            "blend_alpha": item["alpha"],
            "detail_probability": item["detail_probability"],
            "text_alignment_gain": float(statistics["text_alignment_gain"][0]),
            "edge_cosine": float(statistics["edge_cosine"][0]),
            "low_frequency_l1": float(statistics["low_frequency_l1"][0]),
            "candidate_regression_clip_similarity": float(
                statistics["candidate_regression_clip_similarity"][0]
            ),
            "feedback_used": int(item["feedback"]),
            "guarded_text_utility": item["guarded_text_utility"],
            "protected_text_selected": int(item["protected_text_selected"]),
            "probe_text_quality": item["probe_text_quality"],
            "protected_text_quality": item["protected_text_quality"],
            "channel_text_floor": item["channel_floor"],
            "allocated_text_tokens": item["text_tokens"],
            "text_repetition_factor": item["text_repeat"],
            "probe_text_accuracy": item["probe_text_accuracy"],
            "protected_text_accuracy": item["protected_text_accuracy"],
            "adaptive_text_accuracy": item["adaptive_text_accuracy"],
            "probe_bleu2": sentence_bleu2(str(item["source_caption"]), str(item["probe_caption"])),
            "protected_bleu2": sentence_bleu2(str(item["source_caption"]), str(item["protected_caption"])),
            "adaptive_bleu2": sentence_bleu2(str(item["source_caption"]), str(item["received_caption"])),
            "regression_psnr": float(regression_metrics["psnr"][0]),
            "candidate_psnr": float(candidate_metrics["psnr"][0]),
            "selected_psnr": float(selected_metrics["psnr"][0]),
            "regression_ssim": float(regression_metrics["ssim"][0]),
            "candidate_ssim": float(candidate_metrics["ssim"][0]),
            "selected_ssim": float(selected_metrics["ssim"][0]),
            "regression_clip_image": float(regression_clip[0]),
            "candidate_clip_image": float(candidate_clip[0]),
            "selected_clip_image": float(selected_clip[0]),
            "generation_latency_ms": generation_latency_ms,
            "verification_latency_ms": verification_latency_ms,
            "training_data_overlap": int(int(item["image_id"]) in training_ids),
            "received_caption": item["received_caption"],
        }
        for name, value in check_values.items():
            row["check_" + name] = int(value)
        stem = "id%s_snr%s" % (item["image_id"], str(item["snr"]).replace(".", "p"))
        _panel(
            original.cpu(),
            regression.cpu(),
            raw.cpu(),
            candidate.cpu(),
            selected.cpu(),
            invoked,
            accepted,
            route_probability,
            acceptance_probability,
        ).save(output_dir / (stem + ".png"))
        row["panel"] = stem + ".png"
        rows.append(row)
        print(
            "[%d/%d] %s invoke=%d accept=%d route-p=%.2f verify-p=%.2f PSNR=%.2f->%.2f CLIP=%.3f->%.3f"
            % (
                index + 1,
                len(cached),
                stem,
                int(invoked),
                int(accepted),
                route_probability,
                acceptance_probability,
                row["regression_psnr"],
                row["selected_psnr"],
                row["regression_clip_image"],
                row["selected_clip_image"],
            )
        )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(rows, output_dir / "summary.csv")
    print("Done: %s" % output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate V6.5 generate-verify-reject receiver")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--verifier-checkpoint", required=True)
    parser.add_argument("--verifier-thresholds", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--diffusion-model-path", required=True)
    parser.add_argument("--controlnet-path", required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--vae-path")
    parser.add_argument("--text-model-path")
    parser.add_argument("--output-dir", default="experiment_results/v651_holdout_seed642")
    parser.add_argument("--config", default="configs/v61_bandwidth.yaml")
    parser.add_argument("--preset", default="original_4608")
    parser.add_argument("--channel-type", default="awgn", choices=["awgn", "rayleigh", "gamma_gamma", "fso"])
    parser.add_argument("--snr-list", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 20])
    parser.add_argument("--probe-utility", type=float, default=0.5)
    parser.add_argument("--feedback-deadband", type=float, default=0.10)
    parser.add_argument("--text-selection-margin", type=float, default=0.005)
    parser.add_argument("--route-threshold", type=float, default=0.70)
    parser.add_argument("--text-safety", choices=["hybrid", "heuristic_max", "probe_floor", "none"], default="hybrid")
    parser.add_argument("--control-scale", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=642)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-training-data-overlap", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
