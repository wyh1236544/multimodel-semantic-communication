"""Collect source-supervised, receiver-only evidence for the V6.5 verifier.

The source image is used only after a candidate has been produced, to create
delta-PSNR, delta-SSIM and delta-CLIP labels.  Every feature stored for the
deployed verifier is available at the receiver without the source image.
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
from models.controlnet_fusion_policy import load_controlnet_policy
from models.quality_adaptive_recovery import (
    FEATURE_NAMES,
    QualityAdaptivePolicy,
    receiver_features,
)
from utils.checkpoint import load_checkpoint, load_state_dict_flexible
from utils.config import load_config, prepare_config
from utils.experiment import create_model, seed_everything, select_device
from utils.metrics import batch_image_metrics
from utils.v65_selective import (
    VERIFIER_FEATURE_NAMES,
    hybrid_text_utility,
    raw_verification_statistics,
    select_received_text,
    verification_features,
)
from v63_controlnet_fusion_pilot import _fuse, _generate_controlnet, _load_adapter


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("V6.5 verifier data collection requires CUDA")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = load_controlnet_policy(args.policy_checkpoint, device)
    policy_data_ids = set(getattr(policy, "policy_data_image_ids", set()))
    text_guard = QualityAdaptivePolicy().to(device).eval()

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
            probe_alignment = scorer.similarity(probe["image"], probe_captions)
            probe_features = receiver_features(
                probe, snr, image_text_alignment=probe_alignment
            )
            probe_policy = policy(probe_features)
            heuristic = text_guard(probe_features)["text_utility"]
            recommended, safety = hybrid_text_utility(
                probe_features,
                probe_policy["text_utility"],
                heuristic,
                snr,
                probe_utility=args.probe_utility,
                generation_probability=probe_policy["generation_probability"],
            )
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
            alignment = scorer.similarity(adaptive["image"], captions)
            route_features = receiver_features(
                adaptive, snr, image_text_alignment=alignment
            )
            route = policy(route_features)
            choose_protected, probe_text_quality, protected_text_quality = select_received_text(
                probe_features, route_features, minimum_gain=args.text_selection_margin
            )
            for index, image_id in enumerate(batch["image_ids"]):
                probability = float(route["generation_probability"][index])
                if probability < args.candidate_probability_floor:
                    continue
                semantic_source = adaptive if bool(choose_protected[index]) else probe
                cached.append(
                    {
                        "image_id": int(image_id),
                        "snr": float(snr),
                        "original": images[index].detach().cpu(),
                        "regression": adaptive["image"][index].detach().cpu(),
                        "received_caption": (
                            captions[index]
                            if bool(choose_protected[index])
                            else probe_captions[index]
                        ),
                        "protected_text_selected": bool(choose_protected[index]),
                        "probe_text_quality": float(probe_text_quality[index]),
                        "protected_text_quality": float(protected_text_quality[index]),
                        "route_features": route_features[index].detach().cpu(),
                        "generation_probability": probability,
                        "strength": float(route["diffusion_strength"][index]),
                        "alpha": float(route["blend_alpha"][index]),
                        "detail_probability": float(route["detail_probability"][index]),
                        "text_utility": float(actual_utility[index]),
                        "channel_floor": float(safety["channel_floor"][index]),
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
            print("Cached SNR=%g candidates=%d" % (snr, len(cached)))
    if not cached:
        raise RuntimeError(
            "No candidates met --candidate-probability-floor; lower the floor"
        )

    v61.to("cpu")
    scorer.to("cpu")
    policy.to("cpu")
    text_guard.to("cpu")
    del v61, scorer, policy, text_guard
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
        "version": "V6.5.1",
        "snr_list": [float(value) for value in args.snr_list],
        "candidate_probability_floor": float(args.candidate_probability_floor),
        "min_oracle_clip_gain": float(args.min_oracle_clip_gain),
        "max_oracle_psnr_drop": float(args.max_oracle_psnr_drop),
        "max_oracle_ssim_drop": float(args.max_oracle_ssim_drop),
        "channel_type": args.channel_type,
        "image_ids": sorted({int(item["image_id"]) for item in cached}),
        "v64_policy_data_image_ids": sorted(
            int(value) for value in policy_data_ids
        ),
    }
    for index, item in enumerate(cached):
        original = item["original"].to(device)
        regression = item["regression"].to(device)
        route_features = item["route_features"].to(device).unsqueeze(0)
        semantic_output = {
            key: value.to(device) for key, value in item["semantic_output"].items()
        }
        semantic_tokens = adapter(semantic_output, route_features)
        generator = torch.Generator(device=device).manual_seed(
            args.seed * 100000 + index * 10 + 1
        )
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
        rgb = _fuse(regression, raw, "rgb", float(item["alpha"]))
        detail = _fuse(regression, raw, "detail", float(item["alpha"]))
        detail_probability = float(item["detail_probability"])
        candidate = (
            (1.0 - detail_probability) * rgb + detail_probability * detail
        ).clamp(0.0, 1.0)

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
        regression_metrics = batch_image_metrics(
            regression.unsqueeze(0), original.unsqueeze(0)
        )
        candidate_metrics = batch_image_metrics(
            candidate.unsqueeze(0), original.unsqueeze(0)
        )
        regression_clip = scorer.image_similarity(
            regression.unsqueeze(0), original.unsqueeze(0)
        )
        candidate_clip = scorer.image_similarity(
            candidate.unsqueeze(0), original.unsqueeze(0)
        )
        delta_psnr = float(candidate_metrics["psnr"][0] - regression_metrics["psnr"][0])
        delta_ssim = float(candidate_metrics["ssim"][0] - regression_metrics["ssim"][0])
        delta_clip = float(candidate_clip[0] - regression_clip[0])
        fidelity_failure = (
            delta_psnr < -args.max_oracle_psnr_drop
            or delta_ssim < -args.max_oracle_ssim_drop
        )
        accept_target = float(
            delta_clip >= args.min_oracle_clip_gain and not fidelity_failure
        )
        statistics = raw_verification_statistics(evidence)
        records.append(
            {
                "route_features": route_features[0].detach().cpu(),
                "verifier_features": evidence[0].detach().cpu(),
                "accept_target": accept_target,
                "hallucination_target": float(fidelity_failure),
                "delta_clip_target": delta_clip,
                "delta_psnr_target": delta_psnr,
                "delta_ssim_target": delta_ssim,
                "image_id": int(item["image_id"]),
                "snr": float(item["snr"]),
            }
        )
        rows.append(
            {
                "image_id": item["image_id"],
                "snr_db": item["snr"],
                "accept_target": int(accept_target),
                "hallucination_target": int(fidelity_failure),
                "primary_generation_probability": item["generation_probability"],
                "diffusion_strength": item["strength"],
                "blend_alpha": item["alpha"],
                "detail_probability": item["detail_probability"],
                "text_utility": item["text_utility"],
                "protected_text_selected": int(item["protected_text_selected"]),
                "probe_text_quality": item["probe_text_quality"],
                "protected_text_quality": item["protected_text_quality"],
                "delta_clip": delta_clip,
                "delta_psnr": delta_psnr,
                "delta_ssim": delta_ssim,
                "candidate_regression_clip": float(candidate_regression_clip[0]),
                "text_alignment_gain": float(statistics["text_alignment_gain"][0]),
                "edge_cosine": float(statistics["edge_cosine"][0]),
                "low_frequency_l1": float(statistics["low_frequency_l1"][0]),
                "received_caption": prompt,
            }
        )
        print(
            "[%d/%d] id=%s snr=%g accept=%d dCLIP=%+.4f dPSNR=%+.3f dSSIM=%+.4f"
            % (
                index + 1,
                len(cached),
                item["image_id"],
                item["snr"],
                int(accept_target),
                delta_clip,
                delta_psnr,
                delta_ssim,
            )
        )
        if (index + 1) % args.save_interval == 0:
            torch.save(
                {
                    "records": records,
                    "route_feature_names": FEATURE_NAMES,
                    "verifier_feature_names": VERIFIER_FEATURE_NAMES,
                    "metadata": metadata,
                },
                output_dir / "verifier_dataset.partial.pth",
            )

    payload = {
        "records": records,
        "route_feature_names": FEATURE_NAMES,
        "verifier_feature_names": VERIFIER_FEATURE_NAMES,
        "metadata": metadata,
    }
    torch.save(payload, output_dir / "verifier_dataset.pth")
    with (output_dir / "oracle_labels.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Done: %s (%d records)" % (output_dir / "verifier_dataset.pth", len(records)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect V6.5 verifier supervision")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--diffusion-model-path", required=True)
    parser.add_argument("--controlnet-path", required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--vae-path")
    parser.add_argument("--text-model-path")
    parser.add_argument("--output-dir", default="experiment_results/v65_verifier_data")
    parser.add_argument("--config", default="configs/v61_bandwidth.yaml")
    parser.add_argument("--preset", default="original_4608")
    parser.add_argument("--channel-type", default="awgn", choices=["awgn", "rayleigh", "gamma_gamma", "fso"])
    parser.add_argument("--snr-list", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 20])
    parser.add_argument("--probe-utility", type=float, default=0.5)
    parser.add_argument("--feedback-deadband", type=float, default=0.10)
    parser.add_argument("--text-selection-margin", type=float, default=0.005)
    parser.add_argument("--candidate-probability-floor", type=float, default=0.30)
    parser.add_argument("--min-oracle-clip-gain", type=float, default=0.003)
    parser.add_argument("--max-oracle-psnr-drop", type=float, default=0.05)
    parser.add_argument("--max-oracle-ssim-drop", type=float, default=0.004)
    parser.add_argument("--control-scale", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=342)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
