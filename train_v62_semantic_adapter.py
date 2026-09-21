"""Train V6.2 semantic context tokens while keeping V6.1 and SD-v1.5 frozen.

This stage teaches the diffusion UNet to use the actually received common,
image-private and text-private semantics.  It replaces the weak prompt-only
img2img baseline without changing the 872 tensors in seed42.pth.
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionImg2ImgPipeline

from data import build_dataloader
from models.clip_semantics import CLIPSemanticSimilarity
from models.quality_adaptive_recovery import SemanticConditionAdapter, receiver_features
from utils.checkpoint import load_checkpoint, load_state_dict_flexible
from utils.config import load_config, prepare_config
from utils.experiment import create_model, seed_everything, select_device
from utils.v62_diffusion import add_semantic_context, prompt_embeddings


def _prepare_sd_images(images: torch.Tensor) -> torch.Tensor:
    images = F.interpolate(
        images.float(), size=(512, 512), mode="bicubic", align_corners=False
    ).clamp(0.0, 1.0)
    return images.mul(2.0).sub(1.0)


def _prediction_target(scheduler, clean_latents, noise, timesteps):
    prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "v_prediction":
        return scheduler.get_velocity(clean_latents, noise, timesteps)
    raise ValueError("Unsupported scheduler prediction_type: %s" % prediction_type)


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Semantic-adapter training requires CUDA")
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
    v61.eval().requires_grad_(False)
    print("Loaded V6.1 exact tensors=%d" % report["exact_count"])
    clip_path = args.clip_model_path or config["model"]["text_model_name"]
    scorer = CLIPSemanticSimilarity(clip_path).to(device).eval()

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.diffusion_model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.vae.to(device).eval().requires_grad_(False)
    pipe.text_encoder.to(device).eval().requires_grad_(False)
    pipe.unet.to(device).eval().requires_grad_(False)
    adapter = SemanticConditionAdapter(
        semantic_dim=int(config["model"].get("hidden_dim", 256)),
        cross_attention_dim=int(pipe.unet.config.cross_attention_dim),
        num_condition_tokens=args.condition_tokens,
    ).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    data = config["data"]
    loader = build_dataloader(
        image_root=data["train_image_root"],
        annotation_file=data["train_annotation_file"],
        image_size=int(config["model"]["image_size"]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=True,
        caption_mode=data.get("train_caption_mode", "random"),
        max_samples=args.train_samples,
        seed=args.seed,
    )
    supported = set(inspect.signature(v61.forward).parameters)
    global_step = 0
    adapter.train()
    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(loader):
            images = batch["images"].to(device, non_blocking=True)
            snr = random.choice(args.snr_list)
            utility = random.choice(args.utility_choices)
            kwargs = {
                "snr_db": snr,
                "disable_modality_dropout": True,
                "utility_weight": utility,
            }
            kwargs = {key: value for key, value in kwargs.items() if key in supported}
            with torch.no_grad():
                outputs = v61(images, batch["texts"], **kwargs)
                received_captions = v61.decode_text_logits(outputs["text_logits"])
                received_alignment = scorer.similarity(
                    outputs["image"], received_captions
                )
                features = receiver_features(
                    outputs,
                    snr,
                    image_text_alignment=received_alignment,
                )
                conditional, _ = prompt_embeddings(pipe, received_captions, device)
                sd_images = _prepare_sd_images(images).to(dtype=pipe.vae.dtype)
                clean_latents = pipe.vae.encode(sd_images).latent_dist.sample()
                scaling = float(getattr(pipe.vae.config, "scaling_factor", 0.18215))
                clean_latents = clean_latents * scaling
                noise = torch.randn_like(clean_latents)
                timesteps = torch.randint(
                    0,
                    int(pipe.scheduler.config.num_train_timesteps),
                    (clean_latents.size(0),),
                    device=device,
                    dtype=torch.long,
                )
                noisy_latents = pipe.scheduler.add_noise(clean_latents, noise, timesteps)
                target = _prediction_target(
                    pipe.scheduler, clean_latents, noise, timesteps
                )

            semantic_tokens = adapter(outputs, features)
            conditional, _ = add_semantic_context(
                conditional, torch.zeros_like(conditional), semantic_tokens
            )
            prediction = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=conditional.to(dtype=pipe.unet.dtype),
            ).sample
            denoising_loss = F.mse_loss(prediction.float(), target.float())
            token_regularizer = semantic_tokens.float().square().mean()
            loss = denoising_loss + args.regularization * token_regularizer
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.max_grad_norm)
            optimizer.step()
            global_step += 1

            if global_step % args.log_interval == 0:
                print(
                    "epoch=%d step=%d loss=%.6f denoise=%.6f gate=%.3f snr=%g text_utility=%.2f"
                    % (
                        epoch + 1,
                        global_step,
                        float(loss.detach()),
                        float(denoising_loss.detach()),
                        float(torch.sigmoid(adapter.gate_logit).detach()),
                        snr,
                        utility,
                    )
                )
            if args.max_steps and global_step >= args.max_steps:
                break
        checkpoint = {
            "adapter": adapter.state_dict(),
            "adapter_config": {
                "semantic_dim": int(config["model"].get("hidden_dim", 256)),
                "cross_attention_dim": int(pipe.unet.config.cross_attention_dim),
                "num_condition_tokens": args.condition_tokens,
            },
            "epoch": epoch + 1,
            "global_step": global_step,
        }
        torch.save(checkpoint, output_dir / "semantic_adapter_latest.pth")
        print("Saved %s" % (output_dir / "semantic_adapter_latest.pth"))
        if args.max_steps and global_step >= args.max_steps:
            break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train V6.2 received-semantic diffusion adapter")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--diffusion-model-path", required=True)
    parser.add_argument("--output-dir", default="V6.2_model_weights/adapter")
    parser.add_argument("--config", default="configs/v61_bandwidth.yaml")
    parser.add_argument("--vae-path")
    parser.add_argument("--text-model-path")
    parser.add_argument("--clip-model-path")
    parser.add_argument("--preset", default="original_4608")
    parser.add_argument("--channel-type", default="awgn", choices=["awgn", "rayleigh", "gamma_gamma", "fso"])
    parser.add_argument("--snr-list", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 20])
    parser.add_argument("--utility-choices", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--condition-tokens", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
