"""Validate that the offline server can execute semantic-conditioned SD."""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys


def _require_file(root: Path, relative: str) -> None:
    target = root / relative
    if not target.is_file():
        raise FileNotFoundError("Required model file is missing: %s" % target)


def main(args: argparse.Namespace) -> None:
    import torch
    import diffusers
    import transformers
    from diffusers import StableDiffusionImg2ImgPipeline

    print("python=%s" % sys.version.replace("\n", " "))
    print("torch=%s cuda=%s" % (torch.__version__, torch.cuda.is_available()))
    print("diffusers=%s" % diffusers.__version__)
    print("transformers=%s" % transformers.__version__)

    parameters = set(inspect.signature(StableDiffusionImg2ImgPipeline.__call__).parameters)
    required = {"prompt_embeds", "negative_prompt_embeds"}
    missing_parameters = sorted(required - parameters)
    if missing_parameters:
        raise RuntimeError(
            "StableDiffusionImg2ImgPipeline is too old for the semantic adapter; "
            "missing call parameters: %s" % missing_parameters
        )
    print("semantic_prompt_api=PASS")

    sd_root = Path(args.diffusion_model_path).expanduser().resolve()
    clip_root = Path(args.clip_model_path).expanduser().resolve()
    vae_root = Path(args.vae_path).expanduser().resolve()
    for relative in (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/tokenizer_config.json",
        "text_encoder/config.json",
        "unet/config.json",
        "vae/config.json",
    ):
        _require_file(sd_root, relative)
    _require_file(clip_root, "config.json")
    _require_file(vae_root, "config.json")
    print("offline_model_layout=PASS")

    if args.load_pipeline:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            str(sd_root),
            torch_dtype=dtype,
            local_files_only=True,
            safety_checker=None,
            requires_safety_checker=False,
        )
        cross_attention_dim = int(pipe.unet.config.cross_attention_dim)
        if cross_attention_dim != 768:
            raise RuntimeError(
                "Expected SD-v1.5 cross_attention_dim=768, got %d"
                % cross_attention_dim
            )
        print("offline_pipeline_load=PASS cross_attention_dim=%d" % cross_attention_dim)
    print("V6.2 diffusion environment: PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check V6.2 offline diffusion environment")
    parser.add_argument(
        "--diffusion-model-path",
        default="pretrained/stable-diffusion-v1-5",
    )
    parser.add_argument(
        "--clip-model-path",
        default="pretrained/clip-vit-base-patch32",
    )
    parser.add_argument("--vae-path", default="pretrained/sd-vae-ft-mse")
    parser.add_argument("--load-pipeline", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
