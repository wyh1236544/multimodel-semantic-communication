"""Offline environment check for the V6.3 ControlNet Tile pilot."""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys

import torch
import diffusers
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline


def _model_layout(path: str) -> None:
    root = Path(path)
    if not root.is_dir():
        raise RuntimeError("ControlNet directory does not exist: %s" % root)
    if not (root / "config.json").is_file():
        raise RuntimeError("Missing ControlNet config.json: %s" % root)
    weights = [
        root / "diffusion_pytorch_model.bin",
        root / "diffusion_pytorch_model.safetensors",
    ]
    if not any(path.is_file() for path in weights):
        raise RuntimeError(
            "Missing diffusion_pytorch_model.bin or .safetensors: %s" % root
        )


def main(args: argparse.Namespace) -> None:
    print("python=%s" % sys.version.replace("\n", " "))
    print("torch=%s cuda=%s" % (torch.__version__, torch.cuda.is_available()))
    print("diffusers=%s" % diffusers.__version__)
    parameters = set(
        inspect.signature(StableDiffusionControlNetImg2ImgPipeline.__call__).parameters
    )
    required = {
        "image",
        "control_image",
        "prompt_embeds",
        "negative_prompt_embeds",
        "controlnet_conditioning_scale",
    }
    missing = sorted(required - parameters)
    if missing:
        raise RuntimeError("ControlNet img2img API missing: %s" % missing)
    print("controlnet_img2img_api=PASS")
    _model_layout(args.controlnet_path)
    print("offline_controlnet_layout=PASS")
    if args.load_model:
        model = ControlNetModel.from_pretrained(
            args.controlnet_path,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        cross_attention_dim = int(model.config.cross_attention_dim)
        del model
        print("offline_controlnet_load=PASS cross_attention_dim=%d" % cross_attention_dim)
    print("V6.3 ControlNet environment: PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlnet-path", required=True)
    parser.add_argument("--load-model", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
