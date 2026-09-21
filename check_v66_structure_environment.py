"""Architecture and offline-model checks for the V6.6 structure pilot."""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys

import diffusers
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

from utils.v66_structure import (
    binary_structure_map,
    tolerant_structure_f1,
    transmit_binary_structure,
)


def _model_layout(path: str) -> None:
    root = Path(path)
    if not root.is_dir():
        raise RuntimeError("Canny ControlNet directory does not exist: %s" % root)
    if not (root / "config.json").is_file():
        raise RuntimeError("Missing Canny ControlNet config.json: %s" % root)
    weights = [
        root / "diffusion_pytorch_model.bin",
        root / "diffusion_pytorch_model.safetensors",
    ]
    if not any(value.is_file() for value in weights):
        raise RuntimeError("Missing ControlNet weights under %s" % root)


def main(args: argparse.Namespace) -> None:
    print("python=%s" % sys.version.replace("\n", " "))
    print("torch=%s cuda=%s" % (torch.__version__, torch.cuda.is_available()))
    print("diffusers=%s" % diffusers.__version__)
    synthetic = torch.zeros(1, 3, 128, 128)
    synthetic[:, :, 24:104, 40:88] = 1.0
    structure = binary_structure_map(synthetic, side=64, edge_quantile=0.88)
    if tuple(structure.shape) != (1, 1, 64, 64):
        raise RuntimeError("Unexpected structure shape: %s" % (tuple(structure.shape),))
    decoded, stats = transmit_binary_structure(
        structure,
        snr_db=20.0,
        repetition=1,
        generator=torch.Generator().manual_seed(66),
    )
    f1 = float(tolerant_structure_f1(decoded, structure)[0])
    if not 0.0 <= stats["structure_ber"] <= 1.0 or f1 < 0.95:
        raise RuntimeError("Structure channel smoke test failed: %s f1=%f" % (stats, f1))
    print(
        "structure_side_channel=PASS shape=%s source_bits=%d ber=%.6f f1=%.4f"
        % (tuple(structure.shape), int(stats["structure_source_bits"]), stats["structure_ber"], f1)
    )
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
    _model_layout(args.canny_controlnet_path)
    print("offline_canny_controlnet_layout=PASS")
    if args.load_model:
        model = ControlNetModel.from_pretrained(
            args.canny_controlnet_path,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        cross_attention_dim = int(model.config.cross_attention_dim)
        del model
        print("offline_canny_controlnet_load=PASS cross_attention_dim=%d" % cross_attention_dim)
    print("V6.6 structure environment: PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canny-controlnet-path", required=True)
    parser.add_argument("--load-model", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
