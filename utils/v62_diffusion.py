"""Version-tolerant Stable-Diffusion helpers used by the V6.2 scripts."""
from __future__ import annotations

import inspect
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch import Tensor


def tensor_to_pil(image: Tensor, size: Optional[int] = None) -> Image.Image:
    array = (
        image.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)
    result = Image.fromarray(array, "RGB")
    if size is not None:
        result = result.resize((size, size), Image.Resampling.BICUBIC)
    return result


def pil_to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


@torch.no_grad()
def prompt_embeddings(pipe, prompts: List[str], device: torch.device) -> Tuple[Tensor, Tensor]:
    """Return conditional/unconditional SD text embeddings without private APIs."""
    tokenizer = pipe.tokenizer
    max_length = int(tokenizer.model_max_length)
    cleaned = [prompt if str(prompt).strip() else "a natural photograph" for prompt in prompts]
    conditional_ids = tokenizer(
        cleaned,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    negative_ids = tokenizer(
        [""] * len(cleaned),
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    conditional = pipe.text_encoder(conditional_ids)[0]
    negative = pipe.text_encoder(negative_ids)[0]
    return conditional, negative


def add_semantic_context(
    conditional: Tensor,
    negative: Tensor,
    semantic_tokens: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    if semantic_tokens is None:
        return conditional, negative
    semantic_tokens = semantic_tokens.to(
        device=conditional.device, dtype=conditional.dtype
    )
    conditional = torch.cat([conditional, semantic_tokens], dim=1)
    negative = torch.cat([negative, torch.zeros_like(semantic_tokens)], dim=1)
    return conditional, negative


@torch.no_grad()
def generate_img2img(
    pipe,
    regression: Tensor,
    prompt: str,
    strength: float,
    steps: int,
    guidance_scale: float,
    generator: torch.Generator,
    output_height: int,
    output_width: int,
    semantic_tokens: Optional[Tensor] = None,
) -> Tensor:
    """Generate one candidate and resize it back to the communication size."""
    call_parameters = set(inspect.signature(pipe.__call__).parameters)
    kwargs = {
        "image": tensor_to_pil(regression, size=512),
        "strength": float(strength),
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "generator": generator,
    }
    if semantic_tokens is not None:
        required = {"prompt_embeds", "negative_prompt_embeds"}
        if not required.issubset(call_parameters):
            version = "unknown"
            try:
                import diffusers

                version = str(diffusers.__version__)
            except Exception:
                pass
            raise RuntimeError(
                "The semantic adapter is loaded, but this diffusers pipeline "
                "does not expose prompt_embeds/negative_prompt_embeds "
                "(diffusers=%s). Refusing to silently fall back to prompt-only "
                "generation. Install a compatible diffusers release or run "
                "without --adapter-checkpoint for the explicit baseline."
                % version
            )
        conditional, negative = prompt_embeddings(pipe, [prompt], regression.device)
        conditional, negative = add_semantic_context(
            conditional, negative, semantic_tokens
        )
        kwargs["prompt_embeds"] = conditional
        kwargs["negative_prompt_embeds"] = negative
    else:
        kwargs["prompt"] = prompt if prompt.strip() else "a natural photograph"
    result = pipe(**kwargs).images[0]
    result = result.resize((output_width, output_height), Image.Resampling.LANCZOS)
    return pil_to_tensor(result)
