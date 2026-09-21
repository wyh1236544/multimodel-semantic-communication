from typing import Optional, Tuple, Union

import torch
from diffusers import AutoencoderKL
from torch import Tensor, nn


class StableDiffusionVAEWrapper(nn.Module):
    """Frozen Stable Diffusion VAE with [0, 1] image range handling."""

    def __init__(
        self,
        model_name_or_path: str,
        freeze: bool = True,
        scaling_factor: Union[str, float] = "auto",
        use_fp16: bool = False,
        detach_latents: bool = False,
    ):
        super().__init__()
        load_dtype = torch.float16 if use_fp16 else torch.float32
        self.vae = AutoencoderKL.from_pretrained(
            model_name_or_path,
            torch_dtype=load_dtype,
        )
        self.freeze = freeze
        self.use_fp16 = use_fp16
        self.detach_latents = detach_latents
        self.scaling_factor = (
            float(getattr(self.vae.config, "scaling_factor", 0.18215))
            if scaling_factor == "auto"
            else float(scaling_factor)
        )

        if freeze:
            self.vae.requires_grad_(False)
            self.vae.eval()

    def train(self, mode: bool = True) -> "StableDiffusionVAEWrapper":
        super().train(mode)
        if self.freeze:
            self.vae.eval()
        return self

    def _vae_dtype(self) -> torch.dtype:
        return next(self.vae.parameters()).dtype

    def encode_images(self, images: Tensor) -> Tensor:
        """Encode [B, 3, H, W] images in [0, 1] to scaled VAE latents."""
        image_for_vae = images.to(dtype=self._vae_dtype()) * 2.0 - 1.0
        latent_dist = self.vae.encode(image_for_vae).latent_dist
        latents = latent_dist.sample() * self.scaling_factor
        if self.detach_latents:
            latents = latents.detach()
        return latents

    def decode_latents(self, latents: Tensor) -> Tensor:
        """Decode scaled VAE latents to [B, 3, H, W] images in [0, 1].

        This method intentionally does not use torch.no_grad(): even with frozen VAE
        weights, image loss must backpropagate through the decoder to channel latents.
        """
        latents = latents.to(dtype=self._vae_dtype())
        decoded = self.vae.decode(latents / self.scaling_factor).sample
        return ((decoded.float() + 1.0) / 2.0).clamp(0.0, 1.0)

    def forward(self, images: Tensor) -> Tensor:
        return self.decode_latents(self.encode_images(images))


class VAELatentTokenizer(nn.Module):
    """Convert [B, C, H, W] VAE latents to tokens and back."""

    def __init__(self, latent_channels: int, hidden_dim: int):
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_dim = hidden_dim
        self.latent_proj = nn.Conv2d(latent_channels, hidden_dim, kernel_size=1)
        self.latent_unproj = nn.Conv2d(hidden_dim, latent_channels, kernel_size=1)

    def latents_to_tokens(self, latents: Tensor) -> Tensor:
        """[B, C, H, W] -> [B, H*W, D]."""
        features = self.latent_proj(latents.float())
        return features.flatten(2).transpose(1, 2)

    def tokens_to_latents(
        self,
        tokens: Tensor,
        spatial_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        """[B, H*W, D] -> [B, C, H, W]."""
        batch_size, num_tokens, hidden_dim = tokens.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected token dimension {self.hidden_dim}, received {hidden_dim}"
            )
        if spatial_size is None:
            side = int(num_tokens**0.5)
            spatial_size = (side, side)
        height, width = spatial_size
        if height * width != num_tokens:
            raise ValueError(
                f"Token count {num_tokens} does not match spatial size {spatial_size}"
            )
        features = tokens.transpose(1, 2).reshape(
            batch_size, hidden_dim, height, width
        )
        return self.latent_unproj(features)

    def forward(self, latents: Tensor) -> Tensor:
        return self.latents_to_tokens(latents)
