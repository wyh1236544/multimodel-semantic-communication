from typing import Dict, List, Optional, Union

import torch
from torch import Tensor, nn

from .channel import SemanticChannel
from .fusion import build_fusion
from .image_decoder import ImageDecoder
from .image_encoder import ImageEncoder
from .semantic_decoder import SemanticDecoder
from .semantic_encoder import SemanticEncoder
from .text_decoder import TextDecoder
from .text_encoder import TextEncoder
from .vae_tokenizer import StableDiffusionVAEWrapper, VAELatentTokenizer


class MultimodalSemanticCommunicationModel(nn.Module):
    """End-to-end image + text -> AWGN channel -> reconstructed image model."""

    def __init__(self, config: Dict):
        super().__init__()
        image_size = int(config["image_size"])
        self.image_backbone = str(config.get("image_backbone", "cnn")).lower()
        hidden_dim = int(config["hidden_dim"])
        latent_dim = int(config["latent_dim"])
        latent_downsample_factor = int(config.get("latent_downsample_factor", 2))
        num_heads = int(config.get("transformer_heads", 4))
        num_layers = int(config.get("transformer_layers", 2))

        if latent_downsample_factor not in {1, 2}:
            raise ValueError("latent_downsample_factor must be 1 or 2")
        if self.image_backbone == "cnn":
            backbone_downsample_factor = ImageEncoder.downsample_factor
        elif self.image_backbone == "vae_latent":
            backbone_downsample_factor = int(config.get("vae_downsample_factor", 8))
        else:
            raise ValueError("image_backbone must be 'cnn' or 'vae_latent'")
        if image_size % backbone_downsample_factor != 0:
            raise ValueError(
                f"image_size must be divisible by {backbone_downsample_factor}"
            )

        self.feature_size = image_size // backbone_downsample_factor
        self.num_image_tokens = self.feature_size**2
        if self.feature_size % latent_downsample_factor != 0:
            raise ValueError(
                "Encoded feature size must be divisible by latent_downsample_factor"
            )
        self.latent_size = self.feature_size // latent_downsample_factor
        self.num_latent_tokens = self.latent_size**2
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_text_decoder = bool(config.get("use_text_decoder", False))
        self.fusion_type = str(config.get("fusion_type", "concat_mlp")).lower()
        self.text_encoder_output = str(
            config.get("text_encoder_output", "pooled")
        ).lower()
        self.text_dropout = float(config.get("text_dropout", 0.0))
        if not 0.0 <= self.text_dropout < 1.0:
            raise ValueError("text_dropout must be in [0, 1)")

        self.image_encoder = None
        self.image_decoder = None
        self.vae_wrapper = None
        self.vae_tokenizer = None
        if self.image_backbone == "cnn":
            self.image_encoder = ImageEncoder(
                in_channels=int(config.get("image_channels", 3)),
                hidden_dim=hidden_dim,
            )
            self.image_decoder = ImageDecoder(
                hidden_dim=hidden_dim,
                output_channels=int(config.get("image_channels", 3)),
                feature_size=self.feature_size,
            )
        else:
            self.vae_wrapper = StableDiffusionVAEWrapper(
                model_name_or_path=config["vae_model_name_or_path"],
                freeze=bool(config.get("freeze_vae", True)),
                scaling_factor=config.get("vae_scaling_factor", "auto"),
                use_fp16=bool(config.get("vae_use_fp16", False)),
                detach_latents=bool(config.get("detach_vae_latents", False)),
            )
            self.vae_tokenizer = VAELatentTokenizer(
                latent_channels=int(config.get("vae_latent_channels", 4)),
                hidden_dim=hidden_dim,
            )
        self.text_encoder = TextEncoder(
            model_name=config["text_model_name"],
            hidden_dim=hidden_dim,
            encoder_type=config.get("text_encoder_type", "clip"),
            freeze=bool(config.get("freeze_text_encoder", True)),
            max_length=int(config.get("text_max_length", 32)),
            output_type=self.text_encoder_output,
        )
        self.null_text_token = None
        if self.fusion_type == "cross_attention":
            self.null_text_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.normal_(self.null_text_token, std=0.02)
        self.fusion = build_fusion(
            self.fusion_type,
            hidden_dim,
            int(config.get("fusion_heads", num_heads)),
            int(config.get("fusion_layers", 1)),
        )
        self.semantic_encoder = SemanticEncoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            feature_size=self.feature_size,
            downsample_factor=latent_downsample_factor,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        self.channel = SemanticChannel()
        self.semantic_decoder = SemanticDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            feature_size=self.feature_size,
            downsample_factor=latent_downsample_factor,
            num_layers=max(1, num_layers // 2),
            num_heads=num_heads,
        )
        self.text_decoder = None
        if self.use_text_decoder:
            self.text_decoder = TextDecoder(
                hidden_dim=hidden_dim,
                vocab_size=self.text_encoder.vocab_size,
                max_length=int(config.get("text_max_length", 32)),
                num_heads=num_heads,
                num_layers=num_layers,
            )

    def forward(
        self,
        images: Tensor,
        texts: Union[List[str], str],
        snr_db: Union[float, int, Tensor],
        use_text: bool = True,
        channel_noise: Optional[Tensor] = None,
        vae_latents_override: Optional[Tensor] = None,
        decode_image: bool = True,
    ) -> Dict[str, Tensor]:
        vae_latents = None
        if self.image_backbone == "cnn":
            image_tokens = self.image_encoder(images)
        else:
            vae_latents = (
                vae_latents_override
                if vae_latents_override is not None
                else self.vae_wrapper.encode_images(images)
            )
            if vae_latents.shape[-2:] != (self.feature_size, self.feature_size):
                raise ValueError(
                    "VAE latent spatial size does not match image_size / "
                    f"vae_downsample_factor: expected {(self.feature_size, self.feature_size)}, "
                    f"received {tuple(vae_latents.shape[-2:])}"
                )
            image_tokens = self.vae_tokenizer.latents_to_tokens(vae_latents)
        if use_text:
            text_outputs = self.text_encoder(texts, images.device)
            pooled_text = text_outputs["pooled"]
            text_tokens = text_outputs["tokens"]
            text_attention_mask = text_outputs["attention_mask"]
        else:
            text_length = self.text_encoder.max_length
            pooled_text = image_tokens.new_zeros(images.size(0), self.hidden_dim)
            null_token = (
                self.null_text_token
                if self.null_text_token is not None
                else image_tokens.new_zeros(1, 1, self.hidden_dim)
            ).to(device=image_tokens.device, dtype=image_tokens.dtype)
            text_tokens = null_token.expand(
                images.size(0),
                text_length,
                -1,
            )
            text_attention_mask = torch.zeros(
                images.size(0),
                text_length,
                dtype=torch.long,
                device=images.device,
            )
            text_attention_mask[:, 0] = 1

        if not use_text:
            drop_samples = torch.ones(
                images.size(0),
                dtype=torch.bool,
                device=images.device,
            )
        elif self.training and self.text_dropout > 0:
            drop_samples = (
                torch.rand(images.size(0), device=images.device)
                < self.text_dropout
            )
        else:
            drop_samples = torch.zeros(
                images.size(0),
                dtype=torch.bool,
                device=images.device,
            )

        if drop_samples.any():
            pooled_text = pooled_text.clone()
            text_tokens = text_tokens.clone()
            text_attention_mask = text_attention_mask.clone()
            pooled_text[drop_samples] = 0
            null_token = (
                self.null_text_token
                if self.null_text_token is not None
                else image_tokens.new_zeros(1, 1, self.hidden_dim)
            ).to(device=text_tokens.device, dtype=text_tokens.dtype)
            text_tokens[drop_samples] = null_token.expand(
                int(drop_samples.sum().item()),
                text_tokens.size(1),
                -1,
            )
            text_attention_mask[drop_samples] = 0
            text_attention_mask[drop_samples, 0] = 1

        if self.fusion_type == "cross_attention":
            fusion_text = (
                text_tokens
                if self.text_encoder_output == "token"
                else pooled_text.unsqueeze(1)
            )
            fusion_mask = (
                text_attention_mask
                if self.text_encoder_output == "token"
                else text_attention_mask.new_ones(images.size(0), 1)
            )
            fused_tokens = self.fusion(
                image_tokens,
                fusion_text,
                fusion_mask,
            )
        else:
            fused_tokens = self.fusion(image_tokens, pooled_text)
        latent = self.semantic_encoder(fused_tokens)
        noisy_latent = self.channel(
            latent,
            snr_db,
            standard_noise=channel_noise,
        )
        decoded_tokens = self.semantic_decoder(noisy_latent)
        recovered_vae_latents = None
        if self.image_backbone == "cnn":
            if not decode_image:
                raise ValueError("decode_image=False is only supported by vae_latent")
            reconstructed_image = self.image_decoder(decoded_tokens)
        else:
            recovered_vae_latents = self.vae_tokenizer.tokens_to_latents(
                decoded_tokens,
                spatial_size=(self.feature_size, self.feature_size),
            )
            reconstructed_image = None
            if decode_image:
                reconstructed_image = self.vae_wrapper.decode_latents(
                    recovered_vae_latents
                )

        outputs = {
            "latent": latent,
            "latent_noisy": noisy_latent,
            "noisy_latent": noisy_latent,
            "semantic_tokens": decoded_tokens,
            "text_assistance_used": latent.new_tensor(float(use_text)),
            "text_dropout_fraction": drop_samples.float().mean(),
        }
        if reconstructed_image is not None:
            outputs["image"] = reconstructed_image
        if self.text_decoder is not None:
            outputs["text_logits"] = self.text_decoder(decoded_tokens)
        if vae_latents is not None and recovered_vae_latents is not None:
            outputs["vae_latents"] = vae_latents
            outputs["recovered_vae_latents"] = recovered_vae_latents
        return outputs

    def num_channel_uses(self) -> int:
        """Return real-valued channel symbols transmitted per image."""
        return self.num_latent_tokens * self.latent_dim

    def direct_vae_reconstruction(self, images: Tensor) -> Tensor:
        if self.image_backbone != "vae_latent":
            raise RuntimeError("Direct VAE reconstruction requires vae_latent backbone")
        latents = self.vae_wrapper.encode_images(images)
        return self.vae_wrapper.decode_latents(latents)
