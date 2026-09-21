from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
from torch import Tensor, nn

from models.channel import SemanticChannel
from models.deepjscc_baseline import DeepJSCCBaseline
from models.shared_bottleneck import QueryTokenCompressor
from models.text_encoder import TextEncoder
from models.text_reconstruction_head import TextReconstructionHead


class IndependentTextJSCC(nn.Module):
    """Text-only JSCC branch with an explicit fixed channel budget."""

    def __init__(self, config: Dict):
        super().__init__()
        self.hidden_dim = int(config.get("hidden_dim", 256))
        self.latent_dim = int(config.get("text_latent_dim", 16))
        self.num_channel_tokens = int(config.get("text_channel_tokens", 48))
        self.text_max_length = int(config.get("text_max_length", 64))
        num_heads = int(config.get("num_heads", 4))
        self.text_encoder = TextEncoder(
            model_name=config["text_model_name"],
            hidden_dim=self.hidden_dim,
            encoder_type=config.get("text_encoder_type", "clip"),
            freeze=bool(config.get("freeze_text_encoder", True)),
            max_length=self.text_max_length,
            output_type="token",
        )
        self.compressor = QueryTokenCompressor(
            self.hidden_dim, self.num_channel_tokens, num_heads
        )
        self.to_channel = nn.Linear(self.hidden_dim, self.latent_dim)
        self.channel = SemanticChannel(config.get("channel_type", "awgn"))
        self.from_channel = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.decoder = TextReconstructionHead(
            hidden_dim=self.hidden_dim,
            vocab_size=self.text_encoder.vocab_size,
            text_max_length=self.text_max_length,
            num_heads=num_heads,
            num_layers=int(config.get("text_decoder_layers", 2)),
        )

    def configure_channel(self, channel_type: str) -> None:
        self.channel.configure(channel_type)

    def num_channel_uses(self) -> int:
        return self.num_channel_tokens * self.latent_dim

    def forward(
        self,
        texts: List[str],
        device: torch.device,
        snr_db: Union[float, int, Tensor],
        standard_noise: Optional[Tensor] = None,
        channel_gain: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        encoded = self.text_encoder(texts, device)
        compressed = self.compressor(
            encoded["tokens"], key_padding_mask=~encoded["attention_mask"].bool()
        )
        latent = self.to_channel(compressed)
        noisy = self.channel(
            latent,
            snr_db,
            standard_noise=standard_noise,
            channel_gain=channel_gain,
        )
        logits = self.decoder(self.from_channel(noisy))
        return {
            "text_logits": logits,
            "text_input_ids": encoded["input_ids"],
            "text_attention_mask": encoded["attention_mask"],
            "text_latent": latent,
            "text_latent_noisy": noisy,
        }

    def decode_text_logits(self, logits: Tensor) -> List[str]:
        return self.text_encoder.tokenizer.batch_decode(
            logits.argmax(dim=-1), skip_special_tokens=True
        )


class SeparateDeepJSCCTexBaseline(nn.Module):
    """Equal-budget image DeepJSCC plus an independent text JSCC link."""

    def __init__(self, config: Dict):
        super().__init__()
        self.image_branch = DeepJSCCBaseline(
            channel_type=config.get("channel_type", "awgn"),
            channel_channels=int(config.get("image_channel_channels", 60)),
        )
        self.text_branch = IndependentTextJSCC(config)
        self.image_size = int(config.get("image_size", 128))

    def configure_channel(self, channel_type: str) -> None:
        self.image_branch.configure_channel(channel_type)
        self.text_branch.configure_channel(channel_type)

    def num_channel_uses(self) -> Dict[str, int]:
        image = self.image_branch.num_channel_uses(self.image_size)
        text = self.text_branch.num_channel_uses()
        return {"image": image, "text": text, "total": image + text}

    def forward(
        self,
        images: Tensor,
        texts: List[str],
        snr_db: Union[float, int, Tensor],
        image_noise: Optional[Tensor] = None,
        text_noise: Optional[Tensor] = None,
        image_gain: Optional[Tensor] = None,
        text_gain: Optional[Tensor] = None,
        image_channel_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        image_outputs = self.image_branch(
            images, snr_db, standard_noise=image_noise, channel_gain=image_gain,
            channel_mask=image_channel_mask,
        )
        text_outputs = self.text_branch(
            texts, images.device, snr_db,
            standard_noise=text_noise, channel_gain=text_gain,
        )
        return {**image_outputs, **text_outputs}
