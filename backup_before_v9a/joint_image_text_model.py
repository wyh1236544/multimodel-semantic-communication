from __future__ import annotations

from typing import Dict, List, Optional, Union

import copy
import torch
from torch import Tensor, nn

from .channel import SemanticChannel
from .fusion import GatedTextToImageFusion
from .joint_multimodal_decoder import JointMultimodalDecoder
from .joint_multimodal_encoder import JointMultimodalEncoder
from .joint_multimodal_heads import (
    CommonBaseSemanticDecoder,
    CommonCoarseLatentHead,
    ImageReconstructionHead,
)
from .shared_bottleneck import FixedBudgetTokenPacker, SharedBottleneck
from .text_encoder import TextEncoder
from .text_reconstruction_head import TextReconstructionHead
from .vae_tokenizer import StableDiffusionVAEWrapper, VAELatentTokenizer
from .residual_adapter import ResidualReliabilityGate, SpatialResidualAdapter
from .bandwidth_config import bandwidth_summary, resolve_bandwidth_config
from .adaptive_bandwidth import AdaptiveBandwidthController


class DecoderTextToImageRefiner(nn.Module):
    """Let decoded image tokens query decoded text tokens before image reconstruction."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int = 1,
        initial_gate_bias: float = -1.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "image_norm": nn.LayerNorm(hidden_dim),
                        "text_norm": nn.LayerNorm(hidden_dim),
                        "attn": nn.MultiheadAttention(
                            hidden_dim,
                            num_heads,
                            batch_first=True,
                        ),
                        "ffn_norm": nn.LayerNorm(hidden_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim * 4),
                            nn.GELU(),
                            nn.Linear(hidden_dim * 4, hidden_dim),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(initial_gate_bias)))

    def forward(
        self,
        image_tokens: Tensor,
        text_tokens: Tensor,
        reliability: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if text_tokens.numel() == 0:
            return image_tokens, image_tokens.new_zeros(())
        gate = torch.sigmoid(self.gate_logit).to(
            device=image_tokens.device,
            dtype=image_tokens.dtype,
        )
        if reliability is not None:
            gate = gate * reliability.to(
                device=image_tokens.device,
                dtype=image_tokens.dtype,
            ).reshape(image_tokens.size(0), 1, 1)
        refined = image_tokens
        for layer in self.layers:
            query = layer["image_norm"](refined)
            memory = layer["text_norm"](text_tokens)
            attended, _ = layer["attn"](query, memory, memory, need_weights=False)
            refined = refined + gate * attended
            refined = refined + gate * layer["ffn"](layer["ffn_norm"](refined))
        return refined, gate


class JointImageTextSemanticCommunicationModel(nn.Module):
    """Transmit image and text through one shared noisy semantic latent."""

    supported_modes = {"joint", "separate", "image_only", "text_only"}
    supported_semantic_ablations = {
        "full",
        "no_common",
        "common_only",
        "common_only_shuffled",
        "common_only_random",
        "no_image_private",
        "no_text_private",
    }

    def __init__(self, config: Dict, modality_dropout: Optional[Dict] = None):
        super().__init__()
        config = resolve_bandwidth_config(config)
        self.resolved_bandwidth = bandwidth_summary(config)
        if str(config.get("image_backbone", "vae_latent")) != "vae_latent":
            raise ValueError("Joint MVP currently requires image_backbone=vae_latent")

        self.image_backbone = "vae_latent"
        self.transmission_mode = str(
            config.get("transmission_mode", "joint")
        ).lower()
        if self.transmission_mode not in self.supported_modes:
            raise ValueError(f"Unsupported transmission mode: {self.transmission_mode}")

        self.image_size = int(config["image_size"])
        self.hidden_dim = int(config["hidden_dim"])
        self.latent_dim = int(config["latent_dim"])
        self.num_common_tokens = int(config["num_common_tokens"])
        self.num_img_tokens = int(config["num_img_tokens"])
        self.num_txt_tokens = int(config["num_txt_tokens"])
        self.text_max_length = int(config.get("text_max_length", 32))
        self.vae_downsample_factor = int(config.get("vae_downsample_factor", 8))
        self.token_grid_size = self.image_size // self.vae_downsample_factor
        self.num_visual_tokens = self.token_grid_size**2
        self.preserve_spatial_image_tokens = bool(
            config.get("preserve_spatial_image_tokens", False)
        )
        self.symmetric_image_text_decoder = bool(
            config.get("symmetric_image_text_decoder", False)
        )
        num_heads = int(config.get("num_heads", 4))
        self.gated_text_image_fusion_enabled = bool(
            config.get("gated_text_image_fusion", False)
        )
        self.decoder_text_to_image_refiner_enabled = bool(
            config.get("decoder_text_to_image_cross_attention", False)
        )
        self.decoder_image_to_text_refiner_enabled = bool(
            config.get("decoder_image_to_text_cross_attention", False)
        )
        self.image_conditioned_text_fallback = bool(
            config.get("image_conditioned_text_fallback", False)
        )
        self.disable_cross_modal_interaction = bool(
            config.get("disable_cross_modal_interaction", False)
        )

        self.vae_wrapper = StableDiffusionVAEWrapper(
            model_name_or_path=config["vae_model_name_or_path"],
            freeze=bool(config.get("freeze_vae", True)),
            scaling_factor=config.get("vae_scaling_factor", "auto"),
            use_fp16=bool(config.get("vae_use_fp16", False)),
            detach_latents=bool(config.get("detach_vae_latents", False)),
        )
        self.vae_tokenizer = VAELatentTokenizer(
            latent_channels=int(config.get("vae_latent_channels", 4)),
            hidden_dim=self.hidden_dim,
        )
        residual_cfg = config.get("residual_adapter", {})
        self.residual_adapter_enabled = bool(residual_cfg.get("enable", False))
        self.residual_adapter = (
            SpatialResidualAdapter(
                hidden_dim=self.hidden_dim,
                latent_channels=int(config.get("vae_latent_channels", 4)),
                initial_alpha=float(residual_cfg.get("initial_alpha", 0.05)),
            )
            if self.residual_adapter_enabled
            else None
        )
        gate_cfg = residual_cfg.get("reliability_gate", {})
        self.residual_gate_enabled = bool(
            self.residual_adapter_enabled and gate_cfg.get("enable", False)
        )
        self.residual_gate = (
            ResidualReliabilityGate(
                hidden_dim=self.hidden_dim,
                latent_channels=int(config.get("vae_latent_channels", 4)),
                snr_prior_strength=float(gate_cfg.get("snr_prior_strength", 0.25)),
            )
            if self.residual_gate_enabled
            else None
        )
        self.text_encoder = TextEncoder(
            model_name=config["text_model_name"],
            hidden_dim=self.hidden_dim,
            encoder_type=config.get("text_encoder_type", "clip"),
            freeze=bool(config.get("freeze_text_encoder", True)),
            max_length=self.text_max_length,
            output_type="token",
            freeze_projection=bool(config.get("freeze_text_projection", False)),
        )
        self.text_to_image_fusion = (
            GatedTextToImageFusion(
                hidden_dim=self.hidden_dim,
                num_heads=int(config.get("text_image_fusion_heads", num_heads)),
                initial_gate_bias=float(
                    config.get("text_image_gate_initial_bias", -2.0)
                ),
            )
            if self.gated_text_image_fusion_enabled
            else None
        )
        self.multimodal_encoder = JointMultimodalEncoder(
            hidden_dim=self.hidden_dim,
            num_visual_tokens=self.num_visual_tokens,
            text_max_length=self.text_max_length,
            num_common_tokens=self.num_common_tokens,
            num_img_tokens=self.num_img_tokens,
            num_txt_tokens=self.num_txt_tokens,
            num_heads=num_heads,
            num_layers=int(config.get("shared_encoder_layers", 2)),
            preserve_spatial_image_tokens=self.preserve_spatial_image_tokens,
            explicit_shared_semantics=bool(
                config.get("explicit_shared_semantics", False)
            ),
            common_residual_gate_initial_bias=float(
                config.get("common_residual_gate_initial_bias", -2.0)
            ),
            private_common_gate_initial_bias=float(
                config.get("private_common_gate_initial_bias", -2.0)
            ),
        )
        self.bottleneck = SharedBottleneck(self.hidden_dim, self.latent_dim)
        adaptive_bandwidth = config.get("adaptive_bandwidth", {})
        self.adaptive_bandwidth_enabled = bool(
            adaptive_bandwidth.get("enable", False)
        )
        self.bandwidth_controller = (
            AdaptiveBandwidthController.from_config(adaptive_bandwidth)
            if self.adaptive_bandwidth_enabled
            else None
        )
        if (
            self.bandwidth_controller is not None
            and self.bandwidth_controller.max_width != self.latent_dim
        ):
            raise ValueError(
                "adaptive bandwidth maximum must equal the model latent_dim"
            )
        allocation = config.get("dynamic_token_allocation", {})
        self.dynamic_allocation_enabled = bool(allocation.get("enable", False))
        self.channel_token_budget = int(
            allocation.get(
                "modality_token_budget",
                self.num_img_tokens + self.num_txt_tokens,
            )
        )
        self.token_packer = (
            FixedBudgetTokenPacker(
                hidden_dim=self.hidden_dim,
                modality_budget=self.channel_token_budget,
                min_text_tokens=int(allocation.get("min_text_tokens", 8)),
                max_text_tokens=int(
                    allocation.get("max_text_tokens", self.num_txt_tokens)
                ),
                fixed_text_tokens=int(
                    allocation.get("fixed_text_tokens", self.num_txt_tokens)
                ),
                strategy=str(allocation.get("strategy", "fixed")),
                text_selection_strategy=str(
                    allocation.get("text_selection_strategy", "importance")
                ),
            )
            if self.dynamic_allocation_enabled
            else None
        )
        self.channel = SemanticChannel()
        self.quota_conditioning_enabled = bool(
            allocation.get("quota_conditioning", False)
        )
        self.quota_condition = (
            nn.Sequential(
                nn.Linear(1, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
            )
            if self.quota_conditioning_enabled
            else None
        )
        decoder_kwargs = dict(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            num_common_tokens=self.num_common_tokens,
            num_img_tokens=self.num_img_tokens,
            num_txt_tokens=self.num_txt_tokens,
            num_heads=num_heads,
            num_layers=int(config.get("shared_decoder_layers", 2)),
            preserve_spatial_image_tokens=self.preserve_spatial_image_tokens,
        )
        self.joint_decoder = JointMultimodalDecoder(
            **decoder_kwargs,
            transmission_mode="joint",
        )
        self.separate_decoder = JointMultimodalDecoder(
            **decoder_kwargs,
            transmission_mode="separate",
        )
        self.decoder_text_to_image_refiner = (
            DecoderTextToImageRefiner(
                hidden_dim=self.hidden_dim,
                num_heads=num_heads,
                num_layers=int(
                    config.get("decoder_text_to_image_layers", 1)
                ),
                initial_gate_bias=float(
                    config.get("decoder_text_to_image_gate_initial_bias", -1.0)
                ),
            )
            if self.decoder_text_to_image_refiner_enabled
            else None
        )
        self.decoder_image_to_text_refiner = (
            DecoderTextToImageRefiner(
                hidden_dim=self.hidden_dim,
                num_heads=num_heads,
                num_layers=int(
                    config.get("decoder_image_to_text_layers", 1)
                ),
                initial_gate_bias=float(
                    config.get("decoder_image_to_text_gate_initial_bias", -1.0)
                ),
            )
            if self.decoder_image_to_text_refiner_enabled
            else None
        )
        self.image_head = ImageReconstructionHead(
            hidden_dim=self.hidden_dim,
            latent_channels=int(config.get("vae_latent_channels", 4)),
            token_grid_size=self.token_grid_size,
            num_heads=num_heads,
            num_layers=int(config.get("image_head_layers", 1)),
            refinement_blocks=int(
                config.get("image_refinement_blocks", 0)
            ),
            refinement_channels=int(
                config.get("image_refinement_channels", self.hidden_dim)
            ),
            spatial_direct=self.preserve_spatial_image_tokens,
            symmetric_cross_attention=self.symmetric_image_text_decoder,
        )
        coarse_cfg = config.get("common_coarse_regression", {})
        self.common_coarse_enabled = bool(coarse_cfg.get("enable", False))
        self.common_coarse_blur_kernel = int(coarse_cfg.get("blur_kernel", 5))
        if self.common_coarse_blur_kernel <= 0 or self.common_coarse_blur_kernel % 2 == 0:
            raise ValueError("common_coarse_regression.blur_kernel must be a positive odd integer")
        self.common_coarse_head = (
            CommonCoarseLatentHead(
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
                latent_channels=int(config.get("vae_latent_channels", 4)),
                token_grid_size=self.token_grid_size,
                num_heads=num_heads,
                num_layers=int(coarse_cfg.get("num_layers", 1)),
                zero_init=bool(coarse_cfg.get("zero_init", True)),
            )
            if self.common_coarse_enabled
            else None
        )
        base_cfg = config.get("common_base_semantics", {})
        self.common_base_enabled = bool(base_cfg.get("enable", False))
        self.common_caption_max_tokens = min(
            max(int(base_cfg.get("caption_max_tokens", 24)), 1),
            self.text_max_length,
        )
        self.true_common_only_transmission = bool(
            config.get("true_common_only_transmission", {}).get("enable", False)
        )
        self.common_base_decoder = (
            CommonBaseSemanticDecoder(
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
                num_heads=num_heads,
                num_layers=int(base_cfg.get("num_layers", 1)),
            )
            if self.common_base_enabled
            else None
        )
        self.common_base_text_head = (
            TextReconstructionHead(
                hidden_dim=self.hidden_dim,
                vocab_size=self.text_encoder.vocab_size,
                text_max_length=self.common_caption_max_tokens,
                num_heads=num_heads,
                num_layers=int(base_cfg.get("text_layers", 1)),
            )
            if self.common_base_enabled
            else None
        )
        self.common_base_semantic_projector = (
            nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            if self.common_base_enabled
            else None
        )
        snr_gate_cfg = config.get("decoder_text_snr_gate", {})
        self.decoder_text_snr_gate_enabled = bool(snr_gate_cfg.get("enable", False))
        self.decoder_text_snr_threshold_db = float(snr_gate_cfg.get("threshold_db", 7.5))
        self.decoder_text_snr_temperature = max(
            float(snr_gate_cfg.get("temperature", 2.5)), 1e-3
        )
        self.decoder_text_snr_floor = min(
            max(float(snr_gate_cfg.get("floor", 0.0)), 0.0), 1.0
        )
        self.text_head = TextReconstructionHead(
            hidden_dim=self.hidden_dim,
            vocab_size=self.text_encoder.vocab_size,
            text_max_length=self.text_max_length,
            num_heads=num_heads,
            num_layers=int(config.get("text_head_layers", 1)),
        )
        self.dedicated_image_caption_decoder = bool(
            config.get("dedicated_image_caption_decoder", False)
        )
        self.image_caption_decoder = (
            copy.deepcopy(self.text_head.decoder)
            if self.dedicated_image_caption_decoder
            else None
        )
        receiver_semantic_dim = int(config.get("receiver_semantic_dim", 0))
        self.receiver_semantic_projector = (
            nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, receiver_semantic_dim),
            )
            if receiver_semantic_dim > 0
            else None
        )
        self.text_direct_gate_logit = nn.Parameter(
            torch.tensor(float(config.get("text_direct_gate_initial_bias", 0.0)))
        )
        self.text_direct_adapter = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        nn.init.zeros_(self.text_direct_adapter[-1].weight)
        nn.init.zeros_(self.text_direct_adapter[-1].bias)

        dropout = modality_dropout or {}
        self.modality_dropout_enabled = bool(dropout.get("enable", False))
        self.text_drop_prob = float(dropout.get("text_drop_prob", 0.0))
        self.image_token_drop_prob = float(
            dropout.get("image_token_drop_prob", 0.0)
        )
        self.null_text_token = nn.Parameter(
            torch.randn(1, 1, self.hidden_dim) * 0.02
        )
        self.null_image_token = nn.Parameter(
            torch.randn(1, 1, self.hidden_dim) * 0.02
        )

    @property
    def num_shared_tokens(self) -> int:
        if self.dynamic_allocation_enabled:
            return self.num_common_tokens + self.channel_token_budget
        return (
            self.num_common_tokens + self.num_img_tokens + self.num_txt_tokens
        )

    def num_channel_uses(
        self,
        transmission_mode: Optional[str] = None,
        channel_width: Optional[int] = None,
    ) -> int:
        mode = transmission_mode or self.transmission_mode
        if mode not in self.supported_modes:
            raise ValueError(f"Unsupported transmission mode: {mode}")
        token_count = (
            self.num_common_tokens + self.channel_token_budget
            if self.dynamic_allocation_enabled
            else self.num_shared_tokens
        )
        if self.bandwidth_controller is not None:
            width = self.bandwidth_controller._validate_requested_width(
                channel_width
                if channel_width is not None
                else self.bandwidth_controller.default_width
            )
        else:
            width = self.latent_dim
            if channel_width is not None and int(channel_width) != width:
                raise ValueError(
                    "channel_width override requires adaptive_bandwidth.enable=true"
                )
        return token_count * width

    def get_bandwidth_info(
        self,
        channel_width: Optional[int] = None,
    ) -> Dict[str, Union[str, int, List[int], Dict]]:
        """Return the effective preset, token count, and real-symbol budget."""
        info = dict(self.resolved_bandwidth)
        if self.bandwidth_controller is not None:
            width = (
                self.bandwidth_controller.default_width
                if channel_width is None
                else self.bandwidth_controller._validate_requested_width(
                    channel_width
                )
            )
            info.update(self.bandwidth_controller.summary())
            info["symbols_per_token"] = width
            info["channel_uses"] = self.num_channel_uses(
                channel_width=width
            )
        info["channel_type"] = self.channel.channel_type
        return info

    def tokenize_texts(
        self,
        texts: Union[List[str], str],
        device: torch.device,
    ) -> Dict[str, Tensor]:
        return self.text_encoder.tokenize(texts, device)

    def decode_text_logits(self, text_logits: Tensor) -> List[str]:
        token_ids = text_logits.argmax(dim=-1).detach().cpu()
        return self.text_encoder.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True,
        )

    def initialize_separate_from_joint(self) -> None:
        """Give Separate branches the trained Joint branch as a fair warm start."""
        joint_encoder_state = self.multimodal_encoder.joint_transformer.state_dict()
        self.multimodal_encoder.image_transformer.load_state_dict(
            joint_encoder_state
        )
        self.multimodal_encoder.text_transformer.load_state_dict(
            joint_encoder_state
        )

        self.separate_decoder.input_projection.load_state_dict(
            self.joint_decoder.input_projection.state_dict()
        )
        self.separate_decoder.output_norm.load_state_dict(
            self.joint_decoder.output_norm.state_dict()
        )
        joint_decoder_state = self.joint_decoder.shared_decoder.state_dict()
        self.separate_decoder.common_decoder.load_state_dict(
            copy.deepcopy(joint_decoder_state)
        )
        self.separate_decoder.image_decoder.load_state_dict(
            copy.deepcopy(joint_decoder_state)
        )
        self.separate_decoder.text_decoder.load_state_dict(
            copy.deepcopy(joint_decoder_state)
        )
        if (
            self.joint_decoder.image_position is not None
            and self.separate_decoder.image_position is not None
        ):
            self.separate_decoder.image_position.data.copy_(
                self.joint_decoder.image_position.data
            )

    def initialize_image_caption_from_text_head(self) -> None:
        if self.image_caption_decoder is None:
            raise ValueError("Dedicated image caption decoder is disabled")
        self.image_caption_decoder.load_state_dict(
            copy.deepcopy(self.text_head.decoder.state_dict())
        )

    def initialize_common_base_from_existing(self) -> None:
        """Warm-start new V8-A heads from the trained V7 receiver modules."""
        if self.common_base_decoder is None or self.common_base_text_head is None:
            return
        self.common_base_decoder.input_projection.load_state_dict(
            copy.deepcopy(self.joint_decoder.input_projection.state_dict())
        )
        self.common_base_decoder.decoder.load_state_dict(
            copy.deepcopy(self.joint_decoder.shared_decoder.state_dict())
        )
        self.common_base_decoder.output_norm.load_state_dict(
            copy.deepcopy(self.joint_decoder.output_norm.state_dict())
        )
        self.common_base_text_head.decoder.load_state_dict(
            copy.deepcopy(self.text_head.decoder.state_dict())
        )
        self.common_base_text_head.output.load_state_dict(
            copy.deepcopy(self.text_head.output.state_dict())
        )
        with torch.no_grad():
            query_count = self.common_base_text_head.queries.size(1)
            self.common_base_text_head.queries.copy_(
                self.text_head.queries[:, :query_count]
            )
            if self.common_base_semantic_projector is not None:
                linear = self.common_base_semantic_projector[-1]
                if linear.weight.shape[0] == linear.weight.shape[1]:
                    nn.init.eye_(linear.weight)
                    nn.init.zeros_(linear.bias)

    def _apply_modality_dropout(
        self,
        visual_tokens: Tensor,
        text_tokens: Tensor,
        text_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size = visual_tokens.size(0)
        text_dropped = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=visual_tokens.device,
        )
        image_drop_mask = torch.zeros(
            visual_tokens.shape[:2],
            dtype=torch.bool,
            device=visual_tokens.device,
        )
        if not (self.training and self.modality_dropout_enabled):
            return (
                visual_tokens,
                text_tokens,
                text_attention_mask,
                text_dropped,
                image_drop_mask,
            )

        if self.text_drop_prob > 0:
            text_dropped = (
                torch.rand(batch_size, device=visual_tokens.device)
                < self.text_drop_prob
            )
            if text_dropped.any():
                text_tokens = text_tokens.clone()
                text_attention_mask = text_attention_mask.clone()
                null_text = self.null_text_token.to(
                    device=text_tokens.device,
                    dtype=text_tokens.dtype,
                )
                text_tokens[text_dropped] = null_text.expand(
                    int(text_dropped.sum().item()),
                    text_tokens.size(1),
                    -1,
                )
                text_attention_mask[text_dropped] = 0
                text_attention_mask[text_dropped, 0] = 1

        if self.image_token_drop_prob > 0:
            image_drop_mask = (
                torch.rand(
                    visual_tokens.shape[:2],
                    device=visual_tokens.device,
                )
                < self.image_token_drop_prob
            )
            if image_drop_mask.any():
                visual_tokens = visual_tokens.clone()
                null_image = self.null_image_token.to(
                    device=visual_tokens.device,
                    dtype=visual_tokens.dtype,
                )
                visual_tokens[image_drop_mask] = null_image[0, 0]

        return (
            visual_tokens,
            text_tokens,
            text_attention_mask,
            text_dropped,
            image_drop_mask,
        )

    def _decoder_memories(
        self,
        decoded: Dict[str, Tensor],
        mode: str,
        text_image_reliability: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        common = decoded["common_tokens"]
        image = decoded["image_tokens"]
        text = decoded["text_tokens"]
        text_for_image = text
        if text_image_reliability is not None:
            text_for_image = text * text_image_reliability.to(
                device=text.device,
                dtype=text.dtype,
            ).reshape(text.size(0), 1, 1)
        if self.preserve_spatial_image_tokens:
            if mode == "text_only":
                return image, torch.cat([common, text], dim=1)
            if mode == "joint":
                shared = torch.cat([common, image, text], dim=1)
                image_shared = torch.cat(
                    [common, image, text_for_image], dim=1
                )
                image_memory = (
                    image_shared if self.symmetric_image_text_decoder else image
                )
                return image_memory, shared
            common_split = self.num_common_tokens // 2
            text_memory = torch.cat(
                [common[:, common_split:], text],
                dim=1,
            )
            return image, text_memory
        if mode == "joint":
            shared = torch.cat([common, image, text], dim=1)
            image_shared = torch.cat([common, image, text_for_image], dim=1)
            return image_shared, shared
        common_split = self.num_common_tokens // 2
        image_memory = torch.cat([common[:, :common_split], image], dim=1)
        text_memory = torch.cat([common[:, common_split:], text], dim=1)
        return image_memory, text_memory

    def forward(
        self,
        images: Tensor,
        texts: Union[List[str], str],
        snr_db: Union[float, int, Tensor],
        transmission_mode: Optional[str] = None,
        channel_noise: Optional[Tensor] = None,
        channel_gain: Optional[Tensor] = None,
        disable_text_input: bool = False,
        disable_image_input: bool = False,
        vae_latents_override: Optional[Tensor] = None,
        disable_modality_dropout: bool = False,
        image_token_drop_mask_override: Optional[Tensor] = None,
        image_token_drop_rate: Optional[float] = None,
        image_token_drop_generator: Optional[torch.Generator] = None,
        text_token_drop_mask_override: Optional[Tensor] = None,
        text_token_drop_rate: Optional[float] = None,
        text_token_drop_generator: Optional[torch.Generator] = None,
        semantic_ablation: str = "full",
        utility_weight: Optional[Union[float, Tensor]] = None,
        channel_width: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        mode = (transmission_mode or self.transmission_mode).lower()
        if mode not in self.supported_modes:
            raise ValueError(f"Unsupported transmission mode: {mode}")
        semantic_ablation = str(semantic_ablation).lower()
        if semantic_ablation not in self.supported_semantic_ablations:
            raise ValueError(
                f"Unsupported semantic ablation: {semantic_ablation}"
            )
        common_only_mode = semantic_ablation in {
            "common_only",
            "common_only_shuffled",
            "common_only_random",
        }

        vae_latents = (
            vae_latents_override
            if vae_latents_override is not None
            else self.vae_wrapper.encode_images(images)
        )
        visual_tokens = self.vae_tokenizer.latents_to_tokens(vae_latents)
        if disable_image_input:
            visual_tokens = self.null_image_token.to(
                device=visual_tokens.device,
                dtype=visual_tokens.dtype,
            ).expand(visual_tokens.size(0), visual_tokens.size(1), -1)
        text_outputs = self.text_encoder(texts, images.device)
        text_tokens = text_outputs["tokens"]
        attention_mask = text_outputs["attention_mask"]
        input_ids = text_outputs["input_ids"]
        if disable_text_input:
            text_tokens = self.null_text_token.to(
                device=text_tokens.device,
                dtype=text_tokens.dtype,
            ).expand(text_tokens.size(0), text_tokens.size(1), -1)
            attention_mask = torch.zeros_like(attention_mask)
            attention_mask[:, 0] = 1

        if disable_modality_dropout:
            text_dropped = torch.zeros(
                visual_tokens.size(0),
                dtype=torch.bool,
                device=visual_tokens.device,
            )
            image_drop_mask = torch.zeros(
                visual_tokens.shape[:2],
                dtype=torch.bool,
                device=visual_tokens.device,
            )
        else:
            (
                visual_tokens,
                text_tokens,
                attention_mask,
                text_dropped,
                image_drop_mask,
            ) = self._apply_modality_dropout(
                visual_tokens,
                text_tokens,
                attention_mask,
            )
        if image_token_drop_mask_override is not None or image_token_drop_rate is not None:
            if image_token_drop_mask_override is None:
                drop_rate = float(image_token_drop_rate or 0.0)
                if drop_rate <= 0.0:
                    external_drop_mask = torch.zeros_like(image_drop_mask)
                elif drop_rate >= 1.0:
                    external_drop_mask = torch.ones_like(image_drop_mask)
                else:
                    external_drop_mask = (
                        torch.rand(
                            visual_tokens.shape[:2],
                            device=visual_tokens.device,
                            generator=image_token_drop_generator,
                        )
                        < drop_rate
                    )
            else:
                external_drop_mask = image_token_drop_mask_override.to(
                    device=visual_tokens.device,
                    dtype=torch.bool,
                )
                if external_drop_mask.shape != visual_tokens.shape[:2]:
                    raise ValueError(
                        "image_token_drop_mask_override must have shape [B, N]"
                    )
            if external_drop_mask.any():
                visual_tokens = visual_tokens.clone()
                null_image = self.null_image_token.to(
                    device=visual_tokens.device,
                    dtype=visual_tokens.dtype,
                )
                visual_tokens[external_drop_mask] = null_image[0, 0]
                image_drop_mask = image_drop_mask | external_drop_mask

        text_token_drop_mask = torch.zeros(
            text_tokens.shape[:2],
            dtype=torch.bool,
            device=text_tokens.device,
        )
        if text_token_drop_mask_override is not None or text_token_drop_rate is not None:
            if text_token_drop_mask_override is None:
                drop_rate = float(text_token_drop_rate or 0.0)
                if drop_rate <= 0.0:
                    external_text_mask = torch.zeros_like(text_token_drop_mask)
                elif drop_rate >= 1.0:
                    external_text_mask = text_outputs["attention_mask"].bool()
                else:
                    external_text_mask = (
                        torch.rand(
                            text_tokens.shape[:2],
                            device=text_tokens.device,
                            generator=text_token_drop_generator,
                        )
                        < drop_rate
                    ) & text_outputs["attention_mask"].bool()
            else:
                external_text_mask = text_token_drop_mask_override.to(
                    device=text_tokens.device,
                    dtype=torch.bool,
                )
                if external_text_mask.shape != text_tokens.shape[:2]:
                    raise ValueError(
                        "text_token_drop_mask_override must have shape [B, L]"
                    )
                external_text_mask &= text_outputs["attention_mask"].bool()
            if external_text_mask.any():
                text_tokens = text_tokens.clone()
                null_text = self.null_text_token.to(
                    device=text_tokens.device,
                    dtype=text_tokens.dtype,
                )
                text_tokens[external_text_mask] = null_text[0, 0]
                text_token_drop_mask |= external_text_mask

        text_image_gate = visual_tokens.new_zeros(
            visual_tokens.size(0),
            visual_tokens.size(1),
            1,
        )
        cross_modal_enabled = (
            mode == "joint" and not self.disable_cross_modal_interaction
        )
        if self.text_to_image_fusion is not None and cross_modal_enabled:
            visual_tokens, text_image_gate = self.text_to_image_fusion(
                visual_tokens,
                text_tokens,
                attention_mask,
            )
        encoded = self.multimodal_encoder(
            visual_tokens,
            text_tokens,
            attention_mask,
            transmission_mode=mode,
        )
        bottleneck = self.bottleneck(
            encoded["common_tokens"],
            encoded["image_tokens"],
            encoded["text_tokens"],
        )
        transmitted_bottleneck = dict(bottleneck)
        if semantic_ablation == "no_common" or common_only_mode:
            if semantic_ablation == "no_common":
                transmitted_bottleneck["z_common"] = torch.zeros_like(
                    bottleneck["z_common"]
                )
            else:
                transmitted_bottleneck["z_img"] = torch.zeros_like(
                    bottleneck["z_img"]
                )
                transmitted_bottleneck["z_txt"] = torch.zeros_like(
                    bottleneck["z_txt"]
                )
                if semantic_ablation == "common_only_shuffled":
                    transmitted_bottleneck["z_common"] = torch.roll(
                        bottleneck["z_common"], shifts=1, dims=0
                    )
                elif semantic_ablation == "common_only_random":
                    transmitted_bottleneck["z_common"] = torch.randn_like(
                        bottleneck["z_common"]
                    )
        elif semantic_ablation == "no_image_private":
            transmitted_bottleneck["z_img"] = torch.zeros_like(
                bottleneck["z_img"]
            )
        elif semantic_ablation == "no_text_private":
            transmitted_bottleneck["z_txt"] = torch.zeros_like(
                bottleneck["z_txt"]
            )
        physical_common_only = bool(
            common_only_mode
            and self.true_common_only_transmission
        )
        allocation_metadata = None
        if physical_common_only:
            # A real 32-token base-layer payload.  Private positions are not
            # sent through the channel, so they cannot return as pure noise.
            z_shared = transmitted_bottleneck["z_common"]
        elif self.token_packer is not None:
            z_shared, allocation_metadata = self.token_packer.pack(
                transmitted_bottleneck["z_common"],
                transmitted_bottleneck["z_img"],
                transmitted_bottleneck["z_txt"],
                encoded["image_tokens"],
                encoded["text_tokens"],
                text_outputs["attention_mask"].sum(dim=1),
                self.text_max_length,
                snr_db,
                utility_weight=utility_weight,
                text_attention_mask=text_outputs["attention_mask"],
            )
        else:
            z_shared = torch.cat(
                [
                    transmitted_bottleneck["z_common"],
                    transmitted_bottleneck["z_img"],
                    transmitted_bottleneck["z_txt"],
                ],
                dim=1,
            )
        if physical_common_only and channel_noise is not None:
            if channel_noise.size(1) < self.num_common_tokens:
                raise ValueError("channel_noise has fewer tokens than the common payload")
            channel_noise = channel_noise[:, : self.num_common_tokens]
        if physical_common_only and channel_gain is not None and channel_gain.ndim >= 2:
            if channel_gain.size(1) not in {1, self.num_common_tokens}:
                if channel_gain.size(1) < self.num_common_tokens:
                    raise ValueError("channel_gain has fewer tokens than the common payload")
                channel_gain = channel_gain[:, : self.num_common_tokens]
        if self.bandwidth_controller is not None:
            active_channel_width = self.bandwidth_controller.select_width(
                snr_db,
                requested_width=channel_width,
                training=self.training,
                channel_type=self.channel.channel_type,
            )
            z_transmitted, active_noise = (
                self.bandwidth_controller.prepare_transmission(
                    z_shared,
                    active_channel_width,
                    standard_noise=channel_noise,
                )
            )
            active_gain = channel_gain
            if channel_gain is not None and channel_gain.shape == z_shared.shape:
                active_gain = channel_gain[..., :active_channel_width]
        else:
            if channel_width is not None and int(channel_width) != self.latent_dim:
                raise ValueError(
                    "channel_width override requires adaptive_bandwidth.enable=true"
                )
            active_channel_width = self.latent_dim
            z_transmitted = z_shared
            active_noise = channel_noise
            active_gain = channel_gain
        z_received = self.channel(
            z_transmitted,
            snr_db,
            standard_noise=active_noise,
            channel_gain=active_gain,
        )
        z_noisy = (
            self.bandwidth_controller.restore_receiver_width(z_received)
            if self.bandwidth_controller is not None
            else z_received
        )
        # Preserve the physically received common payload before the joint
        # decoder can mix information across semantic branches.
        received_common_channel = z_noisy[:, : self.num_common_tokens]
        snr_tensor = torch.as_tensor(
            snr_db,
            device=z_noisy.device,
            dtype=torch.float32,
        )
        if snr_tensor.ndim == 0:
            snr_tensor = snr_tensor.expand(z_noisy.size(0))
        snr_tensor = snr_tensor.reshape(z_noisy.size(0), -1)[:, 0]
        if self.decoder_text_snr_gate_enabled:
            text_channel_reliability = torch.sigmoid(
                (snr_tensor - self.decoder_text_snr_threshold_db)
                / self.decoder_text_snr_temperature
            )
            text_channel_reliability = self.decoder_text_snr_floor + (
                1.0 - self.decoder_text_snr_floor
            ) * text_channel_reliability
        else:
            text_channel_reliability = torch.ones_like(snr_tensor)
        text_channel_reliability = torch.where(
            text_dropped,
            torch.zeros_like(text_channel_reliability),
            text_channel_reliability,
        )
        if disable_text_input or semantic_ablation in {
            "no_text_private",
        }:
            text_channel_reliability = torch.zeros_like(
                text_channel_reliability
            )
        if common_only_mode:
            text_channel_reliability = torch.zeros_like(text_channel_reliability)
        if physical_common_only:
            decoded_common = z_noisy
            decoded_image = z_noisy.new_zeros(
                z_noisy.size(0), self.num_img_tokens, self.latent_dim
            )
            decoded_text = z_noisy.new_zeros(
                z_noisy.size(0), self.num_txt_tokens, self.latent_dim
            )
        elif self.token_packer is not None:
            decoded_common, decoded_image, decoded_text = self.token_packer.unpack(
                z_noisy, allocation_metadata
            )
        else:
            common_end = self.num_common_tokens
            image_end = common_end + self.num_img_tokens
            decoded_common = z_noisy[:, :common_end]
            decoded_image = z_noisy[:, common_end:image_end]
            decoded_text = z_noisy[:, image_end:]
        decoder_input = torch.cat(
            [decoded_common, decoded_image, decoded_text], dim=1
        )
        image_decoder_input = torch.cat(
            [
                decoded_common,
                decoded_image,
                decoded_text
                * text_channel_reliability.to(decoded_text.dtype)[:, None, None],
            ],
            dim=1,
        )
        decoder = (
            self.joint_decoder if mode == "joint" else self.separate_decoder
        )
        decoded = decoder(decoder_input)
        image_decoded = (
            decoder(image_decoder_input)
            if self.decoder_text_snr_gate_enabled and cross_modal_enabled
            else decoded
        )
        if self.quota_condition is not None and allocation_metadata is not None:
            quota_fraction = (
                allocation_metadata["text_counts"].float()
                / max(float(self.num_txt_tokens), 1.0)
            ).unsqueeze(-1)
            quota_embedding = self.quota_condition(quota_fraction).to(
                dtype=decoded["text_tokens"].dtype
            )
            decoded = dict(decoded)
            decoded["text_tokens"] = (
                decoded["text_tokens"] + quota_embedding[:, None, :]
            )
            if image_decoded is not decoded:
                image_decoded = dict(image_decoded)
                image_decoded["text_tokens"] = (
                    image_decoded["text_tokens"] + quota_embedding[:, None, :]
                )
        decoder_text_image_gate = z_noisy.new_zeros(())
        decoder_image_text_gate = z_noisy.new_zeros(())
        base_decoded = decoded
        decoded = image_decoded
        if self.decoder_text_to_image_refiner is not None and cross_modal_enabled:
            refined_image_tokens, decoder_text_image_gate = (
                self.decoder_text_to_image_refiner(
                    base_decoded["image_tokens"],
                    torch.cat(
                        [
                            base_decoded["common_tokens"],
                            base_decoded["text_tokens"],
                        ],
                        dim=1,
                    ),
                    reliability=text_channel_reliability,
                )
            )
            decoded = dict(decoded)
            decoded["image_tokens"] = refined_image_tokens
        if self.decoder_image_to_text_refiner is not None and cross_modal_enabled:
            refined_text_tokens, decoder_image_text_gate = (
                self.decoder_image_to_text_refiner(
                    base_decoded["text_tokens"],
                    torch.cat(
                        [
                            base_decoded["common_tokens"],
                            base_decoded["image_tokens"],
                        ],
                        dim=1,
                    ),
                )
            )
            decoded = dict(decoded)
            decoded["text_tokens"] = refined_text_tokens
        image_memory, text_memory = self._decoder_memories(
            decoded,
            mode,
            text_image_reliability=text_channel_reliability,
        )

        image_private_reliability = 1.0 - image_drop_mask.float().mean(dim=1)
        if disable_image_input or semantic_ablation in {
            "no_image_private",
        }:
            image_private_reliability = torch.zeros_like(
                image_private_reliability
            )
        if common_only_mode:
            image_private_reliability = torch.zeros_like(image_private_reliability)
        valid_text_count = text_outputs["attention_mask"].sum(dim=1).clamp_min(1)
        dropped_valid_text = (
            text_token_drop_mask & text_outputs["attention_mask"].bool()
        ).sum(dim=1)
        text_private_reliability = 1.0 - (
            dropped_valid_text.float() / valid_text_count.float()
        )
        text_private_reliability = torch.where(
            text_dropped,
            torch.zeros_like(text_private_reliability),
            text_private_reliability,
        )
        if disable_text_input or semantic_ablation in {
            "no_text_private",
        }:
            text_private_reliability = torch.zeros_like(
                text_private_reliability
            )
        if common_only_mode:
            text_private_reliability = torch.zeros_like(text_private_reliability)

        common_coarse_latents = (
            self.common_coarse_head(received_common_channel)
            if self.common_coarse_head is not None
            else None
        )
        common_base_tokens = (
            self.common_base_decoder(received_common_channel)
            if self.common_base_decoder is not None
            else None
        )
        common_base_text_logits = (
            self.common_base_text_head(common_base_tokens)
            if self.common_base_text_head is not None
            else None
        )
        common_receiver_semantic = (
            torch.nn.functional.normalize(
                self.common_base_semantic_projector(
                    common_base_tokens.float().mean(dim=1)
                ),
                dim=-1,
            )
            if self.common_base_semantic_projector is not None
            else None
        )
        # The CLIP text backbone is frozen in V8-A; detaching makes this a
        # stable semantic teacher rather than a second path for loss collapse.
        common_text_semantic_target = torch.nn.functional.normalize(
            text_outputs["pooled"].detach().float(), dim=-1
        )
        recovered_vae_latents = self.image_head(
            image_memory,
            spatial_tokens=decoded["image_tokens"],
            spatial_reliability=image_private_reliability,
        )
        residual_latent = None
        residual_gate = None
        residual_global_gate = None
        residual_spatial_gate = None
        if self.residual_adapter is not None:
            base_vae_latents = recovered_vae_latents
            _, residual_latent = self.residual_adapter(
                decoded["image_tokens"], recovered_vae_latents
            )
            if self.residual_gate is not None:
                snr_tensor = torch.as_tensor(
                    snr_db, device=base_vae_latents.device,
                    dtype=base_vae_latents.dtype,
                )
                if snr_tensor.ndim == 0:
                    snr_tensor = snr_tensor.expand(base_vae_latents.size(0))
                damage_rate = image_drop_mask.float().mean(dim=1)
                # At 100% image-token loss there is no trustworthy private
                # visual evidence; disable only this boundary case.
                gate_reliability = image_private_reliability * (
                    1.0 - (damage_rate > 0.999).to(image_private_reliability.dtype)
                )
                residual_gate, residual_global_gate, residual_spatial_gate = self.residual_gate(
                    decoded["image_tokens"], base_vae_latents, residual_latent,
                    damage_rate, snr_tensor, gate_reliability,
                )
                # Reliability boundary protection: clean images need no
                # correction, while 100% token loss has no trustworthy private
                # evidence. The learned gate remains active for 25-75% loss.
                usable_damage = (
                    (damage_rate > 0.001) & (damage_rate < 0.999)
                ).to(residual_gate.dtype)
                residual_gate = residual_gate * usable_damage[:, None, None, None]
                residual_latent = residual_latent * residual_gate
            recovered_vae_latents = base_vae_latents + residual_latent
        image_rec = self.vae_wrapper.decode_latents(recovered_vae_latents)
        text_features_global = self.text_head.decode_features(text_memory)
        text_logits_global = self.text_head.output(text_features_global)
        image_caption_memory = torch.cat(
            [decoded["common_tokens"], decoded["image_tokens"]],
            dim=1,
        )
        if self.image_caption_decoder is not None:
            image_caption_queries = self.text_head.queries.expand(
                image_caption_memory.size(0), -1, -1
            )
            image_conditioned_text_features = self.image_caption_decoder(
                image_caption_queries,
                image_caption_memory,
            )
        else:
            image_conditioned_text_features = self.text_head.decode_features(
                image_caption_memory
            )
        image_conditioned_text_logits = self.text_head.output(
            image_conditioned_text_features
        )
        text_direct_features = (
            decoded["text_tokens"]
            + self.text_direct_adapter(decoded["text_tokens"])
        )
        text_direct_logits = self.text_head.output(text_direct_features)
        transmitted_text_mask = torch.zeros(
            decoded["text_tokens"].shape[:2],
            device=decoded["text_tokens"].device,
            dtype=torch.bool,
        )
        if allocation_metadata is None:
            transmitted_text_mask.fill_(True)
        else:
            for batch_index, text_indices in enumerate(
                allocation_metadata["text_indices"]
            ):
                transmitted_text_mask[batch_index, text_indices] = True
        direct_gate = torch.sigmoid(self.text_direct_gate_logit).to(
            dtype=text_logits_global.dtype
        )
        reliability = text_private_reliability[:, None, None].to(
            text_logits_global.dtype
        )
        if self.image_conditioned_text_fallback and cross_modal_enabled:
            text_features_base = (
                reliability * text_features_global
                + (1.0 - reliability) * image_conditioned_text_features
            )
        else:
            text_features_base = text_features_global
        text_logits_base = self.text_head.output(text_features_base)
        receiver_text_pooled = text_features_base.float().mean(dim=1)
        image_conditioned_text_pooled = (
            image_conditioned_text_features.float().mean(dim=1)
        )
        if self.receiver_semantic_projector is not None:
            receiver_text_pooled = self.receiver_semantic_projector(
                receiver_text_pooled
            )
            image_conditioned_text_pooled = self.receiver_semantic_projector(
                image_conditioned_text_pooled
            )
        receiver_text_semantic = torch.nn.functional.normalize(
            receiver_text_pooled,
            dim=-1,
        )
        image_conditioned_text_semantic = torch.nn.functional.normalize(
            image_conditioned_text_pooled,
            dim=-1,
        )
        text_logits = text_logits_base + (
            direct_gate
            * text_direct_logits
            * transmitted_text_mask.unsqueeze(-1).to(text_direct_logits.dtype)
            * reliability.to(text_direct_logits.dtype)
        )

        return {
            "image": image_rec,
            "text_logits": text_logits,
            "text_logits_global": text_logits_global,
            "image_conditioned_text_logits": image_conditioned_text_logits,
            "receiver_text_semantic": receiver_text_semantic,
            "image_conditioned_text_semantic": image_conditioned_text_semantic,
            "text_direct_logits": text_direct_logits,
            "transmitted_text_mask": transmitted_text_mask,
            "text_direct_gate": direct_gate,
            "text_input_ids": input_ids,
            "text_attention_mask": text_outputs["attention_mask"],
            # Public tokenizer constants, not source-caption side information.
            # The quality router uses the decoded EOS position instead of the
            # ground-truth attention mask to estimate received text length.
            "text_eos_token_id": input_ids.new_tensor(
                int(
                    getattr(
                        self.text_encoder.tokenizer,
                        "eos_token_id",
                        None,
                    )
                    or (self.text_encoder.vocab_size - 1)
                )
            ),
            "vae_latents": vae_latents,
            "recovered_vae_latents": recovered_vae_latents,
            "common_coarse_latents": common_coarse_latents,
            "common_coarse_blur_kernel": input_ids.new_tensor(
                self.common_coarse_blur_kernel
            ),
            "common_base_tokens": common_base_tokens,
            "common_base_text_logits": common_base_text_logits,
            "common_receiver_semantic": common_receiver_semantic,
            "common_text_semantic_target": common_text_semantic_target,
            "common_caption_max_tokens": input_ids.new_tensor(
                self.common_caption_max_tokens
            ),
            "residual_latent": residual_latent,
            "residual_gate": residual_gate,
            "residual_global_gate": residual_global_gate,
            "residual_spatial_gate": residual_spatial_gate,
            "z_shared": z_shared,
            "z_transmitted": z_transmitted,
            "z_shared_noisy": z_noisy,
            "active_channel_width": z_shared.new_tensor(
                float(active_channel_width)
            ),
            "channel_uses": z_shared.new_tensor(
                float(z_transmitted.size(1) * active_channel_width)
            ),
            "physical_common_only": input_ids.new_tensor(
                int(physical_common_only)
            ),
            "z_common": bottleneck["z_common"],
            "z_img": bottleneck["z_img"],
            "z_txt": bottleneck["z_txt"],
            "decoded_common_tokens": decoded["common_tokens"],
            "decoded_image_tokens": decoded["image_tokens"],
            "decoded_text_tokens": decoded["text_tokens"],
            "encoded_common_tokens": encoded["common_tokens"],
            "encoded_image_private_tokens": encoded["image_tokens"],
            "encoded_text_private_tokens": encoded["text_tokens"],
            "common_image_tokens": encoded["common_image_tokens"],
            "common_text_tokens": encoded["common_text_tokens"],
            "semantic_ablation": semantic_ablation,
            "text_dropout_fraction": text_dropped.float().mean(),
            "image_token_dropout_fraction": image_drop_mask.float().mean(),
            "text_token_dropout_fraction": text_token_drop_mask.float().mean(),
            "common_residual_gate": torch.sigmoid(
                self.multimodal_encoder.common_residual_gate_logit
            ),
            "private_common_gate": torch.sigmoid(
                self.multimodal_encoder.private_common_gate_logit
            ),
            "image_private_reliability": image_private_reliability,
            "text_private_reliability": text_private_reliability,
            "text_channel_reliability": text_channel_reliability,
            "text_image_gate_mean": text_image_gate.mean(),
            "decoder_text_image_gate_mean": decoder_text_image_gate.mean(),
            "decoder_image_text_gate_mean": decoder_image_text_gate.mean(),
            "allocated_image_tokens": (
                z_shared.new_zeros(z_shared.size(0))
                if physical_common_only
                else allocation_metadata["image_counts"].float()
                if allocation_metadata is not None
                else z_shared.new_full(
                    (z_shared.size(0),), float(self.num_img_tokens)
                )
            ),
            "allocated_text_tokens": (
                z_shared.new_zeros(z_shared.size(0))
                if physical_common_only
                else allocation_metadata["text_counts"].float()
                if allocation_metadata is not None
                else z_shared.new_full(
                    (z_shared.size(0),), float(self.num_txt_tokens)
                )
            ),
            "valid_text_tokens": (
                allocation_metadata["valid_text_counts"].float()
                if allocation_metadata is not None
                and "valid_text_counts" in allocation_metadata
                else text_outputs["attention_mask"].sum(dim=1).float()
            ),
            "text_redundant_slots": (
                allocation_metadata["text_redundant_slots"].float()
                if allocation_metadata is not None
                and "text_redundant_slots" in allocation_metadata
                else z_shared.new_zeros(z_shared.size(0))
            ),
            "text_repetition_factor": (
                allocation_metadata["text_repetition_factor"].float()
                if allocation_metadata is not None
                and "text_repetition_factor" in allocation_metadata
                else z_shared.new_ones(z_shared.size(0))
            ),
            "predicted_text_quota": (
                allocation_metadata["predicted_text_quota"]
                if allocation_metadata is not None
                else z_shared.new_full(
                    (z_shared.size(0),), float(self.num_txt_tokens)
                )
            ),
            "target_text_quota": (
                allocation_metadata["target_text_quota"]
                if allocation_metadata is not None
                else z_shared.new_full(
                    (z_shared.size(0),), float(self.num_txt_tokens)
                )
            ),
        }
