from __future__ import annotations
import copy
import torch
from torch import nn
from .channels import LinkFrame, PROFILES
from .protocol import DeterministicPacker, PilotSemanticChannel

class V10Model(nn.Module):
    def __init__(self, base, settings):
        super().__init__()
        self.base, self.settings = base, copy.deepcopy(settings)
        if base.latent_dim != 16 or base.bandwidth_controller is not None:
            raise ValueError('V10 uses the fixed original_4608 bottleneck, width16')
        base.token_packer = DeterministicPacker()
        base.channel = PilotSemanticChannel(16, settings.get('uep', True), settings.get('adapter', True), settings.get('interleave', True))
        base.v10_disable_early_fusion = not settings.get('early_fusion', False)
        base.v10_fix_image_gate = True
        base.v10_skip_inactive_heads = True
        self.last_frame = None
        self.initial_trainable = {n for n,p in self.named_parameters() if p.requires_grad}

    def configure_stage(self, stage):
        if stage not in ('adapter', 'joint'):
            raise ValueError('stage must be adapter or joint')
        frozen = ('base.vae_wrapper.', 'base.text_encoder.', 'base.separate_decoder.',
                  'base.common_coarse_head.', 'base.residual_adapter.', 'base.residual_gate.')
        if self.base.v10_disable_early_fusion:
            frozen += ('base.text_to_image_fusion.',)
        warm = ('base.channel.', 'base.bottleneck.', 'base.joint_decoder.',
                'base.common_base_decoder.', 'base.common_base_semantic_projector.',
                'base.common_base_text_head.', 'base.quota_condition.')
        for name, p in self.named_parameters():
            p.requires_grad_(name in self.initial_trainable and not name.startswith(frozen)
                             and (stage == 'joint' or name.startswith(warm)))

    def forward(self, images, texts, profile='awgn', snr_db=10., semantic_ablation='full',
                vae_latents_override=None, common_task_only=False, shuffled_common_payload=None):
        cfg = self.settings
        spec = dict(PROFILES[profile])
        spec.update(cfg.get('profile_overrides', {}).get(profile, {}))
        frame = LinkFrame(spec, snr_db, len(images), images.device,
                          blocks=int(cfg.get('blocks', 1)), pilots=int(cfg.get('pilots', 32)),
                          feedback_snr_db=float(cfg.get('feedback_snr_db', 0.)),
                          control_length=int(cfg.get('control_length', 64)),
                          threshold=float(cfg.get('control_threshold', .2)),
                          csi=cfg.get('csi', 'pilot'), equalizer=cfg.get('equalizer', 'mmse'),
                          fixed_mode=cfg.get('fixed_mode'),
                          generation_threshold_db=float(cfg.get('generation_threshold_db', 2.5)),
                          reconstruction_threshold_db=float(cfg.get('reconstruction_threshold_db', 8.)))
        self.last_frame = frame
        self.base.channel.frame = frame
        self.base.token_packer.frame = frame
        # True SNR NEVER enters encoder/decoder gates. Only pilot observations.
        # Packer ignores the SNR and uses the noisy-feedback mode at the sender.
        outputs = self.base(images, texts, snr_db=frame.routing_snr,
                            disable_modality_dropout=True,
                            semantic_ablation=semantic_ablation,
                            vae_latents_override=vae_latents_override,
                            common_task_only=common_task_only,
                            shuffled_common_payload=shuffled_common_payload)
        outputs['link_stats'] = frame.last_stats
        return outputs

def build_model(config, device, initial_checkpoint=None):
    # Lazy imports let physical/protocol tests run without Transformers/VAE.
    from utils.experiment import create_model
    from utils.checkpoint import load_checkpoint, load_state_dict_flexible
    base = create_model(config, device)
    report = None
    checkpoint = load_checkpoint(initial_checkpoint) if initial_checkpoint else None
    is_v10 = checkpoint is not None and str(checkpoint.get('version', '')).startswith('V10-')
    if checkpoint is not None and not is_v10:
        report = load_state_dict_flexible(base, checkpoint, allow_shape_expansion=False)
    result = V10Model(base, config['v10']).to(device)
    if is_v10:
        result.load_state_dict(checkpoint['model'], strict=True)
        report = {'v10_exact': True}
    return result, report
