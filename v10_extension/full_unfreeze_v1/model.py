"""Process-local full unfreezing and reproducible LIVE posterior encoding.

All registered parameters are eligible in U2. Inactive historical branches are
not activated: optimizer state and observed gradient coverage are separate.
"""
import torch
from fso_jscc_v1.model import parameters as semantic_parameters, ENCODER
from fso_jscc_v1.model import gradient_groups as semantic_gradients
from fso_fluctuating_v2.model import Frame, ContinuousChannel, surrogate_context
from vae_tail_ablation_v1.model import apply_parent, verify_frozen
from full_unfreeze_v1.core import tail_prefixes, parameter_group


def candidates(model):
    prefixes = tail_prefixes(len(model.base.vae_wrapper.vae.decoder.up_blocks))
    named = list(model.named_parameters())
    required = ('base.text_encoder.backbone.', 'base.vae_wrapper.vae.encoder.',
                'base.vae_wrapper.vae.decoder.') + prefixes
    for prefix in required:
        if not any(n.startswith(prefix) for n, _ in named):
            raise ValueError('Required backbone scope missing: '+prefix)
    return named, prefixes


def eligible_names(model, full, prefixes):
    return ({n for n, _ in model.named_parameters()} if full else
            {n for n, _ in semantic_parameters(model)} |
            {n for n, _ in model.named_parameters() if n.startswith(prefixes)})


def stage(model, warm, full, prefixes, frozen=False):
    allowed = set() if frozen else eligible_names(model, full, prefixes)
    for name, param in model.named_parameters():
        param.requires_grad_(name in allowed and not (warm and name.startswith('base.multimodal_encoder.')))
    model.base.text_encoder.freeze = not full or frozen
    model.base.vae_wrapper.freeze = frozen
    model.base.vae_wrapper.detach_latents = False
    model.base.vae_wrapper.use_fp16 = False
    model.eval()  # deterministic dropout in BOTH arms; NOT no_grad


def install_live_encoder(wrapper):
    """Same fixed standard-normal epsilon per source in teacher and both arms.

    The reference is cached only as a detached LOSS target. Model inputs always
    use the CURRENT encoder's mean/std, with its true reparameterized derivative.
    No straight-through estimator and no pretrained model downloads.
    """
    def encode(images):
        distribution = wrapper.vae.encode(images.to(dtype=wrapper._vae_dtype())*2.-1.).latent_dist
        noise = getattr(wrapper, 'posterior_noise', None)
        if noise is None or tuple(noise.shape) != tuple(distribution.mean.shape):
            raise ValueError('Explicit matching posterior noise required')
        noise = noise.to(device=distribution.mean.device, dtype=distribution.mean.dtype)
        return (distribution.mean+distribution.std*noise)*wrapper.scaling_factor
    wrapper.encode_images = encode


def gradients(named, prefixes):
    result = semantic_gradients(named)
    for key, scope in [('vae_tail', prefixes),
            ('vae_encoder', ('base.vae_wrapper.vae.encoder.', 'base.vae_wrapper.vae.quant_conv.')),
            ('vae_decoder', ('base.vae_wrapper.vae.decoder.', 'base.vae_wrapper.vae.post_quant_conv.')),
            ('clip', ('base.text_encoder.backbone.',)),
            ('text_projection', ('base.text_encoder.projection.',))]:
        result[key] = sum(float(p.grad.detach().square().sum()) for n, p in named
                          if n.startswith(scope) and p.grad is not None)
    return result


def optimizer_groups(named, config, full, prefixes):
    buckets = {}
    for name, param in named:
        if not param.requires_grad:
            continue
        key = parameter_group(name, prefixes)
        buckets.setdefault(key, []).append(param)
    rates = dict(encoder=config['encoder_lr'], mapping=config['mapping_lr'],
                 decoder=config['decoder_lr'], tail=config['tail_lr'],
                 backbone=config['backbone_lr'], auxiliary=config['auxiliary_lr'])
    return [dict(params=value, lr=rates[key], weight_decay=1e-5 if key in ('encoder','mapping','decoder') else 0.)
            for key, value in sorted(buckets.items())]
