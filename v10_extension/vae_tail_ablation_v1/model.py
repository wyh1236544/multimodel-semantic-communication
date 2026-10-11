"""Only explicitly selected terminal VAE decoder parameters can be unfrozen."""
import torch
from fso_jscc_v1.model import parameters as semantic_parameters
from fso_jscc_v1.model import stage as semantic_stage, gradient_groups as semantic_gradients, ENCODER
from fso_fluctuating_v2.model import Frame, ContinuousChannel, surrogate_context
from vae_tail_ablation_v1.core import tail_prefixes


def candidates(model):
    decoder = model.base.vae_wrapper.vae.decoder
    prefixes = tail_prefixes(len(decoder.up_blocks))
    tail = [(n,p) for n,p in model.named_parameters() if n.startswith(prefixes)]
    if not tail or any(not any(n.startswith(prefix) for n,_ in tail) for prefix in prefixes):
        raise ValueError('VAE terminal up block/output layers missing; do not silently unfreeze a different module')
    return semantic_parameters(model)+tail, prefixes


def stage(model, warm, unfreeze_tail, prefixes):
    semantic_stage(model, warm)
    # .eval() controls stochastic layers, not gradients. Decoder forward in the
    # original wrapper deliberately has no no_grad/detach: keep that path intact.
    model.base.vae_wrapper.freeze = not unfreeze_tail
    for name,param in model.named_parameters():
        if name.startswith(prefixes):
            param.requires_grad_(unfreeze_tail)
        if (name.startswith('base.vae_wrapper.') and not name.startswith(prefixes)
                and param.requires_grad):
            raise RuntimeError('Unexpected VAE encoder/earlier decoder unfreeze')
        if name.startswith('base.text_encoder.') and param.requires_grad:
            raise RuntimeError('Unexpected CLIP unfreeze')
    model.eval()


def gradients(named, prefixes):
    groups = semantic_gradients(named)
    groups['vae_tail'] = sum(float(p.grad.detach().square().sum())
        for n,p in named if n.startswith(prefixes) and p.grad is not None)
    return groups


def apply_parent(model, named, payload):
    delta = payload.get('delta',{})
    if set(delta) != {n for n,_ in named}:
        raise ValueError('Exact V2 semantic tensor set required')
    for n,p in named:
        v = delta[n]
        if not torch.is_tensor(v) or v.shape != p.shape or v.dtype != p.dtype or not bool(torch.isfinite(v).all()):
            raise ValueError('Wrong/nonfinite parent tensor '+n)
    with torch.no_grad():
        for n,p in named:
            p.copy_(delta[n].to(p.device))


def verify_frozen(model, initial, allowed):
    for n,p in model.named_parameters():
        if n not in allowed and not torch.equal(p.detach().cpu(),initial[n]):
            raise RuntimeError('Frozen parameter changed: '+n)
