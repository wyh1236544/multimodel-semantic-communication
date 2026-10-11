"""Strict inference-only tensor application; no optimizer states restored."""
import hashlib
from fso_jscc_eval_v1.core import check_delta_metadata


def advance_parent_vae_rng(wrapper, zero_image, count, budget=None):
    """Replay omitted train-cache RNG draws WITHOUT reading any training image.

    The parent's frozen VAE samples its posterior once per cached image. A real
    zero-image encode checks which RNG stream/shape is consumed on this runtime;
    only if a same-shaped Gaussian consumes IDENTICAL CPU/device RNG states do
    we replay count draws. Validation input encodes then start at the old state.
    """
    import torch
    if type(count) is not int or not 0 <= count <= 10000: raise ValueError('Invalid parent cache count')
    device=zero_image.device
    def state():
        s=dict(cpu=torch.get_rng_state().clone())
        if device.type=='cuda': s['cuda']=torch.cuda.get_rng_state(device).clone()
        return s
    def restore(s):
        torch.set_rng_state(s['cpu'])
        if device.type=='cuda': torch.cuda.set_rng_state(s['cuda'],device)
    def fingerprints(s): return {k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in s.items()}
    start=state()
    with torch.no_grad():
        try:
            probe=wrapper.encode_images(zero_image); after_encode=state()
            if not torch.isfinite(probe).all(): raise ValueError('Nonfinite VAE probe')
            restore(start); torch.randn_like(probe); after_gaussian=state()
            if any(not torch.equal(after_encode[k],after_gaussian[k]) for k in start):
                raise RuntimeError('VAE RNG consumption differs from one Gaussian; cannot safely reproduce old latent cache')
        finally: restore(start)
        for i in range(count):
            if budget is not None and i%100==0: budget()
            torch.randn_like(probe)
    return dict(parent_omitted_cache_draws=count,latent_shape=list(probe.shape),latent_dtype=str(probe.dtype),
        exact_cpu_and_device_rng_contract_checked=True,start_rng_sha256=fingerprints(start),end_rng_sha256=fingerprints(state()),
        training_images_read=False,training_performed=False)


def apply_delta(model, named, payload, record, source_design, base_hash, a_hash, split_hash):
    import torch
    check_delta_metadata(payload, record, source_design, base_hash, a_hash, split_hash)
    delta = payload.get('delta', {})
    if set(delta) != {n for n, _ in named}:
        raise ValueError('Exact trained tensor names required')
    for n, q in named:
        v = delta[n]
        if not torch.is_tensor(v) or v.shape != q.shape or v.dtype != q.dtype or not bool(torch.isfinite(v).all()):
            raise ValueError('Wrong/nonfinite delta tensor: '+n)
    with torch.no_grad():
        for n, q in named:
            q.copy_(delta[n].to(q.device))
    model.requires_grad_(False); model.eval()
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('Frozen evaluator unexpectedly has trainable parameters')
