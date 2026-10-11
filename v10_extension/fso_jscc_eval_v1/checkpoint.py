"""Strict inference-only tensor application; no optimizer states restored."""
from fso_jscc_eval_v1.core import check_delta_metadata


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
