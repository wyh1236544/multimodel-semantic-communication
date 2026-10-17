"""Differentiable real-symbol AWGN; receiver receives no source masks/SNR truth."""
import hashlib
import math
import numpy as np
from docci_text_channel_v18.core import noise

def tensor_sha(x): return hashlib.sha256(x.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()

def transmit(wire,snr,value):
    import torch
    if wire.ndim!=2 or wire.shape[1]!=18432 or not torch.isfinite(wire).all(): raise ValueError('Public 18432-real frame required')
    if wire.shape[0]!=1: raise ValueError('One independently seeded description per call')
    epsilon=noise(18432,snr,value)
    observed=wire+torch.as_tensor(epsilon[None],device=wire.device)
    power=float(wire.detach().square().mean())
    pn=float(np.mean(epsilon.astype(np.float64)**2))
    if abs(power-1)>.0002: raise ValueError('TX per-slot unit power violated')
    stats=dict(nominal_snr_db=snr,realized_snr_db=None if snr is None else 10*math.log10(power/pn),
        signal_power=power,noise_power=pn,noise_seed=value,epsilon_sha256=hashlib.sha256(epsilon.tobytes()).hexdigest(),
        tx_sha256=tensor_sha(wire),observed_sha256=tensor_sha(observed),post_noise_normalization=False)
    return observed,stats

def objective(model,ids,hidden,observed):
    import torch.nn.functional as F
    from bart_latent_codec_v4.model import prediction
    rx=model.receive(observed)
    flat=ids.flatten(0,1)
    logits,labels=prediction(model.bart,flat,flat.ne(1),rx['memory'].flatten(0,1),rx['receiver_mask'].flatten(0,1))
    total=F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100,reduction='sum')
    return total,int(labels.ne(-100).sum())
