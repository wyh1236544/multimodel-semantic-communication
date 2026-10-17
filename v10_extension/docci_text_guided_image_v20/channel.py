"""Unit-gain AWGN with charged pilots; no noisy-frame renormalization."""
import math
import numpy as np
from docci_text_guided_image_v20.core import seed, array_sha

def sample(n, snr, value):
    if snr is None:
        return np.zeros(n,np.float32)
    if snr not in (0,5,10,15,20):
        raise ValueError('Unsupported primary AWGN condition')
    return (np.random.Generator(np.random.PCG64(value)).standard_normal(n).astype(np.float32)
            *np.float32(10.**(-snr/20.)))

def channel(wire, snr, epoch, eid, kind, phase, repeat=0):
    import torch
    n = 8192 if kind=='image' else 18432 if kind=='text' else 0
    if wire.shape != (1,n) or not n or not torch.isfinite(wire).all():
        raise ValueError('Public single-description frame required')
    power = float(wire.detach().square().mean())
    if abs(power-1.) > .0002:
        raise ValueError('TX unit power violated')
    value = seed(42,'v20-AWGN',phase,epoch,eid,kind,snr,repeat)
    epsilon = sample(n,snr,value)
    # 32 complex pilots per modality = 64 real observations. Pilot mean is known,
    # channel gain is fixed one in THIS AWGN protocol. This is not fading CSI.
    pilot = sample(64,snr,seed(value,'pilot'))
    n0_hat = float(np.mean(pilot.astype(np.float64)**2))
    estimate = 40. if snr is None else max(-40.,min(40.,-10*math.log10(max(n0_hat,1e-12))))
    reliability = 1./(1.+n0_hat)
    observed = wire+torch.as_tensor(epsilon[None],device=wire.device)
    pn = float(np.mean(epsilon.astype(np.float64)**2))
    stats = dict(nominal_snr_db=snr,noise_seed=value,noise_sha256=array_sha(epsilon),
                 pilot_sha256=array_sha(pilot),pilot_noise_variance=n0_hat,
                 estimated_snr_db=None if snr is None else estimate,
                 clean_conditioning_sentinel=40. if snr is None else None,
                 realized_snr_db=None if snr is None else 10*math.log10(power/pn),
                 signal_power=power,noise_power=pn,post_noise_renormalization=False)
    return observed,wire.new_tensor([estimate]),wire.new_tensor([reliability]),stats
