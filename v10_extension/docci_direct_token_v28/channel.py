"""Unit-gain complex-equivalent AWGN. 32 charged complex pilots, R=1."""
import hashlib
import math
import numpy as np
from docci_direct_token_v28.core import WIDTH, seed


def noise(n, snr, value):
    if snr is None:
        return np.zeros(n, np.float32)
    if type(snr) not in (int, float) or not math.isfinite(snr) or not -10 <= snr <= 20:
        raise ValueError('Unsupported finite SNR')
    return (np.random.Generator(np.random.PCG64(value)).standard_normal(n).astype(np.float32)
            *np.float32(10.**(-snr/20.)))


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def constants(task, snr, value):
    epsilon = noise(WIDTH[task], snr, value)
    pilot = noise(64, snr, seed(value, 'pilot'))
    pn = float(np.mean(epsilon.astype(np.float64)**2))
    n0 = float(np.mean(pilot.astype(np.float64)**2))
    estimate = 40. if snr is None else max(-40., min(40., -10*math.log10(max(n0, 1e-12))))
    return epsilon, dict(nominal_snr_db=snr, noise_seed=value, noise_sha256=array_sha(epsilon),
        pilot_sha256=array_sha(pilot), noise_power=pn, pilot_noise_power=n0,
        estimated_snr_db=None if snr is None else estimate,
        conditioning_snr=estimate, reliability=1./(1.+n0),
        clean_sentinel_NOT_measured_SNR=snr is None, post_noise_normalization=False)


def transmit(wire, task, snr, value):
    import torch
    if wire.shape != (1, WIDTH[task]) or not torch.isfinite(wire).all():
        raise ValueError('Wrong finite public symbol geometry')
    power = float(wire.detach().square().mean())
    if abs(power-1.) > .0002:
        raise ValueError('TX power is not one')
    epsilon, stats = constants(task, snr, value)
    # Only the noise is a constant. wire stays in the autograd graph.
    observed = wire+torch.as_tensor(epsilon[None], device=wire.device, dtype=wire.dtype)
    stats.update(signal_power=power, realized_snr_db=None if snr is None else
        10*math.log10(power/stats['noise_power']),
        tx_sha256=array_sha(wire.detach().float().cpu().numpy()),
        observed_sha256=array_sha(observed.detach().float().cpu().numpy()))
    return observed, stats
