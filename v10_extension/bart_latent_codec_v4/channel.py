"""Pilot-estimated symbol proxy for training; fresh full V3 waveforms for eval.

Proxy SNR is nominal unit-symbol energy/noise, not an optical link budget.
The proxy omits DAC/MZM/clock/synchronization/temporal fading; no hardware claim.
"""
import copy
import math
import numpy as np
from fso_jscc_v1.core import observe, seed_for
from fso_fluctuating_v2.core import environment
from optical_realism_v3.transport import RealisticTransport
from optical_realism_v3.reliability import combine
from optical_realism_v3.spec import sampled_count
from bart_latent_codec_v4.core import geometry


def proxy_settings(c):
    return dict(profiles=c['proxy_profiles'], pilots=c['proxy_pilots'])


def proxy_numpy(x, c, profile, snr, field_seed, noise_seed):
    cfg = proxy_settings(c)
    gain, variance, estimate, residual, rel, pilot_snr, details = observe(cfg, profile, snr, field_seed, noise_seed)
    values = np.asarray(x, np.float32)
    if values.ndim != 1 or len(values)%2 or not np.isfinite(values).all(): raise ValueError('Finite even wire required')
    z = (values[::2]+1j*values[1::2])/math.sqrt(2)
    rng = np.random.default_rng(seed_for(noise_seed, 'payload'))
    noise = (rng.standard_normal(len(z))+1j*rng.standard_normal(len(z)))/math.sqrt(2)
    coefficient = 1. if profile == 'clean' else np.conj(estimate)/(abs(estimate)**2+residual*(1+1/cfg['pilots']))
    y = (z*gain+noise*math.sqrt(variance))*coefficient
    restored = np.stack((y.real, y.imag), -1).reshape(-1)*math.sqrt(2)
    stats = dict(details, profile=profile, nominal_snr_db=snr, pilot_snr_db=pilot_snr,
        true_postfade_snr_db=None if variance == 0 else snr+10*math.log10(max(abs(gain)**2, 1e-30)),
        noise_variance_before_fade=variance, payload_complex_symbols=len(z),
        pilot_complex_symbols=cfg['pilots'], forward_complex_symbols=len(z)+cfg['pilots'],
        analog_repeats=1, raw_utf8_wire_bytes=0, receiver_available=True,
        no_waveform_sync_or_receiver_rejection=True, approximate_training_only=True)
    return restored.astype(np.float32), complex(gain*coefficient), float(variance*abs(coefficient)**2), stats


def proxy_tensor(wire, c, update, ids):
    import torch
    coefficients = []; noises = []; stats = []
    profiles = tuple(sorted(c['proxy_profiles']))
    for j, iid in enumerate(ids):
        profile = profiles[update%len(profiles)]; snr = c['proxy_snr_db'][(update//len(profiles))%len(c['proxy_snr_db'])]
        field_seed = seed_for(c['seed'], 'v4-field', update, iid, j)
        noise_seed = seed_for(c['seed'], 'v4-noise', update, iid, j)
        zero = np.zeros(wire.shape[1], np.float32)
        y0, coefficient, _, detail = proxy_numpy(zero, c, profile, snr, field_seed, noise_seed)
        coefficients.append(coefficient); noises.append(y0); stats.append(detail)
    pair = wire.reshape(wire.shape[0], -1, 2)
    coeff = torch.tensor(coefficients, device=wire.device, dtype=torch.complex64)
    symbols = torch.complex(pair[..., 0], pair[..., 1])/math.sqrt(2)
    y = symbols*coeff[:, None]
    restored = torch.stack((y.real, y.imag), -1).reshape_as(wire)*math.sqrt(2)
    return restored+torch.as_tensor(np.stack(noises), device=wire.device, dtype=wire.dtype), stats


def resource(c, width):
    cases, plan, cap = environment(c); w = copy.deepcopy(cases['weak']['waveform'])
    w.update(pilot_interval=plan['hardware']['pilot_interval'], pilot_block=plan['hardware']['pilot_block'])
    count = sampled_count(0, w)+sampled_count(geometry(width)['payload_complex_symbols'], w)
    if count > cap['slot_samples']: raise ValueError('Text frame exceeds old shared slot cap')
    return dict(geometry(width), scheduled_samples=count, scheduled_seconds=count/w['fs_hz'],
        fs_hz=w['fs_hz'], sps=w['sps'], symbol_rate_baud=w['fs_hz']/w['sps'], rolloff=w['rolloff'],
        all_probe_pilot_preamble_tails_guards_charged=True, exact_equal_energy_claim=False)


def send(c, condition, path, seed, width, values):
    cases, plan, cap = environment(c); res = resource(c, width)
    t = RealisticTransport(cases[condition['weather']], seed_for(c['seed'], 'v4-unused-fade'), seed,
        condition['launch_signal_dbm'], cap['slot_samples'], plan['laser_max_dbm'], plan['hardware'])
    t.path_draw = copy.deepcopy(path); pilot = float(t.probe())
    x = np.asarray(values, np.float32)
    if x.shape != (geometry(width)['total_reals'],) or not np.isfinite(x).all(): raise ValueError('Invalid full text payload')
    z = (x[::2]+1j*x[1::2])/math.sqrt(2)
    observed, event = t.burst(z, 'features', policy='analog')
    y, reliability = combine(observed[None], event['pilot_residual_variance_before_gate'],
        event['pilot_correlated_error_floor'], event['receiver_locked'], 'lmmse')
    restored = np.stack((y.real, y.imag), -1).reshape(-1)*math.sqrt(2)
    if not np.isfinite(restored).all(): raise FloatingPointError('Nonfinite waveform receive')
    stats = dict(t.totals(), receiver_available=bool(event['receiver_locked']), pilot_snr_db=pilot,
        analog_reliability=float(reliability['analog_reliability']), analog_repeats=1,
        payload_complex_symbols=len(z), raw_utf8_wire_bytes=0, pre_fec_ber=None, post_fec_ber=None,
        analog_feature_mse=float(np.mean((restored-x)**2)), resources=res)
    return restored.astype(np.float32), stats, t
