"""Full-frame training proxy and fresh learned-payload holdout gate.

Final evaluation always uses the existing full waveform simulator, never proxy.
"""
import math
import numpy as np
import torch
from fso_fluctuating_v2.core import OFFSETS, seed_for, schedule, transport, as_complex, apply_operator
from bart_clip_joint_v1.data import source_wire


def clean(wire):
    b = wire.shape[0]
    return wire, wire.new_ones(b), wire.new_full((b,), 40.), torch.ones(b, device=wire.device, dtype=torch.bool)


def noisy(wire, bank, channel_config, epoch, position):
    if wire.shape != (1, 288, 16): raise ValueError('Single full physical frame required')
    index = schedule(channel_config, epoch, position, bank)
    if index is None: return clean(wire)
    r = bank['records'][index]
    if r['partition'] != 'train': raise ValueError('Holdout operator in training')
    pairs = wire.reshape(-1, 2); x = torch.complex(pairs[:, 0], pairs[:, 1])/math.sqrt(2)
    cols = []
    for v in (x, x.conj()):
        for shift in OFFSETS:
            if shift < 0: z = torch.cat((v[-shift:], v.new_zeros(-shift)))
            elif shift > 0: z = torch.cat((v.new_zeros(shift), v[:-shift]))
            else: z = v
            cols.append(z)
    matrix = torch.stack(cols+[torch.ones_like(x)], -1)
    rng = np.random.default_rng(seed_for(channel_config['seed'], epoch, position, 'new-bart-residual'))
    residual = np.roll(bank['residuals'][index], int(rng.integers(len(x))))*np.exp(1j*rng.uniform(-math.pi, math.pi))
    y = matrix@torch.as_tensor(bank['coefficients'][index], device=x.device, dtype=x.dtype)+torch.as_tensor(residual, device=x.device, dtype=x.dtype)
    out = torch.stack((y.real, y.imag), -1).reshape_as(wire)*math.sqrt(2)
    return out, wire.new_tensor([r['reliability']]), wire.new_tensor([r['probe_snr_db']]), torch.tensor([r['available']], device=x.device)


@torch.no_grad()
def fidelity(model, items, bank, cc, cfg, stage, device, budget):
    rows = []
    for condition in cc['conditions']:
        selected = [r for r in bank['records'] if r['partition'] == 'holdout' and r['condition'] == condition['id']]
        if len(selected) < cfg['fidelity_frames_per_condition']: raise ValueError('Insufficient full-frame holdouts')
        for ix, r in enumerate(selected[:cfg['fidelity_frames_per_condition']]):
            budget(); item = items[ix % len(items)]
            if item['split'] != 'train': raise ValueError('Validation business in calibration')
            tx = source_wire(model, [item], device)[0].cpu().numpy()
            ns = seed_for(cfg['seed'], stage, condition['id'], ix, 'bart-fidelity')
            t = transport(cc, condition, r['path'], ns); t.probe()
            received, _, stats = t.continuous(tx, 1, 'lmmse')
            rng = np.random.default_rng(seed_for(ns, 'residual'))
            residual = np.roll(bank['residuals'][r['index']], int(rng.integers(2304)))*np.exp(1j*rng.uniform(-math.pi, math.pi))
            predicted = apply_operator(as_complex(tx), bank['coefficients'][r['index']], residual)
            rows.append(dict(stage=stage, condition=condition['id'], source_image_id=item['image_id'],
                source_split='train', available_fit=bool(r['available']), available_fresh=bool(stats['analog_receiver_available']),
                actual_evm2=float(np.mean(abs(as_complex(received)-as_complex(tx))**2)),
                predicted_evm2=float(np.mean(abs(predicted-as_complex(tx))**2))))
    groups = {}; passed = True
    for condition in cc['conditions']:
        rr = [r for r in rows if r['condition'] == condition['id']]
        usable = [r for r in rr if r['available_fit'] and r['available_fresh']]
        difference = float(np.mean([r['available_fit'] != r['available_fresh'] for r in rr]))
        a = float(np.mean([r['actual_evm2'] for r in usable])) if usable else None
        p = float(np.mean([r['predicted_evm2'] for r in usable])) if usable else None
        relative = abs(p-a)/max(a, 1e-8) if usable else None
        good = len(usable) >= 2 and relative <= cc['surrogate_evm_relative_tolerance'] and difference <= cc['surrogate_acceptance_difference_tolerance']
        groups[condition['id']] = dict(passed=bool(good), frames=len(rr), usable=len(usable),
                                      relative_mean_evm_error=relative, availability_disagreement=difference)
        passed = passed and good
    return dict(passed=bool(passed), stage=stage, groups=groups, rows=rows,
                training_sources_only=True, fresh_full_waveform=True, no_threshold_tuning=True)


def papr(wire, cc):
    from fso_jscc_v1.core import kernels
    import torch.nn.functional as F
    x = (wire.reshape(1, -1, 2)/math.sqrt(2)).transpose(1, 2).reshape(2, 1, -1)
    taps = torch.as_tensor(kernels(cc), device=x.device).reshape(1, 1, -1)
    waveform = F.conv_transpose1d(x, taps, stride=cc['sps']).reshape(1, 2, -1)
    power = waveform.square().sum(1); relative = power/power.mean(1, keepdim=True).clamp_min(1e-8)
    return F.relu(relative-10**(cc['papr_target_db']/10)).square().mean()
