"""NumPy reference contracts; sampled channels are independent of neural arms."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from physical_link.waveform import rrc

TEMPLATE = Path(__file__).with_name('config.json')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def seed_for(*identity):
    data = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return int.from_bytes(hashlib.sha256(data.encode()).digest()[:8], 'little') % (2**63-1)


def load(path=TEMPLATE):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    template = json.loads(TEMPLATE.read_text(encoding='utf-8'))
    if set(value) != set(template) or value['schema'] != 'fso-jscc-v1':
        raise ValueError('Missing/unknown experiment key')
    for key, lo, hi in [('seed', 0, 2**31-1), ('train_samples', 2, 10000),
                        ('epochs', 1, 10), ('grad_accum', 1, 16),
                        ('warmup_optimizer_steps', 0, 1000), ('val_samples', 2, 500),
                        ('draws', 1, 8), ('save_images', 0, 16),
                        ('physical_val_samples', 1, 32), ('physical_draws', 1, 4),
                        ('bootstrap_resamples', 100, 10000)]:
        if type(value[key]) is not int or not lo <= value[key] <= hi:
            raise ValueError('Invalid integer: '+key)
    if value['save_images'] > value['val_samples'] or value['physical_val_samples'] > value['val_samples']:
        raise ValueError('Examples/physical subset exceed validation budget')
    for key in ('encoder_lr', 'decoder_lr', 'mapping_lr', 'arm_max_hours'):
        if isinstance(value[key], bool) or not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError('Invalid positive parameter: '+key)
    if value['arm_max_hours'] > 6 or not 0 <= value['papr_weight'] <= .1 or not 3 <= value['papr_target_db'] <= 15:
        raise ValueError('Unbounded optimizer/PAPR settings')
    if (value['fs_hz'], value['sps'], value['rolloff'], value['span'], value['pilots']) != (64e9, 4, .1, 32, 256):
        raise ValueError('V1 fixes AWG64GSa/s, 4SPS, RRC.1/span32 and pilot256')
    bands = value['train_snr_bands']
    if not bands or any(len(b) != 3 or not all(math.isfinite(x) for x in b) or
                        not -10 <= b[0] < b[1] <= 30 or b[2] <= 0 for b in bands):
        raise ValueError('Invalid nominal training SNR mixture')
    if abs(sum(b[2] for b in bands)-1) > 1e-9:
        raise ValueError('SNR mixture weights must sum to one')
    if not value['eval_snrs_db'] or len(set(value['eval_snrs_db'])) != len(value['eval_snrs_db']) or any(
            isinstance(x, bool) or not math.isfinite(x) or not -20 <= x <= 40 for x in value['eval_snrs_db']):
        raise ValueError('Invalid evaluation SNR grid')
    if set(value['profiles']) != {'fso_weak', 'fso_moderate', 'fso_strong'}:
        raise ValueError('Fixed three FSO intensity distributions required')
    for p in value['profiles'].values():
        if set(p) != {'alpha', 'beta', 'xi', 'a0', 'loss_db'} or any(
                isinstance(x, bool) or not math.isfinite(x) for x in p.values()):
            raise ValueError('Invalid FSO parameter schema')
        if min(p['alpha'], p['beta'], p['xi']) <= 0 or not 0 < p['a0'] <= 1 or not 0 <= p['loss_db'] <= 60:
            raise ValueError('Invalid turbulence/pointing/path loss')
    if not value['physical_conditions'] or len(set(value['physical_conditions'])) != len(value['physical_conditions']):
        raise ValueError('Need unique physical condition IDs')
    return value


def draw_gain(config, profile, seed):
    if profile in ('clean', 'awgn'):
        return 1.+0j, dict(intensity_turbulence=1., pointing_intensity=1., path_intensity=1.)
    p = config['profiles'][profile]
    rng = np.random.default_rng(seed)
    turbulence = float(rng.gamma(p['alpha'], 1/p['alpha'])*rng.gamma(p['beta'], 1/p['beta']))
    pointing = float(p['a0']*max(rng.random(), 1e-15)**(1/p['xi']**2))
    path = 10**(-p['loss_db']/10)
    # Gamma-Gamma describes IRRADIANCE. Coherent FIELD gain is its square root.
    gain = math.sqrt(turbulence*pointing*path)*np.exp(1j*rng.uniform(-math.pi, math.pi))
    return gain, dict(intensity_turbulence=turbulence, pointing_intensity=pointing, path_intensity=path)


def train_condition(config, epoch, position):
    rng = np.random.default_rng(seed_for(config['seed'], epoch, position, 'condition'))
    b = config['train_snr_bands'][int(rng.choice(len(config['train_snr_bands']), p=[x[2] for x in config['train_snr_bands']]))]
    profile = tuple(config['profiles'])[int(rng.integers(0, 3))]
    return profile, float(rng.uniform(b[0], b[1]))


def grid(config):
    return [('clean', 40.)]+[(p, float(s)) for p in ('awgn',)+tuple(config['profiles']) for s in config['eval_snrs_db']]


def kernels(config):
    t = np.arange(-config['span']*config['sps']//2, config['span']*config['sps']//2+1)/config['sps']
    h = rrc(t, config['rolloff'])
    return (h/math.sqrt(float(np.sum(h*h)))).astype(np.float32)


def observe(config, profile, snr, field_seed, noise_seed):
    """Pilot-only RX CSI. Fixed noise before fading; no per-frame SNR reset."""
    gain, diagnostics = draw_gain(config, profile, field_seed)
    if profile == 'clean':
        return gain, 0., gain, 0., 1., 40., diagnostics
    variance = 10**(-float(snr)/10)
    rng = np.random.default_rng(noise_seed)
    n = (rng.standard_normal(config['pilots'])+1j*rng.standard_normal(config['pilots']))/math.sqrt(2)
    pilot = gain+n*math.sqrt(variance)
    estimated = complex(pilot.mean())
    residual = max(float(np.sum(abs(pilot-estimated)**2)/(config['pilots']-1)), 1e-10)
    error = residual/config['pilots']
    reliability = abs(estimated)**2/(abs(estimated)**2+residual+error)
    pilot_snr = 10*math.log10(max(abs(estimated)**2-error, 1e-12)/(residual+error))
    return gain, variance, estimated, residual, reliability, pilot_snr, diagnostics


def numpy_channel(values, config, profile, snr, field_seed, noise_seed):
    gain, variance, estimate, residual, rel, pilot_snr, details = observe(config, profile, snr, field_seed, noise_seed)
    x = np.asarray(values, np.float32).reshape(-1, 2)
    symbol = (x[:, 0]+1j*x[:, 1])/math.sqrt(2)
    rng = np.random.default_rng(seed_for(noise_seed, 'payload'))
    n = (rng.standard_normal(len(symbol))+1j*rng.standard_normal(len(symbol)))/math.sqrt(2)
    coefficient = 1. if profile == 'clean' else np.conj(estimate)/(abs(estimate)**2+residual*(1+1/config['pilots']))
    restored = (gain*symbol+math.sqrt(variance)*n)*coefficient
    out = np.stack((restored.real, restored.imag), -1).reshape(np.shape(values))*math.sqrt(2)
    return out.astype(np.float32), rel, dict(details, nominal_snr_db=snr,
        true_postfade_snr_db=None if profile == 'clean' else snr+10*math.log10(max(abs(gain)**2, 1e-30)),
        pilot_snr_db=pilot_snr, noise_variance_before_fade=variance)
