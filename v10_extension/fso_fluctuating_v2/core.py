"""Correlated FSO paths and an explicitly approximate stochastic DSP surrogate."""
import copy
import json
import math
from pathlib import Path
import numpy as np
from fso_jscc_v1.core import digest, seed_for
from optical_link_v1.config import load as optical_load, validate as optical_validate
from optical_link_v1.physics import gamma_parameters, aperture_fraction
from optical_realism_v3.spec import load_plan, resources
from optical_realism_v3.transport import RealisticTransport

TEMPLATE = Path(__file__).with_name('config.json')
OFFSETS = (-2, -1, 0, 1, 2)


def load(path=TEMPLATE):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    if set(c) != set(json.loads(TEMPLATE.read_text())) or c['schema'] != 'fso-fluctuating-v2':
        raise ValueError('Unknown/missing configuration key')
    for k, lo, hi in [('seed', 0, 2**31-1), ('train_samples', 8, 10000), ('epochs', 1, 8),
                     ('grad_accum', 1, 8), ('warmup_optimizer_steps', 0, 500),
                     ('bank_train_frames', 8, 256), ('bank_holdout_frames', 4, 64),
                     ('sequence_window_frames', 2, 64), ('val_samples', 2, 128),
                     ('draws', 1, 4), ('save_images', 0, 8), ('bootstrap_resamples', 100, 10000)]:
        if type(c[k]) is not int or not lo <= c[k] <= hi:
            raise ValueError('Invalid integer '+k)
    for k, v in c.items():
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError('Nonfinite '+k)
    if (c['fs_hz'], c['sps'], c['rolloff'], c['span'], c['pilots']) != (64e9, 4, .1, 32, 256):
        raise ValueError('Fixed AWG64GSa/s/4SPS/RRC.1/span32/pilot256')
    if not 0 < c['frame_spacing_s'] <= c['coherence_time_s'] or not 1e-4 <= c['coherence_time_s'] <= .1:
        raise ValueError('Temporal coherence/spacing inconsistent')
    if c['save_images'] > c['val_samples'] or c['sequence_window_frames'] > c['bank_train_frames']:
        raise ValueError('Subset exceeds parent budget')
    for k in ('encoder_lr', 'decoder_lr', 'mapping_lr'):
        if not 0 < c[k] <= .001: raise ValueError('Invalid learning rate')
    if not 0 <= c['clean_anchor_fraction'] <= .5 or not 0 < c['arm_max_hours'] <= 6:
        raise ValueError('Invalid anchor/time budget')
    if not 0 <= c['papr_weight'] <= .01 or not 3 <= c['papr_target_db'] <= 15:
        raise ValueError('Invalid PAPR constraint')
    if not .05 <= c['surrogate_evm_relative_tolerance'] <= 1 or not 0 <= c['surrogate_acceptance_difference_tolerance'] <= .5:
        raise ValueError('Invalid declared surrogate tolerances')
    if set(c['weather']) != {'weak', 'moderate', 'strong'}: raise ValueError('Fixed weather schema')
    for p in c['weather'].values():
        if set(p) != {'cn2', 'alpha', 'beta', 'jitter_rad'}: raise ValueError('Unknown weather parameter')
        if not 0 <= p['jitter_rad'] <= .001: raise ValueError('Invalid pointing jitter')
        if p['cn2'] is None:
            if not all(isinstance(p[k], (int, float)) and math.isfinite(p[k]) and p[k] > 0 for k in ('alpha', 'beta')):
                raise ValueError('Manual Gamma-Gamma needs alpha/beta')
        elif not 0 < p['cn2'] <= 1e-13 or p['alpha'] is not None or p['beta'] is not None:
            raise ValueError('Use either Cn2 or manual alpha/beta')
    ids = []
    for p in c['conditions']:
        if set(p) != {'id', 'weather', 'launch_signal_dbm'} or p['weather'] not in c['weather']:
            raise ValueError('Invalid condition')
        if not p['id'] or any(ch not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for ch in p['id']):
            raise ValueError('Unsafe condition ID')
        if not -40 <= p['launch_signal_dbm'] <= 0: raise ValueError('Launch power outside screen')
        ids.append(p['id'])
    if not 1 <= len(ids) <= 12 or len(set(ids)) != len(ids) or not c['primary_conditions'] or not set(c['primary_conditions']) <= set(ids):
        raise ValueError('Duplicate/missing conditions')
    environment(c)  # Validate SI ranges and plane-wave domain before writing.
    return c


def environment(c):
    _, old, _ = optical_load()
    plan, _ = load_plan(repeats=1)
    cases = {}
    for name, p in c['weather'].items():
        optical = copy.deepcopy(old['good'])
        optical['laser']['wavelength_m'] = c['wavelength_m']
        optical['path'].update(distance_m=c['distance_m'], attenuation_db_per_km=c['attenuation_db_per_km'])
        optical['pointing'].update(divergence_half_angle_rad=c['beam_divergence_half_angle_rad'],
            receiver_aperture_diameter_m=c['receiver_aperture_diameter_m'], jitter_per_axis_rad=p['jitter_rad'])
        optical['turbulence'].update(mode='cn2_plane_wave' if p['cn2'] is not None else 'manual_gamma_gamma',
            cn2_m_minus_2_over_3=p['cn2'], alpha=p['alpha'], beta=p['beta'], coherence_time_s=c['coherence_time_s'])
        optical['waveform']['policy'] = plan['digital_policy']
        optical_validate(optical)
        gamma_parameters(optical['turbulence'], c['distance_m'], c['wavelength_m'])
        cases[name] = optical
    return cases, plan, resources(cases['weak'], plan)


def gamma_step(x, shape, rho, rng):
    """Exact CIR transition: stationary Gamma(shape,scale=1/shape), corr=rho."""
    if rho == 1.: return x
    scale = (1-rho)/(2*shape)
    return float(scale*rng.noncentral_chisquare(2*shape, 2*shape*rho*x/(1-rho)))


def paths(c, weather, count, identity):
    optical = environment(c)[0][weather]
    alpha, beta, rytov = gamma_parameters(optical['turbulence'], c['distance_m'], c['wavelength_m'])
    rng = np.random.default_rng(seed_for(c['seed'], identity, weather, 'weather'))
    rho = math.exp(-c['frame_spacing_s']/c['coherence_time_s'])
    g1, g2 = rng.gamma(alpha, 1/alpha), rng.gamma(beta, 1/beta)
    xy = rng.standard_normal(2)
    p, point = optical['path'], optical['pointing']
    beam = math.hypot(point['beam_waist_m'], p['distance_m']*point['divergence_half_angle_rad'])
    atm = 10**(-p['attenuation_db_per_km']*p['distance_m']/10000)
    out = []
    for i in range(count):
        if i:
            g1, g2 = gamma_step(g1, alpha, rho, rng), gamma_step(g2, beta, rho, rng)
            xy = rho*xy+math.sqrt(1-rho*rho)*rng.standard_normal(2)
        offset = p['distance_m']*point['jitter_per_axis_rad']*xy
        collection = aperture_fraction(beam, point['receiver_aperture_diameter_m']/2, float(np.linalg.norm(offset)))
        scint = float(g1*g2)
        aperture = atm*collection*scint
        coupled = aperture*10**(-p['rx_optics_loss_db']/10)*p['coupling_efficiency']
        phase = float(rng.uniform(-math.pi, math.pi))  # optical phase is NOT millisecond atmospheric amplitude
        out.append(dict(alpha=alpha, beta=beta, rytov_variance=rytov, gamma_gamma_irradiance=scint,
            atmosphere_loss_db=p['attenuation_db_per_km']*p['distance_m']/1000,
            atmosphere_transmission=atm, aperture_pointing_collection=collection,
            aperture_irradiance_gain=aperture, coupled_irradiance_gain=coupled,
            field_phase_rad=phase, field_real=math.sqrt(coupled)*math.cos(phase), field_imag=math.sqrt(coupled)*math.sin(phase),
            frame_index=i, weather_time_s=i*c['frame_spacing_s'], gamma_component_correlation=rho,
            temporal_model='stationary_CIR_Gamma_components_and_OU_pointing; constant_within_frame',
            parameter_status='assumed_not_hardware_calibrated'))
    return out


def transport(c, condition, path, noise_seed):
    cases, plan, resource = environment(c)
    t = RealisticTransport(cases[condition['weather']], seed_for(c['seed'], 'unused_initial_draw'), noise_seed,
        condition['launch_signal_dbm'], resource['slot_samples'], plan['laser_max_dbm'], plan['hardware'])
    t.path_draw = copy.deepcopy(path)  # simulator input only; never passed to RX/DSP
    return t


def public_payload(seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((288, 16))
    # Several public, non-business waveform distributions, no image or caption.
    if seed % 3 == 1: z = np.clip(z, -1.5, 1.5)
    if seed % 3 == 2: z = rng.laplace(size=z.shape)
    for a, b in ((0, 32), (32, 256), (256, 288)):
        z[a:b] /= math.sqrt(float(np.mean(z[a:b]**2)))
    return z.astype(np.float32)


def as_complex(x):
    a = np.asarray(x).reshape(-1, 2)
    return (a[:, 0]+1j*a[:, 1])/math.sqrt(2)


def design_matrix(x):
    x = np.asarray(x, complex).reshape(-1)
    cols = []
    for conjugate in (False, True):
        v = np.conj(x) if conjugate else x
        for shift in OFFSETS:
            z = np.zeros_like(v)
            if shift < 0: z[:shift] = v[-shift:]
            elif shift > 0: z[shift:] = v[:-shift]
            else: z = v.copy()
            cols.append(z)
    return np.stack(cols+[np.ones_like(x)], -1)


def fit_operator(tx, rx):
    x, y = as_complex(tx), as_complex(rx)
    matrix = design_matrix(x)
    coefficient = np.linalg.lstsq(matrix, y, rcond=1e-8)[0]
    residual = y-matrix@coefficient
    return coefficient.astype(np.complex64), residual.astype(np.complex64)


def apply_operator(x, coefficient, residual):
    return design_matrix(x)@coefficient+residual


def schedule(c, epoch, position, bank):
    if np.random.default_rng(seed_for(c['seed'], epoch, position, 'anchor')).random() < c['clean_anchor_fraction']:
        return None
    # Consecutive positions walk a correlated weather window; windows not ARQ.
    block, offset = divmod(position, c['sequence_window_frames'])
    rng = np.random.default_rng(seed_for(c['seed'], epoch, block, 'window'))
    condition = c['conditions'][int(rng.integers(len(c['conditions'])))]['id']
    indices = [i for i, r in enumerate(bank['records']) if r['partition'] == 'train' and r['condition'] == condition]
    start = int(rng.integers(0, len(indices)-c['sequence_window_frames']+1))
    return indices[start+offset]
