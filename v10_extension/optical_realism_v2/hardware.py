"""Assumed device imperfections, no RX access to simulator timing/phase truth."""
import copy
import json
import math
from pathlib import Path
import numpy as np

TEMPLATE = Path(__file__).with_name('hardware.json')
PRESETS = ('ideal', 'carrier', 'clock', 'hardware', 'combined')


def load(path=TEMPLATE, preset='combined'):
    h = json.loads(Path(path).read_text(encoding='utf-8'))
    template = json.loads(TEMPLATE.read_text(encoding='utf-8'))
    if set(h) != set(template) or h['schema'] != 'optical-realism-v2' or preset not in PRESETS:
        raise ValueError('Unknown/incomplete hardware configuration or preset')
    if h['provenance'] != 'assumed_not_hardware_calibrated':
        raise ValueError('This release is not a hardware calibration workflow')
    for key in set(h)-{'schema', 'provenance'}:
        if isinstance(h[key], bool) or not isinstance(h[key], (int, float)) or not math.isfinite(h[key]):
            raise ValueError('Finite number required: '+key)
    bounds = dict(tx_linewidth_hz=(0, 1e6), lo_linewidth_hz=(0, 1e6),
        frequency_offset_hz=(-100e6, 100e6), adc_clock_ppm=(-100, 100),
        fractional_delay_samples=(-1, 1), adc_jitter_rms_seconds=(0, 2e-12),
        dac_effective_bits=(3, 8), adc_effective_bits=(3, 12), tx_iq_gain_db=(-2, 2),
        tx_iq_skew_samples=(-.5, .5), mzm_quadrature_error_deg=(-10, 10),
        pilot_interval=(64, 512), pilot_block=(16, 64), timing_search_samples=(8, 32),
        timing_tracking_radius_samples=(.5, 3), timing_max_ppm=(100, 500),
        minimum_pilot_snr_db=(-15, 20), minimum_sync_score=(.05, .95),
        maximum_pilot_phase_jump_rad=(1, math.pi))
    for key, (lo, hi) in bounds.items():
        if not lo <= h[key] <= hi: raise ValueError('Unsupported hardware value: '+key)
    for key in ('pilot_interval','pilot_block','timing_search_samples'):
        if type(h[key]) is not int or h[key] % 2: raise ValueError('Even integer required: '+key)
    if preset not in ('carrier', 'combined'):
        h.update(tx_linewidth_hz=0., lo_linewidth_hz=0., frequency_offset_hz=0.)
    if preset not in ('clock', 'combined'):
        h.update(adc_clock_ppm=0., fractional_delay_samples=0., adc_jitter_rms_seconds=0.)
    if preset not in ('hardware', 'combined'):
        h.update(dac_effective_bits=8., adc_effective_bits=12., tx_iq_gain_db=0.,
                 tx_iq_skew_samples=0., mzm_quadrature_error_deg=0.)
    h['preset'] = preset
    return h


def optical_config(base, h):
    c = copy.deepcopy(base)
    c['laser']['linewidth_hz'] = h['tx_linewidth_hz']
    c['receiver'].update(lo_linewidth_hz=h['lo_linewidth_hz'], frequency_offset_hz=h['frequency_offset_hz'])
    c['modulator']['quadrature_error_rad'] += math.radians(h['mzm_quadrature_error_deg'])
    # The old QAM DD receiver is NEVER called by this module.
    return c


def interpolate(value, at):
    """Cubic Lagrange interpolation, zero outside captured support; no SciPy."""
    value = np.asarray(value, complex)
    at = np.asarray(at, float)
    n = np.floor(at).astype(np.int64); u = at-n
    weights = (-u*(u-1)*(u-2)/6, (u+1)*(u-1)*(u-2)/2,
               -(u+1)*u*(u-2)/2, (u+1)*u*(u-1)/6)
    result = np.zeros(at.shape, complex)
    for shift, weight in zip((-1, 0, 1, 2), weights):
        ix = n+shift; valid = (ix >= 0) & (ix < len(value))
        result += value[np.clip(ix, 0, len(value)-1)]*weight*valid
    return result


def extra_converter_noise(value, fullscale, bits, effective_bits, rng):
    """White-noise ENOB surrogate, subtract ideal quantizer variance (no double count).

    Not a measured SINAD/noise spectrum; ENOB is referenced to full scale per axis.
    """
    if effective_bits > bits: raise ValueError('ENOB exceeds nominal resolution')
    extra = (2*fullscale)**2/12*(2**(-2*effective_bits)-2**(-2*bits))
    if extra == 0: return np.asarray(value, complex)
    return value + math.sqrt(extra)*(rng.standard_normal(len(value))+1j*rng.standard_normal(len(value)))


def tx_impair(value, h, rng):
    y = extra_converter_noise(value, 4., 8, h['dac_effective_bits'], rng)
    q = interpolate(1j*y.imag, np.arange(len(y))-h['tx_iq_skew_samples']).imag
    # Gain mismatch, bounded driver swing; no source-dependent normalization.
    y = y.real*10**(h['tx_iq_gain_db']/40)+1j*q*10**(-h['tx_iq_gain_db']/40)
    clipped = float(np.mean((abs(y.real)>4)|(abs(y.imag)>4)))
    return np.clip(y.real,-4,4)+1j*np.clip(y.imag,-4,4), clipped


def sample_clock(value, h, fs, rng):
    at = np.arange(len(value))/(1+h['adc_clock_ppm']*1e-6)+h['fractional_delay_samples']
    if h['adc_jitter_rms_seconds']:
        at += rng.standard_normal(len(value))*h['adc_jitter_rms_seconds']*fs
    return interpolate(value, at), at
