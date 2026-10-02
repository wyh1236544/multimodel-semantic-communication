"""Strict configuration, SI units, mutually exclusive physical/manual modes."""
import copy
import hashlib
import json
import math
from pathlib import Path

TEMPLATE = Path(__file__).with_name('optical_config.json')


def merge(base, change):
    result = copy.deepcopy(base)
    for key, value in change.items():
        if key not in result:
            raise ValueError('Unknown configuration key: '+key)
        if isinstance(result[key], dict):
            if not isinstance(value, dict):
                raise ValueError('Expected parameter group: '+key)
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def number(group, key, lo=0., hi=None, positive=False):
    value = group[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Finite number required: '+key)
    if value < lo or (positive and value == lo) or (hi is not None and value > hi):
        raise ValueError('Parameter out of range: '+key)
    return float(value)


def validate(c):
    w, laser, mod, path, turb, point, rx = [c[k] for k in
        ('waveform', 'laser', 'modulator', 'path', 'turbulence', 'pointing', 'receiver')]
    fs = number(w, 'fs_hz', positive=True)
    sps = number(w, 'sps', 2, 32)
    if int(sps) != sps:
        raise ValueError('Integer sps required')
    number(w, 'rolloff', 0, 1)
    for key, lo, hi in [('span', 8, 512), ('guard_samples', 128, 65536), ('packet_bytes', 16, 4096)]:
        value = number(w, key, lo, hi)
        if int(value) != value or (key == 'span' and int(value) % 2):
            raise ValueError('Invalid integer: '+key)
    if w['dac_bits'] not in (8, 10, 12, 16) or w['policy'] not in ('16qam_none', '16qam_conv', '16qam_repeat2', 'qpsk_conv'):
        raise ValueError('Unsupported DAC/digital policy')
    number(laser, 'wavelength_m', 1e-7, 1e-4)
    number(laser, 'power_dbm', -100, 60)
    number(laser, 'linewidth_hz', 0, fs/100)
    if mod['model'] not in ('iq_mzm', 'linear_reference'):
        raise ValueError('Unknown modulator model')
    number(mod, 'insertion_loss_db', 0, 100)
    number(mod, 'vpi_v', positive=True)
    number(mod, 'drive_peak_v', positive=True)
    if mod['model'] == 'linear_reference' and mod['drive_peak_v'] > mod['vpi_v']/2:
        raise ValueError('Linear-reference modulator requires drive_peak<=Vpi/2')
    for key in ('bias_i_v', 'bias_q_v', 'quadrature_error_rad'):
        number(mod, key, -100, 100)
    number(mod, 'bandwidth_hz', 0, fs/2, positive=True)
    number(path, 'distance_m', 1, 100000)
    for key in ('attenuation_db_per_km', 'tx_optics_loss_db', 'rx_optics_loss_db'):
        number(path, key, 0, 100)
    number(path, 'coupling_efficiency', 0, 1, positive=True)
    number(turb, 'coherence_time_s', positive=True)
    if turb['mode'] == 'cn2_plane_wave':
        number(turb, 'cn2_m_minus_2_over_3', 0, 1e-12)
        if turb['alpha'] is not None or turb['beta'] is not None:
            raise ValueError('Cn2 mode must not also specify alpha/beta')
    elif turb['mode'] == 'manual_gamma_gamma':
        if turb['cn2_m_minus_2_over_3'] is not None:
            raise ValueError('Manual alpha/beta requires Cn2=null')
        number(turb, 'alpha', positive=True)
        number(turb, 'beta', positive=True)
    elif turb['mode'] != 'disabled':
        raise ValueError('Unknown turbulence mode')
    if point['mode'] == 'aperture_integral':
        if point['a0'] is not None or point['xi'] is not None:
            raise ValueError('Physical pointing must not also specify a0/xi')
        for key in ('beam_waist_m', 'receiver_aperture_diameter_m'):
            number(point, key, 0, 100, positive=True)
        for key in ('divergence_half_angle_rad', 'jitter_per_axis_rad'):
            number(point, key, 0, .01)
        for key in ('boresight_x_rad', 'boresight_y_rad'):
            number(point, key, -.01, .01)
        if point['divergence_half_angle_rad'] < laser['wavelength_m']/(math.pi*point['beam_waist_m']):
            raise ValueError('Gaussian effective divergence is below the diffraction limit')
    elif point['mode'] == 'manual_a0_xi':
        for key in ('beam_waist_m', 'divergence_half_angle_rad', 'receiver_aperture_diameter_m',
                    'jitter_per_axis_rad', 'boresight_x_rad', 'boresight_y_rad'):
            if point[key] is not None:
                raise ValueError('Manual pointing requires physical parameters=null')
        number(point, 'a0', 0, 1, positive=True)
        number(point, 'xi', positive=True)
    else:
        raise ValueError('Unknown pointing mode')
    for name in ('tx_edfa', 'rx_edfa'):
        ed = c[name]
        if type(ed['enabled']) is not bool or ed['control'] not in ('gain', 'total_power'):
            raise ValueError('Invalid EDFA enable/control')
        number(ed, 'gain_db', 0, 60)
        number(ed, 'noise_figure_db', 10*math.log10(2), 30)
        number(ed, 'target_total_output_dbm', -100, 60)
        number(ed, 'max_total_output_dbm', -100, 60)
    for key in ('optical_filter_full_bandwidth_hz',):
        number(rx, key, 0, fs, positive=True)
    number(rx, 'electrical_bandwidth_hz', 0, fs/2, positive=True)
    for key in ('lo_linewidth_hz',):
        number(rx, key, 0, fs/100)
    number(rx, 'lo_power_dbm', -100, 40)
    number(rx, 'frequency_offset_hz', -fs/(100*sps), fs/(100*sps))
    for key in ('responsivity_a_per_w', 'pd_saturation_a', 'tia_v_per_a', 'adc_fullscale_v'):
        number(rx, key, positive=True)
    for key in ('dark_current_a', 'thermal_density_a_per_sqrt_hz_per_pd'):
        number(rx, key)
    if type(rx['shot_noise']) is not bool or rx['adc_bits'] not in (8, 10, 12, 16):
        raise ValueError('Invalid receiver noise/ADC')
    number(rx, 'dd_phase_step', 0, .2)
    number(rx, 'osnr_reference_bandwidth_nm', 0, 1, positive=True)
    ref_hz = 299792458./laser['wavelength_m']**2*rx['osnr_reference_bandwidth_nm']*1e-9
    if not fs/32768 <= ref_hz <= fs:
        raise ValueError('OSNR reference bandwidth must fit the modeled optical frequency grid')
    if rx['carrier_recovery'] not in ('pilot_constant', 'pilot_cfo_dd'):
        raise ValueError('Unknown carrier recovery')
    if rx['carrier_recovery'] == 'pilot_constant' and (laser['linewidth_hz'] or rx['lo_linewidth_hz'] or rx['frequency_offset_hz']):
        raise ValueError('Nonzero linewidth/CFO requires pilot_cfo_dd recovery')
    return c


def load(path=TEMPLATE):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if set(data) != {'schema', 'parameter_status', 'base', 'cases'} or data['schema'] != 'coherent-fso-si-v1':
        raise ValueError('Unknown/incomplete configuration schema')
    if data['parameter_status'] not in ('assumed_not_hardware_calibrated', 'measured_parameter_inputs'):
        raise ValueError('Explicit parameter provenance required')
    template = json.loads(TEMPLATE.read_text(encoding='utf-8'))['base']
    # A complete base is required: do not silently fill a misspelled/missing parameter.
    if set(data['base']) != set(template) or any(set(data['base'][k]) != set(template[k]) for k in template):
        raise ValueError('Base must contain exactly the documented parameter groups/keys')
    validate(data['base'])
    if not isinstance(data['cases'], dict) or not data['cases']:
        raise ValueError('Need at least one case')
    cases = {}
    for name, override in data['cases'].items():
        if not name or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in name):
            raise ValueError('Unsafe case name')
        cases[name] = validate(merge(data['base'], override))
    canonical = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return data, cases, hashlib.sha256(canonical.encode()).hexdigest()
