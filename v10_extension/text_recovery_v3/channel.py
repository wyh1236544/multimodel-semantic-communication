"""Fresh whole-frame V3 evaluation; header and all synchronization charged."""
import copy
import math
import numpy as np
from fso_fluctuating_v2.core import environment, seed_for
from optical_realism_v3.transport import RealisticTransport
from optical_realism_v3.reliability import combine
from optical_realism_v3.spec import sampled_count
from text_recovery_v3.core import geometry


def resource(c, arm):
    cases, plan, cap = environment(c); w = copy.deepcopy(cases['weak']['waveform'])
    w.update(pilot_interval=plan['hardware']['pilot_interval'], pilot_block=plan['hardware']['pilot_block'])
    count = sampled_count(0, w)+sampled_count(geometry(arm)['payload_complex_symbols'], w)
    if count > cap['slot_samples']: raise ValueError('Frame exceeds shared slot cap')
    return dict(geometry(arm), scheduled_samples=count, scheduled_seconds=count/w['fs_hz'],
        fs_hz=w['fs_hz'], sps=w['sps'], symbol_rate_baud=w['fs_hz']/w['sps'], rolloff=w['rolloff'],
        note='Header, probe, tracking, preamble, pulse tails and guards charged; not equal actual energy')


def send(c, condition, path, seed, arm, values):
    cases, plan, cap = environment(c); resource(c, arm)
    t = RealisticTransport(cases[condition['weather']], seed_for(c['seed'], 'unused-text-fade'), seed,
        condition['launch_signal_dbm'], cap['slot_samples'], plan['laser_max_dbm'], plan['hardware'])
    t.path_draw = copy.deepcopy(path); pilot = float(t.probe())
    x = np.asarray(values, np.float32)
    if x.shape != (geometry(arm)['total_reals'],) or not np.isfinite(x).all(): raise ValueError('Invalid public payload')
    z = (x[::2]+1j*x[1::2])/math.sqrt(2)
    received, event = t.burst(z, 'features', policy='analog')
    y, observed = combine(received[None, :], event['pilot_residual_variance_before_gate'],
        event['pilot_correlated_error_floor'], event['receiver_locked'], 'lmmse')
    restored = np.stack((y.real, y.imag), -1).reshape(-1)*math.sqrt(2)
    if not np.isfinite(restored).all(): raise FloatingPointError('Nonfinite physical receive')
    event.update(observed)
    stats = dict(t.totals(), receiver_available=bool(event['receiver_locked']), pilot_snr_db=pilot,
        analog_reliability=float(observed['analog_reliability']), analog_repeats=1,
        payload_complex_symbols=len(z), length_header_complex_symbols=geometry(arm)['length_header_complex_symbols'],
        pre_fec_ber=None, post_fec_ber=None, raw_utf8_wire_bytes=0, header_has_crc=False,
        analog_feature_mse=float(np.mean((restored-x)**2)), resources=resource(c, arm))
    return restored.astype(np.float32), stats, t
