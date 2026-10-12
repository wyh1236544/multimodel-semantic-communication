"""Explicit new3072-symbol frame, not the old2304 calibration bank."""
import copy
import math
import numpy as np
from clip_bart_joint_v5.core import geometry
from bart_latent_codec_v4.channel import proxy_tensor, proxy_numpy
from fso_fluctuating_v2.core import environment, seed_for
from optical_realism_v3.transport import RealisticTransport
from optical_realism_v3.reliability import combine
from optical_realism_v3.spec import sampled_count


def resource(c):
    cases, plan, _ = environment(c); w = copy.deepcopy(cases['weak']['waveform'])
    w.update(pilot_interval=plan['hardware']['pilot_interval'], pilot_block=plan['hardware']['pilot_block'])
    samples = sampled_count(0, w)+sampled_count(geometry()['payload_complex_symbols'], w)
    return dict(geometry(), scheduled_samples=samples, slot_samples=samples,
        scheduled_seconds=samples/w['fs_hz'], fs_hz=w['fs_hz'], sps=w['sps'],
        symbol_rate_baud=w['fs_hz']/w['sps'], rolloff=w['rolloff'],
        all_probe_pilot_preamble_tails_guards_charged=True,
        resource_scope='new equal V5 frame for all arms; NOT old2304 budget or equal measured energy')


def send(c, condition, path, seed, values):
    cases, plan, _ = environment(c); res = resource(c)
    t = RealisticTransport(cases[condition['weather']], seed_for(c['seed'], 'unused-v5-fade'), seed,
        condition['launch_signal_dbm'], res['slot_samples'], plan['laser_max_dbm'], plan['hardware'])
    t.path_draw = copy.deepcopy(path); pilot = float(t.probe())
    x = np.asarray(values, np.float32)
    if x.shape != (geometry()['total_reals'],) or not np.isfinite(x).all(): raise ValueError('Full finite C/I/T wire required')
    z = (x[::2]+1j*x[1::2])/math.sqrt(2)
    observed, event = t.burst(z, 'features', policy='analog')
    y, reliability = combine(observed[None], event['pilot_residual_variance_before_gate'],
        event['pilot_correlated_error_floor'], event['receiver_locked'], 'lmmse')
    restored = np.stack((y.real, y.imag), -1).reshape(-1)*math.sqrt(2)
    if not np.isfinite(restored).all(): raise FloatingPointError('Nonfinite physical receive')
    stats = dict(t.totals(), receiver_available=bool(event['receiver_locked']), pilot_snr_db=pilot,
        analog_reliability=float(reliability['analog_reliability']), payload_complex_symbols=len(z),
        analog_repeats=1, raw_utf8_wire_bytes=0, pre_fec_ber=None, post_fec_ber=None,
        resources=res, analog_feature_mse=float(np.mean((restored-x)**2)))
    return restored.astype(np.float32), stats, t
