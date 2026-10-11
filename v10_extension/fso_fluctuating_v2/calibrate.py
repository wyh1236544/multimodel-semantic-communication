"""Public-waveform simulator calibration, no COCO or source text access."""
import argparse
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT))
from fso_fluctuating_v2 import VERSION
from fso_fluctuating_v2.core import (load, digest, seed_for, paths, transport,
    public_payload, as_complex, fit_operator, apply_operator, environment, TEMPLATE)
from v10.paths import write_path


def calibrate(config_path, output, smoke=False):
    c = load(config_path)
    if smoke: c.update(bank_train_frames=8, bank_holdout_frames=4, sequence_window_frames=4)
    output = write_path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    records, coefficients, residuals, fidelity = [], [], [], []
    for condition in c['conditions']:
        for partition, count in [('train', c['bank_train_frames']), ('holdout', c['bank_holdout_frames'])]:
            weather = paths(c, condition['weather'], count, ('calibration', partition, condition['weather']))
            for ix, path in enumerate(weather):
                if time.monotonic()-started > 1800: raise TimeoutError('30min public calibration cap; preserve partial directory')
                key = (condition['id'], partition, ix)
                seed = seed_for(c['seed'], key, 'noise')
                payload = public_payload(seed_for(c['seed'], key, 'payload'))
                t = transport(c, condition, path, seed)
                probe = t.probe()
                received, rel, stats = t.continuous(payload, 1, 'lmmse')
                coefficient, residual = fit_operator(payload, received)
                record = dict(index=len(records), partition=partition, condition=condition['id'], weather=condition['weather'],
                    path=path, noise_seed=seed, probe_snr_db=float(probe), reliability=float(rel), stats=stats,
                    available=bool(stats['analog_receiver_available']))
                records.append(record); coefficients.append(coefficient); residuals.append(residual)
                # Fresh waveform and fresh noise on SAME path: operator fitting must
                # not claim fidelity from reconstructing its own calibration target.
                if partition == 'holdout':
                    x2 = public_payload(seed_for(c['seed'], key, 'fresh_payload'))
                    fresh = transport(c, condition, path, seed_for(seed, 'fresh_noise'))
                    fresh.probe()
                    y2, _, s2 = fresh.continuous(x2, 1, 'lmmse')
                    replay = np.roll(residual, int(seed % len(residual)))*np.exp(1j*(seed % 6283)/1000)
                    predicted = apply_operator(as_complex(x2), coefficient, replay)
                    fidelity.append(dict(condition=condition['id'], frame=ix,
                        available_fit=record['available'], available_fresh=bool(s2['analog_receiver_available']),
                        actual_evm2=float(np.mean(abs(as_complex(y2)-as_complex(x2))**2)),
                        predicted_evm2=float(np.mean(abs(predicted-as_complex(x2))**2))))
                if len(records) == 1 or len(records) % 12 == 0:
                    print('CALIBRATION frames={} {} {} pilotSNR={:.2f} lock={} elapsed={:.1f}s'.format(
                        len(records), condition['id'], partition, probe, record['available'], time.monotonic()-started), flush=True)
    groups = {}
    passed = True
    for condition in c['conditions']:
        cid = condition['id']; hold = [r for r in fidelity if r['condition'] == cid]
        # Acceptance mismatch measured on all frames. EVM fidelity only on both
        # available, because gated-zero EVM=1 falsely makes a useless bank fit.
        usable = [r for r in hold if r['available_fit'] and r['available_fresh']]
        availability_difference = float(np.mean([r['available_fit'] != r['available_fresh'] for r in hold]))
        actual = float(np.mean([r['actual_evm2'] for r in usable])) if usable else None
        predicted = float(np.mean([r['predicted_evm2'] for r in usable])) if usable else None
        relative = abs(predicted-actual)/max(actual, 1e-8) if usable else None
        train_rows = [r for r in records if r['condition'] == cid and r['partition'] == 'train']
        group_ok = (len(usable) >= 2 and relative <= c['surrogate_evm_relative_tolerance'] and
            availability_difference <= c['surrogate_acceptance_difference_tolerance'] and any(r['available'] for r in train_rows))
        passed = passed and group_ok
        groups[cid] = dict(holdout_count=len(hold), jointly_available=len(usable), actual_evm2=actual,
            predicted_evm2=predicted, relative_mean_evm_error=relative, availability_disagreement=availability_difference,
            train_lock_fraction=float(np.mean([r['available'] for r in train_rows])),
            train_pilot_snr_quantiles_db=[float(x) for x in np.quantile([r['probe_snr_db'] for r in train_rows], [.1,.5,.9])],
            passed=bool(group_ok))
    with (output/'bank.npz').open('xb') as stream:
        np.savez_compressed(stream, coefficients=np.asarray(coefficients), residuals=np.asarray(residuals))
    value = dict(version=VERSION, config=c, config_sha256=digest(config_path), records=records,
        bank_sha256=digest(output/'bank.npz'), holdout_fidelity=fidelity, groups=groups,
        calibration_passed=bool(passed), smoke=smoke, wall_seconds=time.monotonic()-started,
        source_sha256={str(f.relative_to(EXT)): digest(f) for name in ('fso_fluctuating_v2','fso_jscc_v1',
            'optical_realism_v3','optical_link_v1','physical_link','business_link','final_compare_v1','optical_realism_v2')
            for f in sorted((EXT/name).rglob('*')) if f.suffix in ('.py','.json','.yaml') and f.name != 'MANIFEST.json'},
        environment=environment(c), public_waveforms_only=True, real_measurements=False,
        model='per-frame five-tap widely-linear operator + replayed correlated stochastic residual; NOT full differentiable hardware',
        holdout_streams_disjoint=True, tolerance_is_engineering_gate_not_hardware_certification=True)
    with (output/'bank.json').open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=bool(passed), groups=groups), indent=2), flush=True)
    if not passed: raise RuntimeError('Surrogate fidelity gate failed; NO training. Preserve bank; do not auto-tune thresholds.')
    with (output/'COMPLETE.json').open('x') as stream:
        json.dump(dict(version=VERSION, calibration_passed=True,
            result_sha256={n: digest(output/n) for n in ('bank.npz','bank.json')}), stream, indent=2)


def read_bank(directory, config_path=None):
    d = Path(directory)
    complete = json.loads((d/'COMPLETE.json').read_text())
    if complete.get('version') != VERSION or complete.get('calibration_passed') is not True:
        raise ValueError('Incomplete/unverified simulator calibration')
    for n, sha in complete['result_sha256'].items():
        if n not in ('bank.npz','bank.json') or digest(d/n) != sha: raise ValueError('Changed calibration files')
    if set(complete['result_sha256']) != {'bank.npz','bank.json'}: raise ValueError('Missing calibration contract')
    value = json.loads((d/'bank.json').read_text())
    if config_path and digest(config_path) != value['config_sha256']: raise ValueError('Different experiment configuration')
    if value['version'] != VERSION or not value['calibration_passed'] or not value['public_waveforms_only'] or value['real_measurements']:
        raise ValueError('Wrong calibration identity/provenance')
    for n, sha in value['source_sha256'].items():
        if Path(n).is_absolute() or '..' in Path(n).parts or digest(EXT/n) != sha: raise ValueError('Calibration source drift: '+n)
    with np.load(d/'bank.npz', allow_pickle=False) as arrays:
        value['coefficients'] = arrays['coefficients'].copy()
        value['residuals'] = arrays['residuals'].copy()
    count = len(value['records'])
    if value['coefficients'].shape != (count, 11) or value['residuals'].shape != (count,2304):
        raise ValueError('Wrong surrogate tensor dimensions')
    if not np.isfinite(value['coefficients']).all() or not np.isfinite(value['residuals']).all(): raise ValueError('Nonfinite bank')
    expected = {(c['id'],p,i) for c in value['config']['conditions'] for p,n in [('train',value['config']['bank_train_frames']),
        ('holdout',value['config']['bank_holdout_frames'])] for i in range(n)}
    actual = {(r['condition'],r['partition'],r['path']['frame_index']) for r in value['records']}
    if actual != expected or count != len(expected): raise ValueError('Incomplete/duplicate calibration frames')
    return value


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('output'); p.add_argument('--experiment-config',default=str(TEMPLATE)); p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); calibrate(a.experiment_config,a.output,a.smoke)
