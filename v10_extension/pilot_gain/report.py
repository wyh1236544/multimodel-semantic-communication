"""Strict lattice validation and image-cluster paired descriptive bootstrap."""
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from pilot_gain.run import STRATEGIES

METRICS = ('psnr', 'ssim', 'bleu2', 'token_acc', 'common_caption_bleu2',
           'common_cosine', 'header_rejected')
PAIRED = ('source_caption_evaluation_only', 'pilot_draw_sha256', 'payload_rng_sha256',
          'payload_shape', 'physical_unit', 'payload_uses', 'pilot_uses', 'header_uses',
          'forward_uses', 'reverse_rf_real_uses')


def quantile(values, q):
    values = sorted(values)
    pos = (len(values)-1)*q
    lo = int(pos)
    hi = min(lo+1, len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(pos-lo)


def effect(differences, draws, seed):
    n = len(differences)
    rng = random.Random(seed)
    boot = [sum(differences[rng.randrange(n)] for _ in range(n))/n for _ in range(draws)]
    return {'mean_delta': statistics.mean(differences), 'image_delta_sd': statistics.stdev(differences),
            'ci95_pointwise': [quantile(boot, .025), quantile(boot, .975)],
            'image_win_fraction': sum(x > 0 for x in differences)/n}


def read_json(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)


def make_report(root, draws=1000):
    root = Path(root)
    plan = read_json(root/'plan.json')
    a = plan['args']
    expected = a['samples']*a['repeats']*len(a['profiles'])*len(a['snr_list'])
    data, identities, summaries, input_hashes = {}, {}, {}, {}
    for strategy, mode in STRATEGIES.items():
        directory = root/strategy
        complete = read_json(directory/'COMPLETE.json')
        if complete['rows'] != expected or complete['expected_rows'] != expected:
            raise ValueError('Incomplete evaluation: '+strategy)
        identity = read_json(directory/'run_identity.json')
        if identity['status'] != 'complete':
            raise ValueError('Missing completion identity')
        cfg = identity['config']['v10']
        if cfg.get('fixed_mode') != mode or cfg.get('csi') != 'pilot' or cfg.get('equalizer') != 'mmse':
            raise ValueError('Unexpected channel configuration')
        settings = dict(cfg)
        settings.pop('fixed_mode', None)
        identities[strategy] = (identity['checkpoint_sha256'], identity['split_sha256'],
                                identity['source_sha256'], settings, identity['config']['model'] if 'model' in identity['config'] else None)
        args = identity['args']
        for field in ('samples', 'repeats', 'seed', 'profiles', 'snr_list', 'data_root', 'vae_path', 'text_model_path'):
            if args[field] != a[field]:
                raise ValueError('Run/plan mismatch: '+field)
        if args['split'] != 'validation' or not args['no_amp'] or args['ablation'] != 'full':
            raise ValueError('Unexpected split/numerical/ablation setting')
        rows = {}
        sample_path = directory/'samples.jsonl'
        input_hashes[strategy] = hashlib.sha256(sample_path.read_bytes()).hexdigest()
        with sample_path.open(encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                key = (r['image_id'], r['profile'], float(r['nominal_snr_db']), r['repeat'])
                if key in rows:
                    raise ValueError('Duplicate sample identity')
                if r['strategy'] != strategy:
                    raise ValueError('Unexpected strategy label')
                for field in METRICS:
                    if not math.isfinite(float(r[field])):
                        raise ValueError('Nonfinite metric')
                for field in PAIRED:
                    if field not in r:
                        raise ValueError('Missing pairing evidence: '+field)
                requested = {0: 'reconstruction', 1: 'hybrid', 2: 'generation'}
                if mode is not None and r['requested_mode'] != requested[mode]:
                    raise ValueError('Fixed mode was not requested')
                rows[key] = r
        if len(rows) != expected:
            raise ValueError('Wrong actual row count')
        data[strategy] = rows
        summaries[strategy] = read_json(directory/'summary.json')
        if summaries[strategy]['diffusion_executed'] or summaries[strategy]['source_caption_used_by_receiver']:
            raise ValueError('Unexpected generation/source-caption usage')
    ref = data['adaptive']
    ids = sorted({k[0] for k in ref})
    lattice = {(i, p, float(s), r) for i in ids for p in a['profiles']
               for s in a['snr_list'] for r in range(a['repeats'])}
    if len(ids) != a['samples'] or set(ref) != lattice:
        raise ValueError('Missing/unplanned lattice condition')
    for strategy in STRATEGIES:
        if identities[strategy] != identities['adaptive'] or set(data[strategy]) != lattice:
            raise ValueError('Checkpoint/split/source/settings/sample mismatch')
        for key in lattice:
            if any(data[strategy][key][f] != ref[key][f] for f in PAIRED):
                raise ValueError('Unpaired channel draws/resources/source at '+str(key))
    groups = {}
    for profile in a['profiles']:
        for snr in a['snr_list']:
            keys = [k for k in ref if k[1] == profile and k[2] == float(snr)]
            group = {'images': len(ids), 'rows': len(keys), 'strategies': {}, 'gains': {}}
            for strategy, rows in data.items():
                values = [rows[k] for k in keys]
                mean = {f: statistics.mean(float(v[f]) for v in values) for f in METRICS}
                mean['sender_mode_fraction'] = {m: sum(v['sender_mode'] == m for v in values)/len(values)
                                                 for m in ('reconstruction', 'hybrid', 'generation')}
                mean['frame_psnr_below_10db_rate'] = sum(v['psnr'] < 10 for v in values)/len(values)
                mean['forward_uses'] = values[0]['forward_uses']
                mean['physical_unit'] = values[0]['physical_unit']
                # This is the existing evaluator's gallery-wide metric, not a per-image bootstrap.
                skey = profile+'/'+str(float(snr))
                mean['common_to_text_retrieval_r1'] = summaries[strategy]['summary'][skey]['common_to_text_retrieval_r1']
                group['strategies'][strategy] = mean
            for strategy in list(STRATEGIES)[1:]:
                comparison = {}
                for field in METRICS:
                    differences = [statistics.mean(float(ref[(i, profile, float(snr), r)][field])-
                                                     float(data[strategy][(i, profile, float(snr), r)][field])
                                                     for r in range(a['repeats'])) for i in ids]
                    seed = int(hashlib.sha256((profile+str(snr)+strategy+field).encode()).hexdigest()[:8], 16)
                    comparison[field] = effect(differences, draws, seed)
                group['gains'][strategy] = comparison
            groups[profile+'/'+str(float(snr))] = group
    return {'paired_rows': expected, 'unique_images': len(ids), 'bootstrap_draws': draws,
            'input_samples_sha256': input_hashes, 'groups': groups,
            'delta_definition': 'adaptive minus named fixed baseline; header_rejected lower is better',
            'limitations': ['One trained checkpoint; no uncertainty across training seeds.',
                            'Average repeats within each image, then resample paired image differences.',
                            '95% intervals are pointwise exploratory, not multiplicity-adjusted significance tests.',
                            'Fixed strategies retain pilots, receiver reliability gates, MMSE, UEP, feedback and header.',
                            'Only route selection changes; not total pilot/equalizer gain, not no-CSI comparison.',
                            'Same checkpoint zero-shot fixed modes, not separately retrained optimum baselines.',
                            'Original continuous RF/IMDD channel; not coherent QAM waveform, BER or hardware-time test.',
                            'Generation names a token allocation; no diffusion is executed.']}


def write_tables(root, report):
    root = Path(root)
    with (root/'gain_table.csv').open('x', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['condition', 'baseline', 'metric', 'adaptive_mean', 'fixed_mean',
                         'delta', 'ci95_low', 'ci95_high', 'image_delta_sd', 'image_win_fraction'])
        for condition, group in report['groups'].items():
            for baseline, effects in group['gains'].items():
                for metric, value in effects.items():
                    writer.writerow([condition, baseline, metric, group['strategies']['adaptive'][metric],
                                     group['strategies'][baseline][metric], value['mean_delta'],
                                     *value['ci95_pointwise'], value['image_delta_sd'], value['image_win_fraction']])
    lines = ['# Pilot-driven allocation gain (validation diagnostic)', '',
             'Positive delta means adaptive is higher. PSNR in dB; BLEU2 in [0,1].',
             'Paired image bootstrap; exploratory pointwise intervals; no training-seed uncertainty.', '',
             '| Condition | Fixed baseline | PSNR delta [95% CI] | BLEU2 delta [95% CI] |',
             '|---|---|---|---|']
    for condition, group in report['groups'].items():
        for baseline, effects in group['gains'].items():
            cells = []
            for field in ('psnr', 'bleu2'):
                e = effects[field]
                cells.append('{:+.4f} [{:+.4f}, {:+.4f}]'.format(e['mean_delta'], *e['ci95_pointwise']))
            lines.append('| {} | {} | {} | {} |'.format(condition, baseline, *cells))
    lines += ['', '## Limits', '']+['- '+s for s in report['limitations']]
    with (root/'REPORT.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('root')
    p.add_argument('--bootstrap', type=int, default=1000)
    a = p.parse_args()
    # Read-only independent re-verification, even when report outputs exist.
    print(json.dumps(make_report(a.root, a.bootstrap), indent=2, allow_nan=False))
