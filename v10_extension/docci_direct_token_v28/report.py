"""Independent CPU verifier: matrices, full text metrics, noise, paired budgets."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_direct_token_v28.core import (read, rows, sha, atomic, safe, check_source,
    PARENTS, ARMS, WIDTH, seed, load, schedule, training_snr, trials, geometry)
from docci_direct_token_v28.channel import constants, array_sha


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=4e-6, abs_tol=4e-7)


def stats_check(s, task, snr, value, tx=None):
    epsilon, expected = constants(task, snr, value)
    for k, v in expected.items():
        if isinstance(v, float):
            if not close(s[k], v): raise ValueError('Noise statistic differs: '+k)
        elif s[k] != v: raise ValueError('Noise identity differs: '+k)
    if abs(s['signal_power']-1) > .0002: raise ValueError('Wrong TX power')
    if snr is None:
        if s['realized_snr_db'] is not None or s['tx_sha256'] != s['observed_sha256']:
            raise ValueError('Clean channel is not identity')
    elif not close(s['realized_snr_db'], 10*math.log10(s['signal_power']/s['noise_power'])):
        raise ValueError('Realized SNR differs')
    if tx is not None:
        if tx.shape != (WIDTH[task],) or array_sha(tx) != s['tx_sha256']:
            raise ValueError('Actual TX frame differs')
        if array_sha((tx+epsilon).astype(np.float32)) != s['observed_sha256']:
            raise ValueError('Actual observed channel frame differs')


def verify_inputs(root, task, c, budget, full):
    spec = read(root/'INPUTS.json')
    if spec['task'] != task or spec['parent_sha256'] != PARENTS[task]['sha256'] or not spec['arrays_verified']:
        raise ValueError('Wrong input protocol')
    data = {}
    for split, count in (('train', c['train_samples']), ('qual_dev', c['val_samples'])):
        if sha(root/(split+'.json')) != spec['selected'][split]: raise ValueError('Input rows changed')
        values = read(root/(split+'.json'))
        if len(values) != count or len({r['example_id'] for r in values}) != count:
            raise ValueError('Incomplete input matrix')
        if any(r['official_split'] != split for r in values): raise ValueError('Wrong source split')
        data[split] = values
        if full:
            from docci_baseline_v14.data import Dataset
            items = Dataset(values, 1)
            for i in range(len(items)): budget(); items[i]
    if full:
        for name, h in spec['historical_files'].items():
            budget()
            if sha(name) != h: raise ValueError('Historical source/weight changed: '+name)
    return data


def verify_train(root, task, arm, c, train, budget, full):
    identity = read(root/'identity.json'); done = read(root/'TRAIN_COMPLETE.json')
    stream = schedule(c, task); hist = rows(root/'history.jsonl')
    ids = [r['example_id'] for r in train]; lookup = {r['example_id']: r for r in train}
    if (identity['task'], identity['arm'], identity['parent_sha256'], identity['config']) != (
            task, arm, PARENTS[task]['sha256'], c): raise ValueError('Training identity differs')
    if identity['train_ids'] != ids or not identity['two_views'] or identity['cross_modal']:
        raise ValueError('Training cohort/view protocol differs')
    if len(hist) != len(stream) or done['updates'] != len(stream) or not done['fixed_final'] or not done['actual_disk_roundtrip']:
        raise ValueError('Partial/unverified training')
    for j, (r, (epoch, ix)) in enumerate(zip(hist, stream)):
        budget(); wanted = [ids[i] for i in ix]
        denom = sum(lookup[e]['source_tokens']+16 for e in wanted) if task == 'text' else len(ix)
        lr = c[task+'_lr']*min(1., (j+1)/min(100, max(1, len(stream)//10)))*(.1+.9*(1-j/len(stream)))
        if (r['update'], r['total_updates'], r['epoch'], r['batch_ids'], r['denominator']) != (
                j+1, len(stream), epoch, wanted, denom) or not close(r['lr'], lr):
            raise ValueError('Schedule/denominator/LR differs')
        if not close(r['loss'], c['clean_weight']*r['clean_loss']+(1-c['clean_weight'])*r['noisy_loss']):
            raise ValueError('Two-view loss differs')
        if not math.isfinite(r['gradient_norm']) or [x['example_id'] for x in r['observations']] != wanted:
            raise ValueError('Missing gradient/per-description observations')
        for entry in r['observations']:
            eid = entry['example_id']
            if len(entry['views']) != 2: raise ValueError('Missing two-view training')
            for s, view, snr, weight in zip(entry['views'], ('clean', 'noisy'),
                    (None, training_snr(c, arm, epoch, eid)), (c['clean_weight'], 1-c['clean_weight'])):
                if s['view'] != view or s['weight'] != weight: raise ValueError('View protocol differs')
                stats_check(s, task, snr, seed(c['seed'], 'train', task, epoch, eid, view))
    audit = read(root/'learning_audit.json')
    if not audit['frozen_unchanged'] or audit['backbone_before'] != audit['backbone_after']:
        raise ValueError('Frozen backbone audit failed')
    if not all(v['parameters'] and v['gradient'] and v['changed'] for v in audit['groups'].values()):
        raise ValueError('TX/RX learning failed')
    if sum(v['parameters'] for v in audit['groups'].values()) != identity['trainable_parameters']:
        raise ValueError('Parameter audit differs')
    if not read(root/'REPLAY.json')['passed']: raise ValueError('Actual saved parent replay failed')
    if full and sha(root/'last.pth') != done['checkpoint_sha256']: raise ValueError('Saved weight changed')
    return identity


def verify_eval(root, task, arm, c, val, budget, full):
    from docci_text_continue_v11.metrics import metrics
    identity = read(root/'identity.json'); done = read(root/'EVAL_COMPLETE.json')
    ids = [x['example_id'] for x in val]; lookup = {r['example_id']: r for r in val}
    expected = {(eid, s, r, control) for eid in ids for s, r, control in trials(c)}
    values = rows(root/'rows.jsonl')
    keys = [(r['example_id'], r['snr_db'], r['repeat'], r['control']) for r in values]
    if len(keys) != len(expected) or set(keys) != expected or done['rows'] != len(expected):
        raise ValueError('Incomplete/duplicate evaluation matrix')
    if sha(root/'rows.jsonl') != done['rows_sha256'] or sha(root/'identity.json') != done['identity_sha256']:
        raise ValueError('Evaluation changed after completion')
    if (identity['task'], identity['arm'], identity['config'], identity['ids'], identity['geometry']) != (
            task, arm, c, ids, geometry(task)): raise ValueError('Evaluation identity differs')
    expected_ck = PARENTS[task]['sha256'] if arm == 'frozen' else read(root.parent/('train_'+arm)/'TRAIN_COMPLETE.json')['checkpoint_sha256']
    if identity['checkpoint_sha256'] != expected_ck: raise ValueError('Evaluation checkpoint differs')
    frames = {}
    if full:
        for eid in ids:
            with np.load(root/'frames'/(eid+'.npz'), allow_pickle=False) as z: frames[eid] = z['tx'].copy()
    for r in values:
        budget(); eid = r['example_id']; source = lookup[eid]; s = r['snr_db']
        donor = ids[(ids.index(eid)+1) % len(ids)] if r['control'] == 'shuffled' else eid
        if (r['official_split'], r['task'], r['arm'], r['donor_id'], r['geometry'], r['input_truncated']) != (
                'qual_dev', task, arm, donor, geometry(task), False): raise ValueError('RX/source identity differs')
        stats_check(r['stats'], task, s, seed(c['seed'], 'eval', task, donor, s, r['repeat']), frames.get(donor))
        rxhash = array_sha(np.zeros(WIDTH[task], np.float32)) if r['control'] == 'zero' else r['stats']['observed_sha256']
        if r['receiver_wire_sha256'] != rxhash: raise ValueError('Delivered payload differs')
        if task == 'text':
            if (r['source_text'], r['source_text_sha256'], r['source_tokens']) != (
                    source['text'], source['text_sha256'], source['source_tokens']): raise ValueError('Original text differs')
            for k, v in metrics(r['source_text'], r['received_caption']).items():
                if isinstance(v, float):
                    if not close(r[k], v): raise ValueError('Recomputed text metric differs: '+k)
                elif r[k] != v: raise ValueError('Recomputed text metric differs: '+k)
            if len(r['chunk_generated_token_ids']) != 9: raise ValueError('Nonpublic frame count')
        else:
            if r['source_image_sha256'] != source['image_sha256'] or not math.isfinite(r['psnr']) or not -1 <= r['ssim'] <= 1:
                raise ValueError('Image identity/metric differs')
            for name, h in [('image_file', 'image_sha256'), ('source_file', 'source_view_sha256')]:
                if name in r and sha(root/r[name]) != r[h]: raise ValueError('Saved PNG changed')
    return values


def effect(a, b, draws):
    # Positive means low-SNR training helps: image PSNR increases / text WER decreases.
    delta = np.asarray(a, np.float64)-np.asarray(b, np.float64)
    rng = np.random.default_rng(42)
    boot = delta[rng.integers(0, len(delta), (draws, len(delta)))].mean(1)
    return dict(mean=float(delta.mean()), paired95CI=np.quantile(boot, [.025, .975]).tolist(),
                units='dB or WER percentage points', independent_descriptions=len(delta))


def aggregate(allrows, c, task):
    output = {}; effects = {}
    for snr in [None]+c['eval_snr_db']:
        label = 'clean' if snr is None else str(snr); output[label] = {}; byarm = {}
        for arm, values in allrows.items():
            matched = [r for r in values if r['snr_db'] == snr and r['control'] == 'matched']
            if not matched: continue
            grouped = {}
            for r in matched: grouped.setdefault(r['example_id'], []).append(r)
            names = ['psnr', 'ssim'] if task == 'image' else ['normalized_wer', 'text_cer', 'bleu2']
            if task == 'image' and all(r['LPIPS'] is not None for r in matched): names.append('LPIPS')
            scores = {k: float(np.mean([np.mean([r[k] for r in v]) for v in grouped.values()])) for k in names}
            if task == 'text':
                from docci_text_continue_v11.metrics import diagnostic, words
                scores.update(diagnostics=diagnostic(matched),
                    corpus_WER=sum(r['normalized_word_errors'] for r in matched)/max(1, sum(r['normalized_source_words'] for r in matched)),
                    adjacent_word_repeat=float(np.mean([sum(a == b for a, b in zip(w, w[1:]))/max(1, len(w)-1)
                        for w in [words(r['received_caption']) for r in matched]])))
            scores.update(descriptions=len(grouped), noise_draws_per_description=len(matched)//len(grouped))
            output[label][arm] = scores
            key = 'psnr' if task == 'image' else 'normalized_wer'
            byarm[arm] = {eid: float(np.mean([r[key] for r in values])) for eid, values in grouped.items()}
        if set(byarm) == {'frozen', 'normal_range', 'low_snr'}:
            ids = sorted(byarm['low_snr']); effects[label] = {}
            for other in ('normal_range', 'frozen'):
                low = np.asarray([byarm['low_snr'][i] for i in ids]); ref = np.asarray([byarm[other][i] for i in ids])
                effects[label]['low_vs_'+other] = effect(low if task == 'image' else ref*100,
                    ref if task == 'image' else low*100, c['bootstrap_draws'])
                effects[label]['low_vs_'+other]['units'] = 'dB' if task == 'image' else 'WER percentage points'
    primary = effects.get('-5', {}).get('low_vs_normal_range')
    clean = effects.get('clean', {}).get('low_vs_normal_range')
    threshold, regression = (.3, -.3) if task == 'image' else (2., -2.)
    screen = bool(primary and clean and primary['mean'] >= threshold and primary['paired95CI'][0] > 0
                  and clean['paired95CI'][0] >= regression)
    return dict(scores=output, effects=effects, primary='-5dB low_snr vs normal_range',
        numerical_screen=screen, semantic_success=False, digital_superiority=False,
        facts='NOT_MEASURED', FSO='NOT_RUN', reused_qual_dev=True,
        uncertainty='description-paired, repeats averaged within description; one training seed; pointwise unadjusted CIs')


def verify(root, full=False, max_minutes=60, only_profile=None):
    root = safe(root); check_source(); state = read(root/'PIPELINE.json')
    started = time.monotonic()
    def budget():
        if time.monotonic()-started > max_minutes*60: raise TimeoutError('Verification budget exhausted')
    if not math.isfinite(max_minutes) or not 0 < max_minutes <= 180: raise ValueError('Verification budget 0..180min')
    result = dict(status=state['status'], tasks={}, completed=False,
                  weights_and_arrays_rechecked_here=full, training=False, full_semantic_success=False)
    if sha(root/'source_snapshot/MANIFEST.json') != sha(Path(__file__).with_name('MANIFEST.json')):
        raise ValueError('Wrong run source release')
    m = read(root/'source_snapshot/MANIFEST.json')
    for base, spec in [('source_snapshot', m['files']), ('dependency_snapshot', m['dependencies'])]:
        for name, h in spec.items():
            budget()
            if sha(root/base/name) != h: raise ValueError('Source snapshot differs: '+name)
    if only_profile is not None and only_profile not in state['profiles']:
        raise ValueError('Requested profile absent from pipeline')
    for profile in ([only_profile] if only_profile else state['profiles']):
        c = load(root/'config.json', profile)
        for task in state['tasks']:
            group = root/profile/task; budget()
            data = verify_inputs(group/'inputs', task, c, budget, full)
            identities = [verify_train(group/('train_'+a), task, a, c, data['train'], budget, full) for a in ARMS]
            for key in ('initial_hashes', 'trainable_parameters', 'train_ids', 'config', 'parent_sha256'):
                if identities[0][key] != identities[1][key]: raise ValueError('Arms not matched: '+key)
            allrows = {a: verify_eval(group/('eval_'+a), task, a, c, data['qual_dev'], budget, full)
                       for a in ('frozen',)+ARMS}
            result['tasks'][profile+'/'+task] = aggregate(allrows, c, task)
    result['completed'] = only_profile is None and state['status'] in ('COMPLETE', 'SMOKE_COMPLETE', 'TRAIN_AND_EVAL_FINISHED', 'COLLECTION_FAILED')
    result['subset_verified'] = only_profile
    if result['completed']: result['status'] = state.get('target_status', state['status'])
    return result


def render(data):
    parts = ['<!doctype html><meta charset="utf-8"><title>V28 direct tokens</title>',
        '<style>body{font:17px system-ui;max-width:1150px;margin:35px auto}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:9px}pre{white-space:pre-wrap}</style>',
        '<h1>V28：独立连续 token，低 SNR 训练</h1><p>没有图文互促；单次传输。AWGN，不是物理 FSO。历史 qual_dev 已重复使用；语义事实和数字优势尚未证明。</p>']
    for name, group in data['tasks'].items():
        parts.append('<h2>'+html.escape(name)+'</h2><table><tr><th>SNR</th><th>冻结父模型</th><th>0–20 dB 继续训练</th><th>含负 SNR 训练</th></tr>')
        for snr, arms in group['scores'].items():
            parts.append('<tr><td>'+snr+'</td>')
            for arm in ('frozen',)+ARMS:
                x = arms.get(arm, {}); value = '{:.3f} dB'.format(x['psnr']) if 'psnr' in x else '{:.2f}% WER'.format(x['normalized_wer']*100) if 'normalized_wer' in x else '—'
                parts.append('<td>'+value+'</td>')
            parts.append('</tr>')
        parts.append('</table><pre>'+html.escape(json.dumps(group['effects'], ensure_ascii=False, indent=2))+'</pre>')
    return ''.join(parts)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('--full', action='store_true')
    p.add_argument('--only-profile', choices=('smoke', 'pilot', 'full'))
    p.add_argument('--max-minutes', type=float, default=60); a = p.parse_args()
    data = verify(a.root, a.full, a.max_minutes, a.only_profile)
    print(json.dumps(data, ensure_ascii=False, indent=2))
