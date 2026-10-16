"""Independent CPU-only validation. No Torch, model loading, or quality relabelling."""
import argparse
import math
from pathlib import Path
import statistics
import sys
import time
import numpy as np
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_segment_bridge_v17.core import (read, sha, lines, load, verify_seal, atomic, ARMS,
    CACHE_INDEX, controls, eval_conditions, batches, kind, geometry, condition, seed, noise_seed,
    donor_index, expected_parameters, parameter_groups)
from docci_rx_bridge_v16.report import verify_metrics, paired
from docci_text_continue_v11.metrics import summary, body, words


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def channel_stats(stats, channel, eid, phase='validation', epoch=0):
    nominal = None if channel == 'clean' else int(channel.split('_')[1])
    variance = 0. if nominal is None else 10.**(-nominal/10.)
    if (stats['nominal_snr_db'] != nominal or stats['noise_variance'] != variance or
        stats['noise_seed'] != noise_seed(phase, epoch, eid, channel) or
        not stats['gate_recomputed_from_received_memory'] or stats['clean_gate_reused']):
        raise ValueError('Channel/receiver-only gate identity differs')
    if not finite(stats['signal_power']) or abs(stats['signal_power']-1) > .0002:
        raise ValueError('Original wire power differs')
    if channel == 'clean':
        if stats['noise_power'] != 0 or stats['realized_snr_db'] is not None or stats['observed_sha256'] != stats['tx_sha256']:
            raise ValueError('Clean channel is not identity')
    elif not finite(stats['noise_power']) or stats['noise_power'] <= 0 or not math.isclose(
            stats['realized_snr_db'], 10*math.log10(stats['signal_power']/stats['noise_power']), rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError('Realized symbol SNR accounting differs')


def verify_inputs(root, arrays, budget):
    verify_seal(root, budget)
    identity, c = read(root/'identity.json'), read(root/'config.json')
    if identity['stage'] != 'inputs' or c != load(identity['smoke']) or identity['cache_index_sha256'] != CACHE_INDEX:
        raise ValueError('Input stage/config identity differs')
    cache = root/'cache_reference'
    if sha(cache/'index.json') != CACHE_INDEX:
        raise ValueError('Different cached dataset')
    from docci_rx_bridge_v16_fix1.compat import installed
    with installed() as validate:
        old = validate(cache, arrays=False, max_minutes=15)
    receipt = read(root/'cache_receipt.json')
    if not receipt['arrays_verified'] or receipt['smoke'] or receipt['light_manifest_sha256'] != old['light_manifest_sha256']:
        raise ValueError('Full original cache was not independently checked')
    ix = read(cache/'index.json')['records']
    train = [r for r in ix if r['official_split'] == 'train'][:c['train_samples']]
    val = [r for r in ix if r['official_split'] == 'qual_dev'][:c['val_samples']]
    spec = read(root/'TEACHER.json')
    if not spec['supervision_only'] or spec['validation_teachers'] or spec['shape'] != [len(train), 10, identity['dimensions']]:
        raise ValueError('Teacher scope/geometry differs')
    if [(r['example_id'], r['text_sha256']) for r in spec['index']] != [(r['example_id'], r['text_sha256']) for r in train]:
        raise ValueError('Teacher source order differs')
    if arrays:
        if sha(root/spec['file']) != spec['sha256']:
            raise ValueError('Native TRAIN teacher array changed')
        with np.load(root/spec['file'], allow_pickle=False) as z:
            if set(z.files) != {'teacher', 'counts'} or z['teacher'].shape != tuple(spec['shape']) or z['counts'].shape != (len(train), 9):
                raise ValueError('Teacher array geometry differs')
            if not np.isfinite(z['teacher']).all() or z['counts'].dtype.kind not in 'iu' or (z['counts'] < 0).any() or (z['counts'] > 62).any():
                raise ValueError('Teacher array finite/count contract differs')
            if not np.array_equal(z['counts'].sum(1)+2, [r['source_tokens'] for r in train]):
                raise ValueError('Teacher coverage differs from original full source')
            if (np.linalg.norm(z['teacher'][:, :9], axis=-1)[z['counts'] > 0] < 1e-6).any() or (np.linalg.norm(z['teacher'][:, 9], axis=-1) < 1e-6).any():
                raise ValueError('Nonempty native teacher cannot be a zero placeholder')
    observations = read(root/'observations.json')
    expected = {(r['example_id'], channel) for r in val for channel in eval_conditions(identity['smoke'])}
    if len(observations) != len(expected) or {(r['example_id'], r['condition']) for r in observations} != expected:
        raise ValueError('Incomplete observed validation matrix')
    source = {r['example_id']: r for r in val}
    for r in observations:
        budget()
        s = source[r['example_id']]
        if r['source_text'] != s['text'] or r['source_text_sha256'] != s['text_sha256'] or r['source_tokens'] != s['source_tokens'] or r['official_split'] != 'qual_dev':
            raise ValueError('Observation source differs')
        channel_stats(r['stats'], r['condition'], r['example_id'])
        if arrays:
            if sha(root/r['file']) != r['sha256']:
                raise ValueError('Observed memory file changed')
            with np.load(root/r['file'], allow_pickle=False) as z:
                if set(z.files) != {'memory', 'gate'} or z['memory'].shape != (9, 32, identity['dimensions']) or z['gate'].shape != (9,):
                    raise ValueError('Observation geometry differs')
                for key in ('memory', 'gate'):
                    if not np.isfinite(z[key]).all() or __import__('hashlib').sha256(z[key].astype(np.float32).tobytes()).hexdigest() != r['stats'][key+'_sha256']:
                        raise ValueError('Observation finite/hash differs')
                if (z['gate'] < 0).any() or (z['gate'] > 1).any():
                    raise ValueError('Received EOS gate range')
    baseline = lines(root/'baseline.jsonl')
    if len(baseline) != len(expected) or {(r['example_id'], r['condition']) for r in baseline} != expected:
        raise ValueError('Native baseline matrix incomplete')
    old_baseline = {r['example_id']: r for r in lines(cache/'baseline.jsonl')}
    observation_map = {(r['example_id'], r['condition']): r for r in observations}
    for r in baseline:
        obs = observation_map[r['example_id'], r['condition']]
        if any(r[k] != obs[k] for k in ('source_text', 'source_text_sha256', 'source_tokens', 'official_split', 'stats', 'file', 'sha256')):
            raise ValueError('Native baseline source/observation differs')
        verify_metrics(r)
        if r['condition'] == 'clean' and r['chunk_generated_token_ids'] != old_baseline[r['example_id']]['chunk_generated_token_ids']:
            raise ValueError('Original V14 token baseline changed')
    f = read(root/'frozen_audit.json')
    if f['before'] != f['after'] or not f['unchanged']:
        raise ValueError('Original frozen preparation backbone changed')
    return dict(stage='inputs', smoke=identity['smoke'], train=len(train), validation=len(val),
                baseline_rows=len(baseline), arrays_verified=arrays, facts='PENDING', full_semantic_success=False)


def verify_arm(root, inputs, arrays, budget):
    verify_seal(root, budget)
    identity, c = read(root/'identity.json'), read(root/'config.json')
    arm, smoke = identity['arm'], identity['smoke']
    if arm not in ARMS or c != load(smoke) or c != read(inputs/'config.json') or identity['inputs_light_sha256'] != sha(inputs/'LIGHT_MANIFEST.json'):
        raise ValueError('Arm/config/input provenance differs')
    ix = read(inputs/'cache_reference/index.json')['records']
    train = [r for r in ix if r['official_split'] == 'train'][:c['train_samples']]
    val = [r for r in ix if r['official_split'] == 'qual_dev'][:c['val_samples']]
    ids = [r['example_id'] for r in train]
    info = read(root/'trainability.json')
    expected_updates = 5 if smoke else math.ceil(len(train)/c['ce_accumulation'])*c['ce_epochs']
    expected_align = math.ceil(len(train)/c['alignment_batch'])*c['alignment_epochs']
    if info['TRAIN_ids'] != ids or info['ce_updates'] != expected_updates or info['alignment_updates'] != expected_align or info['trainable_parameters'] != expected_parameters():
        raise ValueError('Parameter/schedule matrix differs')
    history, alignment = lines(root/'history.jsonl'), lines(root/'alignment.jsonl')
    if len(history) != expected_updates or len(alignment) != expected_align:
        raise ValueError('Partial training is not a complete arm')
    schedule = list(batches(len(train), c['ce_accumulation'], c['ce_epochs'], c['seed']))[:expected_updates]
    for step, (r, (epoch, indices)) in enumerate(zip(history, schedule)):
        budget()
        if r['update'] != step+1 or r['total_updates'] != expected_updates or r['epoch'] != epoch+1 or r['batch_ids'] != [ids[i] for i in indices] or r['valid_target_tokens'] != sum(train[i]['source_tokens']+16 for i in indices):
            raise ValueError('CE source order/token denominator differs')
        lr = c['ce_lr']*min(1., (step+1)/min(100, max(1, expected_updates//10)))*(.1+.9*(1-step/max(1, expected_updates)))
        if not all(finite(r[k]) for k in ('loss', 'CE', 'lr', 'gradient_norm')) or not math.isclose(lr, r['lr'], rel_tol=1e-8, abs_tol=1e-12):
            raise ValueError('Invalid loss/LR record')
        expected_channels = [condition(arm, 'ce', epoch, ids[i]) for i in indices]
        if r['conditions'] != expected_channels or len(r['observations']) != len(indices):
            raise ValueError('Common channel schedule differs')
        expected_donors = [ids[(i+1+seed('ce-donor', epoch, ids[i]) % (len(train)-1)) % len(train)] for i in indices]
        if r['donor_ids'] != expected_donors:
            raise ValueError('TRAIN donor schedule differs')
        for eid, channel, stats in zip(r['batch_ids'], r['conditions'], r['observations']):
            channel_stats(stats, channel, eid, 'ce', epoch)
    align_schedule = list(batches(len(train), c['alignment_batch'], c['alignment_epochs'], c['seed']))
    for step, (r, (epoch, indices)) in enumerate(zip(alignment, align_schedule)):
        if r['update'] != step+1 or r['batch_ids'] != [ids[i] for i in indices] or r['conditions'] != [condition(arm, 'align', epoch, ids[i]) for i in indices] or not all(finite(r[k]) for k in ('loss', 'contrast', 'segment', 'matching')):
            raise ValueError('Real contrastive batch source/order differs')
    learning, frozen, complete = (read(root/name) for name in ('learning_audit.json', 'frozen_audit.json', 'TRAIN_COMPLETE.json'))
    if (set(learning['groups']) != {'resampler', 'repair', 'memory_output', 'residual_gate', 'matching', 'semantic_image', 'semantic_segment'} or
        not learning['passed'] or not all(v['changed'] and v['nonzero_gradient'] for v in learning['groups'].values()) or
        {k: v['parameters'] for k, v in learning['groups'].items()} != parameter_groups() or
        sum(v['parameters'] for v in learning['groups'].values()) != info['trainable_parameters']):
        raise ValueError('Required bridge group failed to learn')
    if not frozen['unchanged'] or frozen['before'] != frozen['after'] or not complete['actual_disk_roundtrip'] or not complete['fixed_final'] or complete['updates'] != expected_updates or complete['alignment_updates'] != expected_align:
        raise ValueError('Frozen/checkpoint/completion audit differs')
    seen_ids = {eid for r in history for eid in r['batch_ids']}
    if complete['unique_training_ids'] != len(seen_ids) or (not smoke and len(seen_ids) != len(train)):
        raise ValueError('TRAIN coverage differs')
    if arrays and sha(root/'last.pth') != complete['checkpoint_sha256']:
        raise ValueError('Saved final checkpoint changed')
    matrix = read(root/'matrix.json')
    if matrix['ids'] != [r['example_id'] for r in val] or matrix['controls'] != list(controls(smoke)) or matrix['conditions'] != list(eval_conditions(smoke)) or matrix['geometry'] != geometry(arm):
        raise ValueError('Evaluation protocol matrix changed')
    rows = lines(root/'rows.jsonl')
    expected_keys = {(r['example_id'], channel, control) for r in val for channel in eval_conditions(smoke) for control in controls(smoke)}
    if len(rows) != len(expected_keys) or {(r['example_id'], r['condition'], r['control']) for r in rows} != expected_keys:
        raise ValueError('Incomplete evaluation matrix')
    baseline = {(r['example_id'], r['condition']): r for r in lines(inputs/'baseline.jsonl')}
    observations = {(r['example_id'], r['condition']): r for r in read(inputs/'observations.json')}
    sources = {r['example_id']: r for r in val}
    for r in rows:
        budget()
        eid, channel, control = r['example_id'], r['condition'], r['control']
        s, obs = sources[eid], observations[eid, channel]
        if r['source_text'] != s['text'] or r['source_text_sha256'] != s['text_sha256'] or r['source_tokens'] != s['source_tokens'] or r['arm'] != arm or r['official_split'] != 'qual_dev' or r['input_truncated'] or r['oracle'] != (kind(arm) == 'oracle_clip'):
            raise ValueError('Output source/arm boundary differs')
        if r['own_observation_sha256'] != obs['sha256'] or r['own_stats'] != obs['stats']:
            raise ValueError('Own observation changed across controls')
        g = np.asarray(r['first_free_content_probability'], dtype=np.float32)
        if g.shape != (9,) or not np.isfinite(g).all() or (g < 0).any() or (g > 1).any() or __import__('hashlib').sha256(g.tobytes()).hexdigest() != obs['stats']['gate_sha256']:
            raise ValueError('Receiver content gate differs')
        pos = matrix['ids'].index(eid)
        donor = val[donor_index(pos, len(val), control)] if control.startswith('donor') else s
        if r['auxiliary_donor_id'] != (donor['example_id'] if control.startswith('donor') else None) or r['auxiliary_cache_sha256'] != donor[('text' if kind(arm) == 'self' else 'image')+'_file_sha256']:
            raise ValueError('Real auxiliary donor differs')
        if control == 'off' and r['chunk_generated_token_ids'] != baseline[eid, channel]['chunk_generated_token_ids']:
            raise ValueError('Off baseline tokens changed')
        if r['off_reused'] != (control == 'off'):
            raise ValueError('Off baseline reuse provenance differs')
        empty = list(range((s['source_tokens']-2+61)//62, 9))
        if r['empty_slots'] != empty or r['empty_false_activation'] != sum(bool(body(r['chunk_generated_token_ids'][j])[0]) for j in empty):
            raise ValueError('Empty-slot diagnostic differs')
        verify_metrics(r)
    if read(root/'EVAL_COMPLETE.json')['rows'] != len(rows):
        raise ValueError('Generation completion differs')
    scores = {channel: {control: summary([r for r in rows if r['condition'] == channel and r['control'] == control])
                       for control in controls(smoke)} for channel in eval_conditions(smoke)}
    return dict(stage='arm', arm=arm, smoke=smoke, scores=scores, rows=len(rows),
                trainable_parameters=info['trainable_parameters'], arrays_verified=arrays,
                facts='PENDING', full_semantic_success=False)


def word_repeat(rows):
    values = [words(r['received_caption']) for r in rows]
    return statistics.mean(sum(a == b for a, b in zip(v, v[1:]))/max(1, len(v)-1) for v in values)


def compare(visual_rows, self_rows, channel):
    a = [r for r in visual_rows if r['condition'] == channel and r['control'] == 'matched']
    b = [r for r in self_rows if r['condition'] == channel and r['control'] == 'matched']
    donors = [r for r in visual_rows if r['condition'] == channel and r['control'].startswith('donor')]
    wrong = []
    for r in a:
        group = [d for d in donors if d['example_id'] == r['example_id']]
        wrong.append(dict(example_id=r['example_id'], normalized_wer=statistics.mean(d['normalized_wer'] for d in group)))
    vs_self, vs_donor = paired(a, b, 10000), paired(a, wrong, 10000)
    empty = sum(r['empty_false_activation'] for r in a)
    ra, rb = word_repeat(a), word_repeat(b)
    sa, sb = summary(a), summary(b)
    screen = (vs_self['gain_pp'] >= 2 and vs_self['paired95CI_pp'][0] > 0 and
              vs_donor['gain_pp'] >= 1 and vs_donor['paired95CI_pp'][0] > 0 and
              empty == 0 and ra <= rb and sa['first_segment_adjacent_repeat'] <= sb['first_segment_adjacent_repeat'])
    return dict(vs_same_size_self=vs_self, vs_mean_of_fixed_donors=vs_donor,
                correct_word_adjacent_repeat=ra, self_word_adjacent_repeat=rb,
                empty_false_activation=empty, numerical_screen_pass=screen,
                facts='PENDING', full_semantic_success=False,
                donor_unit='mean donors within description, NOT independent extra descriptions')


def pipeline(root, arrays=True, max_minutes=60):
    start = time.monotonic()
    def budget():
        if time.monotonic()-start > max_minutes*60:
            raise TimeoutError('Independent CPU verification budget exhausted')
    state = read(root/'PIPELINE.json')
    results = {}
    for name in ('smoke_inputs', 'inputs'):
        if state['stages'].get(name) == 'VERIFIED':
            results[name] = verify_inputs(root/name, arrays, budget)
    for name in ['smoke_'+a for a in ARMS]+list(ARMS):
        if state['stages'].get(name) == 'VERIFIED':
            inp = root/('smoke_inputs' if name.startswith('smoke_') else 'inputs')
            results[name] = verify_arm(root/name, inp, arrays, budget)
    all_four = all(a in results for a in ARMS)
    if state['status'] == 'COMPLETE' and (not all_four or 'inputs' not in results or not all('smoke_'+a in results for a in ARMS)):
        raise ValueError('Incomplete pipeline cannot be labelled complete')
    comparisons = {}
    for visual, own in (('B_oracle_clean', 'A_self_clean'), ('D_oracle_mixed', 'C_self_mixed')):
        if visual in results and own in results:
            vr, sr = lines(root/visual/'rows.jsonl'), lines(root/own/'rows.jsonl')
            comparisons[visual] = {channel: compare(vr, sr, channel) for channel in eval_conditions(False)}
    for prefix in ('', 'smoke_'):
        for aa, bb in (('A_self_clean', 'B_oracle_clean'), ('C_self_mixed', 'D_oracle_mixed')):
            a, b = prefix+aa, prefix+bb
            if a not in results or b not in results:
                continue
            for name in ('history.jsonl', 'alignment.jsonl'):
                x, y = lines(root/a/name), lines(root/b/name)
                for i, j in zip(x, y):
                    if i['conditions'] != j['conditions'] or i['batch_ids'] != j['batch_ids']:
                        raise ValueError('Paired arms did not use common data/channel order')
                    if name == 'history.jsonl' and i['observations'] != j['observations']:
                        raise ValueError('Paired arms did not use identical own receiver observations')
    return dict(pipeline_status=state['status'], completed_stages=list(results), arm_results=results,
                comparisons=comparisons, primary='D_oracle_mixed vs C_self_mixed at awgn_0',
                secondary='B_oracle_clean vs A_self_clean at clean; other channels exploratory',
                oracle_only=True, real_received_image='NOT_RUN', FSO='NOT_RUN', digital='NOT_RUN',
                facts='PENDING', full_semantic_success=False, arrays_verified=arrays,
                limits='one training seed; reused qual_dev; no final independent test; asymmetric ideal image diagnostic')


def main():
    p = argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('--inputs'); p.add_argument('--light', action='store_true')
    p.add_argument('--max-minutes', type=float, default=60)
    p.add_argument('--receipt', choices=('VERIFIED.json', 'SMOKE_VERIFIED.json'), default='VERIFIED.json'); a = p.parse_args()
    if not math.isfinite(a.max_minutes) or not 0 < a.max_minutes <= 180:
        p.error('Finite0..180 minute CPU cap')
    root = Path(a.root).resolve(strict=True); start = time.monotonic()
    def budget():
        if time.monotonic()-start > a.max_minutes*60:
            raise TimeoutError('Independent stage verification timed out')
    if (root/'PIPELINE.json').is_file():
        result = pipeline(root, not a.light, a.max_minutes)
    elif read(root/'identity.json')['stage'] == 'inputs':
        result = verify_inputs(root, not a.light, budget)
    else:
        result = verify_arm(root, Path(a.inputs).resolve(strict=True), not a.light, budget)
    receipt = root/a.receipt
    if receipt.exists():
        if read(receipt) != result:
            raise FileExistsError('Existing different verification receipt preserved')
    else:
        atomic(receipt, result)
    print('Independent V17 CPU VERIFIED; facts PENDING; no automatic quality claim', flush=True)


if __name__ == '__main__':
    main()
