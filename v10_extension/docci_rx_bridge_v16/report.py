"""Independent CPU checks, paired exploratory intervals; no Torch/pickle loading."""
import argparse
import html
import json
import math
from pathlib import Path
import statistics
import sys
import time
import numpy as np
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_rx_bridge_v16.core import FOLDER, read, lines, sha, atomic, ARMS, CONTROLS, geometry, batches, PARENTS, DOCCI_SHA256, parameter_groups, load
from docci_text_continue_v11.metrics import metrics, body, summary, words


def inside(root, name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('Unsafe artifact member')
    target = root/p
    if target.is_symlink() or any(parent.is_symlink() for parent in target.parents if parent != root.parent):
        raise ValueError('Symlink artifact rejected')
    target.resolve(strict=True).relative_to(root.resolve())
    return target


def verify_seal(root, budget):
    done = read(root/'FINISHED.json')
    if done['manifest_sha256'] != sha(root/'LIGHT_MANIFEST.json'):
        raise ValueError('Light manifest seal differs')
    for name, digest in read(root/'LIGHT_MANIFEST.json').items():
        budget()
        if sha(inside(root, name)) != digest:
            raise ValueError('Sealed file differs: '+name)
    m = read(root/'source_snapshot/MANIFEST.json')
    if sha(root/'source_snapshot/MANIFEST.json') != sha(FOLDER/'MANIFEST.json') or read(root/'identity.json')['source_manifest_sha256'] != sha(FOLDER/'MANIFEST.json'):
        raise ValueError('Run does not use this trusted immutable source release')
    for name, digest in m['files'].items():
        if sha(inside(root, 'source_snapshot/'+name)) != digest:
            raise ValueError('Run source snapshot differs')
    for name, digest in m['dependencies'].items():
        if sha(inside(root, 'dependency_snapshot/'+name)) != digest:
            raise ValueError('Run historical dependency snapshot differs')


def index_check(root, arrays, budget):
    c, ix = read(root/'config.json'), read(root/'index.json')['records']
    source_identity = read(root/'source_identity.json')
    if source_identity['annotation_sha256'] != DOCCI_SHA256 or source_identity['official_counts'] != dict(train=9647, test=5000, qual_dev=100, qual_test=100):
        raise ValueError('DOCCI source identity differs')
    seen = set()
    for row in ix:
        budget()
        eid = row['example_id']
        if eid in seen or row['official_split'] not in ('train', 'qual_dev'):
            raise ValueError('Duplicate/forbidden source split')
        seen.add(eid)
        if __import__('hashlib').sha256(row['text'].encode()).hexdigest() != row['text_sha256']:
            raise ValueError('Original description hash differs')
        if not 2 <= row['source_tokens'] <= 512:
            raise ValueError('No-truncation token capacity differs')
        if arrays:
            for kind in ('text', 'image'):
                path = inside(root, row[kind+'_file'])
                if sha(path) != row[kind+'_file_sha256']:
                    raise ValueError('Frozen feature cache changed')
                with np.load(path, allow_pickle=False) as z:
                    expected = (dict(memory=(9, 32, 768), gate=(9,), ids=(9, 64), wire=(18432,))
                                if kind == 'text' else dict(wire=(8192,), rx_i0=(196, 96),
                                rx_clip=(196, 768), oracle_clip=(196, 768)))
                    if set(z.files) != set(expected):
                        raise ValueError('Cache array keys differ')
                    for key, shape in expected.items():
                        if z[key].shape != shape or not np.isfinite(z[key]).all():
                            raise ValueError('Cache geometry/finite contract differs: '+key)
                    if abs(float(np.mean(z['wire']**2))-1.) > .0002:
                        raise ValueError('Cache wire power differs')
                    if kind == 'text':
                        ids = z['ids']
                        if ids.dtype.kind not in 'iu' or (ids < 0).any() or (ids >= 50265).any():
                            raise ValueError('Invalid source token IDs')
                        count = 0
                        for slot in ids:
                            if slot[0] != 0 or list(slot).count(2) != 1:
                                raise ValueError('Source framing differs')
                            end = list(slot).index(2)
                            if not np.all(slot[end+1:] == 1) or np.any(slot[1:end] <= 2):
                                raise ValueError('Source PAD/EOS framing differs')
                            count += end-1
                        if count+2 != row['source_tokens'] or (z['gate'] < 0).any() or (z['gate'] > 1).any():
                            raise ValueError('Source coverage/public content-gate differs')
    if (sum(r['official_split'] == 'train' for r in ix), sum(r['official_split'] == 'qual_dev' for r in ix)) != (c['train_samples'], c['val_samples']):
        raise ValueError('Frozen pair counts differ')
    parents = read(root/'parents.json')
    if set(parents) != set(PARENTS) or any(parents[k]['sha256'] != PARENTS[k]['sha256'] for k in PARENTS):
        raise ValueError('Approved parents differ')
    for task in ('text', 'image'):
        audit = read(root/(task+'_frozen_audit.json'))
        if audit['before'] != audit['after'] or not audit['unchanged']:
            raise ValueError('Frozen feature pipeline differs')
    baseline = lines(root/'baseline.jsonl')
    expected_ids = [r['example_id'] for r in ix if r['official_split'] == 'qual_dev']+[r['example_id'] for r in ix if r['official_split'] == 'train'][:8]
    if len(baseline) != len(expected_ids) or {r['example_id'] for r in baseline} != set(expected_ids):
        raise ValueError('Incomplete frozen baseline matrix')
    for r in baseline:
        origin = next(e for e in ix if e['example_id'] == r['example_id'])
        if r['source_text'] != origin['text'] or r['source_text_sha256'] != origin['text_sha256'] or r['source_tokens'] != origin['source_tokens'] or r['official_split'] != origin['official_split']:
            raise ValueError('Frozen baseline source/length identity differs')
        verify_metrics(r)
    return ix, baseline


def verify_metrics(row):
    calculated = metrics(row['source_text'], row['received_caption'])
    for key, expected in calculated.items():
        actual = row[key]
        if isinstance(expected, (float, int)) and not isinstance(expected, bool):
            if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=2e-7, abs_tol=2e-8):
                raise ValueError('Independent text metric differs: '+key)
        elif expected != actual:
            raise ValueError('Independent text metric differs: '+key)
    joined, capacity = [], False
    if len(row['chunk_generated_token_ids']) != 9:
        raise ValueError('Public9-frame layout differs')
    for slot in row['chunk_generated_token_ids']:
        if not 3 <= len(slot) <= 65 or slot[:2] != [2, 0] or any(type(t) is not int or t < 0 or t >= 50265 for t in slot):
            raise ValueError('Invalid generated frame')
        content, eos = body(slot)
        joined.extend(content)
        capacity = capacity or not eos
    if joined != row['generated_token_ids'] or capacity != row['stopped_by_capacity']:
        raise ValueError('Joined frame/capacity differs')


def validate_eval(folder, ix, baseline, arm):
    matrix = read(folder/'matrix.json')
    if matrix['controls'] != list(CONTROLS) or matrix['geometry'] != geometry(arm) or matrix['arm'] != arm:
        raise ValueError('Evaluation intervention contract differs')
    rows = lines(folder/'rows.jsonl')
    if len(rows) != len(ix)*4 or matrix['ids'] != [r['example_id'] for r in ix]:
        raise ValueError('Evaluation pair matrix incomplete')
    by_id = {r['example_id']: r for r in ix}
    base = {r['example_id']: r for r in baseline}
    seen, own_gates = set(), {}
    for row in rows:
        key = row['example_id'], row['control']
        if key in seen or key[0] not in by_id or key[1] not in CONTROLS:
            raise ValueError('Duplicate/unknown evaluation row')
        seen.add(key)
        source = by_id[key[0]]
        if row['source_text'] != source['text'] or row['source_text_sha256'] != source['text_sha256'] or row['official_split'] != source['official_split'] or row['source_tokens'] != source['source_tokens'] or row['arm'] != arm:
            raise ValueError('Wrong source text or split')
        position = matrix['ids'].index(key[0])
        donor = ix[(position+1) % len(ix)] if key[1] == 'shuffled' else source
        if row['auxiliary_donor_id'] != (donor['example_id'] if key[1] == 'shuffled' else None):
            raise ValueError('Auxiliary donor differs')
        if row['own_text_cache_sha256'] != source['text_file_sha256'] or row['auxiliary_cache_sha256'] != donor[('text' if arm == 'self' else 'image')+'_file_sha256']:
            raise ValueError('Own memory/donor provenance differs')
        if row['condition'] != 'ideal_clean' or row['snr_db'] is not None or row['oracle'] != (arm == 'oracle_clip') or row['input_truncated']:
            raise ValueError('Channel/oracle boundary differs')
        gate = row['first_free_content_probability']
        if len(gate) != 9 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in gate):
            raise ValueError('Invalid receiver-only EOS content probability')
        if key[0] in own_gates and gate != own_gates[key[0]]:
            raise ValueError('Content gate changed across auxiliary controls')
        own_gates[key[0]] = gate
        if key[1] == 'off' and row['chunk_generated_token_ids'] != base[key[0]]['chunk_generated_token_ids']:
            raise ValueError('Gate-off tokens differ from original V14')
        empty = list(range(math.ceil((source['source_tokens']-2)/62), 9))
        active = [j for j in range(9) if j not in empty]
        generated = row['chunk_generated_token_ids']
        if row['empty_slots'] != empty or row['nonempty_slots'] != active or row['empty_false_activation'] != sum(bool(body(generated[j])[0]) for j in empty) or row['nonempty_suppressed'] != sum(not body(generated[j])[0] for j in active):
            raise ValueError('Empty/nonempty diagnostics differ')
        verify_metrics(row)
    result = {}
    for control in CONTROLS:
        group = [r for r in rows if r['control'] == control]
        score = summary(group)
        score['empty_false_activation'] = sum(r['empty_false_activation'] for r in group)
        score['empty_slots'] = sum(len(r['empty_slots']) for r in group)
        score['empty_activation_fraction'] = score['empty_false_activation']/max(1, score['empty_slots'])
        score['nonempty_suppressed'] = sum(r['nonempty_suppressed'] for r in group)
        result[control] = score
    return rows, result


def paired(rows_a, rows_b, draws, seed=42):
    a = {r['example_id']: r['normalized_wer'] for r in rows_a}
    b = {r['example_id']: r['normalized_wer'] for r in rows_b}
    if not a or set(a) != set(b) or len(a) != len(rows_a) or len(b) != len(rows_b):
        raise ValueError('Paired cohort differs/duplicates')
    difference = np.asarray([b[k]-a[k] for k in sorted(a)], dtype=np.float64)*100
    rng = np.random.default_rng(seed)
    # Positive means candidate A has LESS error. Chunked to bound RAM.
    sampled = []
    for start in range(0, draws, 1000):
        ix = rng.integers(0, len(difference), (min(1000, draws-start), len(difference)))
        sampled.extend(difference[ix].mean(1).tolist())
    return dict(gain_pp=float(difference.mean()), paired95CI_pp=np.quantile(sampled, [.025, .975]).tolist(),
                pairs=len(a), seed=seed, training_seed_uncertainty=False, reused_qual_dev_exploratory=True)


def compare(candidate, self_rows, c):
    matched = [r for r in candidate if r['control'] == 'matched']
    self_matched = [r for r in self_rows if r['control'] == 'matched']
    wrong = [r for r in candidate if r['control'] == 'shuffled']
    zero = [r for r in candidate if r['control'] == 'zero']
    against_self = paired(matched, self_matched, c['bootstrap_draws'])
    against_wrong = paired(matched, wrong, c['bootstrap_draws'])
    against_zero = paired(matched, zero, c['bootstrap_draws'])
    sm, ss = summary(matched), summary(self_matched)
    def word_repeat(rows):
        values = [words(r['received_caption']) for r in rows]
        return statistics.mean(sum(a == b for a, b in zip(v, v[1:]))/max(1, len(v)-1) for v in values)
    repeated, self_repeated = word_repeat(matched), word_repeat(self_matched)
    empty_count = sum(r['empty_false_activation'] for r in matched)
    # This is a numerical SCREEN, never the final factual/semantic success rule.
    passed = (against_self['gain_pp'] >= c['minimum_wer_gain_pp'] and
              against_self['paired95CI_pp'][0] > 0 and
              against_wrong['gain_pp'] >= c['minimum_donor_gain_pp'] and
              against_wrong['paired95CI_pp'][0] > 0 and
              sm['first_segment_adjacent_repeat'] <= ss['first_segment_adjacent_repeat'] and
              repeated <= self_repeated and
              empty_count == 0)
    return dict(vs_same_size_self=against_self, vs_real_wrong_auxiliary=against_wrong,
                vs_zero_auxiliary=against_zero, empty_false_activation=empty_count,
                numerical_screen_pass=passed, facts='PENDING', full_semantic_success=False,
                paragraph_word_adjacent_repeat=repeated,
                self_paragraph_word_adjacent_repeat=self_repeated,
                primary_success_requires_independent_fact_improvement=True)


def verify(root, arrays=True, max_minutes=60):
    if not math.isfinite(max_minutes) or not 0 < max_minutes <= 180:
        raise ValueError('Finite bounded CPU verification time required')
    root = Path(root).resolve(strict=True)
    start = time.monotonic()
    def budget():
        if time.monotonic()-start > max_minutes*60:
            raise TimeoutError('Independent verification time cap; preserve records')
    verify_seal(root, budget)
    identity = read(root/'identity.json')
    c = read(root/'config.json')
    raw = dict(c)
    if identity['smoke']:
        for key in ('train_samples', 'val_samples', 'alignment_epochs', 'alignment_batch',
                    'ce_epochs', 'ce_accumulation', 'save_examples'):
            raw[key] = load()[key]
    if load(raw, identity['smoke']) != c:
        raise ValueError('Resolved runtime configuration contract differs')
    if identity['action'] == 'cache':
        ix, baseline = index_check(root, arrays, budget)
        return dict(stage='cache', pairs=len(ix), arrays_verified=arrays,
                    light_manifest_sha256=sha(root/'LIGHT_MANIFEST.json'),
                    baseline_validation=summary([r for r in baseline if r['official_split'] == 'qual_dev']),
                    full_semantic_success=False, smoke=identity['smoke'])
    cache = (root/'cache_reference' if not arrays and (root/'cache_reference').is_dir()
             else Path(identity['cache'])).resolve(strict=True)
    expected_arms = {'smoke_train': list(ARMS), 'oracle': ['self', 'oracle_clip'],
                     'actual': ['rx_clip', 'rx_i0']}
    if identity['action'] not in expected_arms or identity['arms'] != expected_arms[identity['action']] or identity['smoke'] != (identity['action'] == 'smoke_train'):
        raise ValueError('Stage/smoke arm matrix incomplete or changed')
    # Training records refer to the frozen cache; never discover a different run.
    if sha(cache/'index.json') != identity['cache_index_sha256']:
        raise ValueError('Training cache identity changed')
    verify_seal(cache, budget)
    ix, baseline = index_check(cache, arrays, budget)
    if read(cache/'config.json') != c:
        raise ValueError('Train/cache configuration differs')
    pipeline = read(root/'original_pipeline_audit.json')
    if not pipeline['frozen_codec_and_BART_unchanged'] or pipeline['before'] != pipeline['after']:
        raise ValueError('Original codec/BART changed')
    scores, evaluated, param_counts = {}, {}, []
    for arm in identity['arms']:
        budget()
        folder = root/arm
        t = read(folder/'trainability.json')
        history = lines(folder/'history.jsonl')
        align = lines(folder/'alignment.jsonl')
        required_updates = 3 if identity['smoke'] else math.ceil(c['train_samples']/c['ce_accumulation'])*c['ce_epochs']
        if t['ce_updates'] != required_updates or t['alignment_updates'] != math.ceil(c['train_samples']/c['alignment_batch'])*c['alignment_epochs']:
            raise ValueError('Declared update count changed')
        expected = list(batches(c['train_samples'], c['ce_accumulation'], c['ce_epochs'], c['seed']))[:t['ce_updates']]
        if len(history) != len(expected) or len(align) != t['alignment_updates']:
            raise ValueError('Training schedule incomplete')
        training_ix = [r for r in ix if r['official_split'] == 'train']
        ids = [r['example_id'] for r in training_ix]
        if t['training_ids'] != ids:
            raise ValueError('Training identities differ')
        for k, (row, (epoch, indices)) in enumerate(zip(history, expected)):
            valid = sum((min(62, max(0, training_ix[i]['source_tokens']-2-j*62))+2) for i in indices for j in range(9))
            lr = c['ce_lr']*min(1., (k+1)/min(100, max(1, t['ce_updates']//10)))*(.1+.9*(1-k/max(1, t['ce_updates'])))
            if row['update'] != k+1 or row['total_updates'] != t['ce_updates'] or row['epoch'] != epoch+1 or row['batch_ids'] != [ids[i] for i in indices] or row['valid_target_tokens'] != valid or not math.isfinite(row['loss']) or not math.isclose(row['lr'], lr, rel_tol=1e-8, abs_tol=1e-12) or not math.isfinite(row['gradient_norm']):
                raise ValueError('TRAIN order/token denominator differs')
        align_expected = list(batches(c['train_samples'], c['alignment_batch'], c['alignment_epochs'], c['seed']))
        for k, (row, (_, indices)) in enumerate(zip(align, align_expected)):
            if row['update'] != k+1 or row['batch_ids'] != [ids[i] for i in indices] or not math.isfinite(row['loss']):
                raise ValueError('Real contrastive batch schedule differs')
        learning = read(folder/'learning_audit.json')
        frozen = read(folder/'frozen_audit.json')
        complete = read(folder/'TRAIN_COMPLETE.json')
        if set(learning['groups']) != {'resampler', 'cross0', 'cross1'} or not learning['passed'] or not all(v['parameters'] == parameter_groups(c)[g] and v['changed'] and v['nonzero_gradient'] for g, v in learning['groups'].items()) or sum(v['parameters'] for v in learning['groups'].values()) != t['trainable_parameters']:
            raise ValueError('Bridge learning audit failed')
        if not frozen['unchanged'] or frozen['before'] != frozen['after'] or not complete['actual_disk_roundtrip']:
            raise ValueError('Frozen/checkpoint roundtrip audit failed')
        if complete['updates'] != t['ce_updates'] or complete['alignment_updates'] != t['alignment_updates'] or not complete['fixed_final']:
            raise ValueError('Checkpoint completion schedule differs')
        if arrays and sha(folder/'last.pth') != complete['checkpoint_sha256']:
            raise ValueError('Actual bridge checkpoint differs')
        param_counts.append(t['trainable_parameters'])
        scores[arm] = {}
        for split in ('train', 'qual_dev'):
            subset = [r for r in ix if r['official_split'] == split]
            if split == 'train':
                subset = subset[:8]
            ev, score = validate_eval(folder/split, subset, baseline, arm)
            scores[arm][split] = score
            if split == 'qual_dev':
                evaluated[arm] = ev
    if len(set(param_counts)) != 1:
        raise ValueError('Trainable parameters not exactly matched')
    self_rows = evaluated.get('self')
    if self_rows is None:
        reference = root/'self_reference'
        ref = read(reference/'identity.json')
        if ref['cache_index_sha256'] != identity['cache_index_sha256'] or ref['config'] != c or ref['trainable_parameters'] != param_counts[0]:
            raise ValueError('Saved self baseline is not matched')
        self_rows, _ = validate_eval(reference/'qual_dev', [r for r in ix if r['official_split'] == 'qual_dev'], baseline, 'self')
    comparisons = {arm: compare(evaluated[arm], self_rows, c) for arm in evaluated if arm != 'self'}
    return dict(stage=identity['action'], smoke=identity['smoke'], scores=scores, comparisons=comparisons,
                arrays_verified=arrays, frozen_BART_verified=True, trainable_parameters=param_counts[0],
                primary_actual='rx_clip', rx_i0='exploratory, not posthoc confirmed winner',
                facts='PENDING', full_semantic_success=False, digital='NOT_RUN', FSO='NOT_RUN',
                limits='single training replicate, repeatedly used qual_dev; no final independent test')


def render(root, result):
    parts = ['<!doctype html><meta charset="utf-8"><title>V16 actual recovery</title>',
             '<style>body{font:16px system-ui;margin:30px;max-width:1300px}td{vertical-align:top;padding:12px;border:1px solid #ccc}table{border-collapse:collapse;width:100%}pre{white-space:pre-wrap}img{width:240px}</style>',
             '<h1>V16 图像→文本接收辅助：真实输出</h1><p>clean，SNR不适用；oracle是未计资源的诊断。事实审核未完成，不宣称互促成功。</p>',
             '<pre>'+html.escape(json.dumps(result, ensure_ascii=False, indent=2))+'</pre>']
    for folder in sorted(root.glob('*/qual_dev')):
        rows = lines(folder/'rows.jsonl')
        parts.append('<h2>'+html.escape(folder.parent.name)+'</h2>')
        ids = list(dict.fromkeys(r['example_id'] for r in rows))[:4]
        for eid in ids:
            group = [r for r in rows if r['example_id'] == eid]
            parts.append('<h3>'+html.escape(eid)+'</h3>')
            image = 'images/'+eid+'__received.png'
            if (root/image).is_file():
                parts.append('<img src="'+image+'" alt="unfused received image">')
            parts.append('<p>原文（仅评价）：'+html.escape(group[0]['source_text'])+'</p><table><tr>')
            for row in group:
                parts.append('<td><b>'+html.escape(row['control'])+'</b><p>WER {:.2f}%</p><p>'.format(row['normalized_wer']*100)+html.escape(row['received_caption'])+'</p></td>')
            parts.append('</tr></table>')
    return ''.join(parts)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root')
    p.add_argument('--max-minutes', type=float, default=60)
    a = p.parse_args()
    data = verify(a.root, max_minutes=a.max_minutes)
    atomic(Path(a.root)/'VERIFIED.json', data)
    print('Independent CPU verification passed; facts PENDING, no quality promotion', flush=True)
