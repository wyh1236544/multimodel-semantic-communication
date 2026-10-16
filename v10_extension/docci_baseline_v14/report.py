"""Independent light-record verification; never imports Torch or executes weights."""
import hashlib
import html
import json
import math
from pathlib import Path
import random
import statistics
import sys
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_baseline_v14.core import read, rows, sha, geometry, DECODE, batches, updates
from docci_text_continue_v11.metrics import metrics, summary, body


def inside(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Unsafe relative artifact path')
    p = root
    for part in Path(name).parts:
        p = p/part
        if p.is_symlink():
            raise ValueError('Symlink artifact rejected')
    return p


def expected_outputs(design):
    tasks = ['text', 'image'] if design['task']=='both' else [design['task']]
    result = []
    for task in tasks:
        result.append(task+'/native_validation')
        arms = design['text_arms'] if task=='text' else ['data64', 'expanded']
        if design['mode']=='calibrate':
            arms = ['historical_reference']
        for arm in arms:
            if design['mode'] != 'calibrate':
                result.append(task+'/'+arm+'/diagnostic_train')
            result.append(task+'/'+arm+'/validation')
    return result


def finite(x):
    return type(x) in (float, int) and math.isfinite(x)


def same(x, y):
    if type(y) in (int, bool, str):
        return type(x) == type(y) and x == y
    return finite(x) and math.isclose(x, y, abs_tol=1e-10, rel_tol=1e-9)


def evaluation(root, name, c, reference):
    import numpy as np
    folder = inside(root, name)
    m, done = read(folder/'matrix.json'), read(folder/'EVAL_COMPLETE.json')
    rr = rows(folder/'rows.jsonl')
    if done['rows_sha256'] != sha(folder/'rows.jsonl') or done['matrix_sha256'] != sha(folder/'matrix.json'):
        raise ValueError('Evaluation hash mismatch')
    task = name.split('/')[0]
    native = '/native_' in name
    expected_cohort = 'diagnostic_train' if name.endswith('/diagnostic_train') else 'validation'
    entries = read(root/task/('cache_train_identity.json' if expected_cohort=='diagnostic_train' else 'cache_validation_identity.json'))['records']
    if expected_cohort == 'diagnostic_train':
        entries = entries[:min(c['train_eval_samples'], len(entries))]
    ids = [e['example_id'] for e in entries]
    if m['task'] != task or m['cohort'] != expected_cohort or m['example_ids'] != ids:
        raise ValueError('Source/cohort matrix differs')
    controls = ['native'] if native else ['matched', 'shuffled', 'zero']
    strategies = list(DECODE) if task=='text' else ['image']
    if m['controls'] != controls or m['strategies'] != strategies:
        raise ValueError('Evaluation controls/decoding differ')
    if m['native_privileged_NOT_transmission'] != native or m['geometry'] != (None if native else geometry(task)):
        raise ValueError('Public resource/native contract differs')
    expected = {(i, k, s) for i in ids for k in controls for s in strategies}
    if len(rr) != len(expected) or done['rows'] != len(rr):
        raise ValueError('Incomplete evaluation')
    lookup = {e['example_id']: e for e in entries}
    frames = {}
    if not native:
        for eid in ids:
            path = folder/'frames'/(eid+'.npz')
            with np.load(path, allow_pickle=False) as z:
                if z.files != ['observed']:
                    raise ValueError('Source information leaked into RX frame')
                arr = z['observed']
                if arr.dtype != np.float32 or arr.shape != (geometry(task)['payload_reals'],) or not np.isfinite(arr).all():
                    raise ValueError('Invalid frame array')
                if abs(float(np.mean(arr**2))-1.) > .0002:
                    raise ValueError('Invalid transmitted power')
                frames[eid] = dict(file=path.relative_to(folder).as_posix(), sha256=sha(path),
                                   array_sha256=hashlib.sha256(arr.tobytes()).hexdigest())
    for row in rr:
        key = (row['example_id'], row['control'], row['strategy'])
        if key not in expected:
            raise ValueError('Duplicate/unknown evaluation row')
        expected.remove(key)
        eid, control, _ = key
        entry = lookup[eid]
        if row['official_split'] != entry['official_split'] or row['task'] != task or row['cohort'] != expected_cohort:
            raise ValueError('Row source/split differs')
        if row['condition'] != 'ideal_clean' or row['snr_db'] is not None or row['native'] != native:
            raise ValueError('Mislabelled channel/native condition')
        if not native:
            f = frames[eid]
            if row['frame_file'] != f['file'] or row['frame_sha256'] != f['sha256'] or row['tx_array_sha256'] != f['array_sha256']:
                raise ValueError('Stored observed frame identity differs')
            donor = ids[(ids.index(eid)+1) % len(ids)] if control=='shuffled' else None
            if row['donor_example_id'] != donor or row['received_array_sha256'] != frames[donor or eid]['array_sha256']:
                raise ValueError('Shuffled control did not use independent donor frame')
        if task == 'text':
            if row['source_text'] != entry['text'] or row['source_text_sha256'] != hashlib.sha256(row['source_text'].encode()).hexdigest():
                raise ValueError('Original text changed')
            if row['source_tokens'] != entry['source_tokens'] or row['input_truncated']:
                raise ValueError('Source truncation/count differs')
            actual = metrics(row['source_text'], row['received_caption'])
            if any(not same(row[k], v) for k, v in actual.items()):
                raise ValueError('Independently recalculated text metrics differ')
            chunks = row['chunk_generated_token_ids']
            if len(chunks) != 9 or any(len(x) > 65 for x in chunks):
                raise ValueError('Public fixed chunk layout changed')
            joined, stopped = [], False
            for seq in chunks:
                tokens, eos = body(seq)
                joined.extend(tokens)
                stopped = stopped or not eos
            if joined != row['generated_token_ids'] or stopped != row['stopped_by_capacity']:
                raise ValueError('Generated token concatenation/EOS differs')
        else:
            if row['source_image_sha256'] != entry['image_sha256']:
                raise ValueError('Image source changed')
            if not finite(row['psnr']) or not 0 <= row['psnr'] <= 150 or not finite(row['ssim']) or not -1 <= row['ssim'] <= 1:
                raise ValueError('Invalid image metrics')
            measured = m['LPIPS_status']=='MEASURED_OFFLINE'
            if measured != (row['LPIPS'] is not None) or (measured and (not finite(row['LPIPS']) or row['LPIPS'] < 0)):
                raise ValueError('Unmeasured/nonfinite perceptual score')
            for field in ('restored_image', 'source_view'):
                if row[field+'_file'] is not None and sha(inside(folder, row[field+'_file'])) != row[field+'_sha256']:
                    raise ValueError('Saved actual image changed')
        refkey = (task, expected_cohort, eid)
        value = row['source_text_sha256'] if task=='text' else row['source_image_sha256']
        if refkey in reference and reference[refkey] != value:
            raise ValueError('Same-example comparisons used different original sources')
        reference[refkey] = value
    groups = {}
    for strategy in strategies:
        for control in controls:
            group = [r for r in rr if r['strategy']==strategy and r['control']==control]
            if task == 'text':
                groups[strategy+'/'+control] = summary(group)
            else:
                groups[strategy+'/'+control] = dict(n=len(group), mean_PSNR=statistics.mean(r['psnr'] for r in group),
                    mean_SSIM=statistics.mean(r['ssim'] for r in group),
                    mean_LPIPS=statistics.mean(r['LPIPS'] for r in group) if group[0]['LPIPS'] is not None else None)
    return dict(groups=groups, rows=rr, task=task, matrix=m)


def paired(left, right, key):
    if [r['example_id'] for r in left] != [r['example_id'] for r in right] or not left:
        raise ValueError('Paired comparison IDs differ')
    d = [b[key]-a[key] for a, b in zip(left, right)]
    rng = random.Random(42)
    resamples = sorted(statistics.mean(d[rng.randrange(len(d))] for _ in d) for _ in range(2000))
    return dict(right_minus_left=statistics.mean(d), CI95=[resamples[49], resamples[1949]],
                n=len(d), sampling='description/image-paired bootstrap; fixed trained model; not training-seed uncertainty')


def verify(root, budget=lambda: None):
    root = Path(root).resolve(strict=True)
    if (root/'FAILED.json').exists():
        raise ValueError('Failed run is not complete')
    done = read(root/'RUN_FINISHED.json')
    if done['manifest_sha256'] != sha(root/'LIGHT_MANIFEST.json'):
        raise ValueError('Light manifest changed')
    manifest = read(root/'LIGHT_MANIFEST.json')
    if not manifest:
        raise ValueError('Empty record bundle')
    for name, expected in manifest.items():
        budget()
        if sha(inside(root, name)) != expected:
            raise ValueError('Light file changed: '+name)
    design, c, identity = read(root/'design.json'), read(root/'config.json'), read(root/'run_identity.json')
    outputs = expected_outputs(design)
    if identity['outputs'] != outputs or identity['schema'] != 'docci-baseline-v14':
        raise ValueError('Design/evaluated matrix differs')
    source = read(root/'source_snapshot/MANIFEST.json')
    if identity['source_manifest_sha256'] != sha(root/'source_snapshot/MANIFEST.json'):
        raise ValueError('Executed source identity differs')
    for name, expected in source['files'].items():
        if sha(inside(root/'source_snapshot', name)) != expected:
            raise ValueError('Executed source snapshot changed')
    for name, expected in source['dependencies'].items():
        if sha(inside(root/'dependency_snapshot', name)) != expected:
            raise ValueError('Historical dependency snapshot differs')
    reference, result = {}, {}
    for name in outputs:
        budget()
        print('VERIFY '+name, flush=True)
        result[name] = evaluation(root, name, c, reference)
    for task in ('text', 'image'):
        if not (root/task).exists():
            continue
        frozen = read(root/task/'identity.json')
        if frozen['backbone_signature_before'] != frozen['backbone_signature_after']:
            raise ValueError('Frozen backbone audit differs')
    if design['mode'] != 'calibrate':
        for task in ('text', 'image'):
            if not (root/task).exists():
                continue
            entries = read(root/task/'cache_train_identity.json')['records']
            arms = design['text_arms'] if task=='text' else ['data64', 'expanded']
            initial = []
            for arm in arms:
                budget()
                folder = root/task/arm/'train'
                audit = read(folder/'learning_audit.json')
                complete = read(folder/'TRAIN_COMPLETE.json')
                train = read(folder/'trainability.json')
                hist = rows(folder/'history.jsonl')
                effective = c['text_microbatch']*c['text_accumulation'] if task=='text' else c['image_accumulation']
                total = 2 if design['mode']=='smoke' else updates(len(entries), effective, c[task+'_epochs'])
                if len(hist) != total or complete['updates'] != total or train['total_updates'] != total:
                    raise ValueError('Partial training cannot be complete')
                if not audit['passed'] or not audit['groups'] or any(not all(g[k] for k in ('parameters', 'nonzero_gradient', 'changed')) for g in audit['groups'].values()):
                    raise ValueError('Learning audit failed')
                selected = entries[:min(64, len(entries))] if task=='image' and arm=='data64' else entries
                if train['unique_training_ids'] != [e['example_id'] for e in selected]:
                    raise ValueError('Training data scale differs')
                if any(e['official_split'] != 'train' for e in selected):
                    raise ValueError('Validation/test data entered optimizer')
                schedule = list(batches(len(entries), effective, c[task+'_epochs'], c['seed']))
                if design['mode']=='smoke':
                    schedule = [(0, list(range(min(effective, len(selected))))) for _ in range(2)]
                seen = set()
                token_map = {e['example_id']: e.get('source_tokens', 0)+16 for e in entries}
                for i, (row, (_, ix)) in enumerate(zip(hist, schedule)):
                    wanted = [selected[x % len(selected)]['example_id'] for x in ix]
                    seen.update(wanted)
                    if row['update'] != i+1 or row['total_updates'] != total or row['batch_example_ids'] != wanted or row['unique_examples_seen'] != len(seen):
                        raise ValueError('Actual training presentations differ from predeclared schedule')
                    if not finite(row['loss']) or not finite(row['gradient_norm']):
                        raise ValueError('Nonfinite training history')
                    if task=='text':
                        if row['valid_target_tokens'] != sum(token_map[x] for x in wanted):
                            raise ValueError('Valid-token weighted CE accounting differs')
                if complete['unique_examples_seen'] != len(seen) or not complete['actual_disk_roundtrip']:
                    raise ValueError('Checkpoint completion/audit differs')
                if (folder/'last.pth').exists() and sha(folder/'last.pth') != complete['checkpoint_sha256']:
                    raise ValueError('Saved checkpoint changed')
                initial.append(read(root/task/arm/'identity.json')['initial_hashes'])
            if task=='image' and initial[0] != initial[1]:
                raise ValueError('Image data-size arms did not share identical initialization')
    comparisons = {}
    if design['mode'] != 'calibrate':
        pairs = []
        if 'text/legacy64/validation' in result:
            pairs += [('text/legacy64/validation', 'text/'+arm+'/validation', 'normalized_wer', strategy)
                      for arm in design['text_arms'] if arm!='legacy64' for strategy in DECODE]
        if 'image/data64/validation' in result:
            pairs.append(('image/data64/validation', 'image/expanded/validation', 'psnr', 'image'))
        for left, right, key, strategy in pairs:
            get = lambda name: [r for r in result[name]['rows'] if r['control']=='matched' and r['strategy']==strategy]
            comparisons[right+' minus '+left+' / '+strategy] = paired(get(left), get(right), key)
    public = {name: value['groups'] for name, value in result.items()}
    return dict(status='BASELINE_RECORDS_VERIFIED_NOT_QUALITY_SUCCESS', mode=design['mode'],
                groups=public, comparisons=comparisons, evaluations=result,
                facts='PENDING_SEPARATE_HUMAN_AUDIT', digital='NOT_RUN', FSO=False,
                historical_quality_gates='UNCHANGED; this module does not promote to FSO/joint')


def render(data):
    parts = ['<!doctype html><meta charset="utf-8"><title>V14真实基线结果</title>',
             '<style>body{font:17px sans-serif;max-width:1300px;margin:30px auto}td,th{border:1px solid #bbb;padding:9px;vertical-align:top}table{border-collapse:collapse;width:100%}img{max-width:45%}pre{white-space:pre-wrap}</style>',
             '<h1>V14 单分支基线：实际输出</h1><p>理想无噪声；SNR不适用；未使用Common、FSO或扩散。工程完成不代表恢复达标。贪心与部署解码必须分开。原生骨干为未传输参照。</p>',
             '<pre>'+html.escape(json.dumps({k:v for k,v in data.items() if k!='evaluations'}, ensure_ascii=False, indent=2))+'</pre>']
    for name, value in data['evaluations'].items():
        parts.append('<h2>'+html.escape(name)+'</h2>')
        for row in value['rows']:
            if row['task']=='text' and row['control'] in ('matched','native'):
                parts.append('<p><b>'+html.escape(row['example_id']+' / '+row['strategy'])+'</b></p><table><tr><th>原文（仅评价）</th><th>恢复</th></tr><tr><td>'+html.escape(row['source_text'])+'</td><td>'+html.escape(row['received_caption'])+'</td></tr></table>')
            if row['task']=='image' and row['restored_image_file']:
                parts.append('<p>'+html.escape(row['example_id']+' / '+row['control'])+' PSNR={:.3f}dB</p>'.format(row['psnr']))
                for key in ('source_view_file', 'restored_image_file'):
                    parts.append('<img src="records/'+html.escape(name+'/'+row[key], quote=True)+'">')
    return '\n'.join(parts)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('root')
    a = p.parse_args()
    data = verify(a.root)
    print(json.dumps({k:v for k,v in data.items() if k!='evaluations'}, ensure_ascii=False, indent=2))
