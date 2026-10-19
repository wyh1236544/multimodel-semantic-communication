"""Independent output matrix, byte/metric validation and paired PRIVATE reports."""
import argparse
import csv
import html
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_pretrained_checks_v22.core import (CONFIG, read, rows, sha, digest, local,
    atomic, safe, case_key, prepared, verify_files, metric, seed)

CONTROLS = dict(caption=('received_text', 'caption_only', 'joint_correct', 'joint_wrong_image', 'joint_blank_image'),
    enhance=('no_text', 'correct_text', 'wrong_text'),
    digital=tuple(c+'_'+p for c in ('jpeg', 'bpg') for p in ('qpsk12', 'qpsk34', 'qam16_12', 'qam16_34')))

def matrix(data, wanted):
    found = {(case_key(r), r['control']): r for r in data}
    if len(found) != len(data) or set(found) != wanted:
        raise ValueError('Missing/duplicate/unexpected result matrix')
    return found

def verify(root, prepared_root=None, target_root=None, artifact_roots=None):
    receipt = verify_files(root)
    stage = receipt['stage']
    if stage == 'prepare':
        index, data, examples = prepared(root)
        if len(data) != receipt['metadata']['rows']:
            raise ValueError('Preparation count differs')
        # Independent PNG/hash checks apply also to truth; original text is never a receiver argument.
        for ex in examples.values():
            if sha(local(root, ex['source_image'])) != ex['source_image_sha256'] or digest(ex['source_text'].encode()) != ex['source_text_sha256']:
                raise ValueError('Evaluation truth identity differs')
        return dict(stage=stage, rows=len(data), training=False, semantic_success=False)
    if stage in ('inventory', 'assets'):
        return dict(stage=stage, training=False, semantic_success=False)
    if not prepared_root:
        raise ValueError('Explicit verified preparation required')
    index, base, examples = prepared(prepared_root)
    identity = read(Path(root)/'identity.json')
    if identity['prepared_sha256'] != sha(Path(prepared_root)/'COMPLETE.json'):
        raise ValueError('Different preparation lineage')
    data = rows(Path(root)/'rows.jsonl')
    if len(data) != receipt['metadata']['rows']:
        raise ValueError('Recorded row count differs')
    base_by = {case_key(r): r for r in base}
    if stage in CONTROLS:
        controls = CONTROLS[stage]
        if stage=='digital':
            controls = tuple(x for x in controls if x.startswith(identity['codec']+'_'))
        matrix(data, {(k, c) for k in base_by for c in controls})
    elif stage=='score':
        target = Path(target_root or identity['target_path'])
        target_receipt = verify_files(target)
        if sha(target/'COMPLETE.json') != identity['target_sha256']:
            raise ValueError('Score target lineage differs')
        target_rows = rows(target/'rows.jsonl')
        target_map = {(case_key(r), r.get('control', 'continuous')):r for r in target_rows}
        matrix(data, set(target_map))
    else:
        raise ValueError('Unknown stage')
    for r in data:
        if case_key(r) not in base_by:
            raise ValueError('Unexpected evaluation identity')
        src = base_by[case_key(r)]
        if stage in ('enhance', 'digital'):
            if sha(local(root, r['image_file'])) != r['image_sha256']:
                raise ValueError('Output image bytes differ')
            with Image.open(local(root, r['image_file'])) as im:
                if im.size != (512, 512) or im.mode != 'RGB':
                    raise ValueError('Output image geometry differs')
        if stage=='digital':
            if r['source_image_at_RX'] or r['source_text_at_RX'] or r['allocation']['total_symbols'] != 4128:
                raise ValueError('Digital receiver/resource leak')
            if abs(r['actual_average_power']-1) > 2e-6:
                raise ValueError('Digital energy budget differs')
            if r['post_fec_errors'] is None and r['post_fec_bits'] != 0:
                raise ValueError('Unknown BER marked as measured')
            if r['snr_db'] is None and not r['rx']['packet_crc']:
                raise ValueError('Clean digital source/packet gate failed')
        if stage=='enhance':
            donor = base_by[(src['donor_id'], src['snr_db'], src['repeat'])]
            prompt = '' if r['control']=='no_text' else (donor if r['control']=='wrong_text' else src)['received_caption']
            receiver = identity['receiver']
            expected_image = src['received_image_sha256']
            if receiver['stage']=='digital':
                target = Path((artifact_roots or {}).get(receiver['receipt_sha256'], identity['receiver_root']))
                if sha(target/'COMPLETE.json') != receiver['receipt_sha256']:
                    raise ValueError('Digital generator input lineage differs')
                matches = [x for x in rows(target/'rows.jsonl') if case_key(x)==case_key(r) and x['control']==receiver['control']]
                if len(matches) != 1:
                    raise ValueError('Digital generator input row differs')
                expected_image = matches[0]['image_sha256']
            if (r['prompt_sha256'] != digest(prompt.encode()) or r['received_image_sha256'] != expected_image
                    or r['random_seed'] != seed(42, 'V22-diffusion', *case_key(r)) % (2**63-1)
                    or r['source_text_at_RX'] or r['source_image_at_RX'] or r['prompt_plan']['truncated']):
                raise ValueError('Prompt/observation/randomness intervention differs')
        if stage=='caption':
            if (r['output_sha256'] != digest(r['output_text'].encode()) or r['source_text_at_RX']
                    or r['received_caption_sha256'] != src['received_caption_sha256']):
                raise ValueError('Caption receiver identity differs')
            if r['control']=='received_text' and r['output_text'] != src['received_caption']:
                raise ValueError('Baseline text changed')
            expected_prompt = None
            expected_image = None
            if r['control'] != 'received_text':
                if r['control']=='caption_only':
                    expected_prompt = 'Describe the visible objects, attributes, counts and spatial relationships. Do not invent details.'
                else:
                    expected_prompt = ('Correct this received description using the image. Preserve supported facts, '
                        'do not invent details. Output only the complete corrected description. Received description: '
                        +json.dumps(src['received_caption'])+' Corrected description:')
                if r['control']=='joint_blank_image':
                    expected_image = digest(np.full((512, 512, 3), 128, np.uint8).tobytes())
                else:
                    observed = base_by[(src['donor_id'], src['snr_db'], src['repeat'])] if r['control']=='joint_wrong_image' else src
                    with Image.open(local(prepared_root, observed['received_image'])) as im:
                        expected_image = digest(np.asarray(im.convert('RGB')).tobytes())
            if (r['conditioning_image_sha256'] != expected_image or r['prompt_sha256'] !=
                    (None if expected_prompt is None else digest(expected_prompt.encode()))):
                raise ValueError('Caption image/prompt intervention identity differs')
        if stage=='score':
            if any(not np.isfinite(r[k]) for k in ('MSE', 'LPIPS')) or r['MSE'] < 0 or r['LPIPS'] < 0:
                raise ValueError('Invalid perceptual score')
            if r['MSE'] > 0 and (r['psnr'] is None or abs(r['psnr']+10*np.log10(r['MSE'])) > 1e-8):
                raise ValueError('PSNR/MSE mismatch')
            origin = target_map[(case_key(r), r['control'])]
            name = origin['received_image'] if target_receipt['stage']=='prepare' else origin['image_file']
            with Image.open(local(target, name)) as im:
                output = np.asarray(im.convert('RGB')).copy()
            with Image.open(local(prepared_root, examples[r['example_id']]['source_image'])) as im:
                truth = np.asarray(im.convert('RGB')).copy()
            if (r['image_sha256'] != sha(local(target, name)) or
                    abs(metric(truth, output, examples[r['example_id']]['rectangle'])['MSE']-r['MSE']) > 1e-14):
                raise ValueError('Independent PNG MSE/identity differs')
    return dict(stage=stage, rows=len(data), training=False, semantic_success=False, facts='PENDING')

def paired(a, b, metric_name, alpha=.05):
    if set(a) != set(b):
        raise ValueError('Paired matrix differs')
    groups = {}
    for eid, rep in a:
        if a[eid, rep][metric_name] is None or b[eid, rep][metric_name] is None:
            return dict(mean=None, ci=None, reason='identical-image PSNR is infinite; inspect MSE')
        delta = a[eid, rep][metric_name]-b[eid, rep][metric_name]
        groups.setdefault(eid, []).append(-delta if metric_name=='LPIPS' else delta)
    values = np.array([np.mean(groups[x]) for x in sorted(groups)])
    if len(values) < 2:
        raise ValueError('Need at least two independent images')
    rng = np.random.Generator(np.random.PCG64(42))
    draw = values[rng.integers(0, len(values), size=(CONFIG['bootstrap_draws'], len(values)))].mean(1)
    return dict(mean=float(values.mean()), ci=np.quantile(draw, [alpha/2, 1-alpha/2]).tolist(),
        images=len(values), repeats_averaged_within_image=True, alpha=alpha,
        bootstrap_draws=CONFIG['bootstrap_draws'], training_seeds=1,
        reused_development_set=True, independent_test=False)

def render(a):
    root = safe(a.output)
    if root.exists():
        raise FileExistsError('Existing report preserved')
    index, base, examples = prepared(a.prepared)
    targets = {sha(Path(a.prepared)/'COMPLETE.json'):Path(a.prepared)}
    targets.update({sha(Path(p)/'COMPLETE.json'):Path(p) for p in a.stages})
    stages = []
    for path in a.stages:
        receipt = read(Path(path)/'COMPLETE.json')
        if receipt['stage']=='enhance':
            receiver = read(Path(path)/'identity.json')['receiver']
            if receiver['receipt_sha256'] not in targets:
                raise ValueError('Include the generator receiver-input stage in --stages for portable verification')
        target = None if receipt['stage'] != 'score' else targets.get(read(Path(path)/'identity.json')['target_sha256'])
        if receipt['stage']=='score' and target is None:
            raise ValueError('Include each scored target in --stages for a portable independent report')
        result = verify(path, a.prepared, target, targets)
        if result['stage'] in ('inventory', 'assets', 'prepare'):
            raise ValueError('Use --prepared for preparation, --stages for evaluation/score stages')
        stages.append((Path(path), result['stage'], rows(Path(path)/'rows.jsonl')))
    root.mkdir(parents=True, exist_ok=False)
    summary = dict(evaluation='exploratory fixed dev; single training seed; no final test',
        facts='PENDING', semantic_success=False, new_training=False, aggregates={}, effects={},
        BERTScore='NOT_RUN', CAPTURE='NOT_RUN', digital_postprocessing='NOT_RUN unless separately evaluated',
        policy='No inference of mutual help from caption-only success or image appearance alone')
    scored = {}
    for path, stage, data in stages:
        if stage=='score':
            identity = read(path/'identity.json')
            scored[identity['target_sha256']] = data
            for r in data:
                key = '{} / {} / {}'.format(targets[identity['target_sha256']].name, r['control'], r['snr_db'])
                summary['aggregates'].setdefault(key, []).append({k:r[k] for k in ('psnr', 'MSE', 'LPIPS')})
        if stage=='caption':
            from docci_text_continue_v11.metrics import metrics
            for r in data:
                m = metrics(examples[r['example_id']]['source_text'], r['output_text'])
                key = '{} / {} / {}'.format(path.name, r['control'], r['snr_db'])
                summary['aggregates'].setdefault(key, []).append(dict(normalized_WER=m['normalized_wer']))
    for key, values in list(summary['aggregates'].items()):
        summary['aggregates'][key] = dict(rows=len(values), **{k:None if any(v[k] is None for v in values)
            else float(np.mean([v[k] for v in values])) for k in values[0]})
    # Only compare baseline and enhancement after BOTH exact target-linked LPIPS stages exist.
    for path, stage, _ in stages:
        if stage != 'enhance' or sha(path/'COMPLETE.json') not in scored:
            continue
        enhancement_identity = read(path/'identity.json')
        receiver = enhancement_identity['receiver']
        base_scores = scored.get(receiver['receipt_sha256'])
        if not base_scores:
            continue
        data = scored[sha(path/'COMPLETE.json')]
        for snr in CONFIG['snr_db']:
            own = {(r['example_id'], r['repeat']):r for r in data if r['snr_db']==snr and r['control']=='correct_text'}
            for control in ('input', 'no_text', 'wrong_text'):
                reference = base_scores if control=='input' else data
                reference_control = receiver['control'] if control=='input' else control
                other = {(r['example_id'], r['repeat']):r for r in reference if r['snr_db']==snr and r['control']==reference_control}
                # Three primary LPIPS contrasts at -5dB use Bonferroni simultaneous intervals.
                alpha = .05/3 if snr == -5 else .05
                summary['effects']['{} / {} / correct_over_{}'.format(path.name, snr, control)] = {
                    name:paired(own, other, name, alpha if name=='LPIPS' else .05) for name in ('psnr', 'LPIPS')}
        effects = summary['effects']
        primary = {control:effects['{} / -5 / correct_over_{}'.format(path.name, control)] for control in ('input', 'no_text', 'wrong_text')}
        checks = {control+'_LPIPS_adjusted_CI_positive':value['LPIPS']['ci'][0] > 0 for control, value in primary.items()}
        baseline = np.mean([r['LPIPS'] for r in data if r['snr_db']==-5 and r['control']=='no_text'])
        checks['relative_LPIPS_gain_at_least_2percent'] = primary['no_text']['LPIPS']['mean'] >= .02*baseline
        checks['PSNR_noninferiority'] = (primary['input']['psnr']['ci'] is not None and
                                         primary['input']['psnr']['ci'][0] >= -.5)
        summary.setdefault('numerical_screens', {})[path.name] = dict(
            status='NUMERIC_ONLY; FACT_REVIEW_PENDING', checks={k:bool(v) for k, v in checks.items()},
            numeric_eligible=bool(all(checks.values())), full_semantic_pass=False)
    # Blind human audit IDs and a separate PRIVATE decoding key. Facts are not auto-guessed.
    with (root/'fact_review.csv').open('x', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['blind_id', 'source_text', 'received_or_generated_text', 'source_atomic_fact',
                    'fact_type', 'preserved_0_1', 'unsupported_added_fact', 'reviewer'])
        keys = []
        candidates = [('continuous', r, r['received_caption']) for r in base]
        for path, stage, data in stages:
            if stage=='caption':
                candidates.extend((path.name+'/'+r['control'], r, r['output_text']) for r in data)
        first_review = {r['example_id'] for r in index['examples'][:min(10, len(examples))]}
        for label, r, text in candidates:
            if r['repeat'] or r['example_id'] not in first_review:
                continue
            bid = str(seed('V22-human', label, *case_key(r)))
            w.writerow([bid, examples[r['example_id']]['source_text'], text, '', '', '', '', ''])
            keys.append(dict(blind_id=bid, control=label, case=list(case_key(r))))
    atomic(root/'fact_review_PRIVATE_key.json', keys)
    atomic(root/'summary.json', summary)
    parts = ['<!doctype html><meta charset="utf-8"><title>V22 PRIVATE results</title>',
        '<style>body{font:16px system-ui;margin:30px;max-width:1400px}img{width:310px}td{vertical-align:top;padding:12px;border:1px solid #ddd}pre{white-space:pre-wrap}</style>',
        '<h1>V22：免训练可行性检查（PRIVATE）</h1>',
        '<p>AWGN；clean 无噪声，0 / −5 dB 为名义信道 SNR，不是图像 PSNR。负 SNR 是分布外推理。人工事实未审核，不宣布互促成功。</p>',
        '<pre>'+html.escape(json.dumps(summary, ensure_ascii=False, indent=2))+'</pre>']
    import shutil
    (root/'images').mkdir()
    def portable(path):
        name = sha(path)+'.png'
        target = root/'images'/name
        if not target.exists():
            shutil.copyfile(path, target)
        return 'images/'+name
    image_keys = []
    with (root/'image_fact_review.csv').open('x', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['blind_id', 'source_image', 'output_image', 'source_atomic_fact',
                    'fact_type', 'preserved_0_1', 'unsupported_added_fact', 'reviewer'])
        candidates = [('continuous', r, Path(a.prepared), r['received_image']) for r in base]
        for path, stage, data in stages:
            if stage in ('digital', 'enhance'):
                candidates.extend((path.name+'/'+r['control'], r, path, r['image_file']) for r in data)
        for label, r, folder, name in sorted(candidates, key=lambda x:seed('V22-blind-order', x[0], *case_key(x[1]))):
            if r['repeat'] or r['example_id'] not in first_review:
                continue
            bid = str(seed('V22-image-human', label, *case_key(r)))
            original = local(a.prepared, examples[r['example_id']]['source_image'])
            w.writerow([bid, portable(original), portable(local(folder, name)), '', '', '', '', ''])
            image_keys.append(dict(blind_id=bid, control=label, case=list(case_key(r))))
    atomic(root/'image_fact_review_PRIVATE_key.json', image_keys)
    first = {x['example_id'] for x in index['examples'][:2]}
    for r in base:
        if r['example_id'] not in first or r['repeat']:
            continue
        ex = examples[r['example_id']]
        parts.append('<h2>'+html.escape(str(case_key(r)))+'</h2><table><tr>')
        for label, path in (('原图（仅评价）', local(a.prepared, ex['source_image'])),
                            ('连续接收', local(a.prepared, r['received_image']))):
            parts.append('<td>'+label+'<br><img src="'+portable(path)+'"></td>')
        for path, stage, data in stages:
            if stage in ('enhance', 'digital'):
                for out in data:
                    if case_key(out)==case_key(r):
                        parts.append('<td>'+html.escape(path.name+'/'+out['control'])+'<br><img src="'+portable(local(path, out['image_file']))+'"></td>')
        parts.append('</tr></table><p>原文：'+html.escape(ex['source_text'])+'</p><p>实际接收文本：'+html.escape(r['received_caption'])+'</p>')
        for _, stage, data in stages:
            if stage=='caption':
                for out in data:
                    if case_key(out)==case_key(r):
                        parts.append('<p>'+html.escape(out['control']+': '+out['output_text'])+'</p>')
    (root/'index.html').write_text('\n'.join(parts), encoding='utf-8')
    print('PRIVATE report: '+str(root/'index.html'), flush=True)

if __name__=='__main__':
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action', choices=('verify', 'render'))
    p.add_argument('root', nargs='?')
    p.add_argument('--prepared')
    p.add_argument('--target')
    p.add_argument('--stages', nargs='*', default=[])
    p.add_argument('--output')
    a = p.parse_args()
    if a.action=='verify':
        print(json.dumps(verify(a.root, a.prepared, a.target), indent=2))
    else:
        if not a.prepared or not a.output:
            p.error('render requires --prepared and --output')
        render(a)
