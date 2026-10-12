"""Independent record verification. A negative diagnostic screen is not success."""
import argparse
import html
import json
from pathlib import Path
import sys
EXT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXT))
from text_recovery_v3 import VERSION
from text_recovery_v3.core import digest, row, metrics, summarize, gate, geometry


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def lines(path): return [json.loads(v) for v in Path(path).read_text(encoding='utf-8').splitlines() if v.strip()]


def safe(root, name):
    p = Path(name)
    if p.is_absolute() or not p.parts or any(v in ('.', '..') for v in p.parts): raise ValueError('Unsafe record path')
    q = root
    for part in p.parts:
        q = q/part
        if q.is_symlink(): raise ValueError('Symlink records not allowed')
    return q


def check_text(entries, expected_ids, expected_controls, split):
    expected = {(control, iid) for control in expected_controls for iid in expected_ids}
    seen = set()
    for v in entries:
        key = (v['control'], v['image_id'])
        if key in seen or key not in expected or v['source_split'] != split: raise ValueError('Text matrix duplicate/foreign source')
        seen.add(key)
        scores = metrics(v['source_text'], v['covered_text'], v['received_caption'])
        if any(v[k] != value for k, value in scores.items()): raise ValueError('Text metrics do not recompute')
    if seen != expected: raise ValueError('Missing text rows')


def verify(root, sealed=True):
    root = Path(root).resolve(strict=True); identity = read(root/'run_identity.json'); design = read(root/'design.json')
    if identity['version'] != VERSION or identity['status'] not in ('smoke_complete', 'screen_complete') or (root/'FAILED.json').exists():
        raise ValueError('No completed screen / execution failure')
    if sealed:
        seal_record = read(root/'RECORDS_MANIFEST.json')
        marker = 'SMOKE_COMPLETE.json' if identity['smoke'] else 'SCREEN_COMPLETE.json'
        done = read(root/marker)
        if done['version'] != VERSION or done['manifest_sha256'] != digest(root/'RECORDS_MANIFEST.json') or done['not_a_quality_success_marker'] is not True:
            raise ValueError('Screen marker identity mismatch')
        current = {str(f.relative_to(root)):digest(f) for f in sorted(root.rglob('*')) if f.is_file() and f.name not in ('RECORDS_MANIFEST.json', 'SCREEN_COMPLETE.json', 'SMOKE_COMPLETE.json') and f.suffix != '.pth'}
        if current != seal_record['files']: raise ValueError('Record file missing/extra/hash drift')
        for name, info in seal_record['weights'].items():
            f = safe(root, name)
            if f.exists() and digest(f) != info['sha256']: raise ValueError('Saved weight hash mismatch')
    c = design['config']; ids = design['diagnostic_caption_view']['image_ids']
    if len(ids) != c['diagnostic_samples'] or len(set(ids)) != len(ids): raise ValueError('Wrong TRAIN diagnosis count')
    native = lines(root/'native_controls.jsonl')
    check_text(native, ids, ('native_masked', 'native_all128', 'native_pad_zero'), 'train')
    expected_native = {k:summarize([v for v in native if v['control']==k]) for k in ('native_masked', 'native_all128', 'native_pad_zero')}
    ng = read(root/'native_gate.json')
    if (root/'native_lora_controls.jsonl').is_file():
        fitted = lines(root/'native_lora_controls.jsonl'); check_text(fitted, ids, ('matched',), 'train')
        expected_native['native_lora'] = summarize(fitted)
    if expected_native != ng['groups']: raise ValueError('Native metrics summary mismatch')
    last = expected_native.get('native_lora', expected_native['native_masked'])
    expected_good = last['covered_wer'] <= c['overfit_wer_gate'] and last['capacity_stop_fraction']==0
    if ng['passed'] != expected_good or ng['quality_limit'] != c['overfit_wer_gate']: raise ValueError('Native quality gate mismatch')
    audit = read(root/'bart_interface_audit.json')
    if not audit['passed'] or max(audit['incremental_max_logit_error'], audit['cached_source_max_logit_error']) > c['forward_tolerance']:
        raise ValueError('BART interface failed')
    if read(root/'frozen_verification.json')['passed'] is not True: raise ValueError('Frozen audit missing')
    statuses = read(root/'arm_status.json')
    if set(statuses) != set(design['arms']) or design['arms'] != identity['arms']: raise ValueError('Missing arm status')
    groups_checked = 0
    for arm, status in statuses.items():
        if status['status'] == 'native_stop':
            if ng['passed'] or identity['smoke'] or (root/arm).exists(): raise ValueError('Wrong native stop')
            continue
        if not ng['passed'] and not identity['smoke']: raise ValueError('Training after failed native gate')
        folder = root/arm; entries = lines(folder/'diagnostic_controls.jsonl')
        if not (folder/'diagnostic/TRAIN_COMPLETE.json').is_file(): raise ValueError('Diagnostic training marker missing')
        check_text(entries, ids, ('matched', 'zero', 'shuffled'), 'train')
        groups = {k:summarize([v for v in entries if v['control']==k]) for k in ('matched', 'zero', 'shuffled')}
        expected = gate(groups, c); actual = read(folder/'diagnostic_gate.json')
        if any(actual[k] != v for k,v in expected.items()) or actual['groups'] != groups or status['diagnostic_passed'] != expected['passed']:
            raise ValueError('Arm diagnostic quality mismatch')
        if any(v['geometry'] != geometry(arm) for v in entries): raise ValueError('Payload/header resource mismatch')
        groups_checked += 1
        if status['expanded']:
            if not (folder/'clean/TRAIN_COMPLETE.json').is_file(): raise ValueError('Clean training marker missing')
            if not identity['expand'] or arm not in ('A2_ramp', 'A3_lora') or not (expected['passed'] or identity['smoke']):
                raise ValueError('Unauthorized expansion after negative gate')
            views = design['main_caption_views'][arm]; train_ids = views['train']['image_ids']; val_ids = views['validation']['image_ids']
            if len(train_ids)!=c['train_samples'] or len(val_ids)!=c['val_samples'] or set(train_ids)&set(val_ids): raise ValueError('Invalid main split')
            ve = lines(folder/'validation_controls.jsonl'); check_text(ve, val_ids, ('matched', 'zero', 'shuffled'), 'validation')
            vg = {k:summarize([v for v in ve if v['control']==k]) for k in ('matched', 'zero', 'shuffled')}
            vd = gate(vg, c, False); va = read(folder/'validation_gate.json')
            if any(va[k]!=v for k,v in vd.items()) or status['validation_passed']!=vd['passed']: raise ValueError('Validation gate mismatch')
            if status['status'] == 'waveform_evaluated':
                if not (vd['passed'] or identity['smoke']): raise ValueError('Physical evaluation after failed clean gate')
                wd = read(folder/'waveform_design.json'); cc = wd['config']; we = lines(folder/'waveform.jsonl')
                check_text(we, val_ids, [v['id'] for v in cc['conditions']], 'validation')
                expected_events = {(cid, iid) for cid in [v['id'] for v in cc['conditions']] for iid in val_ids}
                events = lines(folder/'events.jsonl')
                if len(events)!=2*len(expected_events) or any(sum(v['condition']==cid and v['image_id']==iid for v in events)!=2 for cid,iid in expected_events):
                    raise ValueError('Missing probe/features event')
                for v in we:
                    if v['stats']['payload_complex_symbols']!=geometry(arm)['payload_complex_symbols'] or v['stats']['raw_utf8_wire_bytes']!=0:
                        raise ValueError('Physical header/resources not charged')
                    if v['receiver_failure'] and v['received_caption']: raise ValueError('Erased frame silently restored')
                    if v['received_length_correct'] != (v['received_length']==v['covered_tokens']):
                        raise ValueError('Length diagnostic mismatch')
    for complete in root.rglob('TRAIN_COMPLETE.json'):
        folder = complete.parent; done = read(complete); hist = lines(folder/'history.jsonl'); audit = read(folder/'learning_audit.json')
        if len(hist)!=done['steps'] or [v['step'] for v in hist]!=list(range(1, done['steps']+1)) or hist[-1]['optimizer_updates']!=done['optimizer_updates']:
            raise ValueError('Incomplete/duplicated training steps')
        if not audit['passed'] or not audit['frozen_unchanged'] or any(not all(v[k] for k in ('tensors', 'nonzero_gradient', 'changed')) for v in audit['groups'].values()):
            raise ValueError('Gradient/update audit failed')
        if not read(folder/'reload_audit.json')['passed']: raise ValueError('Actual reload audit missing')
        expected_steps = c['native_steps'] if folder.name=='native_lora_overfit' else c['diagnostic_steps'] if folder.name=='diagnostic' else c['train_samples']*c['clean_epochs']
        if done['steps']!=expected_steps: raise ValueError('Training budget differs from contract')
        if folder.name != 'native_lora_overfit':
            consistency = read(folder/'forward_consistency.json')
            if not consistency['passed'] or any(v>c['forward_tolerance'] for v in consistency['errors'].values()): raise ValueError('Bridge forward inconsistent')
    if 'native_lora' in expected_native and not (root/'native_lora_overfit/TRAIN_COMPLETE.json').is_file(): raise ValueError('Native LoRA training missing')
    return dict(passed=True, diagnostic_arms_checked=groups_checked, smoke=identity['smoke'],
        quality_success={k:v['diagnostic_passed'] for k,v in statuses.items()}, whole_run_status=identity['status'])


def render(root):
    root = Path(root); statuses = read(root/'arm_status.json'); ng = read(root/'native_gate.json')
    result = dict(native=ng, arms={})
    rows = lines(root/'native_controls.jsonl')
    for arm, state in statuses.items():
        result['arms'][arm] = dict(state)
        if (root/arm/'diagnostic_gate.json').exists():
            result['arms'][arm]['gate'] = read(root/arm/'diagnostic_gate.json')
            rows += lines(root/arm/'diagnostic_controls.jsonl')
        if (root/arm/'validation_controls.jsonl').exists(): rows += lines(root/arm/'validation_controls.jsonl')
        if (root/arm/'waveform.jsonl').exists(): rows += lines(root/arm/'waveform.jsonl')
    if (root/'native_lora_controls.jsonl').exists(): rows += lines(root/'native_lora_controls.jsonl')
    from v10.io import atomic_json
    atomic_json(root/'summary.json', result)
    page = ['<!doctype html><meta charset="utf-8"><title>文字恢复V3实际结果</title>',
        '<style>body{font:16px sans-serif;padding:24px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:10px;vertical-align:top}td{max-width:460px;overflow-wrap:anywhere}</style>',
        '<h1>文字恢复V3：实际输出，不禁止重复、不筛选好看样例</h1>',
        '<p>Native为训练源特征诊断，不是等资源通信方案。A4更多符号；A5仅文本公共分支，不是图文联合证据。</p>',
        '<table><tr><th>分支／条件／ID</th><th>原文（评价用）</th><th>实际恢复</th><th>WER／结束</th></tr>']
    for v in rows:
        page.append('<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.3f}／{}</td></tr>'.format(
            html.escape('{} / {} / {}'.format(v['arm'], v['control'], v['image_id'])), html.escape(v['source_text']),
            html.escape(v['received_caption']), v['covered_wer'], '达到上限' if v['stopped_by_capacity'] else '接收失败' if v['receiver_failure'] else '正常结束'))
    page.append('</table>')
    with (root/'actual_text_recovery.html').open('x', encoding='utf-8') as f: f.write('\n'.join(page))
    with (root/'REPORT.md').open('x', encoding='utf-8') as f:
        f.write('# Text recovery V3 screen\n\nCompletion means the bounded screen ran, NOT that text recovery improved.\n\n')
        f.write('Actual outputs: actual_text_recovery.html; metrics/gates: summary.json; fixed-last-step checkpoints stay server.\n\n')
        for arm, state in result['arms'].items(): f.write('- {}: {}\n'.format(arm, json.dumps(state, ensure_ascii=False)))


def seal(root):
    from v10.io import atomic_json
    root = Path(root); checked = verify(root, sealed=False)
    files = {str(f.relative_to(root)):digest(f) for f in sorted(root.rglob('*')) if f.is_file() and f.suffix!='.pth'}
    weights = {str(f.relative_to(root)):dict(sha256=digest(f), bytes=f.stat().st_size) for f in sorted(root.rglob('*.pth'))}
    atomic_json(root/'RECORDS_MANIFEST.json', dict(version=VERSION, files=files, weights=weights))
    name = 'SMOKE_COMPLETE.json' if checked['smoke'] else 'SCREEN_COMPLETE.json'
    atomic_json(root/name, dict(version=VERSION, result=checked, manifest_sha256=digest(root/'RECORDS_MANIFEST.json'),
        not_a_quality_success_marker=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root'); args=p.parse_args(); print(json.dumps(verify(args.root), indent=2))
