"""Independent matrices, metrics, quality stops, training budgets and file hashes."""
import argparse
import html
import json
from pathlib import Path
import sys
EXT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXT))
from bart_latent_codec_v4 import VERSION, ARMS
from bart_latent_codec_v4.core import metrics, summarize, gate, digest, geometry, batch_indices


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p): return [json.loads(v) for v in Path(p).read_text(encoding='utf-8').splitlines() if v.strip()]


def check_rows(rows, ids, controls, split, width, native=False):
    expected = {(control, iid) for control in controls for iid in ids}; seen = set()
    for r in rows:
        key = (r['control'], r['image_id'])
        if key in seen or key not in expected or r['source_split'] != split: raise ValueError('Duplicate/foreign text row')
        seen.add(key)
        if r['geometry'] != geometry(width): raise ValueError('Wire resource/mask mismatch')
        if r['raw_text_exact'] != (r['source_text'] == r['received_caption']): raise ValueError('Exact result mismatch')
        if any(r.get(k) != v for k, v in metrics(r['source_text'], r['covered_text'], r['received_caption']).items()):
            raise ValueError('Text metrics do not recompute')
        if r['receiver_failure'] and r['received_caption']: raise ValueError('Unavailable frame returned text')
        if native and not r.get('source_memory_oracle_train_only'): raise ValueError('Native oracle mislabeled')
        if not native and r.get('source_memory_oracle_train_only'): raise ValueError('Source oracle in communication rows')
        if r['control'] == 'shuffled' and (r.get('donor_image_id') not in ids or r['donor_image_id'] == r['image_id']):
            raise ValueError('Batch-one shuffle/no foreign donor')
    if seen != expected: raise ValueError('Missing text rows')


def groups(rows):
    return {k: summarize([r for r in rows if r['control'] == k]) for k in ('matched', 'zero', 'shuffled')}


def check_training(root, arm, stage, c, width, ids):
    folder = root/arm/stage
    done = read(folder/'TRAIN_COMPLETE.json'); history = lines(folder/'history.jsonl')
    expected = c['diagnostic_updates'] if stage == 'diagnostic' else c['clean_updates'] if stage == 'clean' else c['continue_updates']
    if done['updates'] != expected or [r['update'] for r in history] != list(range(1, expected+1)):
        raise ValueError('Missing/duplicate/wrong update budget')
    for index, r in enumerate(history):
        expected_ids = [ids[i] for i in batch_indices(len(ids), c['batch_size'], index, c['seed'])]
        if r['batch_image_ids'] != expected_ids or r['total_updates'] != expected: raise ValueError('Wrong data order/batch')
        if stage != 'fso_continue' and r['proxy']: raise ValueError('Noise in clean stage')
        if stage == 'fso_continue':
            if len(r['proxy']) != c['batch_size']: raise ValueError('Missing training channel events')
            if any(v['payload_complex_symbols'] != geometry(width)['payload_complex_symbols'] or v['analog_repeats'] != 1 or v['raw_utf8_wire_bytes'] != 0 for v in r['proxy']):
                raise ValueError('Hidden resources in proxy training')
    audit = read(folder/'learning_audit.json')
    if not audit['passed'] or not audit['frozen_unchanged'] or set(audit['groups']) != {'compressor', 'tx_normalization', 'receiver', 'memory'}:
        raise ValueError('Required groups/frozen audit failed')
    if any(not all(v[k] for k in ('tensors', 'nonzero_gradient', 'changed')) for v in audit['groups'].values()):
        raise ValueError('No observed learning in required group')
    if not read(folder/'reload_audit.json')['passed']: raise ValueError('Reload audit missing')
    consistency = read(folder/'forward_consistency.json')
    if consistency['tolerance'] != c['forward_tolerance'] or max(consistency['errors'].values()) > c['forward_tolerance']:
        raise ValueError('Forward mismatch')
    ta = read(folder/'trainability.json')
    if ta['updates'] != expected or ta['batch_size'] != c['batch_size'] or ta['example_presentations'] != expected*c['batch_size'] or ta['decoder_unfrozen'] or ta['original_position_MSE']:
        raise ValueError('Training protocol mismatch')
    weight = folder/'last.pth'
    if weight.exists() and digest(weight) != done['checkpoint_sha256']: raise ValueError('Checkpoint changed')


def verify(root, sealed=True):
    root = Path(root).resolve(strict=True)
    if any(f.is_symlink() for f in root.rglob('*')): raise ValueError('No symlink artifacts')
    identity = read(root/'run_identity.json'); design = read(root/'design.json')
    if identity['version'] != VERSION or identity['status'] not in ('screen_complete', 'smoke_complete') or (root/'FAILED.json').exists():
        raise ValueError('No completed screen / execution failure')
    c = identity['config']; width = identity['width']; statuses = read(root/'arm_status.json')
    if c != design['config'] or width != design['width'] or design['arms'] != list(ARMS) or set(statuses) != set(ARMS):
        raise ValueError('Missing arm/design identity')
    if design['protocol'] != geometry(width) or not design['text_only'] or not design['decoder_frozen']:
        raise ValueError('Design/public wire mismatch')
    ids = design['diagnostic_view']['image_ids']
    if len(ids) != c['diagnostic_samples'] or len(set(ids)) != len(ids): raise ValueError('Wrong diagnosis IDs')
    native = lines(root/'native.jsonl'); check_rows(native, ids, ['matched'], 'train', width, True)
    ns = summarize(native); ng = read(root/'native_gate.json')
    native_ok = ns['covered_wer'] <= c['overfit_wer_gate'] and ns['capacity_stop_fraction'] == 0
    if ng['passed'] != native_ok or ng['groups'] != ns or ng['smoke_quality_bypass'] != identity['smoke']:
        raise ValueError('Native quality differs')
    bi = read(root/'bart_interface_audit.json')
    if not bi['passed'] or max(bi['incremental_max_logit_error'], bi['cached_source_max_logit_error']) > c['forward_tolerance']:
        raise ValueError('BART interface audit failed')
    fv = read(root/'frozen_verification.json')
    if not fv['passed'] or not fv['BART_unchanged'] or not fv['original_data_split_unchanged']: raise ValueError('Frozen evidence missing')
    initial = None; checked = 0
    for arm in ARMS:
        s = statuses[arm]; folder = root/arm
        if s['status'] == 'native_stop':
            if native_ok or identity['smoke'] or folder.exists() or s['expanded'] or s['diagnostic_passed']:
                raise ValueError('Invalid native stop')
            continue
        if not native_ok and not identity['smoke']: raise ValueError('Training after native quality failure')
        current = read(folder/'initial_state_hashes.json')
        if initial is not None and initial != current: raise ValueError('Paired initialization differs')
        initial = current; check_training(root, arm, 'diagnostic', c, width, ids)
        rows = lines(folder/'diagnostic_controls.jsonl')
        check_rows(rows, ids, ['matched', 'zero', 'shuffled', 'matched_beam4'], 'train', width)
        decision = gate(groups(rows), c)
        if read(folder/'diagnostic_gate.json') != decision or s['diagnostic_passed'] != decision['passed']:
            raise ValueError('Greedy diagnostic gate differs')
        checked += 1
        if not s['expanded']:
            if s['status'] != ('diagnostic_passed' if decision['passed'] else 'overfit_stop') or (folder/'clean').exists():
                raise ValueError('Incorrect quality stop')
            continue
        if not identity['expand'] or identity['smoke'] or not decision['passed']: raise ValueError('Unauthorized expansion')
        tr = design['main_train_view']['image_ids']; vi = design['validation_view']['image_ids']
        if len(tr) != c['train_samples'] or len(vi) != c['val_samples'] or set(tr)&set(vi): raise ValueError('Invalid main split')
        check_training(root, arm, 'clean', c, width, tr)
        rows = lines(folder/'validation_controls.jsonl')
        check_rows(rows, vi, ['matched', 'zero', 'shuffled', 'matched_beam4'], 'validation', width)
        vg = gate(groups(rows), c, False)
        if read(folder/'validation_gate.json') != vg or s['validation_passed'] != vg['passed']: raise ValueError('Clean validation gate differs')
        if not vg['passed']:
            if s['status'] != 'clean_validation_stop' or (folder/'fso_continue').exists() or (folder/'eval_clean').exists():
                raise ValueError('Expanded through negative validation')
            continue
        if s['status'] != 'waveform_evaluated': raise ValueError('Missing expanded stage status')
        for stage in ('clean', 'clean_continue', 'fso_continue'):
            if stage != 'clean':
                check_training(root, arm, stage, c, width, tr)
                sr = lines(folder/(stage+'_controls.jsonl')); check_rows(sr, vi, ['matched', 'zero', 'shuffled', 'matched_beam4'], 'validation', width)
            ef = folder/('eval_'+stage); ed = read(ef/'eval_design.json')
            pr = lines(ef/'proxy.jsonl'); wr = lines(ef/'waveform.jsonl')
            check_rows(pr, vi, ed['proxy_conditions'], 'validation', width)
            check_rows(wr, vi, ed['waveform_conditions'], 'validation', width)
            from bart_latent_codec_v4.channel import resource
            expected_resource = resource(ed['waveform_config'], width)
            if ed['resources'] != expected_resource: raise ValueError('Waveform scheduling resource mismatch')
            events = lines(ef/'events.jsonl')
            if len(events) != 2*len(wr): raise ValueError('Missing physical events')
            for r in wr:
                st = r['stats']
                if st['payload_complex_symbols'] != geometry(width)['payload_complex_symbols'] or st['used_samples'] != expected_resource['scheduled_samples'] or st['analog_repeats'] != 1 or st['raw_utf8_wire_bytes'] != 0:
                    raise ValueError('Physical resources mismatch')
                if r['receiver_failure'] != (not st['receiver_available']): raise ValueError('Erasure mismatch')
                if len([v for v in events if v['image_id'] == r['image_id'] and v['condition'] == r['control']]) != 2:
                    raise ValueError('Probe/features pair missing')
    for name, sha in identity['source_sha256'].items():
        if Path(name).name != name or digest(root/'source_snapshot'/name) != sha: raise ValueError('Runtime source snapshot mismatch')
    if sealed:
        sm = read(root/'RECORDS_MANIFEST.json'); marker = 'SMOKE_COMPLETE.json' if identity['smoke'] else 'SCREEN_COMPLETE.json'
        done = read(root/marker)
        if done['version'] != VERSION or done['manifest_sha256'] != digest(root/'RECORDS_MANIFEST.json') or not done['not_a_quality_success_marker']:
            raise ValueError('Screen marker mismatch')
        current = {str(f.relative_to(root)): digest(f) for f in sorted(root.rglob('*')) if f.is_file() and f.suffix != '.pth' and f.name not in ('RECORDS_MANIFEST.json', 'SMOKE_COMPLETE.json', 'SCREEN_COMPLETE.json')}
        if current != sm['files']: raise ValueError('Missing/extra/drifted records')
    return dict(passed=True, diagnostic_arms_checked=checked, smoke=identity['smoke'],
        quality_success={arm: statuses[arm]['diagnostic_passed'] for arm in ARMS})


def finish(root):
    from v10.io import atomic_json
    root = Path(root); verify(root, False)
    summary = dict(native=read(root/'native_gate.json'), arms={}, image_psnr=None, image_recovery_executed=False)
    entries = lines(root/'native.jsonl')
    for arm, s in read(root/'arm_status.json').items():
        summary['arms'][arm] = dict(s)
        for filename in ('diagnostic_gate.json', 'validation_gate.json'):
            if (root/arm/filename).exists(): summary['arms'][arm][filename] = read(root/arm/filename)
        for file in sorted((root/arm).rglob('*controls.jsonl')): entries += lines(file)
        for file in sorted((root/arm).glob('eval_*/*.jsonl')):
            if file.name == 'events.jsonl': continue
            rows = lines(file); entries += rows
            summary['arms'][arm][str(file.relative_to(root/arm))] = {k: summarize([v for v in rows if v['control'] == k]) for k in sorted({v['control'] for v in rows})}
    atomic_json(root/'summary.json', summary)
    page = ['<!doctype html><meta charset="utf-8"><title>BART压缩潜空间V4实际恢复</title>',
        '<style>body{font:16px sans-serif;padding:24px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:9px;vertical-align:top}td{max-width:430px;overflow-wrap:anywhere}</style>',
        '<h1>BART压缩潜空间V4：全部实际文本输出</h1>',
        '<p>Native仅为训练源特征诊断。matched为无约束贪心；matched_beam4为辅助束搜索，不能替代质量门禁。原文只用于评价。本轮不恢复图片。</p>',
        '<table><tr><th>分支/阶段/条件/ID</th><th>输入原文（评价用）</th><th>实际恢复</th><th>WER/结束</th></tr>']
    for r in entries:
        page.append('<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.3f}/{} </td></tr>'.format(
            html.escape('{} / {} / {} / {}'.format(r['arm'], r.get('stage', r['source_split']), r['control'], r['image_id'])),
            html.escape(r['source_text']), html.escape(r['received_caption']), r['covered_wer'],
            '接收失败' if r['receiver_failure'] else '达到长度上限' if r['stopped_by_capacity'] else '自然EOS'))
    page.append('</table>')
    with (root/'actual_text_recovery.html').open('x', encoding='utf-8') as f: f.write('\n'.join(page))
    with (root/'REPORT.md').open('x', encoding='utf-8') as f:
        f.write('# BART latent codec V4\n\nScreen complete is NOT a text-quality success marker.\n\n')
        f.write('All texts: actual_text_recovery.html; gates and metrics: summary.json. Frozen BART, no images, no demonstrated C/I/T joint gain.\n\n')
        for arm, result in summary['arms'].items(): f.write('- {}: {}\n'.format(arm, result['status']))
    files = {str(f.relative_to(root)): digest(f) for f in sorted(root.rglob('*')) if f.is_file() and f.suffix != '.pth'}
    weights = {str(f.relative_to(root)): dict(sha256=digest(f), bytes=f.stat().st_size) for f in sorted(root.rglob('*.pth'))}
    atomic_json(root/'RECORDS_MANIFEST.json', dict(version=VERSION, files=files, weights=weights))
    name = 'SMOKE_COMPLETE.json' if read(root/'run_identity.json')['smoke'] else 'SCREEN_COMPLETE.json'
    atomic_json(root/name, dict(version=VERSION, manifest_sha256=digest(root/'RECORDS_MANIFEST.json'), not_a_quality_success_marker=True))
    verify(root)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root'); a = p.parse_args(); print(json.dumps(verify(a.root), indent=2))
