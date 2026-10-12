"""Independent row/gate/hash verification and escaped image/text report."""
import argparse
import html
import json
import math
from pathlib import Path
import sys
from collections import defaultdict
EXT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXT))
from bart_clip_joint_v1 import VERSION, ARMS, CANDIDATES
from bart_clip_joint_v1.core import digest, file_under, summarize, metrics, quality_gate


def read_lines(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def checked_groups(path):
    rows = read_lines(path); groups = defaultdict(list); seen = set()
    for r in rows:
        key = (r['arm'], r['image_id'])
        if key in seen: raise ValueError('Duplicate memory control')
        seen.add(key)
        recomputed = metrics(r['source_text'], r['covered_text'], r['received_caption'])
        if any(isinstance(v, (float, int)) and abs(r[k]-v) > 1e-9 for k, v in recomputed.items() if not isinstance(v, bool)):
            raise ValueError('Text metrics do not match actual received text')
        groups[r['arm']].append(r)
    if set(groups) != {'matched', 'zero', 'shuffled'}: raise ValueError('Missing memory control')
    ids = [{r['image_id'] for r in groups[k]} for k in ('matched', 'zero', 'shuffled')]
    if not ids[0] == ids[1] == ids[2] or len(ids[0]) < 2: raise ValueError('Unpaired controls')
    return {k: summarize(v) for k, v in groups.items()}


def inspect(root):
    root = Path(root); identity = json.loads((root/'run_identity.json').read_text())
    design = json.loads((root/'design.json').read_text()); cfg = design['config']; status = identity['status']
    if identity['version'] != VERSION or design['version'] != VERSION: raise ValueError('Wrong run version')
    allowed = {'diagnostic_ready', 'overfit_stop', 'clean_validation_stop', 'smoke_complete', 'complete'}
    if status not in allowed: raise ValueError('Incomplete/failed run, logs cannot claim success')
    if set(design['train_ids']) & set(design['validation_ids']): raise ValueError('Split overlap')
    if design['payload_complex_symbols'] != 2304 or design['analog_repeats'] != 1 or design['raw_utf8_wire_bytes'] != 0:
        raise ValueError('Physical resource contract differs')
    groups = checked_groups(root/'overfit_controls.jsonl'); gate = quality_gate(groups, cfg)
    stored = json.loads((root/'overfit_gate.json').read_text())
    if stored['passed'] != gate['passed']: raise ValueError('Overfit gate inconsistent with actual outputs')
    if abs(stored['matched_wer']-gate['matched_wer']) > 1e-9: raise ValueError('Changed gate metric')
    if status == 'overfit_stop' and gate['passed']: raise ValueError('Stop status inconsistent')
    if status in ('diagnostic_ready', 'complete', 'clean_validation_stop') and not gate['passed']:
        raise ValueError('Production bypassed quality gate')
    if status == 'smoke_complete' and design['smoke'] is not True: raise ValueError('Smoke flag missing')
    if status == 'complete' and (design['smoke'] or design['stage'] != 'screen'): raise ValueError('Not a formal screen')
    if not (root/'text_private_overfit').is_dir(): raise ValueError('No actual diagnostic training evidence')
    if status == 'complete':
        for name in tuple(c+'_clean' for c in CANDIDATES)+ARMS:
            if not (root/name).is_dir(): raise ValueError('Missing trained arm')
    frozen = json.loads((root/'frozen_verification.json').read_text())
    if not all(frozen.get(k) is True for k in ('bart_and_reference_unchanged', 'old_unchanged', 'clip_unchanged', 'no_test_source_used', 'gradients_verified')):
        raise ValueError('Frozen/source split verification missing')
    real = json.loads((root/'real_backbone_gradient_gate.json').read_text())
    if real.get('passed') is not True or not all(real.get('groups', {}).values()): raise ValueError('Actual-backbone gradient gate missing')
    for name in ('text_private_overfit',)+tuple(c+'_clean' for c in CANDIDATES)+ARMS:
        target = root/name
        if not target.exists(): continue
        audit = json.loads((target/'learning_audit.json').read_text())
        if not audit['passed'] or not audit['frozen_unchanged']: raise ValueError('Required learning/frozen audit failed')
        if not audit['groups'] or any(not v['parameters'] or not v['nonzero_gradient'] or not v['changed'] for v in audit['groups'].values()):
            raise ValueError('Bridge group learning evidence missing')
        history = read_lines(target/'history.jsonl')
        end = json.loads((target/'TRAIN_COMPLETE.json').read_text())
        if len(history) != end['steps'] or [r['step'] for r in history] != list(range(1, len(history)+1)):
            raise ValueError('Training steps incomplete')
        expected_steps = cfg['diagnostic_updates'] if name == 'text_private_overfit' else cfg['train_samples']*(cfg['clean_epochs'] if name.endswith('_clean') else cfg['continue_epochs'])
        if len(history) != expected_steps or end['optimizer_updates'] != math.ceil(expected_steps/cfg['grad_accum']):
            raise ValueError('Incorrect fixed training/update budget')
    rows = read_lines(root/'samples.jsonl') if (root/'samples.jsonl').is_file() else []
    if status == 'complete' or (status == 'smoke_complete' and design['stage'] == 'screen'):
        for c in CANDIDATES:
            g = checked_groups(root/(c+'_clean_controls.jsonl'))
            cg = quality_gate(g, cfg, diagnostic=False)
            if not cg['passed'] and not design['smoke']: raise ValueError('Clean validation gate bypassed')
        for arm in ARMS:
            for stage in ('before', 'after'):
                proof = json.loads((root/(arm+'_fidelity_'+stage+'.json')).read_text())
                cc = design['channel_config']; recomputed_pass = True
                for condition in cc['conditions']:
                    rr = [r for r in proof['rows'] if r['condition'] == condition['id']]
                    if len(rr) != cfg['fidelity_frames_per_condition'] or any(r['source_split'] != 'train' or r['source_image_id'] not in design['train_ids'] for r in rr):
                        raise ValueError('Unpaired/training-source fidelity contract')
                    usable = [r for r in rr if r['available_fit'] and r['available_fresh']]
                    mismatch = sum(r['available_fit'] != r['available_fresh'] for r in rr)/len(rr)
                    actual = sum(r['actual_evm2'] for r in usable)/len(usable) if usable else None
                    predicted = sum(r['predicted_evm2'] for r in usable)/len(usable) if usable else None
                    good = len(usable) >= 2 and abs(predicted-actual)/max(actual, 1e-8) <= cc['surrogate_evm_relative_tolerance'] and mismatch <= cc['surrogate_acceptance_difference_tolerance']
                    recomputed_pass = recomputed_pass and good
                if not proof['passed'] or not recomputed_pass: raise ValueError('Fidelity gate failed')
        expected = {(arm, cid, iid) for arm in ARMS for cid in ['clean']+[r['id'] for r in design['channel_config']['conditions']] for iid in design['validation_ids']}
        actual = {(r['arm'], r['condition'], r['image_id']) for r in rows}
        if actual != expected or len(rows) != len(expected): raise ValueError('Incomplete/duplicate evaluation matrix')
    for r in rows:
        if r['source_sha256'] != design['source_sha256'][str(r['image_id'])] or r['source_split'] != 'validation': raise ValueError('Wrong source identity')
        if r['payload_complex_symbols'] != 2304 or r['analog_repeats'] != 1 or r['raw_utf8_wire_bytes'] != 0: raise ValueError('Unequal payload')
        for k in ('psnr', 'ssim', 'covered_wer'):
            if not math.isfinite(r[k]): raise ValueError('Nonfinite evaluation metric')
        recomputed = metrics(r['source_text'], r['covered_text'], r['received_caption'])
        if abs(recomputed['covered_wer']-r['covered_wer']) > 1e-9: raise ValueError('Actual text disagrees with metric')
        if not r['stats']['receiver_available'] and r['received_caption'] != '': raise ValueError('Failure text was hallucinated instead of erased')
        for k in ('image_file', 'constellation_file'):
            if k in r and not file_under(root, r[k]).is_file(): raise ValueError('Missing visual evidence')
    return identity, design, rows, groups


def report(root):
    root = Path(root); identity, design, rows, controls = inspect(root)
    from v10.io import atomic_json
    groups = defaultdict(list)
    for r in rows: groups[(r['arm'], r['condition'])].append(r)
    summary = {}
    for (arm, cid), rr in groups.items():
        summary[arm+'__'+cid] = dict(summarize(rr), psnr=sum(r['psnr'] for r in rr)/len(rr),
            ssim=sum(r['ssim'] for r in rr)/len(rr), receiver_available_fraction=sum(r['stats']['receiver_available'] for r in rr)/len(rr))
    pairs = []; lookup = {(r['arm'], r['condition'], r['image_id']): r for r in rows}
    comparisons = [('J1_joint_fso', 'J2_separate_fso', 'joint_minus_separate'),
                   ('J1_joint_fso', 'J0_no_clip_input_fso', 'clip_input_gain')]+[(c+'_fso', c+'_clean_continue', c+'_noise_training_gain') for c in CANDIDATES]
    for left, right, name in comparisons:
        for cid in ['clean']+[c['id'] for c in design['channel_config']['conditions']]:
            vv = [(lookup[(left, cid, iid)], lookup[(right, cid, iid)]) for iid in design['validation_ids'] if (left, cid, iid) in lookup and (right, cid, iid) in lookup]
            if vv: pairs.append(dict(comparison=name, condition=cid, images=len(vv),
                psnr_gain_db=sum(a['psnr']-b['psnr'] for a, b in vv)/len(vv),
                wer_reduction=sum(b['covered_wer']-a['covered_wer'] for a, b in vv)/len(vv)))
    value = dict(version=VERSION, status=identity['status'], smoke=design['smoke'], rows=len(rows),
                 text_private_controls=controls, groups=summary, paired_differences=pairs,
                 limits=design['limitations'], actual_analog_ber='undefined, not fabricated')
    atomic_json(root/'summary.json', value)
    esc = lambda value: html.escape(str(value), quote=True)
    output = ['<!doctype html><meta charset="utf-8"><title>BART CLIP C/I/T</title><style>body{font:15px system-ui;margin:32px;color:#172338}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd7e6;padding:9px;vertical-align:top}img{width:160px}pre{white-space:pre-wrap}small{color:#64748b}</style>',
        '<h1>CLIP + BART + VAE 候选实验</h1><p>状态：'+esc(identity['status'])+'</p>',
        '<p>预检或训练集记忆结果不是泛化结果；只有 complete 才有完整 FSO 对照。图片读接收 C/I，不使用原文或接收文字旁路。全部恢复文字是实际自回归生成。</p>',
        '<h2>文字压缩预检：正常／置零／错配接收记忆</h2><pre>'+esc(json.dumps(controls, ensure_ascii=False, indent=2))+'</pre>']
    for label, file in [('未经压缩的预训练 BART（TRAIN诊断）', 'pretrained_uncompressed.jsonl'), ('T32压缩恢复（TRAIN诊断）', 'overfit_controls.jsonl')]:
        output += ['<h2>'+esc(label)+'</h2><table><tr><th>组</th><th>原文，仅供评价</th><th>实际恢复</th><th>覆盖 WER</th></tr>']
        for r in read_lines(root/file):
            output.append('<tr><td>'+esc(r['arm'])+'</td><td>'+esc(r['source_text'])+'</td><td>'+esc(r['received_caption'])+'</td><td>'+esc(round(r['covered_wer'], 4))+'</td></tr>')
        output.append('</table>')
    output += ['<h2>同资源配对差值</h2><pre>'+esc(json.dumps(pairs, ensure_ascii=False, indent=2))+'</pre>',
               '<h2>图文恢复样例</h2><table><tr><th>配置/链路</th><th>原图</th><th>恢复图片</th><th>原文与恢复文本</th><th>指标</th></tr>']
    for r in rows:
        if 'image_file' not in r: continue
        original = 'images/source_'+str(r['image_id'])+'.png'
        details = dict(psnr=r['psnr'], ssim=r['ssim'], covered_wer=r['covered_wer'],
            pilot_snr_db=r['stats'].get('pilot_snr_db'), physical_symbol_snr_db=r['stats'].get('physical_symbol_snr_db'),
            scheduled_seconds=r['stats'].get('scheduled_seconds'), launch_signal_energy_j=r['stats'].get('launch_signal_energy_j'))
        output.append('<tr><td>'+esc(r['arm']+' / '+r['condition'])+'</td><td><img src="'+esc(original)+'"></td><td><img src="'+esc(r['image_file'])+'"></td><td>原文：'+esc(r['source_text'])+'<hr>恢复：'+esc(r['received_caption'])+'</td><td><pre>'+esc(json.dumps(details, ensure_ascii=False, indent=2))+'</pre></td></tr>')
    output += ['</table><p>'+esc(design['limitations'])+'</p>']
    path = root/'report.html'
    if path.exists():
        if (root/'COMPLETE.json').exists() or (root/'DIAGNOSTIC_COMPLETE.json').exists():
            if path.read_text() != '\n'.join(output): raise ValueError('Sealed report differs; preserve it')
            return value
        raise FileExistsError('Report already exists before sealing')
    with path.open('x', encoding='utf-8') as f: f.write('\n'.join(output))
    return value


def seal(root):
    root = Path(root); identity, _, _, _ = inspect(root)
    marker = 'COMPLETE.json' if identity['status'] == 'complete' else 'DIAGNOSTIC_COMPLETE.json'
    files = [f for f in root.rglob('*') if f.is_file() and f.name not in ('COMPLETE.json', 'DIAGNOSTIC_COMPLETE.json')]
    value = dict(version=VERSION, status=identity['status'], wall_seconds=identity['wall_seconds'],
        result_sha256={str(f.relative_to(root)): digest(f) for f in files if f.suffix != '.pth'},
        checkpoint_sha256={str(f.relative_to(root)): digest(f) for f in files if f.suffix == '.pth'})
    with (root/marker).open('x', encoding='utf-8') as f: json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def verify(root, weights=True):
    root = Path(root); identity, _, _, _ = inspect(root)
    marker = 'COMPLETE.json' if identity['status'] == 'complete' else 'DIAGNOSTIC_COMPLETE.json'
    end = json.loads((root/marker).read_text())
    if end['version'] != VERSION or end['status'] != identity['status']: raise ValueError('Wrong completion evidence')
    for n, sha in end['result_sha256'].items():
        if digest(file_under(root, n)) != sha: raise ValueError('Result hash changed: '+n)
    if weights:
        for n, sha in end['checkpoint_sha256'].items():
            if digest(file_under(root, n)) != sha: raise ValueError('Checkpoint hash changed: '+n)
    return identity['status']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('run'); a = p.parse_args()
    print('Verified '+verify(a.run))
