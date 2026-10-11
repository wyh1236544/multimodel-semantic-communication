"""Whole-matrix independent audit and predeclared new-image paired comparisons."""
import argparse
import csv
import html
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT))
from fso_jscc_v1 import ARMS, VERSION as PARENT_VERSION
from fso_jscc_v1.report import safe_file, read_rows, curve_svg, matrix as parent_matrix
from fso_jscc_eval_v1 import VERSION
from fso_jscc_eval_v1.core import matrix, image_stats, contrast, digest, expected_counts, proxy_conditions, validate


def verify(directory, complete=False):
    root = Path(directory)
    d = json.loads((root/'design.json').read_text(encoding='utf-8'))
    validate(d['config'])
    if d['expected_counts'] != expected_counts(d['config']):
        raise ValueError('Changed declared matrix counts')
    receipt = json.loads((root/'input_receipt.json').read_text(encoding='utf-8'))
    source = json.loads((root/'source_evidence/design.json').read_text(encoding='utf-8'))
    source_done = json.loads((root/'source_evidence/COMPLETE.json').read_text(encoding='utf-8'))
    source_identity = json.loads((root/'source_evidence/run_identity.json').read_text(encoding='utf-8'))
    required_source = {'COMPLETE.json','design.json','run_identity.json','checkpoint_load.json'}
    required_source.update('train_'+a+'/COMPLETE.json' for a in ARMS)
    required_source.update('train_'+a+'/history.jsonl' for a in ARMS[1:])
    if not required_source.issubset(receipt['source_copy_sha256']) or source['version'] != PARENT_VERSION or source['smoke'] or source_identity['status'] != 'complete' or len(source['validation_ids']) != 32 or source['config']['val_samples'] != 32:
        raise ValueError('Missing/incomplete original training provenance')
    for name in required_source-{'COMPLETE.json'}:
        if source_done['result_sha256'].get(name) != receipt['source_copy_sha256'][name]:
            raise ValueError('Copied source evidence differs from original completion hash')
    weights = receipt['weights_before']
    rng = json.loads((root/'latent_rng_replay.json').read_text(encoding='utf-8'))
    if (rng['parent_omitted_cache_draws'] != source['config']['train_samples'] or not rng['exact_cpu_and_device_rng_contract_checked'] or
            rng['training_images_read'] or rng['training_performed']):
        raise ValueError('Missing/changed VAE latent-cache RNG replay contract')
    if weights['base'] != source_identity['checkpoint_sha256'] or weights['A'] != source_identity['a_delta_sha256']:
        raise ValueError('Wrong original base/A identity')
    histories = []
    for a in ARMS:
        r = json.loads((root/'source_evidence'/('train_'+a)/'COMPLETE.json').read_text(encoding='utf-8'))
        if r['arm'] != a:
            raise ValueError('Wrong source training identity')
        if a == ARMS[0]:
            if r['trained'] or r['steps'] or r['optimizer_steps']:
                raise ValueError('Source M0 was trained')
        else:
            history = read_rows(root/'source_evidence'/('train_'+a)/'history.jsonl')
            if (not r['trained'] or r['steps'] != source['train_forward_steps'] or len(history) != r['steps'] or
                    r['optimizer_steps'] != source['optimizer_steps_per_trained_arm'] or
                    min(r['nonzero_gradient_steps'].values()) <= 0 or min(r['parameter_update_squared_l2_by_group'].values()) <= 0 or
                    weights[a] != r['checkpoint_sha256'] or weights[a] != source_done['checkpoint_sha256']['train_'+a+'/last_delta.pth']):
                raise ValueError('Wrong/incomplete original trained checkpoint')
            if [x['forward_step'] for x in history] != list(range(1,r['steps']+1)) or history[-1]['optimizer_step'] != r['optimizer_steps']:
                raise ValueError('Wrong original training step sequence')
            histories.append([(x['epoch'],x['image_id'],x['nominal_snr_db']) for x in history])
    if any(h != histories[0] for h in histories[1:]):
        raise ValueError('Original training groups were not paired')
    from v10.data import validate_manifest
    split = json.loads((root/'split_evidence.json').read_text(encoding='utf-8')); validate_manifest(split)
    if digest(root/'split_evidence.json') != source_identity['split_sha256'] or d['validation_ids'] != split['validation'][:d['config']['samples']]:
        raise ValueError('Changed frozen split or validation order')
    known = {c['id']:c for c in source['physical_plan']['conditions']}
    if (d['proxy_conditions'] != proxy_conditions(d['config']) or
            d['physical_conditions'] != [known[c] for c in d['config']['physical_conditions']] or
            d['physical_plan'] != source['physical_plan'] or d['physical_cases'] != source['physical_cases'] or
            d['physical_resource'] != source['physical_resource']):
        raise ValueError('Changed channel grid or inherited physical plan')
    if d['version'] != VERSION or tuple(d['arms']) != ARMS or d['split'] != 'validation' or d['training_performed']:
        raise ValueError('Wrong evaluator version/arms/split or training was performed')
    ids, old, new = d['validation_ids'], d['pilot_ids'], d['new_ids']
    if (len(ids) != d['config']['samples'] or len(set(ids)) != len(ids) or set(ids) & set(source['train_ids']) or
            set(old) & set(new) or set(ids) != set(old)|set(new) or set(old) != set(ids)&set(source['validation_ids'])):
        raise ValueError('Invalid/overlapping pilot/new/validation image IDs')
    if not d['smoke'] and not new:
        raise ValueError('No new verification images')
    for name, expected in receipt['source_copy_sha256'].items():
        if digest(safe_file(root/'source_evidence', name)) != expected:
            raise ValueError('Changed original source evidence: '+name)
    if receipt['weights_before'] != receipt['weights_after'] or not receipt['parent_verified_at_start_and_end']:
        raise ValueError('Input weights changed or parent run not verified')
    loads = json.loads((root/'checkpoint_load.json').read_text(encoding='utf-8'))
    if (not loads['base_exact'] or not loads['a_exact'] or loads['base_sha256'] != weights['base'] or
            loads['a_delta_sha256'] != weights['A'] or set(loads['arms']) != set(ARMS)):
        raise ValueError('Missing loaded checkpoint arm')
    for a, r in loads['arms'].items():
        if not r['exact'] or not r['all_parameters_frozen'] or r['state_before'] != r['state_after']:
            raise ValueError('Weights changed during evaluation: '+a)
        if a != ARMS[0] and r['checkpoint_sha256'] != weights[a]:
            raise ValueError('Wrong checkpoint loaded for arm')
    rows = read_rows(root/'samples.jsonl')
    seen, paired, events = set(), defaultdict(list), defaultdict(list)
    for e in read_rows(root/'events.jsonl'):
        events[(e['arm'], e['image_id'], e['engine'], e['condition'], e['draw'])].append(e)
    for r in rows:
        key = (r['arm'], r['image_id'], r['engine'], r['condition'], r['draw'])
        if key in seen:
            raise ValueError('Duplicate result row')
        seen.add(key)
        if any(not math.isfinite(r[k]) for k in ('psnr', 'ssim', 'normalized_wer', 'token_acc', 'bleu2', 'cer')):
            raise ValueError('Nonfinite metric')
        if r['source_sha256'] != d['validation_source_sha256'][str(r['image_id'])]:
            raise ValueError('Changed image/caption source')
        if r['source_latent_sha256'] != d['validation_latent_sha256'][str(r['image_id'])]:
            raise ValueError('Changed paired cached VAE latent')
        if r['image_subset'] != ('pilot' if r['image_id'] in old else 'new'):
            raise ValueError('Incorrect old/new image label')
        if r['analog_repeats'] != 1 or r['raw_utf8_wire_bytes'] != 0 or r['pre_fec_ber'] is not None or r['post_fec_ber'] is not None:
            raise ValueError('Hidden repeats/text oracle or fictitious analog BER')
        expected_ok = r['receiver_available'] and r['psnr'] >= d['config']['image_psnr_target_db'] and r['normalized_wer'] <= d['config']['normalized_wer_target']
        if r['diagnostic_joint_success'] != bool(expected_ok):
            raise ValueError('Incorrect diagnostic quality target status')
        paired[key[1:]].append((r['source_sha256'], r['source_latent_sha256'], r['field_seed'], r['noise_seed']))
        for k in ('source_image_file', 'image_file', 'constellation_file'):
            if k in r and digest(safe_file(root, r[k])) != r[k+'_sha256']:
                raise ValueError('Changed visual artifact')
        if r['engine'] == 'proxy':
            if r['payload_complex_symbols'] != 2304 or r['forward_complex_symbols'] != 2560 or events.get(key):
                raise ValueError('Wrong proxy resource accounting')
        elif r['engine'] == 'waveform_v3':
            ev = events.get(key, [])
            if len(ev) != r['event_count'] or [e['purpose'] for e in ev] != ['probe', 'features']:
                raise ValueError('Missing events/hidden retry')
            used = sum(e['awg_samples']+e['guard_samples'] for e in ev)
            if used != r['used_samples'] or used != d['physical_resource']['analog_samples'] or used > r['slot_samples']:
                raise ValueError('Wrong waveform samples/time accounting')
            if r['launch_signal_energy_j'] > r['signal_energy_budget_j']*(1+1e-8):
                raise ValueError('Common peak-envelope energy cap exceeded')
    if seen != matrix(d):
        raise ValueError('Missing/extra matrix cells: missing {}, extra {}'.format(len(matrix(d)-seen), len(seen-matrix(d))))
    if set(events)-{k for k in seen if k[2] == 'waveform_v3'}:
        raise ValueError('Orphan waveform event')
    if any(len(v) != 4 or len(set(v)) != 1 for v in paired.values()):
        raise ValueError('Unpaired arm source/channel draws')
    replay = json.loads((root/'replay.json').read_text(encoding='utf-8'))
    if replay['matched_rows'] != len(matrix(d) & parent_matrix(source)) or replay['matched_rows'] <= 0 or replay['changed_receiver_availability'] or any(
            v > replay['tolerances'][k] for k, v in replay['max_absolute_metric_difference'].items()):
        raise ValueError('Failed frozen-weight original-result replay')
    if complete:
        done = json.loads((root/'COMPLETE.json').read_text(encoding='utf-8'))
        identity = json.loads((root/'run_identity.json').read_text(encoding='utf-8'))
        if done['version'] != VERSION or done['rows'] != len(rows) or identity['status'] != 'complete' or (root/'FAILED.json').exists():
            raise ValueError('Failed/incomplete evaluation')
        required = {'design.json', 'input_receipt.json', 'checkpoint_load.json', 'samples.jsonl', 'events.jsonl',
                    'replay.json', 'latent_rng_replay.json', 'timing.json', 'split_evidence.json', 'run_identity.json', 'summary.json', 'gain_table.csv', 'REPORT.md', 'report.html'}
        required.update('source_evidence/'+n for n in receipt['source_copy_sha256'])
        if not required.issubset(done['result_sha256']):
            raise ValueError('Essential verification evidence missing')
        for name, expected in done['result_sha256'].items():
            if Path(name).suffix in ('.pth', '.pt') or digest(safe_file(root, name)) != expected:
                raise ValueError('Unsafe result manifest/hash mismatch: '+name)
    return d, rows


def write_report(directory):
    from v10.io import atomic_json
    root = Path(directory); d, rows = verify(root)
    groups = defaultdict(list)
    for r in rows:
        for subset in ('all', r['image_subset']):
            groups[(subset, r['engine'], r['condition'], r['arm'])].append(r)
    summary = []
    for (subset, engine, condition, arm), v in sorted(groups.items()):
        entry = dict(subset=subset, engine=engine, condition=condition, arm=arm, trials=len(v),
            **{k:image_stats(v, k) for k in ('psnr', 'ssim', 'normalized_wer', 'token_acc', 'diagnostic_joint_success')},
            receiver_unavailable_fraction=float(np.mean([not r['receiver_available'] for r in v])),
            psnr_trial_p10=float(np.quantile([r['psnr'] for r in v], .1)))
        if engine == 'waveform_v3':
            entry.update(mean_received_optical_signal_dbm=float(np.mean([r['received_optical_signal_dbm'] for r in v])),
                mean_rx_selected_noise_only_snr_db=float(np.mean([r['physical_symbol_snr_db'] for r in v])),
                mean_pilot_estimated_snr_db=float(np.mean([r['estimated_snr_db'] for r in v])),
                mean_actual_launch_signal_energy_j=float(np.mean([r['launch_signal_energy_j'] for r in v])))
        summary.append(entry)
    ci = []
    for subset in ('all', 'pilot', 'new'):
        selected = [r for r in rows if subset == 'all' or r['image_subset'] == subset]
        scopes = {'waveform_low_power_primary': [r for r in selected if r['engine'] == 'waveform_v3' and r['condition'] in d['config']['primary_physical_conditions']],
                  'proxy_FSO_minus5_0_5': [r for r in selected if r['engine'] == 'proxy' and r['condition'].startswith('fso_') and float(r['condition'].split('_snr')[-1]) in (-5, 0, 5)]}
        scopes.update({'waveform_'+c['id']: [r for r in selected if r['engine'] == 'waveform_v3' and r['condition'] == c['id']] for c in d['physical_conditions']})
        for scope, values in scopes.items():
            if not values:
                continue
            for reference in (ARMS[0], ARMS[2]):
                for metric in ('psnr', 'ssim', 'normalized_wer', 'diagnostic_joint_success'):
                    ci.append(dict(subset=subset, scope=scope, reference=reference, candidate=ARMS[3], metric=metric,
                        predeclared_primary=(subset == 'new' and scope == 'waveform_low_power_primary' and reference == ARMS[2] and metric == 'psnr'),
                        **contrast(values, metric, reference, ARMS[3], d['config']['bootstrap_resamples'])))
    # Offline-SNR strata use M0 to define the bins for ALL arms. Exploratory only.
    cuts = [-1000., -10., -5., 0., 5., 10., 1000.]
    bins = {(r['image_id'], r['condition'], r['draw']): int(np.searchsorted(cuts, r['physical_symbol_snr_db'], side='right')-1)
            for r in rows if r['arm'] == ARMS[0] and r['engine'] == 'waveform_v3'}
    strata = []
    for b in range(len(cuts)-1):
        v = [r for r in rows if r['engine'] == 'waveform_v3' and bins[(r['image_id'], r['condition'], r['draw'])] == b]
        if v:
            strata.append(dict(reference_bin_db=[None if abs(x)==1000 else x for x in cuts[b:b+2]],
                definition='M0 offline RX-selected projection noise-only SNR; not exact hardware SNR; all failures retained',
                rows=len(v), psnr_gain=contrast(v, resamples=d['config']['bootstrap_resamples'])))
    result = dict(version=VERSION, rows=len(rows), expected_counts=expected_counts(d['config']), groups=summary,
                  contrasts=ci, snr_strata_exploratory=strata, training_performed=False,
                  primary='M3-M2 PSNR on NEW images, average fixed low-power conditions and draws per image',
                  limitations=['one training replicate; validation not final test', 'pointwise image bootstrap, not training uncertainty or multiple-comparison test',
                               'same symbols/samples/peak-envelope caps, NOT strict equal actual energy',
                               'no independent semantic correctness metric', 'not joint-vs-separate or analog-vs-digital comparison'])
    atomic_json(root/'summary.json', result)
    with (root/'gain_table.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['subset','scope','reference','candidate','metric','predeclared_primary','mean','ci95','images','confidence_scope'])
        writer.writeheader(); writer.writerows(ci)
    lines = ['# 扩大验证：四组现有权重，零重新训练', '',
        '状态：完整矩阵通过独立验证。所有权重冻结；旧结果、模型权重和划分均未改写。', '',
        '验证图片 {} 张：原方向筛选 {} 张、新增 {} 张。新增图片指未用于这次 32 图筛选，不保证没有用于更早的项目历史。'.format(len(d['validation_ids']),len(d['pilot_ids']),len(d['new_ids'])),
        '主比较：新增图片，{}，M3−M2 PSNR；每图先平均固定条件及独立信道试次，再做图片 bootstrap。'.format(', '.join(d['config']['primary_physical_conditions'])),
        'R=1；两次 draw 是两个独立发送试验，不合并恢复。失锁样本全部进入指标。WER 越低越好且可超过 1，不等于准确率。', '',
        '|子集|范围|参考|指标|M3−参考均值|95% 图片 CI|图片数|', '|---|---|---|---|---:|---|---:|']
    for c in ci:
        if c['scope'] in ('waveform_low_power_primary', 'proxy_FSO_minus5_0_5'):
            lines.append('|{}|{}|{}|{}|{:+.5f}|{}|{}|'.format(c['subset'],c['scope'],c['reference'],c['metric'],c['mean'],c['ci95'],c['images']))
    lines += ['', '区间不包含训练随机性；次要条件/指标区间未做多重比较校正。不直接用一个正区间宣称所有场景均优。', '',
        '查看 summary.json：逐条件 PSNR/SSIM/WER 的图片间 SD、尾部 PSNR、不可用率、光功率、离线投影 SNR、导频 SNR、实际能量，以及按 M0 定义的探索性 SNR 分层。', '',
        'diagnostic_joint_success 仅为接收可用且 PSNR≥{} dB、WER≤{} 的固定诊断目标，不是独立语义或 CRC 成功认证。'.format(d['config']['image_psnr_target_db'], d['config']['normalized_wer_target']),
        '现有图文联合架构没有增加显式文字引导，也没有数字/JPEG 或分别传输新对照；不能由此宣称连续或联合传输优越。',
        '名义 SNR 在衰落之前。详细波形 SNR 为仿真接收投影噪声诊断，不能称硬件实测。公共时间槽/峰值上限相同，不代表实际能量逐帧相等。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    parts = ['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>扩大 FSO 验证</title>',
        '<style>body{font:16px/1.6 system-ui;max-width:1400px;margin:30px auto;padding:20px}img{max-width:100%}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.card{background:#f4f6f8;padding:12px;overflow-wrap:anywhere}pre{white-space:pre-wrap}@media(max-width:850px){.grid{grid-template-columns:repeat(2,1fr)}}</style>',
        '<h1>扩大验证：旧 32 图与新增图片分开统计</h1><pre>'+html.escape('\n'.join(lines))+'</pre>']
    for subset in ('all','new'):
        selected = [r for r in rows if subset=='all' or r['image_subset']==subset]
        for profile in ('fso_weak','fso_moderate','fso_strong'):
            for metric in ('psnr','normalized_wer'):
                svg = curve_svg(selected, profile, metric)
                if svg:
                    name = subset+'_'+profile+'_'+metric+'.svg'
                    (root/name).write_text(svg, encoding='utf-8'); parts.append('<h2>{}</h2><img src="{}">'.format(subset,name))
    pictures = defaultdict(list)
    for r in rows:
        if 'image_file' in r:
            pictures[(r['image_id'],r['engine'],r['condition'],r['draw'])].append(r)
    for key, v in sorted(pictures.items()):
        parts.append('<h2>{} · {}</h2><p>原文（仅评估）：{}</p><img width="180" src="{}"><div class="grid">'.format(
            html.escape(str(key)), v[0]['image_subset'], html.escape(v[0]['source_caption_evaluation_only']), v[0]['source_image_file']))
        for r in sorted(v,key=lambda x:x['arm']):
            parts.append('<div class="card"><h3>{}</h3><img width="256" src="{}"><p>PSNR {:.2f} / SSIM {:.3f} / WER {:.3f} / 接收可用 {}</p><p>{}</p></div>'.format(
                r['arm'],r['image_file'],r['psnr'],r['ssim'],r['normalized_wer'],r['receiver_available'],html.escape(r['received_caption'])))
        parts.append('</div>')
    (root/'report.html').write_text('\n'.join(parts),encoding='utf-8')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False); p.add_argument('directory'); p.add_argument('--verify-only',action='store_true')
    a=p.parse_args()
    if a.verify_only:
        verify(a.directory,complete=True); print('Frozen weights, parent replay, old/new subsets, matrix, resources and hashes verified.')
    else:
        write_report(a.directory)
