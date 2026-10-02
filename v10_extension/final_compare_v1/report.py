"""Independent complete-matrix/resource verifier and image-cluster paired summaries."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT))
from final_compare_v1 import VERSION, METHODS, REFERENCES
from final_compare_v1.spec import sampled_count, canonical_hash, resources, digital_samples, packet_symbols
from final_compare_v1.metrics import caption_metrics, task_metrics
from business_link.engine import WIRE
from physical_link.digital import fec_size
from optical_link_v1.physics import dbm_to_w, ratio_db


def read_lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def check_close(a, b, name):
    if not isinstance(a, (float, int)) or not math.isfinite(a) or not math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-16):
        raise ValueError('Accounting mismatch: '+name)


def verify(directory, require_complete=False):
    d = Path(directory).resolve(strict=True)
    design = json.loads((d/'design.json').read_text(encoding='utf-8'))
    identity = json.loads((d/'run_identity.json').read_text(encoding='utf-8'))
    ids = json.loads((d/'validation_ids.json').read_text(encoding='utf-8'))
    if design['version'] != VERSION or design['split'] != 'validation' or design['trained'] is not False:
        raise ValueError('Not this frozen-A validation design')
    if identity['plan_sha256'] != canonical_hash(design['plan']) or identity['optical_config_sha256'] != canonical_hash(design['optical_config']):
        raise ValueError('Plan/physical configuration identity mismatch')
    if len(ids) != design['val_samples'] or len(set(ids)) != len(ids) or any(type(i) is not int for i in ids):
        raise ValueError('Wrong or duplicate source IDs')
    conditions = {c['id']: c for c in design['conditions']}
    expected = {(i, 'clean', 0, m) for i in ids for m in REFERENCES}
    expected.update((i, c, j, m) for i in ids for c in conditions for j in range(design['draws']) for m in METHODS)
    rows, events = read_lines(d/'samples.jsonl'), read_lines(d/'packets.jsonl')
    key = lambda r: (r['image_id'], r['condition'], r['draw'], r['method'])
    if len(rows) != len(expected) or len(rows) != design['expected_rows'] or {key(r) for r in rows} != expected:
        raise ValueError('Incomplete/duplicate/extra comparison matrix; partial data cannot be called complete')
    grouped = {}
    event_keys = set()
    for event in events:
        k = key(event)
        ek = k+(event['purpose'], event['sequence'], event['attempt'])
        if k not in expected or k[1] == 'clean' or ek in event_keys:
            raise ValueError('Extra/duplicate packet event')
        event_keys.add(ek)
        grouped.setdefault(k, []).append(event)
    fingerprints, shared_path, shared_probe = {}, {}, {}
    for row in rows:
        k = key(row)
        for name in ('psnr', 'ssim', 'normalized_wer'):
            if not isinstance(row[name], (int, float)) or not math.isfinite(row[name]):
                raise ValueError('Invalid metric: '+name)
        if not -1 <= row['ssim'] <= 1 or row['normalized_wer'] < 0:
            raise ValueError('Metric out of range')
        text = row['source_caption_evaluation_only'].encode('utf-8')
        if hashlib.sha256(text).hexdigest() != row['source_caption_sha256']:
            raise ValueError('Source caption hash mismatch')
        measured=caption_metrics(row['source_caption_evaluation_only'],row['received_caption'])
        for name in ('wer','cer','normalized_wer'):
            check_close(row[name],measured[name],name)
        if row['text_exact'] != measured['text_exact']:
            raise ValueError('Text exactness was not computed against full source')
        for name,value in task_metrics(row,design['plan']).items():
            if row[name] != value: raise ValueError('Public task threshold mismatch')
        fingerprint = (row['source_caption_sha256'], row['source_rgb_sha256'], row['normalized_payload_sha256'])
        if fingerprints.setdefault(row['image_id'], fingerprint) != fingerprint:
            raise ValueError('Unpaired source business or semantic features')
        for name in ('image_file', 'source_image_file', 'constellation_file'):
            if name in row:
                path = Path(row[name])
                if path.is_absolute() or '..' in path.parts or not (d/path).is_file() or (d/path).resolve().parent not in (d/'images', d/'constellations'):
                    raise ValueError('Missing/unsafe visual artifact')
                if digest(d/path) != row[name+'_sha256']:
                    raise ValueError('Visual artifact changed after evaluation')
        is_reference = row['method'] in REFERENCES
        if bool(row['reference']) != is_reference or is_reference != (row['condition'] == 'clean'):
            raise ValueError('Wrong reference label')
        if is_reference:
            if k in grouped: raise ValueError('Clean reference unexpectedly transmitted')
            continue
        optical = design['effective_cases'][conditions[row['condition']]['case']]
        w = optical['waveform']
        contract=resources(optical,design['plan'])
        if row['slot_samples'] != contract['slot_samples'] or row['launch_target_signal_dbm'] != conditions[row['condition']]['launch_signal_dbm']:
            raise ValueError('Public slot/power changed between methods')
        ev = grouped.get(k, [])
        if len(ev) != row['event_count'] or not ev or ev[0]['purpose'] != 'probe':
            raise ValueError('Missing charged pilot or packet events')
        if row['nominal_snr_db'] is not None:
            raise ValueError('This SI link must not inject a nominal SNR')
        path_key = k[:3]
        if shared_path.setdefault(path_key, row['path_draw']) != row['path_draw']:
            raise ValueError('Methods did not share the same fading realization')
        if shared_probe.setdefault(path_key, row['routing_probe_snr_db']) != row['routing_probe_snr_db']:
            raise ValueError('Methods did not share the same independent pilot observation')
        if row['method'] == 'semantic_analog':
            if row['post_fec_ber'] is not None or row['pre_fec_ber'] is not None or row['feature_object_crc_accepted'] is not None:
                raise ValueError('Analog cannot certify bitwise BER/CRC delivery')
            if len(ev) != 2 or ev[1]['data_symbols'] != 2304*row['analog_repeats']:
                raise ValueError('Uncharged analog repetition')
            if row['analog_repeats'] != contract['analog_repeats']:
                raise ValueError('Continuous redundancy differs from public budget rule')
        elif row['method']=='semantic_digital':
            if row['used_samples'] != contract['semantic_digital_samples']:
                raise ValueError('Semantic digital object/packet overhead differs from public protocol')
            if row['payload_sha256'] != row['normalized_payload_sha256']:
                raise ValueError('Digital feature payload differs from the shared frozen source')
        elif not row.get('resource_refused_before_payload',False):
            if row['used_samples'] != digital_samples([row['compressed_text_wire_bytes'],row['image_wire_bytes']],w,design['plan']['digital_policy']):
                raise ValueError('Traditional object/packet overhead missing')
        elif len(ev)!=1:
            raise ValueError('Refused business unexpectedly transmitted payload')
        used = sum(e['awg_samples']+e['guard_samples'] for e in ev)
        for e in ev:
            if e['awg_samples']+e['guard_samples'] != sampled_count(e['data_symbols'], w):
                raise ValueError('Wire symbols, pulse tails, pilot or guard omitted')
            if e['guard_samples'] != w['guard_samples']:
                raise ValueError('Guard interval changed')
            if e['purpose']!='probe' and row['method']!='semantic_analog':
                policy=design['plan']['digital_policy']
                bits=(WIRE.size+w['packet_bytes']+4)*8
                wire_bits=(bits if policy.endswith('_none') else fec_size(bits))*(2 if policy=='16qam_repeat2' else 1)
                if e['policy']!=policy or e['data_symbols']!=packet_symbols(w,policy) or e['post_bits']!=bits or e['pre_bits']!=wire_bits:
                    raise ValueError('Digital modulation/FEC/repetition/CRC bit accounting changed')
            check_close(e['power_stages']['launch']['signal_w'], dbm_to_w(row['launch_target_signal_dbm']), 'launch power')
            for bits, errors in (('pre_bits', 'pre_errors'), ('post_bits', 'post_errors')):
                if not 0 <= e[errors] <= e[bits]: raise ValueError('Invalid bit errors')
        if used != row['used_samples'] or used > row['slot_samples'] or row['unused_slot_samples'] != row['slot_samples']-used:
            raise ValueError('Time cap violation')
        check_close(row['scheduled_seconds'], used/w['fs_hz'], 'scheduled time')
        energy = sum(e['power_stages']['launch']['signal_w']*e['waveform_seconds'] for e in ev)
        check_close(row['launch_signal_energy_j'], energy, 'signal energy')
        check_close(row['signal_energy_budget_j'],dbm_to_w(row['launch_target_signal_dbm'])*row['slot_samples']/w['fs_hz'],'public signal energy cap')
        if energy > row['signal_energy_budget_j']*(1+1e-8): raise ValueError('Energy cap violation')
        for bits, errors, ber in (('pre_bits', 'pre_errors', 'pre_fec_ber'), ('post_bits', 'post_errors', 'post_fec_ber')):
            n, error = sum(e[bits] for e in ev), sum(e[errors] for e in ev)
            if row[bits] != n or row[errors] != error: raise ValueError('Pooled BER counts differ')
            if n: check_close(row[ber], error/n, ber)
        for signal, noise, output in (('snr_signal_energy_a2','snr_noise_energy_a2','physical_symbol_snr_db'),
                                      ('pilot_signal_energy_v2','pilot_noise_energy_v2','estimated_snr_db')):
            a, b = sum(e[signal] for e in ev), sum(e[noise] for e in ev)
            check_close(row[signal], a, signal)
            check_close(row[noise], b, noise)
            result = ratio_db(a, b)
            if result is None:
                if row[output] is not None: raise ValueError('Undefined physical ratio must be null')
            else: check_close(row[output], result, output)
    if require_complete:
        complete = json.loads((d/'COMPLETE.json').read_text(encoding='utf-8'))
        if identity['status'] != 'complete' or complete['rows'] != len(rows) or complete['version'] != VERSION:
            raise ValueError('Missing completed identity')
        mandatory={'design.json','validation_ids.json','checkpoint_load.json','samples.jsonl','packets.jsonl','summary.json','REPORT.md','report.html'}
        if not mandatory.issubset(complete['result_sha256']):
            raise ValueError('Missing completed immutable evidence hashes')
        for name, expected_hash in complete['result_sha256'].items():
            if Path(name).name != name or digest(d/name) != expected_hash:
                raise ValueError('Result changed since independent completion verification')
    return design, rows, events


def paired_ci(rows, a, b, condition, count, seed=42):
    grouped = {}
    for r in rows:
        if r['condition'] == condition and r['method'] in (a, b):
            grouped.setdefault(r['image_id'], {}).setdefault(r['method'], []).append(r)
    deltas = []
    for image_id, methods in sorted(grouped.items()):
        if set(methods) != {a, b}: raise ValueError('Unpaired comparison')
        deltas.append([np.mean([r[key] for r in methods[a]])-np.mean([r[key] for r in methods[b]])
                       for key in ('psnr', 'normalized_wer', 'joint_task_success')])
    values = np.asarray(deltas, float)
    indices = np.random.default_rng(seed).integers(0, len(values), (count, len(values)))
    boot = values[indices].mean(1)
    return dict(condition=condition, method=a, reference_method=b, independent_images=len(values),
        psnr_delta_db=float(values[:,0].mean()), psnr_ci95=[float(v) for v in np.quantile(boot[:,0],[.025,.975])],
        normalized_wer_delta=float(values[:,1].mean()), normalized_wer_ci95=[float(v) for v in np.quantile(boot[:,1],[.025,.975])],
        joint_success_delta=float(values[:,2].mean()), joint_success_ci95=[float(v) for v in np.quantile(boot[:,2],[.025,.975])],
        bootstrap_unit='image; average repeated channel draws within image BEFORE resampling',
        exploratory_pointwise_ci_not_multiple_comparison_confirmatory=True,
        practical_psnr_candidate=bool(np.quantile(boot[:,0],.025) > .5),
        small_sample_warning=len(values)<20)


def summarize(design, rows):
    groups, conditions = [], {c['id']:c for c in design['conditions']}
    for condition, method in sorted(set((r['condition'],r['method']) for r in rows)):
        sub = [r for r in rows if r['condition']==condition and r['method']==method]
        row = dict(condition=condition, method=method, rows=len(sub), independent_images=len(set(r['image_id'] for r in sub)),
            case=conditions[condition]['case'] if condition!='clean' else 'clean',
            launch_signal_dbm=conditions[condition]['launch_signal_dbm'] if condition!='clean' else None)
        for key in ('psnr','ssim','normalized_wer','wer','cer','text_exact'):
            row[key] = float(np.mean([r[key] for r in sub]))
        for key in ('image_task_success','text_task_success','joint_task_success'):
            row[key+'_rate'] = float(np.mean([r[key] for r in sub]))
        if condition!='clean':
            for key in ('pre_fec_ber','post_fec_ber'):
                prefix = 'pre' if key.startswith('pre_') else 'post'
                n, error = sum(r[prefix+'_bits'] for r in sub), sum(r[prefix+'_errors'] for r in sub)
                row[key] = error/n if n else None
            row.update(scheduled_microseconds=float(np.mean([r['scheduled_seconds'] for r in sub]))*1e6,
                mean_signal_energy_j=float(np.mean([r['launch_signal_energy_j'] for r in sub])),
                physical_symbol_snr_db=ratio_db(sum(r['snr_signal_energy_a2'] for r in sub),sum(r['snr_noise_energy_a2'] for r in sub)),
                estimated_snr_db=ratio_db(sum(r['pilot_signal_energy_v2'] for r in sub),sum(r['pilot_noise_energy_v2'] for r in sub)),
                physical_snr_p05_p50_p95=[float(v) for v in np.quantile([r['physical_symbol_snr_db'] for r in sub if r['physical_symbol_snr_db'] is not None],[.05,.5,.95])] if any(r['physical_symbol_snr_db'] is not None for r in sub) else None,
                mean_received_signal_w=float(np.mean([r['power_stages']['detector_input']['signal_w'] for r in sub])))
            for key in ('image_object_crc_accepted','text_object_crc_accepted','feature_object_crc_accepted'):
                vals = [r[key] for r in sub if r.get(key) is not None]
                row[key+'_rate'] = float(np.mean(vals)) if vals else None
        groups.append(row)
    pairs = [paired_ci(rows,a,b,c,design['plan']['bootstrap_resamples']) for c in conditions
             for a,b in [('semantic_digital','traditional_digital'),('semantic_analog','traditional_digital'),('semantic_analog','semantic_digital')]]
    return dict(version=VERSION, groups=groups, paired_comparisons=pairs, all_failures_included=True,
                inference_only=True, fixed_trained_checkpoint=True, hardware_calibrated=False,
                conclusion_scope='frozen A, specified codec/DSP/slot/power/FSO cases; NOT all analog communication or all digital codecs',
                decision_rule='Use paired task/image/text/time/energy tradeoffs. PSNR gain alone cannot certify joint benefit. Null BER is not zero BER.',
                sample_note='Channel draws are repeated observations, not new independent images/training seeds; CIs exploratory.')


def write_report(directory):
    d = Path(directory)
    design, rows, events = verify(d)
    report = summarize(design, rows)
    (d/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines = ['# 图文光链路最终方向对照', '',
        '冻结A权重；无训练、无测试集选择、无原文进入语义接收端。三条路线使用同一128×128图像和完整caption。',
        '64 GSa/s、4 SPS、16 GBd；数字策略{}，调度上限{} µs。具体RRC/DSP/物理参数见design.json。比较相同带宽、活动平均发射信号光功率、时间/信号能量上限，不是假称实际消耗完全相等。'.format(design['plan']['digital_policy'],design['plan']['slot_microseconds']), '',
        'JPEG质量仅按含导频/包头/CRC/FEC/重复/填充/滤波尾/保护间隔的字节预算选择，不按PSNR挑选。传统分支分别传JPEG与压缩原文；两者CRC独立。语义分支没有原文旁路。',
        '连续分支使用相同归一化语义特征、公开最大整数重复次数、同一慢衰落场；不获得免费时间/能量或独立衰落分集。该A权重适配过数字量化接口，因此连续分支是冻结接口兼容性测试，不是针对连续信道重训优化的DeepJSCC。', '',
        '## 全样本汇总', '', '| 工况 | 路线 | PSNR/dB | 归一化WER | 原文完全一致 | 图文联合达标 | 调度/µs | 后FEC BER |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    f = lambda x: '不适用' if x is None else '{:.4g}'.format(x)
    for r in report['groups']:
        lines.append('| {} | {} | {:.3f} | {:.3f} | {:.1%} | {:.1%} | {} | {} |'.format(
            r['condition'],r['method'],r['psnr'],r['normalized_wer'],r['text_exact'],r['joint_task_success_rate'],f(r.get('scheduled_microseconds')),f(r.get('post_fec_ber'))))
    lines += ['', '## 配对收益与不确定性', '', '| 工况 | 路线−参照 | PSNR差/dB | 图像聚类bootstrap 95%区间 | WER差（负数更好） |', '|---|---|---:|---|---:|']
    for r in report['paired_comparisons']:
        lines.append('| {} | {}−{} | {:.3f} | [{:.3f}, {:.3f}] | {:.3f} |'.format(r['condition'],r['method'],r['reference_method'],r['psnr_delta_db'],*r['psnr_ci95'],r['normalized_wer_delta']))
    lines += ['', '## 如何决定下一条研发路线', '',
        '1. 先看浮点无信道和Q8无信道：若文字在这里已经差，问题在表示/截断/解码或量化，不应归咎于光信道。传统文字分支的完全一致性是不同任务能力，不能偷换为语义token准确率。',
        '2. 看同一工况的全样本图文达标比例，以及配对PSNR/WER差、实际时间和能量；失败输出也在平均值中。只在CRC通过样本上计算会高估数字链路。',
        '3. 如果连续分支没有可重复的资源−质量优势，将它保留为对照，主研发采用更有证据的数字路线；不能由冻结A的失败推导“所有连续语义通信无用”。',
        '4. 本轮是validation方向筛选，不能当最终测试或论文可发表证明。方向/参数锁定后，才在未使用的test图像上做独立确认；不把test加入本次自动流水线。', '',
        '## 指标和仿真边界', '',
        '- PSNR/SSIM：相对同一输入图的像素/结构保真；20 dB仅是公开业务筛选阈值，不是普适恢复标准。',
        '- 原始WER/CER与完全一致性：完整原文任务；归一化WER仅忽略大小写/标点，不等于语义正确率；token_acc仅对神经分支定义。',
        '- 前/后FEC BER和包CRC：只有数字分支有二进制真值；连续分支null，不等于0。BER是真值离线诊断，CRC判决不读发送端。',
        '- 实际物理SNR：仿真可分离有用信号投影与噪声的离线符号比值（不是接收端知道的真值）。总SNR按线性能量合并；SNDR/EVM另包含失真。导频估计与物理SNR不必相等，深衰落有偏置。',
        '- 每个业务包含一个导频及全部线速开销；时间为调度光样本+保护间隔，另列一次L/c传播时延；GPU/CPU执行时间不是光传输时间。有效业务吞吐量使用业务质量阈值，不能当无损字节goodput。',
        '- 激光/MZM/TX EDFA/路径损耗/Gamma-Gamma/指向误差/RX EDFA/ASE/LO/探测/热噪声/RRC/DAC/ADC在现有SI模型中；未建模设备电功耗、空闲ASE、激光/CFO漂移、完整实验仪器响应。没有硬件标定，不声称与OptiSystem完全一致。',
        '- 信号光功率扫点不是人为给接收端塞入SNR。每个burst按已知发射端MZM/EDFA模型设置激光功率，失败则停止；不使用接收功率/衰落/恢复真值做发射控制。',
        '- 本轮只能得到离散扫点的经验范围。32图×3信道重复不能认证普适的突变阈值或95%可靠恢复门限。', '',
        '所有原文/恢复文字、图片、星座、功率与逐包记录见report.html、samples.jsonl和packets.jsonl。']
    (d/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    from final_compare_v1.plots import gallery
    gallery(d,rows,report)
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory'); p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    verify(args.directory,require_complete=True)
    if not args.verify_only:
        raise SystemExit('Completed artifacts are immutable; report created by run.py before COMPLETE. Use --verify-only.')
    print('Independent complete matrix, packet resources and immutable result hashes verified.')
