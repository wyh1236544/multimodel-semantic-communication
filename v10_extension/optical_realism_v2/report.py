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
from optical_realism_v2 import VERSION, METHODS, REFERENCES
from optical_realism_v2.spec import sampled_count, canonical_hash, resources, digital_samples, packet_symbols
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
    fingerprints, shared_path, shared_probe, shared_laser = {}, {}, {}, {}
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
        if shared_laser.setdefault(path_key,row['fixed_laser_dbm']) != row['fixed_laser_dbm']:
            raise ValueError('Fixed common laser working point changed between methods')
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
            if e['power_stages']['launch']['signal_w'] > row['launch_peak_signal_bound_w']*(1+1e-8):
                raise ValueError('Passive hardware peak bound exceeded')
            check_close(e['fixed_laser_dbm'],row['fixed_laser_dbm'],'fixed laser')
            count = 256 + (math.ceil(e['data_symbols']/w['pilot_interval'])*w['pilot_block'] if e['data_symbols'] else 0)
            if e['pilot_symbols'] != count or e['tracking_pilot_symbols'] != count-256:
                raise ValueError('Uncharged interspersed tracking pilots')
            if not e['receiver_locked'] and e['crc_accepted']:
                raise ValueError('Unlocked digital packet falsely accepted')
            for bits, errors in (('pre_bits', 'pre_errors'), ('post_bits', 'post_errors')):
                if not 0 <= e[errors] <= e[bits]: raise ValueError('Invalid bit errors')
        if used != row['used_samples'] or used > row['slot_samples'] or row['unused_slot_samples'] != row['slot_samples']-used:
            raise ValueError('Time cap violation')
        check_close(row['scheduled_seconds'], used/w['fs_hz'], 'scheduled time')
        energy = sum(e['power_stages']['launch']['signal_w']*e['waveform_seconds'] for e in ev)
        check_close(row['launch_signal_energy_j'], energy, 'signal energy')
        check_close(row['signal_energy_budget_j'],row['launch_peak_signal_bound_w']*row['slot_samples']/w['fs_hz'],'public signal energy cap')
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
    report = summarize(design,rows)
    for group in report['groups']:
        sub = [r for r in rows if r['condition']==group['condition'] and r['method']==group['method']]
        if group['condition'] != 'clean':
            group.update(receiver_lock_failures=sum(r['receiver_lock_failures'] for r in sub),
                business_all_bursts_locked_rate=float(np.mean([r['receiver_all_bursts_locked'] for r in sub])),
                mean_tracking_pilot_symbols=float(np.mean([r['tracking_pilot_symbols'] for r in sub])))
    report.update(hardware=design['plan']['hardware'],
        launch_axis='public reference calibration power, not actual equal mean transmitted power',
        fixed_laser_peak_envelope_and_time_budget=True,not_identical_spent_energy=True)
    (d/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\\n',encoding='utf-8')
    lines=['# 相干光链路真实性 V2', '',
        '冻结 A；无训练；仅 validation；完整保留失败。硬件参数为假设，不是实测。', '',
        '每工况以公共 QPSK 波形标定后固定激光。不同业务的实际平均光功率/能量可以不同，不能声称严格等平均发射功率。横轴是校准参考功率；实际值见每行 power_stages。', '',
        '导频间隔/线宽/频偏/时钟/ENOB/IQ 参数见 design.json 的 plan.hardware。接收端只读观测波形和公开导频，未使用 QAM 判决跟踪连续载荷。长帧缓冲后拟合时钟、插值相位；不是实时因果 DSP 延迟测量。', '',
        '| 工况 | 方法 | PSNR | WER | 文字完全一致 | 调度µs | 光信号能量J | 全部突发锁定率 | 后FEC BER |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    f=lambda x: '不适用' if x is None else '{:.4g}'.format(x)
    for g in report['groups']:
        lines.append('| {} | {} | {:.3f} | {:.3f} | {:.1%} | {} | {} | {} | {} |'.format(
            g['condition'],g['method'],g['psnr'],g['normalized_wer'],g['text_exact'],
            f(g.get('scheduled_microseconds')),f(g.get('mean_signal_energy_j')),
            f(g.get('business_all_bursts_locked_rate')),f(g.get('post_fec_ber'))))
    lines += ['', '## 解释边界', '',
        '- 数字采用 Gray QPSK 或 16QAM、软信息、K7 1/2 卷积码；16qam_repeat2 再重复2次。并非 LDPC/Polar 或最优数字方案；JPEG/语义对象仍按整对象 CRC 拒收。',
        '- 连续重复和跟踪导频均计入样点/时间/能量；同慢衰落块内重复不是衰落分集。连续 BER=null，不是0。',
        '- 导频残差含同步/量化/器件失真；离线物理 SNR 分离光学/散粒/热噪声，未把 ENOB 噪声塞入这一分项。后者及全部损伤体现在 EVM/SNDR，二者不能混叫同一个真实 SNR。',
        '- 锁定判断是预设导频阈值，不是能检测所有周跳的完备状态机；拒收保留计分。深衰落下全部路线都可能失锁，连续没有无限低 SNR 保底承诺。',
        '- 8-bit DAC / 12-bit ADC 与附加 ENOB 白噪声仅为替代模型；不等于测得 SINAD/频响。保留旧零相位带宽模型，尚未实现实测复频响、空间波前与跨业务时间相关耦合。',
        '- 64GSa/s / 4SPS=16GBd，RRC滚降0.1。净业务吞吐须扣除导频/包头/CRC/FEC/填充/保护间隔/失败。CPU仿真墙钟不是光学传输或硬件DSP耗时。',
        '- 当前A无信道文字仍可能有表示损失；本版没有新增原文旁路或训练，不能承诺文字质量改善。',
        '- 配对区间见 summary.json；按图像聚类，通道重复不是新增独立图片。方向筛选不构成最终test或论文结论。']
    (d/'REPORT.md').write_text('\\n'.join(lines)+'\\n',encoding='utf-8')
    from optical_realism_v2.plots import gallery
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
