"""Strict fixed-task coverage; BER ratios use BIT counts, failures stay in PSNR."""
import csv
import itertools
import json
import math
from pathlib import Path
import numpy as np


def key(row):
    return (row['image_id'], row['profile'], float(row['nominal_snr_db']),
            row['repeat'], row['policy'], row['receiver'])


def mean(values, name):
    data = [float(v[name]) for v in values if v.get(name) is not None]
    return float(np.mean(data)) if data else None


def make_report(rows, design):
    expected = set(itertools.product(design['image_ids'], design['profiles'],
        map(float, design['snr_list']), range(design['repeats']), design['policies'], design['receivers']))
    if len(rows) != len(expected) or len({key(r) for r in rows}) != len(rows) or {key(r) for r in rows} != expected:
        raise ValueError('Missing/duplicate/unplanned task rows; report refused')
    paired, task_inputs = {}, {}
    for row in rows:
        if row['business_bytes'] <= 0 or row['sequential_link_seconds'] <= 0:
            raise ValueError('Invalid business byte/time record')
        numeric = [v for v in row.values() if isinstance(v, (int, float))]
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError('Nonfinite result row')
        image_id = row['image_id']
        input_record = (row['input_sha256'], row['business_bytes'])
        if image_id in task_inputs and input_record != task_inputs[image_id]:
            raise ValueError('Unmatched business input across conditions')
        task_inputs[image_id] = input_record
        scenario = (image_id, row['profile'], float(row['nominal_snr_db']), row['repeat'])
        identity = (row['field_identity'], row['payload_sha256'], row['sender_mode'])
        if scenario in paired and paired[scenario] != identity:
            raise ValueError('Unpaired field / source payload / sender mode across policies')
        paired[scenario] = identity
    if sum(v[1] for v in task_inputs.values()) != design['business_bytes']:
        raise ValueError('Business volume differs from frozen manifest')
    groups = {}
    for profile, snr, policy, receiver in itertools.product(design['profiles'], design['snr_list'], design['policies'], design['receivers']):
        values = [r for r in rows if (r['profile'], float(r['nominal_snr_db']), r['policy'], r['receiver']) == (profile, float(snr), policy, receiver)]
        name = '{} / {} dB / {} / {}'.format(profile, snr, policy, receiver)
        flights = {v['one_way_propagation_seconds'] for v in values}
        if len(flights) != 1:
            raise ValueError('Inconsistent path delay within fixed task')
        # Business tasks stream consecutively: propagation latency charged once
        # per complete business burst, not once per image. Stop-and-wait ARQ and
        # mode-feedback waits are already inside serialization/control seconds.
        total_seconds = sum(v['serialization_and_control_seconds'] for v in values)+next(iter(flights))*design['repeats']
        pre_bits, post_bits = sum(v['pre_bits'] for v in values), sum(v['post_bits'] for v in values)
        attempts = sum(v['packet_attempts'] for v in values)
        success_bytes = sum(v['business_bytes']*int(v['task_quality_success']) for v in values)
        delivered_bytes = sum(v['business_bytes']*int(bool(v['input_delivery_complete'])) for v in values)
        fields = ('psnr', 'ssim', 'text_cer', 'text_wer', 'text_exact_match_diagnostic',
                  'fusion_used', 'estimated_snr_db', 'evm_rms', 'source_quantization_mse',
                  'source_payload_fraction', 'task_quality_success', 'feature_object_crc_accepted',
                  'text_crc_accepted', 'dac_clip_fraction_max', 'gross_modulation_bps')
        g = {field: mean(values, field) for field in fields}
        g.update(profile=profile, nominal_snr_db=snr, policy=policy, receiver=receiver,
            rows=len(values), unique_images=len(design['image_ids']),
            original_business_bytes=design['business_bytes']*design['repeats'],
            scheduled_attempt_time_seconds=total_seconds,
            all_tasks_quality_success=all(v['task_quality_success'] for v in values),
            completion_time_seconds=(total_seconds if all(v['task_quality_success'] for v in values) else None),
            waveform_seconds=sum(v['waveform_seconds'] for v in values),
            normalized_waveform_energy_power_seconds=sum(v.get('waveform_energy_power_seconds', 0) for v in values),
            inference_plus_simulation_wall_seconds=sum(v['inference_plus_channel_simulation_wall_seconds'] for v in values),
            pre_fec_ber=sum(v['pre_errors'] for v in values)/pre_bits if pre_bits else None,
            post_fec_ber=sum(v['post_errors'] for v in values)/post_bits if post_bits else None,
            packet_error_rate=sum(v['crc_failures'] for v in values)/attempts if attempts else None,
            transmitted_coded_bits=pre_bits, post_fec_wire_bits_tested=post_bits,
            effective_wire_code_rate=post_bits/pre_bits if pre_bits else None,
            quality_business_goodput_bps=success_bytes*8/total_seconds,
            delivered_business_goodput_bps=(None if policy == 'analog' else delivered_bytes*8/total_seconds),
            analog_feature_ber_not_defined=(policy == 'analog'),
            task_psnr_threshold=design['task_psnr_threshold'])
        groups[name] = g
    gains = {}
    if {'baseline', 'received_text'} <= set(design['receivers']):
        lookup = {key(r): r for r in rows}
        for profile, snr, policy in itertools.product(design['profiles'], design['snr_list'], design['policies']):
            diffs = []
            for image_id in design['image_ids']:
                differences = []
                for repeat in range(design['repeats']):
                    k = (image_id, profile, float(snr), repeat, policy)
                    a, b = lookup[k+('baseline',)], lookup[k+('received_text',)]
                    for field in ('pre_errors', 'post_errors', 'sequential_link_seconds', 'text_crc_accepted'):
                        if a[field] != b[field]:
                            raise ValueError('Receiver comparison used different wire transmissions')
                    differences.append(b['psnr']-a['psnr'])
                diffs.append(float(np.mean(differences)))
            rng = np.random.default_rng(615)
            bootstrap = np.mean(rng.choice(diffs, (1000, len(diffs))), axis=1)
            gains['{} / {} dB / {}'.format(profile, snr, policy)] = {
                'delta_psnr_db': float(np.mean(diffs)),
                'image_cluster_bootstrap_ci95': np.quantile(bootstrap, [.025, .975]).tolist() if len(diffs) >= 2 else None,
                'images': len(diffs), 'no_training_seed_uncertainty': True}
    return {'schema': 'business-summary-v3', 'rows': len(rows), 'expected_rows': len(expected),
        'fixed_business_bytes': design['business_bytes'], 'image_count': len(design['image_ids']),
        'text_count': len(design['image_ids']), 'summary': groups, 'received_text_fusion_gains': gains,
        'model_retrained': False, 'hardware_calibrated': False, 'diffusion_executed': False,
        'ber_truth_used_only_for_diagnostics': True,
        'failure_rows_included': True, 'same_coherent_field_shared_by_modalities_and_retries': True,
        'waveform_time_is_simulated_not_measured_throughput': True,
        'business_goodput_is_source_equivalent_not_raw_qam_line_rate': True,
        'receiver_fusion_is_untrained_zero_shot_diagnostic': True,
        'arq_assumption': ('charged_error_free_RF_ACK' if design['settings']['max_attempts'] > 1 else 'no_ARQ'),
        'automatic_fec_selection': False}


def write_report(directory, report):
    directory = Path(directory)
    with (directory/'business_summary.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    values = list(report['summary'].values())
    with (directory/'business_table.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    lines = ['# V3 固定业务量链路结果', '',
        '同一批 {} 张图片 + {} 段文字；RGB8 模型输入与 UTF8 业务基准合计 {} 字节。'.format(
            report['image_count'], report['text_count'], report['fixed_business_bytes']), '',
        '所有失败保留；BER 按错误比特总数/测试比特总数计算。传输时间为模拟值，不是 GPU 耗时或实测光通信吞吐。', '',
        '| 条件 | PSNR dB | 文本逐字一致率 | Pre BER | Post BER | PER | 业务合格率 | 合格业务等效 Gbps |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v):
        return 'N/A' if v is None else '{:.4g}'.format(v)
    for name, group in report['summary'].items():
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} |'.format(name,
            fmt(group['psnr']), fmt(group['text_exact_match_diagnostic']), fmt(group['pre_fec_ber']),
            fmt(group['post_fec_ber']), fmt(group['packet_error_rate']), fmt(group['task_quality_success']),
            fmt(group['quality_business_goodput_bps']/1e9)))
    lines += ['', '连续特征支路没有 bit BER/CRC；其 BER 仅覆盖数字文字/模式包。业务等效 goodput 含源压缩，不能当作物理线路比特率。', '',
        '只有全部任务达到预先指定的文字一致 + PSNR 阈值时，completion_time_seconds 才有值；失败条件只报告已花费的尝试时间。', '',
        'received_text 是未再训练的接收端融合诊断，不预先承诺图像增益。没有执行扩散、LDPC、自动 FEC 选择或实测 AWG。']
    with (directory/'REPORT.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('output')
    args = p.parse_args()
    d = Path(args.output)
    rows = [json.loads(line) for line in (d/'samples.jsonl').read_text(encoding='utf-8').splitlines() if line]
    result = make_report(rows, json.loads((d/'design.json').read_text(encoding='utf-8')))
    print(json.dumps({'rows': result['rows'], 'verified': True}, indent=2))
