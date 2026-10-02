"""Full failure-preserving reports; physical SNR pooled in linear energy units."""
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from optical_link_v1.physics import ratio_db, w_to_dbm


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['case']].append(row)
    result = {}
    for name, group in groups.items():
        accepted = [r for r in group if r.get('feature_object_crc_accepted', r.get('object_accepted', False))]
        values = dict(rows=len(group), object_delivery_rate=len(accepted)/len(group))
        for metric in ('psnr', 'ssim', 'token_acc', 'bleu2', 'text_wer', 'text_cer'):
            scored = [r[metric] for r in group if r.get(metric) is not None]
            if scored:
                values[metric+'_all_mean'] = sum(scored)/len(scored)
                delivered = [r[metric] for r in accepted if r.get(metric) is not None]
                values[metric+'_delivered_mean'] = sum(delivered)/len(delivered) if delivered else None
        for errors, bits, key in [('pre_errors','pre_bits','pre_fec_ber'), ('post_errors','post_bits','post_fec_ber')]:
            denominator = sum(r.get(bits, 0) for r in group)
            values[key] = sum(r.get(errors, 0) for r in group)/denominator if denominator else None
        energy_s = sum(r.get('snr_signal_energy_a2', 0) for r in group)
        energy_n = sum(r.get('snr_noise_energy_a2', 0) for r in group)
        values['physical_symbol_snr_db_energy_pooled'] = ratio_db(energy_s, energy_n) if energy_s else None
        seconds = sum(r.get('scheduled_seconds') or 0 for r in group)
        values['scheduled_seconds_total'] = seconds
        values['semantic_body_goodput_bps'] = sum(r.get('semantic_body_bytes', 0)*8 for r in accepted)/seconds if seconds else None
        physical = [r for r in group if r.get('power_stages')]
        if physical:
            duration = sum(r['waveform_seconds'] for r in physical)
            values['power_stages_active_time_pooled'] = {}
            for stage in physical[0]['power_stages']:
                entry = {k:sum(r['power_stages'][stage][k]*r['waveform_seconds'] for r in physical)/duration for k in ('signal_w','ase_w','total_w')}
                entry.update(signal_dbm=w_to_dbm(entry['signal_w']), total_dbm=w_to_dbm(entry['total_w']))
                values['power_stages_active_time_pooled'][stage] = entry
        result[name] = values
    return result


def write_report(directory, rows, scope):
    directory = Path(directory)
    summary = dict(scope=scope, groups=summarize(rows),
                   conclusion='completion is not improvement, hardware calibration or publication evidence')
    (directory/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['# 物理光链路结果', '', '范围：'+scope, '',
             '参数是仿真假设，不是硬件实测。无预设SNR；接收端只用导频估计。', '',
             '| 条件 | 行数 | 整对象通过率 | 物理符号SNR/dB | FEC前BER | FEC后BER | 图片PSNR/dB |',
             '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(value):
        return '未测' if value is None else '{:.5g}'.format(value)
    for name, group in summary['groups'].items():
        lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(name, group['rows'],
            fmt(group['object_delivery_rate']), fmt(group['physical_symbol_snr_db_energy_pooled']),
            fmt(group['pre_fec_ber']), fmt(group['post_fec_ber']), fmt(group.get('psnr_all_mean'))))
    lines += ['', 'SNR按线性信号/噪声能量池化，不取dB算术平均；失败样本全部保留。',
              'PSNR是图像指标，不是电SNR。只成功对象的指标另外记录，不能替代全业务结果。',
              '光功率定义和参考位置见README。光链路耗时与程序墙钟时间分开。',
              '所有文字恢复保存在samples.jsonl；已保存图文与星座图见gallery.html。', '']
    (directory/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    cards = []
    for row in rows:
        if not row.get('image_file'):
            continue
        title = '{} / {} / repeat{}'.format(row['image_id'], row['case'], row.get('repeat', 0))
        metrics = 'PSNR={} dB；SNR={} dB；导频估计={} dB；CRC={}'.format(
            fmt(row.get('psnr')), fmt(row.get('physical_symbol_snr_db')), fmt(row.get('estimated_snr_db')), row['feature_object_crc_accepted'])
        cards.append('<section><h2>'+html.escape(title)+'</h2><p>'+html.escape(metrics)+'</p>'
            +'<img src="'+html.escape(row['source_image_file'], quote=True)+'"><img src="'+html.escape(row['image_file'], quote=True)+'">'
            +'<p>原文（仅评价）：'+html.escape(row.get('source_caption_evaluation_only',''))+'</p>'
            +'<p>接收恢复文字：'+html.escape(row.get('received_neural_caption',''))+'</p>'
            +('<a href="'+html.escape(row['constellation_file'], quote=True)+'">均衡前后星座图</a>' if row.get('constellation_file') else '')+'</section>')
    document = '<!doctype html><meta charset="utf-8"><title>物理光链路图文结果</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}section{border-top:1px solid #aaa;padding:16px}img{width:256px;image-rendering:auto;margin-right:12px}p{white-space:pre-wrap}</style><h1>物理光链路：全部条件展示</h1><p>左：原图；右：恢复图。包含CRC失败；非硬件实测。</p>'+''.join(cards)
    (directory/'gallery.html').write_text(document, encoding='utf-8')
    return summary
