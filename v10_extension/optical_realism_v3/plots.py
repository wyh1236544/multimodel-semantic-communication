"""Offline native SVG/HTML. Escape every caption; no CDN/network dependencies."""
import html
import json
import math
from pathlib import Path

COLORS = {'traditional_digital':'#2478bf', 'semantic_digital':'#cf6b1b', 'semantic_analog':'#298652'}
LABELS = {'traditional_digital':'JPEG+原文数字传输', 'semantic_digital':'语义特征数字传输',
          'semantic_analog':'语义特征连续传输', 'traditional_clean':'JPEG无信道',
          'semantic_float_clean':'语义浮点无信道', 'semantic_q8_clean':'语义Q8无信道'}


def chart(path, summaries, key, label):
    groups = {}
    for row in summaries:
        if row['case'] == 'moderate' and row.get(key) is not None:
            groups.setdefault(row['method'], []).append((row['launch_signal_dbm'], row[key]))
    points = [p for values in groups.values() for p in values]
    if not points:
        return False
    xs, ys = zip(*points)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax: xmin, xmax = xmin-1, xmax+1
    pad = max((ymax-ymin)*.1, .01)
    ymin, ymax = ymin-pad, ymax+pad
    px = lambda x: 72+(x-xmin)/(xmax-xmin)*516
    py = lambda y: 290-(y-ymin)/(ymax-ymin)*230
    bits = ['<svg xmlns="http://www.w3.org/2000/svg" width="680" height="390" viewBox="0 0 680 390">',
            '<rect width="680" height="390" fill="white"/>',
            '<g font-family="sans-serif" font-size="12" fill="#263238">',
            '<text x="72" y="26">'+html.escape(label)+'（包含全部失败样本）</text>']
    for i in range(6):
        y = ymin+(ymax-ymin)*i/5
        bits += ['<path d="M72 {} H588" stroke="#e0e5eb"/>'.format(py(y)),
                 '<text x="12" y="{}">{:.3g}</text>'.format(py(y)+4, y)]
    for x in sorted(set(xs)):
        bits.append('<text x="{}" y="312" text-anchor="middle">{:g}</text>'.format(px(x), x))
    bits.append('<text x="330" y="335" text-anchor="middle">公共校准参考功率 / dBm（实际发射功率另列）</text>')
    for i, (method, values) in enumerate(sorted(groups.items())):
        values.sort()
        color = COLORS[method]
        coords = ' '.join('{:.2f},{:.2f}'.format(px(x), py(y)) for x, y in values)
        bits.append('<polyline points="{}" fill="none" stroke="{}" stroke-width="2"/>'.format(coords, color))
        for x, y in values:
            bits.append('<circle cx="{}" cy="{}" r="3" fill="{}"/>'.format(px(x), py(y), color))
        bits.append('<text x="{}" y="368" fill="{}">{}</text>'.format(18+i*220, color, LABELS[method]))
    bits.append('</g></svg>')
    Path(path).write_text('\n'.join(bits), encoding='utf-8')
    return True


def gallery(directory, rows, report):
    directory = Path(directory)
    plotted = []
    for key, label in [('psnr', '图片PSNR / dB'), ('normalized_wer', '归一化文字WER'),
                       ('joint_task_success_rate', '图文联合达标比例'), ('post_fec_ber', '后FEC误码率（连续方式不适用）'),
                       ('waveform_noise_only_snr_db', '波形分项噪声SNR / dB'),
                       ('physical_symbol_snr_db', '接收选择后投影噪声SNR / dB'), ('scheduled_microseconds', '实际光链路调度时间 / µs')]:
        name = key+'.svg'
        if chart(directory/name, report['groups'], key, label):
            plotted.append(name)
    e = lambda value: html.escape(str(value), quote=True)
    conditions = sorted(set(r['condition'] for r in rows))
    methods = sorted(set(r['method'] for r in rows))
    document = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>图文光链路真实性V3</title>',
        '<style>body{font:15px system-ui;margin:28px;color:#24333d;background:#f5f7fa}select,input{padding:8px;margin:8px}',
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}.card{background:white;padding:16px;border-radius:9px}',
        'img{max-width:100%}.pair{display:flex;gap:8px}.pair img{width:46%;image-rendering:auto}p{overflow-wrap:anywhere}.note{background:#fff1d6;padding:14px}</style>',
        '<h1>固定硬件、计费导频的图文光链路对照</h1>',
        '<p class="note">这是冻结A的方向筛选，不是优化后的DeepJSCC或硬件实测。连续传输的BER/CRC不适用；所有失败均保留。同一激光设置不代表同一实际平均光功率；用时/能量另列。示例没有按恢复效果挑选。</p>',
        '<p><a href="REPORT.md">完整解释与统计</a> · <a href="summary.json">汇总数据</a> · <a href="samples.jsonl">全部逐样本图片/文字记录</a></p>',
        '<div class="grid">']
    for name in plotted:
        document.append('<div class="card"><img src="{}" alt="{}"></div>'.format(e(name), e(name)))
    document += ['</div><h2>实际恢复样例</h2>', '<label>工况 <select id="condition"><option value="">全部</option>']
    document += ['<option value="{}">{}</option>'.format(e(v), e(v)) for v in conditions]
    document += ['</select></label><label>方法 <select id="method"><option value="">全部</option>']
    document += ['<option value="{}">{}</option>'.format(e(v), e(LABELS[v])) for v in methods]
    document += ['</select></label><label>图片编号 <input id="imageid" placeholder="例如 12345"></label>',
                 '<div class="grid" id="cards">']
    for row in rows:
        if 'image_file' not in row:
            continue
        metric = 'PSNR {:.2f} dB · SSIM {:.3f} · WER {:.3f} · 原文完全一致 {}'.format(
            row['psnr'], row['ssim'], row['normalized_wer'], row['text_exact'])
        document += ['<section class="card" data-condition="{}" data-method="{}" data-imageid="{}">'.format(
            e(row['condition']), e(row['method']), row['image_id']),
            '<h3>{} · {} · id {} · draw {}</h3>'.format(e(LABELS[row['method']]), e(row['condition']), row['image_id'], row['draw']),
            '<div class="pair"><img src="{}" alt="原图"><img src="{}" alt="恢复图"></div>'.format(e(row['source_image_file']), e(row['image_file'])),
            '<p>{}</p><p>原文：{}</p><p>恢复：{}</p>'.format(e(metric), e(row['source_caption_evaluation_only']), e(row['received_caption']))]
        if not row['reference']:
            document.append('<p>调度 {:.3f} µs · 接收选择后投影SNR {} dB · 探测导频估计 {} dB · 后FEC BER {} · 接收光功率 {} dBm</p>'.format(
                row['scheduled_seconds']*1e6, e(row['physical_symbol_snr_db']), e(row['routing_probe_snr_db']),
                e(row['post_fec_ber']), e(row['received_optical_signal_dbm'])))
            document.append('<p>波形分项噪声SNR {} dB · 探测锁定 {} · 载荷全部锁定 {} · 发射光信号能量 {:.4g} nJ</p>'.format(
                e(row['waveform_noise_only_snr_db']),e(row['probe_receiver_locked']),e(row['payload_receiver_locked']),row['launch_signal_energy_j']*1e9))
            if row['method']=='semantic_analog':
                document.append('<p>特征MSE（仅离线诊断） {:.4g} · 接收可靠性 {:.4g} · 共同残差下限 {:.4g} · 重复分半估计 {}</p>'.format(
                    row['analog_feature_mse'],row['analog_reliability'],row['analog_correlated_error_floor'],e(row['analog_split_half_variance'])))
        if 'constellation_file' in row:
            document.append('<details><summary>发送/接收星座与均衡效果</summary><img src="{}" alt="星座"></details>'.format(e(row['constellation_file'])))
        document.append('</section>')
    document += ['</div><script>function filter(){const c=document.getElementById("condition").value,m=document.getElementById("method").value,i=document.getElementById("imageid").value.trim();document.querySelectorAll("#cards .card").forEach(x=>{x.hidden=!!((c&&x.dataset.condition!==c)||(m&&x.dataset.method!==m)||(i&&x.dataset.imageid!==i));});}document.querySelectorAll("select,input").forEach(x=>x.addEventListener("input",filter));</script></html>']
    (directory/'report.html').write_text('\n'.join(document), encoding='utf-8')
