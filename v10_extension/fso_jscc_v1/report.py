"""Independent matrix/gradient/resource validation, paired image bootstrap, visual report."""
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
from fso_jscc_v1 import VERSION, ARMS
from fso_jscc_v1.core import digest, seed_for


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def safe_file(root, name):
    name = Path(name)
    if name.is_absolute() or '..' in name.parts:
        raise ValueError('Unsafe record path')
    target = root/name
    for parent in [target]+list(target.parents):
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ValueError('Symlink record refused')
    target.resolve().relative_to(root.resolve())
    return target


def matrix(design):
    cfg = design['config']
    keys = set()
    for arm in ARMS:
        for ix, image_id in enumerate(design['validation_ids']):
            for c in design['proxy_conditions']:
                for draw in range(1 if c['profile'] == 'clean' else cfg['draws']):
                    keys.add((arm, image_id, 'proxy', c['id'], draw))
            if ix < cfg['physical_val_samples']:
                for c in design['physical_conditions']:
                    for draw in range(cfg['physical_draws']):
                        keys.add((arm, image_id, 'waveform_v3', c['id'], draw))
    return keys


def verify(directory, complete=False):
    d = Path(directory)
    design = json.loads((d/'design.json').read_text(encoding='utf-8'))
    if design['version'] != VERSION or tuple(design['arms']) != ARMS or design['split'] != 'validation':
        raise ValueError('Wrong version/arms/split')
    cfg = design['config']
    ids = design['validation_ids']
    if len(ids) != cfg['val_samples'] or len(set(ids)) != len(ids) or set(ids) & set(design['train_ids']):
        raise ValueError('Wrong/overlapping image split')
    rows, seen = read_rows(d/'samples.jsonl'), set()
    paired = defaultdict(list)
    event_rows = read_rows(d/'events.jsonl')
    events = defaultdict(list)
    for e in event_rows:
        events[(e['arm'], e['image_id'], e['engine'], e['condition'], e['draw'])].append(e)
    for r in rows:
        key = (r['arm'], r['image_id'], r['engine'], r['condition'], r['draw'])
        if key in seen:
            raise ValueError('Duplicate sample: '+str(key))
        seen.add(key)
        for metric in ('psnr', 'ssim', 'token_acc', 'normalized_wer', 'bleu2'):
            if not math.isfinite(r[metric]):
                raise ValueError('Nonfinite metric')
        if r['analog_repeats'] != 1 or r['raw_utf8_wire_bytes'] != 0 or r['pre_fec_ber'] is not None or r['post_fec_ber'] is not None:
            raise ValueError('Repeated payload/original-text side channel/fictitious analog BER')
        if r['source_sha256'] != design['validation_source_sha256'][str(r['image_id'])]:
            raise ValueError('Mismatched source identity')
        paired[key[1:]].append((r['field_seed'], r['noise_seed'], r['source_sha256']))
        for name in ('image_file', 'source_image_file', 'constellation_file'):
            if name in r and digest(safe_file(d, r[name])) != r[name+'_sha256']:
                raise ValueError('Changed visual artifact')
        if r['engine'] == 'proxy':
            if r['payload_complex_symbols'] != 2304 or r['forward_complex_symbols'] != 2304+cfg['pilots']:
                raise ValueError('Uncharged proxy resource')
            if events.get(key):
                raise ValueError('Unexpected waveform events for analytic proxy')
        elif r['engine'] == 'waveform_v3':
            ev = events.get(key, [])
            if len(ev) != r['event_count'] or [e['purpose'] for e in ev] != ['probe', 'features']:
                raise ValueError('Missing probe/payload events or hidden retransmission')
            used = sum(e['awg_samples']+e['guard_samples'] for e in ev)
            if used != r['used_samples'] or used != design['physical_resource']['analog_samples'] or used > r['slot_samples']:
                raise ValueError('Wrong detailed waveform sample accounting')
            if r['launch_signal_energy_j'] > r['signal_energy_budget_j']*(1+1e-8):
                raise ValueError('Exceeded common peak-envelope energy cap')
    if seen != matrix(design):
        raise ValueError('Incomplete/unexpected matrix: missing {}, extra {}'.format(len(matrix(design)-seen), len(seen-matrix(design))))
    if set(events)-{k for k in seen if k[2] == 'waveform_v3'}:
        raise ValueError('Orphan waveform event')
    if any(len(v) != len(ARMS) or len(set(v)) != 1 for v in paired.values()):
        raise ValueError('Unpaired sources/channel draws across arms')
    training_histories = []
    for arm in ARMS:
        arm_dir = d/('train_'+arm)
        record = json.loads((arm_dir/'COMPLETE.json').read_text(encoding='utf-8'))
        if record['arm'] != arm:
            raise ValueError('Wrong arm record')
        if arm == ARMS[0]:
            if record['trained'] or record['optimizer_steps'] != 0 or record['steps'] != 0:
                raise ValueError('Frozen A was trained')
        else:
            history = read_rows(arm_dir/'history.jsonl')
            expected = design['train_forward_steps']
            updates = design['optimizer_steps_per_trained_arm']
            if (not record['trained'] or record['steps'] != expected or len(history) != expected or
                    record['optimizer_steps'] != updates or min(record['nonzero_gradient_steps'].values()) <= 0 or
                    min(record['parameter_update_squared_l2_by_group'].values()) <= 0):
                raise ValueError('Incomplete training/absent module gradient')
            if [r['forward_step'] for r in history] != list(range(1, expected+1)) or history[-1]['optimizer_step'] != updates:
                raise ValueError('Training history step mismatch')
            if any(r['arm'] != arm or not math.isfinite(r['loss']) for r in history):
                raise ValueError('Wrong training identity/nonfinite loss')
            if any(r['image_id'] not in design['train_ids'] for r in history):
                raise ValueError('Training outside frozen subset')
            expected_profile = 'clean' if arm == ARMS[1] else 'awgn' if arm == ARMS[2] else None
            if any(r['profile'] != expected_profile if expected_profile else r['profile'] not in cfg['profiles'] for r in history):
                raise ValueError('Wrong training channel for arm')
            training_histories.append([(r['epoch'], r['image_id'], r['nominal_snr_db']) for r in history])
            checkpoint = arm_dir/'last_delta.pth'
            # Collected source/result archive intentionally omits large weights.
            if checkpoint.exists() and digest(checkpoint) != record['checkpoint_sha256']:
                raise ValueError('Changed saved checkpoint')
    if any(h != training_histories[0] for h in training_histories[1:]):
        raise ValueError('Unpaired training order/SNR budget')
    if complete:
        done = json.loads((d/'COMPLETE.json').read_text(encoding='utf-8'))
        identity = json.loads((d/'run_identity.json').read_text(encoding='utf-8'))
        if done['version'] != VERSION or done['rows'] != len(rows) or identity['status'] != 'complete' or (d/'FAILED.json').exists():
            raise ValueError('Incomplete/failed run')
        required = {'design.json', 'samples.jsonl', 'events.jsonl', 'checkpoint_load.json', 'run_identity.json',
                    'summary.json', 'REPORT.md', 'report.html', 'gain_table.csv'}
        required.update('train_'+arm+'/COMPLETE.json' for arm in ARMS)
        required.update('train_'+arm+'/history.jsonl' for arm in ARMS[1:])
        if not required.issubset(done['result_sha256']):
            raise ValueError('Completion manifest missing essential evidence')
        for name, expected in done['result_sha256'].items():
            if digest(safe_file(d, name)) != expected:
                raise ValueError('Completion hash mismatch: '+name)
    return design, rows


def paired_ci(rows, reference=ARMS[2], candidate=ARMS[3], resamples=2000, metric='psnr'):
    lookup = {(r['arm'], r['image_id'], r['engine'], r['condition'], r['draw']): r for r in rows}
    groups = defaultdict(list)
    for key, r in lookup.items():
        if r['arm'] != candidate:
            continue
        old = lookup.get((reference,)+key[1:])
        if old is None:
            raise ValueError('Unpaired contrast')
        groups[r['image_id']].append(r[metric]-old[metric])
    values = np.array([np.mean(groups[i]) for i in sorted(groups)])
    if not len(values):
        return dict(mean=None, ci95=None, images=0, confidence_scope='no selected samples')
    rng = np.random.default_rng(seed_for('bootstrap', reference, candidate, metric))
    boot = np.mean(values[rng.integers(0, len(values), size=(resamples, len(values)))], axis=1)
    return dict(mean=float(values.mean()), ci95=[float(x) for x in np.quantile(boot, [.025, .975])],
                images=len(values), unit='dB' if metric == 'psnr' else metric,
                confidence_scope='image-cluster sampling only, single trained model per arm; NOT training-seed uncertainty')


def curve_svg(rows, profile, metric):
    # SVG is a browser/report vector, not a publication-ready PDF claim.
    data = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['engine'] == 'proxy' and r['condition'].startswith(profile+'_snr'):
            data[r['arm']][float(r['condition'].split('_snr')[-1])].append(r)
    xs = sorted({x for series in data.values() for x in series})
    if not xs:
        return ''
    ymin, ymax = (0., 50.) if metric == 'psnr' else (0., max(1.5, max(r[metric] for series in data.values() for group in series.values() for r in group)))
    xp = lambda x: 65+600*(x-xs[0])/max(xs[-1]-xs[0], 1)
    yp = lambda y: 345-270*(y-ymin)/(ymax-ymin)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="880" height="420" viewBox="0 0 880 420">',
             '<rect width="880" height="420" fill="white"/><g font-family="sans-serif" font-size="13">',
             '<text x="65" y="25">{}: {} | R=1; bars=image SD, not training SD</text>'.format(profile, metric)]
    for y in np.linspace(ymin, ymax, 6):
        parts += ['<path d="M65 {} H665" stroke="#ddd"/><text x="8" y="{}">{:.2g}</text>'.format(yp(y), yp(y)+4, y)]
    for x in xs:
        parts += ['<text x="{}" y="370">{:g}</text>'.format(xp(x)-8, x)]
    colors = ('#000000', '#E69F00', '#0072B2', '#CC79A7')
    for index, arm in enumerate(ARMS):
        points = []
        for x in xs:
            items = defaultdict(list)
            for r in data[arm][x]:
                items[r['image_id']].append(r[metric])
            values = np.array([np.mean(v) for v in items.values()])
            mean, sd = float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.
            points.append((xp(x), yp(mean)))
            parts += ['<path d="M{} {} V{}" stroke="{}"/><circle cx="{}" cy="{}" r="4" fill="{}"/>'.format(
                xp(x), yp(max(ymin, mean-sd)), yp(min(ymax, mean+sd)), colors[index], xp(x), yp(mean), colors[index])]
        parts += ['<polyline points="{}" fill="none" stroke="{}" stroke-width="2" stroke-dasharray="{}"/>'.format(
            ' '.join('{},{}'.format(x, y) for x, y in points), colors[index], ('none', '8 3', '3 3', '10 3 2 3')[index]),
            '<text x="690" y="{}" fill="{}">{}</text>'.format(80+index*30, colors[index], arm)]
    parts += ['<text x="230" y="402">Nominal unit-symbol SNR (dB), BEFORE FSO fade</text></g></svg>']
    return ''.join(parts)


def write_report(directory):
    from v10.io import atomic_json
    d = Path(directory)
    design, rows = verify(d)
    groups = defaultdict(list)
    for r in rows:
        groups[(r['arm'], r['engine'], r['condition'])].append(r)
    summary = []
    for key, values in sorted(groups.items()):
        summary.append(dict(arm=key[0], engine=key[1], condition=key[2], trials=len(values),
            images=len({r['image_id'] for r in values}), mean_psnr=float(np.mean([r['psnr'] for r in values])),
            mean_ssim=float(np.mean([r['ssim'] for r in values])), mean_normalized_wer=float(np.mean([r['normalized_wer'] for r in values])),
            token_acc=float(np.mean([r['token_acc'] for r in values])), rejected_fraction=float(np.mean([not r['receiver_available'] for r in values])),
            common_to_source_text_cosine=float(np.mean([r['common_to_source_text_cosine'] for r in values])) if all('common_to_source_text_cosine' in r for r in values) else None,
            receiver_text_to_source_text_cosine=float(np.mean([r['receiver_text_to_source_text_cosine'] for r in values])) if all('receiver_text_to_source_text_cosine' in r for r in values) else None,
            psnr_trial_p10=float(np.quantile([r['psnr'] for r in values], .1))))
    primary_rows = [r for r in rows if r['engine'] == 'proxy' and r['condition'].startswith('fso_') and
                    float(r['condition'].split('_snr')[-1]) in (-5., 0., 5.)]
    contrasts = []
    for reference in ARMS[:-1]:
        for metric in ('psnr', 'normalized_wer'):
            contrasts.append(dict(reference=reference, candidate=ARMS[3], metric=metric,
                scope='proxy_FSO_nominal_minus5_0_5', **paired_ci(primary_rows, reference, ARMS[3], design['config']['bootstrap_resamples'], metric)))
    for c in design['physical_conditions']:
        selected = [r for r in rows if r['engine'] == 'waveform_v3' and r['condition'] == c['id']]
        contrasts.append(dict(reference=ARMS[2], candidate=ARMS[3], metric='psnr', scope='waveform_'+c['id'],
            **paired_ci(selected, ARMS[2], ARMS[3], design['config']['bootstrap_resamples'])))
    result = dict(version=VERSION, rows=len(rows), groups=summary, paired_contrasts=contrasts,
        interpretation='direction-screen only; all failures retained; no p-values or generalized training-seed significance',
        physical_contract='equal symbol/sample length and public peak envelope; measured launch energies may differ',
        continuous_BER=None, no_paper_performance_claim=True)
    atomic_json(d/'summary.json', result)
    with (d/'gain_table.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['reference', 'candidate', 'metric', 'scope', 'mean', 'ci95', 'images', 'confidence_scope', 'unit'])
        writer.writeheader(); writer.writerows(contrasts)
    lines = ['# FSO 单次连续特征带噪训练：方向筛选', '',
             'M0=冻结 A；M1=追加干净微调；M2=AWGN 微调；M3=FSO 微调。M1 不是从未带噪训练的模型。', '',
             '主比较：M3−M2，FSO 名义 −5/0/5dB，按图片聚合信道试次后进行配对 bootstrap。置信区间不包含训练随机性；只训练一个随机种子。', '',
             '没有原文旁路、重复合并或虚构的连续 BER。低功率/失锁样本保留。代理信道 SNR 扫描和 V3 真实参数波形结果必须分开解读。Common/接收文本 latent 的余弦对齐属于同 CLIP 家族指标，不是独立对象或关系正确性证据。', '',
             '|参考|指标|范围|M3−参考均值|95% CI|图片数|', '|---|---|---|---:|---|---:|']
    for c in contrasts:
        lines.append('|{}|{}|{}|{}|{}|{}|'.format(c['reference'], c['metric'], c['scope'], c['mean'], c['ci95'], c['images']))
    lines += ['', '判断方向时同时检查：低 SNR 的 PSNR/文本收益，高 SNR 是否退化，V3 波形链路是否保留收益。此结果不单独证明图文联合优于独立模型，也不证明超越 JPEG/数字通信。', '',
              'VAE/CLIP 冻结；联合编码器、瓶颈、解码器及 TX/RX 小映射参加训练。网络维度不变，4608实数/2304复载荷符号，R=1。', '',
              'FSO 训练层：Gamma-Gamma 光强×指向×路径损耗，复场幅度取平方根；导频估计 CSI，固定发端归一化，噪声不随每帧衰落重设。', '',
              '简化训练层没有复现 DAC/MZM/ASE/时钟同步；V3 独立复测补充这些因素，参数仍是模拟假设，不是实验室校准。']
    (d/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    charts = []
    for profile in design['config']['profiles']:
        for metric in ('psnr', 'normalized_wer'):
            svg = curve_svg(rows, profile, metric)
            if svg:
                name = profile+'_'+metric+'.svg'; (d/name).write_text(svg, encoding='utf-8'); charts.append(name)
    parts = ['<!doctype html><meta charset="utf-8"><title>单次 FSO JSCC</title>',
             '<style>body{font:16px system-ui;max-width:1350px;margin:30px auto;padding:20px}img{max-width:100%}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{background:#f4f6f8;padding:12px;overflow-wrap:anywhere}pre{white-space:pre-wrap}</style>',
             '<h1>单次 FSO 连续传输：四组训练对照</h1><p>R=1。信道 draws 是独立试验，不合并接收。原文只供评估，未送入接收端。</p>',
             '<pre>'+html.escape('\n'.join(lines))+'</pre>']
    for name in charts:
        parts.append('<img src="{}" alt="{}">'.format(name, name))
    images = defaultdict(list)
    for r in rows:
        if 'image_file' in r:
            images[(r['image_id'], r['engine'], r['condition'], r['draw'])].append(r)
    for key, group in sorted(images.items()):
        parts.append('<h2>{}</h2><p>原文（仅评估）：{}</p><img width="160" src="{}"><div class="grid">'.format(
            html.escape(str(key)), html.escape(group[0]['source_caption_evaluation_only']), group[0]['source_image_file']))
        for r in sorted(group, key=lambda x: x['arm']):
            parts.append('<div class="card"><h3>{}</h3><img width="256" src="{}"><p>PSNR {:.2f}dB / WER {:.3f} / RX available {}</p><p>接收文字：{}</p></div>'.format(
                r['arm'], r['image_file'], r['psnr'], r['normalized_wer'], r['receiver_available'], html.escape(r['received_caption'])))
        parts.append('</div>')
    (d/'report.html').write_text('\n'.join(parts), encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('directory'); p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    if args.verify_only:
        verify(args.directory, complete=True); print('Matrix, paired identities, training, images and completion hashes verified.')
    else:
        write_report(args.directory)
