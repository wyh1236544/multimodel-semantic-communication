"""Additive Common-only audit; Python 3.8+, no training or weight writes."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
ROOT = EXT.parent
sys.path.insert(0, str(EXT))
from v10.paths import write_path, local_environment


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def check_sources():
    expected = read(HERE / 'base_hashes.json')
    changed = [p for p, sha in expected.items()
               if not (ROOT / p).is_file() or digest(ROOT / p) != sha]
    if changed:
        raise ValueError('Base implementation differs; do not silently mix experiments: ' + str(changed))
    return expected


def validate_pair(real, random):
    """Require one checkpoint and matching evaluation settings, not just matching rows."""
    for name, item, ablation in [('real', real, 'common_only'),
                                 ('random', random, 'common_only_random')]:
        if item.get('status') != 'complete' or item['args']['ablation'] != ablation:
            raise ValueError('Incomplete or wrong arm: ' + name)
    for key in ('checkpoint_sha256', 'split_sha256', 'config', 'source_sha256'):
        if not real.get(key) or real[key] != random.get(key):
            raise ValueError('Pair mismatch: ' + key)
    for key in ('seed', 'split', 'samples', 'repeats', 'profiles', 'snr_list', 'no_amp'):
        if key not in real['args'] or real['args'][key] != random['args'].get(key):
            raise ValueError('Pair setting mismatch: ' + key)


def commands(args, out):
    common = ['--split', 'validation', '--samples', str(args.samples), '--repeats', '3',
              '--profiles', 'awgn', 'rayleigh', 'fso_moderate',
              '--snr-list', '-5', '0', '10', '--seed', '42', '--no-amp',
              '--batch-size', '1', '--num-workers', '0', '--save-images', '4',
              '--max-hours', str(args.hours_per_arm)]
    return [[sys.executable, '-B', '-u', str(EXT / 'launch_v10.py'), 'eval',
             str(args.checkpoint), str(out / arm), '--ablation', ablation] + common
            for arm, ablation in [('real', 'common_only'), ('random', 'common_only_random')]]


def report(out, full_dirs):
    from verify_v10_run import verify
    from compare_v10 import compare
    real, random = out / 'real', out / 'random'
    verify(real)
    verify(random)
    ri, ni = read(real / 'run_identity.json'), read(random / 'run_identity.json')
    validate_pair(ri, ni)
    # Candidate REAL minus baseline RANDOM. compare also enforces paired rows,
    # physical units, forward/reverse uses and measured average transmit power.
    comparison = compare(random, real, draws=2000, seed=42)
    rs, ns = read(real / 'summary.json'), read(random / 'summary.json')
    full = {}
    for label, directory in full_dirs.items():
        if not (directory / 'summary.json').is_file():
            full[label] = {'available': False, 'path': str(directory)}
            continue
        verify(directory)
        identity = read(directory / 'run_identity.json')
        if identity['args']['ablation'] != 'full':
            raise ValueError('Expected full evaluation: ' + str(directory))
        full[label] = {'available': True, 'summary': read(directory / 'summary.json'),
                       'checkpoint_sha256': identity['checkpoint_sha256'],
                       'same_checkpoint_as_common': identity['checkpoint_sha256'] == ri['checkpoint_sha256'],
                       'same_split_manifest': identity['split_sha256'] == ri['split_sha256'],
                       'interpretation': 'Absolute context only; not an equal-rate Common comparison.'}
    result = {'version': 'V10 Common audit 1', 'development_only': True,
              'checkpoint_sha256': ri['checkpoint_sha256'],
              'real_common': rs, 'random_common': ns, 'real_minus_random': comparison,
              'existing_full_absolute': full,
              'limitations': [
                  'Validation includes checkpoint-selection images; not final test evidence.',
                  'Gaussian random control tests useful learned content, not strict disentanglement.',
                  'Retrieval uses the same CLIP family as training; add independent semantic evaluation.',
                  'Retrieval R@1 is descriptive; existing evaluator does not retain per-image retrieval hits.',
                  'Bootstrap resamples image clusters, not training seeds; pointwise CIs are not multiplicity-adjusted.',
                  'Common and full have different rates; no equal-rate superiority claim.',
                  'Common short-head BLEU against the full narrative is a limited proxy, not factuality.'
              ]}
    save(out / 'audit_summary.json', result)
    lines = ['# Common 分支开发实验', '', '方向：真实 Common − 随机 Common。没有重新训练。', '',
             '|信道 / SNR|真实 R@1|随机 R@1|图库|Common cosine 差值 [95% CI]|短描述 BLEU2 差值 [95% CI]|',
             '|---|---:|---:|---:|---|---|']
    for group, value in comparison['summary'].items():
        a, b = rs['summary'][group], ns['summary'][group]
        delta = value['delta_candidate_minus_baseline']
        def interval(field):
            d = delta[field]
            return '{:+.4f} [{:+.4f}, {:+.4f}]'.format(d['mean'], *d['ci95'])
        lines.append('|{}|{:.3f}|{:.3f}|{}|{}|{}|'.format(
            group, a['common_to_text_retrieval_r1'], b['common_to_text_retrieval_r1'],
            a['retrieval_gallery_size'], interval('common_cosine'), interval('common_caption_bleu2')))
    lines += ['', 'R@1 是匹配原描述的检索，不是对象分类准确率；随机检索参考值为 1/图库大小，但不是置信区间。',
              '', '## 已有 full 绝对指标（不同传输量，仅作背景）', '',
              '|运行|信道 / SNR|PSNR|SSIM|BLEU2|Common R@1|前向次数 / 单位|', '|---|---|---:|---:|---:|---:|---|']
    for label, item in full.items():
        if not item['available']:
            lines.append('|{}|缺少文件||||||'.format(label))
            continue
        for group, v in item['summary']['summary'].items():
            lines.append('|{}|{}|{:.3f}|{:.4f}|{:.4f}|{}|{} / {}|'.format(
                label, group, v['psnr'], v['ssim'], v['bleu2'],
                v.get('common_to_text_retrieval_r1'), v['forward_uses'], v['physical_unit']))
    lines += ['', '## 解读边界', ''] + ['- ' + x for x in result['limitations']]
    with (out / 'REPORT.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    # Fixed first saved images, repeat 0, all conditions. No score-based cherry-picking.
    random_rows = {}
    for line in (random / 'samples.jsonl').read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        random_rows[(r['image_id'], r['profile'], r['nominal_snr_db'], r['repeat'])] = r
    cards = []
    for line in (real / 'samples.jsonl').read_text(encoding='utf-8').splitlines():
        a = json.loads(line)
        if a['repeat'] != 0 or 'image_file' not in a:
            continue
        b = random_rows[(a['image_id'], a['profile'], a['nominal_snr_db'], a['repeat'])]
        e = lambda x: html.escape(str(x), quote=True)
        cards.append('<section><h2>{} · {} · {} dB</h2><p>原描述（仅评价）：{}</p>'
                     '<div><figure><img src="real/images/{}__source.png"><figcaption>原图</figcaption></figure>'
                     '<figure><img src="real/{}"><figcaption>真实 Common：{}</figcaption></figure>'
                     '<figure><img src="random/{}"><figcaption>随机 Common：{}</figcaption></figure></div></section>'.format(
                         e(a['image_id']), e(a['profile']), e(a['nominal_snr_db']),
                         e(a['source_caption_evaluation_only']), e(a['image_id']),
                         e(a['image_file']), e(a['common_caption']), e(b['image_file']), e(b['common_caption'])))
    with (out / 'captions.html').open('x', encoding='utf-8') as f:
        f.write('<!doctype html><meta charset="utf-8"><title>Common audit</title>'
                '<style>body{font:16px sans-serif;max-width:1100px;margin:40px auto;background:#f3f5fa}'
                'section{background:white;padding:20px;margin:20px 0}div{display:flex}figure{flex:1;margin:10px}'
                'img{width:192px;image-rendering:auto}figcaption{line-height:1.6}</style>'
                '<h1>Common 真实／随机对照</h1><p>固定前4张图、repeat 0；图像仅供诊断，核心看检索和短描述。</p>' + ''.join(cards))
    with zipfile.ZipFile(out / 'return_results.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.json', '.jsonl', '.md', '.html', '.png', '.log'):
                z.write(p, p.relative_to(out).as_posix())
    print('Report and return archive:', out / 'return_results.zip', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('action', choices=['plan', 'run', 'report'])
    p.add_argument('--output', default='experiment_results/v10_common_audit_seed42')
    p.add_argument('--checkpoint', default='experiment_results/v10_pilot_noamp_seed42/best.pth')
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--hours-per-arm', type=float, default=1.0)
    p.add_argument('--full-baseline', default='experiment_results/v10_zero_shot_val100_noamp')
    p.add_argument('--full-pilot', default='experiment_results/v10_pilot_val100_noamp')
    args = p.parse_args()
    os.chdir(str(ROOT))
    if not 2 <= args.samples <= 100 or not 0 < args.hours_per_arm <= 1:
        p.error('Bounded development audit: 2..100 images, 0 < hours-per-arm <= 1')
    out = write_path(args.output)
    args.checkpoint = str(Path(args.checkpoint).resolve())
    if ROOT not in Path(args.checkpoint).parents:
        raise ValueError('Checkpoint must be inside multimodel')
    base = check_sources()
    cmds = commands(args, out)
    if args.action == 'plan':
        print(json.dumps({'commands': cmds, 'total_forward_passes': args.samples * 3 * 3 * 3 * 2,
                          'max_wall_hours': args.hours_per_arm * 2,
                          'training': False, 'writes': str(out)}, indent=2))
        return
    if args.action == 'run':
        if not Path(args.checkpoint).is_file():
            raise FileNotFoundError(args.checkpoint)
        out.mkdir(parents=True, exist_ok=False)
        original = digest(args.checkpoint)
        save(out / 'audit_plan.json', {'commands': cmds, 'checkpoint_sha256': original,
                                     'base_sha256': base, 'samples': args.samples})
        env = local_environment()
        for arm, cmd in zip(['real', 'random'], cmds):
            print('Starting {} Common arm (maximum {:.0f} minutes)'.format(arm, args.hours_per_arm * 60), flush=True)
            started = time.monotonic()
            # No shell, never touch another user's process. The launcher handles its child.
            proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
            try:
                code = proc.wait(timeout=args.hours_per_arm * 3600)
            except BaseException:
                if proc.poll() is None:
                    import signal
                    proc.send_signal(signal.SIGINT)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.terminate()
                        proc.wait(timeout=30)
                raise
            if code:
                raise RuntimeError('{} arm failed (exit {}); evidence retained, no automatic retry'.format(arm, code))
            print('{} completed in {:.1f}s'.format(arm, time.monotonic() - started), flush=True)
        if digest(args.checkpoint) != original:
            raise RuntimeError('Checkpoint changed during audit; refuse report')
    report(out, {'zero_shot': Path(args.full_baseline), 'pilot': Path(args.full_pilot)})


if __name__ == '__main__':
    main()
