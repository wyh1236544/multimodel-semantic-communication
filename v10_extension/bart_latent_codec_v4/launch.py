"""Bounded owned-child launcher; no dependency installation or remote execution."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
EXT = Path(__file__).resolve().parents[1]; PROJECT = EXT.parent; sys.path.insert(0, str(EXT))
from v10.paths import write_path, local_environment
from bart_latent_codec_v4.core import load, TEMPLATE, caption_view
from bart_clip_joint_v1.core import offline_snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('action', choices=('check', 'data-check', 'smoke', 'diagnostic', 'pipeline'))
    p.add_argument('output', nargs='?'); p.add_argument('--max-hours', type=float, default=2.)
    p.add_argument('--width', type=int, choices=(16, 32, 64), default=64)
    p.add_argument('--experiment-config', default=str(TEMPLATE))
    defaults = dict(bart_path=PROJECT/'pretrained/bart-base', data_root=PROJECT/'COCO2017',
        split_manifest=EXT/'manifests/ln_v10_splits.json',
        channel_source_run=PROJECT/'experiment_results/v10_joint_jscc_arch_v1_01/v10_screen')
    for k, v in defaults.items(): p.add_argument('--'+k.replace('_', '-'), default=str(v))
    a = p.parse_args(); c = load(a.experiment_config)
    if not math.isfinite(a.max_hours) or not 0 < a.max_hours <= 24: p.error('Finite 0..24-hour budget')
    if (a.action in ('check', 'data-check')) != (a.output is None): p.error('Run needs NEW output; check/data-check need none')
    out = write_path(a.output, PROJECT) if a.output else None
    if out and (out.exists() or out.with_name(out.name+'.log').exists()): raise FileExistsError('Existing output/log preserved; choose02')
    if out and (out.relative_to(PROJECT).parts[0] != 'experiment_results' or not out.name.startswith('v10_')):
        raise ValueError('Project-local experiment_results/v10_* only')
    if a.action == 'data-check':
        for split, count in [('train', c['train_samples']), ('validation', c['val_samples'])]:
            _, receipt = caption_view(a.data_root, a.split_manifest, split, count, c['seed']); print(receipt)
        offline_snapshot(a.bart_path, 'bart'); print('Offline BART + original COCO labels checked; no training/images.'); return
    exe = os.environ.get('PYTHON_BIN') or sys.executable
    if not Path(exe).is_file(): raise FileNotFoundError('Existing Python interpreter not found')
    env = local_environment(PROJECT); env['CUDA_VISIBLE_DEVICES'] = '1'
    folder = Path(__file__).parent; start = time.monotonic()
    def invoke(command, log=None, cap=600):
        remaining = min(cap, a.max_hours*3600-(time.monotonic()-start))
        if remaining <= 0: raise TimeoutError('Overall launcher cap')
        stream = log.open('x', encoding='utf-8') if log else None; child = None
        try:
            child = subprocess.Popen(command, cwd=str(PROJECT), env=env, stdout=stream,
                stderr=subprocess.STDOUT if stream else None)
            print('Owned PID={}{}'.format(child.pid, ' log='+str(log) if log else ''), flush=True)
            code = child.wait(timeout=remaining)
            if code: raise RuntimeError('Child exit {}; preserve logs; no auto retry'.format(code))
        except BaseException:
            if child is not None and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            raise
        finally:
            if stream: stream.close()
    invoke([exe, '-B', str(folder/'install.py'), '--project-root', str(PROJECT), '--verify-only'])
    for name in ('test_core.py', 'test_report.py', 'test_model.py'): invoke([exe, '-B', str(folder/name)])
    print('CPU contracts passed; real saved BART/GPU memory and recovery still require smoke.', flush=True)
    if a.action == 'check': return
    offline_snapshot(a.bart_path, 'bart')
    caption_view(a.data_root, a.split_manifest, 'train', c['diagnostic_samples'], c['seed'])
    if a.action == 'pipeline':
        parent = Path(a.channel_source_run)
        if not (parent/'COMPLETE.json').is_file() or not (parent/'design.json').is_file():
            raise FileNotFoundError('Completed old non-smoke optical config required')
    out.mkdir(parents=True, exist_ok=False)
    def run(name, smoke):
        remaining = a.max_hours*3600-(time.monotonic()-start)-90
        cap = min(1800, remaining) if smoke else remaining
        if cap <= 0: raise TimeoutError('No budget left for collection')
        cmd = [exe, '-B', '-u', str(folder/'run.py'), '--output-dir', str(out/name),
            '--width', str(a.width), '--max-hours', str(cap/3600), '--experiment-config', str(Path(a.experiment_config).resolve())]
        for k in defaults: cmd += ['--'+k.replace('_', '-'), str(Path(getattr(a, k)).resolve())]
        if smoke: cmd.append('--smoke')
        elif a.action == 'pipeline': cmd.append('--expand')
        invoke(cmd, out/(name+'.log'), cap); invoke([exe, '-B', str(folder/'report.py'), str(out/name)])
        return out/name
    selected = run('v10_smoke', True)
    if a.action != 'smoke': selected = run('v10_screen', False)
    invoke([exe, '-B', str(folder/'collect.py'), str(selected), str(out/'v10_bart_latent_codec_v4_records.tar.gz')])
    print('Send v10_bart_latent_codec_v4_records.tar.gz, including negative quality stops; weights stay server.', flush=True)


if __name__ == '__main__': main()
