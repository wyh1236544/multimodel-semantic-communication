"""Owned-child offline GPU1 launcher, no dependency changes or source overwrite."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
EXT = Path(__file__).resolve().parents[1]; PROJECT = EXT.parent; sys.path.insert(0, str(EXT))
from v10.paths import write_path, local_environment
from bart_clip_joint_v1.core import load, TEMPLATE, offline_snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('action', choices=('check', 'smoke', 'diagnostic', 'pipeline')); p.add_argument('output', nargs='?')
    p.add_argument('--max-hours', type=float, default=6.); p.add_argument('--experiment-config', default=str(TEMPLATE))
    p.add_argument('--bart-path', default=str(PROJECT/'pretrained/bart-base'))
    p.add_argument('--source-run', default=str(PROJECT/'experiment_results/v10_joint_jscc_arch_v1_01/v10_screen'))
    p.add_argument('--legacy-run', default=str(PROJECT/'experiment_results/v10_fso_fluctuating_v2_01/v10_screen'))
    p.add_argument('--calibration', default=str(PROJECT/'experiment_results/v10_fso_fluctuating_v2_01/calibration'))
    p.add_argument('--checkpoint', default=str(PROJECT/'experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth'))
    p.add_argument('--a-delta', default=str(PROJECT/'experiment_results/v10_token_guidance_quick_01/train_A/last_delta.pth'))
    a = p.parse_args(); load(a.experiment_config)
    if not math.isfinite(a.max_hours) or not 0 < a.max_hours <= 24: p.error('Finite stop budget >0 and <=24h')
    if (a.action == 'check') != (a.output is None): p.error('check has no output; other actions require NEW output')
    out = write_path(a.output, PROJECT) if a.output else None
    if out and (out.exists() or out.with_name(out.name+'.log').exists()): raise FileExistsError('Existing output/log preserved; use02')
    if out and (out.relative_to(PROJECT).parts[0] != 'experiment_results' or not out.name.startswith('v10_')):
        raise ValueError('Only existing project experiment_results/v10_* outputs')
    exe = os.environ.get('PYTHON_BIN') or sys.executable
    if not Path(exe).is_file(): raise FileNotFoundError('Interpreter missing: '+exe)
    folder = Path(__file__).parent; env = local_environment(PROJECT); env['CUDA_VISIBLE_DEVICES'] = '1'
    started = time.monotonic()
    def invoke(command, log=None, cap=None):
        remaining = a.max_hours*3600-(time.monotonic()-started)
        if cap is not None: remaining = min(remaining, cap)
        if remaining <= 0: raise TimeoutError('Launcher stop budget reached')
        stream = log.open('x', encoding='utf-8') if log else None; child = None
        try:
            child = subprocess.Popen(command, cwd=str(PROJECT), env=env, stdout=stream,
                                     stderr=subprocess.STDOUT if stream else None)
            print('Owned PID={}{}'.format(child.pid, ' log='+str(log) if log else ''), flush=True)
            code = child.wait(timeout=remaining)
            if code: raise RuntimeError('Owned child exit {}; preserve logs; no automatic rerun'.format(code))
        except BaseException:
            if child is not None and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            raise
        finally:
            if stream: stream.close()
    invoke([exe, '-B', str(folder/'install.py'), '--project-root', str(PROJECT), '--verify-only'], cap=600)
    invoke([exe, '-B', str(folder/'test_core.py')], cap=600)
    invoke([exe, '-B', str(folder/'test_report.py')], cap=600)
    invoke([exe, '-B', str(folder/'test_model.py')], cap=600)
    print('CPU/Torch contracts passed. Actual pretrained weights/GPU/effect still require smoke/diagnostic.', flush=True)
    if a.action == 'check': return
    defaults = dict(source_run=Path(a.source_run), legacy_run=Path(a.legacy_run), calibration=Path(a.calibration),
        checkpoint=Path(a.checkpoint), a_delta=Path(a.a_delta), bart_path=Path(a.bart_path),
        config=EXT/'configs/v10_multichannel.json', split_manifest=EXT/'manifests/ln_v10_splits.json',
        data_root=PROJECT/'COCO2017', vae_path=PROJECT/'pretrained/sd-vae-ft-mse', text_model_path=PROJECT/'pretrained/clip-vit-base-patch32')
    for key, value in defaults.items():
        if not value.exists(): raise FileNotFoundError('Missing '+key+': '+str(value)+'; offline server cannot download automatically')
        defaults[key] = value.resolve(strict=True)
    offline_snapshot(defaults['bart_path'], 'bart'); offline_snapshot(defaults['text_model_path'], 'clip')
    out.mkdir(parents=True, exist_ok=False)
    def run(name, stage, smoke):
        remaining = a.max_hours*3600-(time.monotonic()-started)
        cap = min(1800., max(1., remaining-90.)) if smoke else max(1., remaining-90.)
        command = [exe, '-B', '-u', str(folder/'run.py'), '--output-dir', str(out/name), '--stage', stage,
                   '--max-hours', str(cap/3600), '--experiment-config', str(Path(a.experiment_config).resolve())]
        for key, value in defaults.items(): command += ['--'+key.replace('_', '-'), str(value)]
        if smoke: command.append('--smoke')
        invoke(command, out/(name+'.log'), cap)
        invoke([exe, '-B', str(folder/'report.py'), str(out/name)], cap=600)
        return out/name
    selected = run('v10_smoke', 'diagnostic', True)
    if a.action != 'smoke':
        selected = run('v10_diagnostic' if a.action == 'diagnostic' else 'v10_screen',
                       'diagnostic' if a.action == 'diagnostic' else 'screen', False)
    invoke([exe, '-B', str(folder/'collect.py'), str(selected), str(out/'v10_bart_clip_joint_v1_records.tar.gz')], cap=600)
    print('Send small records archive. diagnostic_ready is NOT completed joint/FSO training.', flush=True)


if __name__ == '__main__': main()
