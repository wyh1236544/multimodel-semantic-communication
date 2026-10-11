"""User-started, project-local pipeline. Physical GPU1; no dependency changes."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
EXT = Path(__file__).resolve().parents[1]
PROJECT = EXT.parent
sys.path.insert(0, str(EXT))
from v10.paths import write_path, local_environment
from fso_jscc_v1.core import load, TEMPLATE


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('action', choices=('check', 'smoke', 'quick', 'pipeline'))
    p.add_argument('output', nargs='?')
    p.add_argument('--max-hours', type=float, default=12.)
    p.add_argument('--experiment-config', default=str(TEMPLATE))
    p.add_argument('--checkpoint', default=str(PROJECT/'experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth'))
    p.add_argument('--a-delta', default=str(PROJECT/'experiment_results/v10_token_guidance_quick_01/train_A/last_delta.pth'))
    a = p.parse_args()
    if not math.isfinite(a.max_hours) or not 0 < a.max_hours <= 24:
        p.error('Budget >0 and <=24h; not an ETA')
    exe = os.environ.get('PYTHON_BIN') or sys.executable  # empty shell variable no longer breaks execution
    if not Path(exe).is_file():
        raise FileNotFoundError('Interpreter missing: '+exe)
    env = local_environment(PROJECT); env['CUDA_VISIBLE_DEVICES'] = '1'
    folder = Path(__file__).resolve().parent
    load(a.experiment_config)
    started = time.monotonic()
    def invoke(command, log=None):
        remaining = a.max_hours*3600-(time.monotonic()-started)
        if remaining <= 0:
            raise TimeoutError('Overall launcher budget reached')
        stream = log.open('x', encoding='utf-8') if log else None
        process = None
        try:
            process = subprocess.Popen(command, cwd=str(PROJECT), env=env,
                stdout=stream, stderr=subprocess.STDOUT if stream else None)
            print('Owned child PID={}{}'.format(process.pid, ' log='+str(log) if log else ''), flush=True)
            code = process.wait(timeout=remaining)
            if code:
                raise RuntimeError('Child exit {}; preserve evidence{}'.format(code, ': '+str(log) if log else ''))
        except BaseException:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            raise
        finally:
            if stream:
                stream.close()
    if a.action in ('check', 'pipeline'):
        for name in ('test_core.py', 'test_model.py'):
            invoke([exe, '-B', str(folder/name)])
        invoke([exe, '-B', str(EXT/'optical_realism_v3/test_dsp.py')])
        invoke([exe, '-B', str(EXT/'optical_realism_v3/test_repairs.py')])
        print('Synthetic contracts passed; real A/base/COCO/GPU still require smoke.', flush=True)
    if a.action == 'check':
        if a.output:
            p.error('check takes no output')
        return
    if not a.output:
        p.error('Specify a NEW experiment_results/v10_* output')
    output = write_path(a.output, PROJECT)
    if output.exists() or output.with_name(output.name+'.log').exists():
        raise FileExistsError('Existing output/log retained: '+str(output))
    if output.relative_to(PROJECT).parts[0] != 'experiment_results' or not output.name.startswith('v10_'):
        raise ValueError('Output must be experiment_results/v10_*')
    defaults = dict(checkpoint=Path(a.checkpoint).resolve(), a_delta=Path(a.a_delta).resolve(),
        config=EXT/'configs/v10_multichannel.json', split_manifest=EXT/'manifests/ln_v10_splits.json',
        data_root=PROJECT/'COCO2017', vae_path=PROJECT/'pretrained/sd-vae-ft-mse',
        text_model_path=PROJECT/'pretrained/clip-vit-base-patch32')
    for key, value in defaults.items():
        if not value.exists():
            raise FileNotFoundError('Missing prerequisite {}: {}'.format(key, value))
    def run(target, smoke):
        remaining = a.max_hours-(time.monotonic()-started)/3600
        command = [exe, '-B', '-u', str(folder/'run.py'), '--output-dir', str(target),
                   '--experiment-config', str(Path(a.experiment_config).resolve()), '--max-hours', str(remaining)]
        for key, value in defaults.items():
            command += ['--'+key.replace('_', '-'), str(value)]
        if smoke:
            command += ['--smoke']
        target.parent.mkdir(parents=True, exist_ok=True)
        invoke(command, target.with_name(target.name+'.log'))
        invoke([exe, '-B', str(folder/'report.py'), str(target), '--verify-only'])
    if a.action == 'pipeline':
        output.mkdir(parents=True, exist_ok=False)
        run(output/'v10_smoke', True)
        print('Real smoke passed; now three SAME-budget training arms. See screen.log for progress.', flush=True)
        run(output/'v10_screen', False)
        selected = output/'v10_screen'
    else:
        run(output, a.action == 'smoke'); selected = output
    archive = output/'v10_fso_jscc_v1_records.tar.gz' if a.action == 'pipeline' else output.with_name(output.name+'_records.tar.gz')
    invoke([exe, '-B', str(folder/'collect.py'), str(selected), str(archive)])
    print('Return this archive (no weights): '+str(archive), flush=True)


if __name__ == '__main__':
    main()
