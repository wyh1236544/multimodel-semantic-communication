"""Offline checks -> discarded real smoke -> fresh fixed-budget training."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_baseline_v14.core import PROJECT, FOLDER, atomic, safe, load, source_check, ARMS


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action', choices=('check', 'calibrate', 'smoke', 'train'))
    p.add_argument('output', nargs='?')
    p.add_argument('--task', choices=('text', 'image', 'both'), default='text')
    p.add_argument('--profile', choices=('pilot', 'full'), default='pilot')
    p.add_argument('--config')
    p.add_argument('--text-arms', default='legacy64,released64')
    p.add_argument('--text-parent')
    p.add_argument('--image-parent')
    p.add_argument('--lpips-assets')
    p.add_argument('--data-root', default=str(PROJECT))
    p.add_argument('--bart-path', default=str(PROJECT/'pretrained/bart-base'))
    p.add_argument('--vae-path', default=str(PROJECT/'pretrained/sd-vae-ft-mse'))
    p.add_argument('--max-hours', type=float, default=8)
    a = p.parse_args()
    if (a.action=='check') != (a.output is None) or not 0 < a.max_hours <= 48:
        p.error('check takes no output; other actions need NEW output and0..48h cap')
    arms = a.text_arms.split(',')
    if len(set(arms)) != len(arms) or any(x not in ARMS for x in arms):
        p.error('Unknown or duplicate text arms')
    load(a.config, a.profile)
    source_check()
    exe = os.environ.get('PYTHON_BIN') or sys.executable
    if not Path(exe).is_file():
        raise FileNotFoundError('Existing interpreter missing; do not install dependencies')
    env = {k:v for k,v in os.environ.items() if k != 'SUDO_PW'}
    env.update(CUDA_VISIBLE_DEVICES='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1', V14_REQUIRE_TORCH='1',
               PYTHONPATH=os.pathsep.join([str(EXT), str(PROJECT)]))
    started = time.monotonic()
    def invoke(cmd, log=None):
        cap = a.max_hours*3600-(time.monotonic()-started)
        if cap <= 0:
            raise TimeoutError('Overall owned-child time budget exhausted')
        stream = log.open('x', encoding='utf-8') if log else None
        child = None
        try:
            child = subprocess.Popen(cmd, cwd=str(PROJECT), env=env, stdout=stream,
                stderr=subprocess.STDOUT if stream else None, start_new_session=True)
            print('Owned PID={}{}'.format(child.pid, ' log='+str(log) if log else ''), flush=True)
            code = child.wait(timeout=cap)
            if code:
                raise RuntimeError('Owned child exit {}; preserve logs; no auto retry'.format(code))
        except BaseException:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise
        finally:
            if stream:
                stream.close()
    for name in ('test_core.py', 'test_report.py', 'test_model.py'):
        invoke([exe, '-B', str(FOLDER/name)])
    print('CPU contracts passed. Actual saved-backbone/GPU/data checks remain mandatory.', flush=True)
    if a.action == 'check':
        return
    output = safe(a.output)
    if output.exists():
        raise FileExistsError('Existing run preserved; choose a NEW attempt')
    output.mkdir(parents=True, exist_ok=False)
    atomic(output/'launch_identity.json', dict(action=a.action, task=a.task, profile=a.profile,
           text_arms=arms, max_hours=a.max_hours, physical_gpu=1, smoke_weights_reused=False,
           collection_is_separate_CPU_step=True, no_global_environment_changes=True))
    def run(mode, name):
        remaining = a.max_hours-(time.monotonic()-started)/3600
        if remaining <= 0:
            raise TimeoutError('No remaining GPU-program budget')
        cmd = [exe, '-B', '-u', str(FOLDER/'run.py'), '--mode', mode, '--task', a.task,
               '--profile', a.profile, '--text-arms', a.text_arms, '--output', str(output/name),
               '--max-hours', str(remaining)]
        for key in ('config', 'text_parent', 'image_parent', 'lpips_assets', 'data_root', 'bart_path', 'vae_path'):
            value = getattr(a, key)
            if value:
                cmd += ['--'+key.replace('_', '-'), str(Path(value).resolve())]
        invoke(cmd, output/(name+'.log'))
        invoke([exe, '-B', str(FOLDER/'report.py'), str(output/name)], output/(name+'_verify.log'))
    try:
        if a.action == 'calibrate':
            # Calibration never calls fit() and never writes new neural weights.
            run('calibrate', 'calibration')
            target = output/'calibration'
        else:
            run('smoke', 'smoke')
            target = output/'smoke'
            if a.action == 'train':
                run('train', 'main')
                target = output/'main'
        atomic(output/'LAUNCH_FINISHED.json', dict(final_root=str(target),
               collection_pending=True, engineering_success_NOT_recovery_claim=True))
        print('FINISHED. Independent PRIVATE collection (no training):', flush=True)
        print('{} -B -u {} {} {}'.format(exe, FOLDER/'collect.py', target,
              output.with_name(output.name+'_records.tar.gz')), flush=True)
    except BaseException as exc:
        atomic(output/'FAILED.json', dict(exception=type(exc).__name__, message=str(exc)))
        raise


if __name__ == '__main__':
    main()
