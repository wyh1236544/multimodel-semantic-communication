"""Never starts V14, never automatically promotes oracle→actual or submits test."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_rx_bridge_v16.core import FOLDER, PROJECT, atomic, safe, check_source, load


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action', choices=('check', 'smoke', 'cache', 'oracle', 'actual'))
    p.add_argument('output', nargs='?')
    p.add_argument('--cache')
    p.add_argument('--oracle')
    p.add_argument('--config')
    p.add_argument('--max-hours', type=float, default=8)
    for key, default in dict(data_root=PROJECT, bart_path=PROJECT/'pretrained/bart-base',
                            vae_path=PROJECT/'pretrained/sd-vae-ft-mse', longclip_path=PROJECT/'longclip-B.pt',
                            text_checkpoint=None, image_checkpoint=None).items():
        p.add_argument('--'+key.replace('_', '-'), default=str(default) if default is not None else None)
    a = p.parse_args()
    if (a.action == 'check') != (a.output is None) or not 0 < a.max_hours <= 24:
        p.error('check takes no output, all other actions require NEW output and0..24h cap')
    if a.action in ('oracle', 'actual') and not a.cache or a.action == 'actual' and not a.oracle:
        p.error('oracle requires --cache; actual additionally requires --oracle')
    load(a.config)
    check_source()
    exe = os.environ.get('PYTHON_BIN') or sys.executable
    if not Path(exe).is_file():
        raise FileNotFoundError('Use existing server interpreter; no dependency installation')
    env = {k: v for k, v in os.environ.items() if k != 'SUDO_PW'}
    env.update(CUDA_VISIBLE_DEVICES='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1', V16_REQUIRE_TORCH='1',
               PYTHONPATH=os.pathsep.join([str(EXT), str(PROJECT)]))
    start = time.monotonic()
    def invoke(cmd, log=None):
        cap = a.max_hours*3600-(time.monotonic()-start)
        if cap <= 0:
            raise TimeoutError('Overall budget exhausted')
        stream = log.open('x', encoding='utf-8') if log else None
        child = None
        try:
            child = subprocess.Popen(cmd, cwd=str(PROJECT), env=env, stdout=stream,
                    stderr=subprocess.STDOUT if stream else None, start_new_session=True)
            print('Owned PID={}{}'.format(child.pid, ' log='+str(log) if log else ''), flush=True)
            code = child.wait(timeout=cap)
            if code:
                raise RuntimeError('Owned child exit {}; preserve logs, no automatic retry'.format(code))
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
    for name in ('test_core.py', 'test_report.py', 'test_integration.py', 'test_model.py'):
        invoke([exe, '-B', str(FOLDER/name)])
    print('CPU contracts passed. Real pinned checkpoints/backbones/GPU still require smoke.', flush=True)
    if a.action == 'check':
        return
    output = safe(a.output)
    output.mkdir(parents=True, exist_ok=False)
    atomic(output/'launch_identity.json', dict(action=a.action, pid=os.getpid(), max_hours=a.max_hours,
           no_existing_job_restarted=True, no_V14_training=True, no_automatic_actual_promotion=True))
    def run(action, label, smoke=False, cache=None, smoke_root=None):
        cmd = [exe, '-B', '-u', str(FOLDER/'run.py'), action, str(output/label),
               '--max-hours', str(a.max_hours-(time.monotonic()-start)/3600)]
        if smoke:
            cmd += ['--smoke']
        for key, value in dict(cache=cache, smoke_root=smoke_root, oracle=a.oracle, config=a.config,
                              data_root=a.data_root, bart_path=a.bart_path, vae_path=a.vae_path,
                              longclip_path=a.longclip_path, text_checkpoint=a.text_checkpoint,
                              image_checkpoint=a.image_checkpoint).items():
            if value:
                cmd += ['--'+key.replace('_', '-'), str(Path(value).resolve())]
        invoke(cmd, output/(label+'.log'))
        invoke([exe, '-B', '-u', str(FOLDER/'report.py'), str(output/label)], output/(label+'_verify.log'))
    try:
        if a.action in ('smoke', 'cache'):
            run('cache', 'smoke_cache', True)
            run('smoke_train', 'smoke_train', True, output/'smoke_cache')
            if a.action == 'cache':
                run('cache', 'main', False, smoke_root=output/'smoke_train')
        else:
            run(a.action, 'main', cache=a.cache)
        atomic(output/'LAUNCH_FINISHED.json', dict(action=a.action, collection_pending=True,
               quality_success=False, next_step_requires_user_start=True))
        print('Launch finished. Keep cache/weights on server; collect PRIVATE light records separately.', flush=True)
    except BaseException as exc:
        atomic(output/'FAILED.json', dict(exception=type(exc).__name__, message=str(exc)))
        raise


if __name__ == '__main__':
    main()
