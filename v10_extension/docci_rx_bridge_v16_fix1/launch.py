"""Resume a completed smoke cache; do not repeat it or any existing training."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_rx_bridge_v16.core import PROJECT, FOLDER as ORIGINAL, read, atomic, safe, load
from docci_rx_bridge_v16_fix1.compat import FOLDER, source_check


def resumed_config(c):
    raw = dict(c)
    for key in ('train_samples', 'val_samples', 'alignment_epochs', 'alignment_batch',
                'ce_epochs', 'ce_accumulation', 'save_examples'):
        raw[key] = load()[key]
    if load(raw, smoke=True) != c:
        raise ValueError('Saved smoke configuration differs from declared contract')
    return raw


def main():
    source_check()
    if len(sys.argv) < 2:
        raise SystemExit('Use resume-smoke SOURCE_CACHE NEW_OUTPUT or check/smoke/cache/oracle/actual ...')
    # Original launcher tests and same GPU/budget settings; only its run/report
    # child entry points are replaced. Other child commands are untouched.
    if sys.argv[1] != 'resume-smoke':
        from docci_rx_bridge_v16 import launch
        old = subprocess.Popen
        def compatible(cmd, *args, **kwargs):
            cmd = list(cmd)
            for name in ('run', 'report'):
                target = str(ORIGINAL/(name+'.py'))
                if target in cmd:
                    at = cmd.index(target)
                    cmd[at:at+1] = [str(FOLDER/'entry.py'), name]
                    break
            return old(cmd, *args, **kwargs)
        subprocess.Popen = compatible
        try:
            launch.main()
        finally:
            subprocess.Popen = old
        return
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action', choices=('resume-smoke',))
    p.add_argument('source_cache')
    p.add_argument('output')
    p.add_argument('--max-hours', type=float, default=1)
    p.add_argument('--bart-path')
    p.add_argument('--text-checkpoint')
    a = p.parse_args()
    if not 0 < a.max_hours <= 2:
        p.error('Smoke only; finite0..2h budget required')
    cache = safe(a.source_cache).resolve(strict=True)
    identity = read(cache/'identity.json')
    if identity['action'] != 'cache' or identity['smoke'] is not True or not (cache/'FINISHED.json').is_file():
        raise ValueError('Require completed original SMOKE cache; do not relabel incomplete/full cache')
    if (cache.parent/'smoke_train').exists():
        raise FileExistsError('Original smoke_train exists: inspect it, do not duplicate training')
    raw = resumed_config(read(cache/'config.json'))
    output = safe(a.output)
    if cache == output or cache in output.parents:
        raise ValueError('New output must be outside preserved source cache')
    output.mkdir(parents=True, exist_ok=False)
    atomic(output/'launch_identity.json', dict(action='resume-smoke', original_cache=str(cache),
           cache_generation_repeated=False, original_failure_preserved=True, max_hours=a.max_hours,
           only_new_file_allowed_in_cache='VERIFIED.json derived CPU receipt', pid=os.getpid()))
    atomic(output/'smoke_input_config.json', raw)
    env = {k: v for k, v in os.environ.items() if k != 'SUDO_PW'}
    env.update(CUDA_VISIBLE_DEVICES='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONDONTWRITEBYTECODE='1', V16_REQUIRE_TORCH='1', TOKENIZERS_PARALLELISM='false',
               PYTHONPATH=os.pathsep.join([str(EXT), str(PROJECT)]))
    exe = os.environ.get('PYTHON_BIN') or sys.executable
    start = time.monotonic()
    def invoke(cmd, log):
        cap = a.max_hours*3600-(time.monotonic()-start)
        if cap <= 0:
            raise TimeoutError('Owned resume budget exhausted')
        child = None
        with log.open('x', encoding='utf-8') as stream:
            try:
                child = subprocess.Popen(cmd, cwd=str(PROJECT), env=env, stdout=stream,
                         stderr=subprocess.STDOUT, start_new_session=True)
                print('Owned PID={} log={}'.format(child.pid, log), flush=True)
                if child.wait(timeout=cap):
                    raise RuntimeError('Owned child failed; preserve logs; no automatic retry')
            except BaseException:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                raise
    try:
        invoke([exe, '-B', str(FOLDER/'test_fix.py')], output/'fix_contracts.log')
        invoke([exe, '-B', str(ORIGINAL/'launch.py'), 'check'], output/'original_contracts.log')
        # This is CPU-only verification, not cache creation or training.
        invoke([exe, '-B', '-u', str(FOLDER/'entry.py'), 'report', str(cache)], output/'cache_verify.log')
        remaining = a.max_hours-(time.monotonic()-start)/3600
        cmd = [exe, '-B', '-u', str(FOLDER/'entry.py'), 'run', 'smoke_train', str(output/'smoke_train'),
               '--smoke', '--cache', str(cache), '--config', str(output/'smoke_input_config.json'),
               '--max-hours', str(remaining)]
        for key in ('bart_path', 'text_checkpoint'):
            if getattr(a, key):
                cmd += ['--'+key.replace('_', '-'), getattr(a, key)]
        invoke(cmd, output/'smoke_train.log')
        invoke([exe, '-B', '-u', str(FOLDER/'entry.py'), 'report', str(output/'smoke_train')], output/'smoke_train_verify.log')
        atomic(output/'LAUNCH_FINISHED.json', dict(cache_reused=True, original_cache=str(cache),
               final_root=str(output/'smoke_train'), collection_pending=True, semantic_success=False))
        print('FIX1 SMOKE FINISHED; collect through fix1/entry.py collect, not original collector', flush=True)
    except BaseException as exc:
        atomic(output/'FAILED.json', dict(exception=type(exc).__name__, message=str(exc)))
        raise


if __name__ == '__main__':
    main()
