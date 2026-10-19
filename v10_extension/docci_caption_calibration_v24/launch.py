"""One owned GPU stage, hard time budget, independent CPU verify/collection."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_caption_calibration_v24.core import FOLDER, Budget, atomic, safe, source_check
from docci_caption_calibration_v24.run import parser

def invoke(command, log, env, deadline):
    if time.monotonic() >= deadline: raise TimeoutError('No remaining budget')
    with log.open('x', encoding='utf-8') as f:
        child = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT,
            env=env, start_new_session=True)
        print('Owned PID={} log={}'.format(child.pid, log), flush=True)
        try:
            while child.poll() is None:
                if time.monotonic() >= deadline: raise TimeoutError('Owned child exceeded budget; no restart')
                time.sleep(.5)
            if child.returncode: raise RuntimeError('Owned child exit {}; preserve {}'.format(child.returncode, log))
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
    return child.pid

def main():
    a = parser().parse_args(); source_check(); Budget(a.max_minutes)
    root = safe(a.output)
    logs = [root.with_name(root.name+s) for s in ('.contracts.log', '.hf_contracts.log', '.log', '.collect.log')]
    archive = root.with_name(root.name+'_records.tar.gz')
    if any(p.exists() for p in [root, archive, archive.with_name(archive.name+'.contents')]+logs):
        raise FileExistsError('Existing output/log/collection preserved; choose a NEW run name')
    root.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ); env.pop('SUDO_PW', None)
    env.update(CUDA_VISIBLE_DEVICES='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    start = time.monotonic(); deadline = start+a.max_minutes*60; status = 'FAILED'; pids = []
    try:
        pids.append(invoke([sys.executable, '-B', str(FOLDER/'test_core.py')], logs[0], env, deadline))
        # This is an actual tiny random HF CPU model, not a downloaded/full BLIP-2.
        pids.append(invoke([sys.executable, '-B', str(FOLDER/'test_hf.py')], logs[1], env, deadline))
        reserve = min(180, a.max_minutes*60*.15)
        pids.append(invoke([sys.executable, '-B', '-u', str(FOLDER/'run.py')]+sys.argv[1:], logs[2], env, deadline-reserve))
        pids.append(invoke([sys.executable, '-B', '-u', str(FOLDER/'collect.py'), str(root), str(archive)], logs[3], env, deadline))
        status = 'VERIFIED'
    finally:
        if root.is_dir():
            atomic(root/'LAUNCH.json', dict(status=status, owned_PIDs=pids,
                elapsed_seconds=time.monotonic()-start, training_updates=0, semantic_success=False,
                facts='PENDING', automatic_next_stage=False, archive=str(archive) if status == 'VERIFIED' else None))
    print('V24 '+status+'; send PRIVATE records, not weights. Human review before QA.', flush=True)

if __name__ == '__main__': main()
