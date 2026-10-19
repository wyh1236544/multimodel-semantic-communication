"""Hard stage timeout, explicit owned process group, never resumes another stage."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_pretrained_checks_v22.core import FOLDER, safe, source_check, atomic, read, Budget
from docci_pretrained_checks_v22.run import parser

def main():
    a = parser().parse_args()
    source_check(); Budget(a.max_hours)
    root = safe(a.output)
    log = root.with_name(root.name+'.log')
    if root.exists() or log.exists():
        raise FileExistsError('Existing result/log preserved; do not repeat the job')
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop('SUDO_PW', None)
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    start = time.monotonic()
    code, timed_out = None, False
    with log.open('x', encoding='utf-8') as f:
        child = subprocess.Popen([sys.executable, '-B', '-u', str(FOLDER/'run.py')]+sys.argv[1:],
            stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        print('Owned PID={} log={}'.format(child.pid, log), flush=True)
        try:
            while child.poll() is None:
                if time.monotonic()-start > a.max_hours*3600:
                    timed_out = True
                    raise TimeoutError('Owned V22 stage timed out')
                time.sleep(1)
            code = child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            if root.is_dir():
                atomic(root/'LAUNCH.json', dict(pid=child.pid, pgid=child.pid,
                    exit_code=child.returncode, timed_out=timed_out,
                    elapsed_seconds=time.monotonic()-start, no_auto_next_stage=True))
    if code:
        raise RuntimeError('Owned child failed; preserve log; no automatic retry')
    from docci_pretrained_checks_v22.report import verify
    result = verify(root, a.prepared)
    print('V22 stage independently verified: '+result['stage'], flush=True)

if __name__=='__main__':
    main()
