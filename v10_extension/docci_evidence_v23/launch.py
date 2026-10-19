"""Owned stage plus independent CPU verifier share one explicit hard wall budget."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_evidence_v23.core import FOLDER,safe,atomic,source_check,Budget
from docci_evidence_v23.run import parser

def invoke(command,log,env,deadline,owners):
    if deadline<=time.monotonic():raise TimeoutError('No remaining stage budget')
    with log.open('x',encoding='utf-8') as f:
        child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        owners.append(child.pid)
        print('Owned PID={} log={}'.format(child.pid,log),flush=True)
        try:
            while child.poll() is None:
                if time.monotonic()>=deadline:raise TimeoutError('Owned stage timed out; no restart')
                time.sleep(.5)
            if child.returncode:raise RuntimeError('Owned child exit {}; preserve log'.format(child.returncode))
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
    return child.pid

def main():
    a=parser().parse_args();source_check();Budget(a.max_hours)
    root=safe(a.output);log=root.with_name(root.name+'.log');vlog=root.with_name(root.name+'.verify.log')
    if any(p.exists() for p in (root,log,vlog)):raise FileExistsError('Existing output/log preserved; choose NEW name')
    log.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ);env.pop('SUDO_PW',None)
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    start=time.monotonic();deadline=start+a.max_hours*3600;status='FAILED';pids=[]
    try:
        invoke([sys.executable,'-B','-u',str(FOLDER/'run.py')]+sys.argv[1:],log,env,deadline,pids)
        command=[sys.executable,'-B','-u',str(FOLDER/'report.py'),'verify',str(root),'--max-hours',str(max(.001,(deadline-time.monotonic())/3600))]
        if a.prepared:command+=['--prepared',a.prepared]
        if a.target:command+=['--target',a.target]
        invoke(command,vlog,env,deadline,pids);status='VERIFIED'
    finally:
        if root.is_dir():atomic(root/'LAUNCH.json',dict(status=status,owned_PIDs=pids,elapsed_seconds=time.monotonic()-start,
            automatic_next_stage=False,no_training=True))
    print('V23 '+status+'; no automatic next stage. Keep arrays/weights on server.',flush=True)

if __name__=='__main__':main()
