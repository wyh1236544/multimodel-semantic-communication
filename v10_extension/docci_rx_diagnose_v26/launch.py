"""Bounded smoke -> frozen diagnosis -> independent PRIVATE collection. No training."""
import argparse
import os
from pathlib import Path
import sys
import time
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_rx_diagnose_v26.core import (FOLDER,SCHEMA,sha,safe,atomic,source_check,Budget)

def main():
    from docci_rx_diagnose_v26.run import parser as worker_parser
    from docci_caption_calibration_v24.launch import invoke
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action',choices=('inventory','pipeline'));p.add_argument('output')
    p.add_argument('--parent');p.add_argument('--bart-path');p.add_argument('--max-hours',type=float,default=2)
    a=p.parse_args();source_check();Budget(a.max_hours)
    extra=[]
    for flag in ('parent','bart_path'):
        if getattr(a,flag):extra+=['--'+flag.replace('_','-'),getattr(a,flag)]
    parsed=worker_parser().parse_args(['inventory',a.output]+extra)
    root=safe(a.output);parent=Path(parsed.parent).resolve()
    if root==parent or parent in root.parents or root in parent.parents:raise ValueError('Output cannot overlap parent')
    archive=root.with_name(root.name+'_records.tar.gz')
    if any(p.exists() for p in (root,archive,archive.with_name(archive.name+'.sha256'),archive.with_name(archive.name+'.contents'))):
        raise FileExistsError('Existing run/archive preserved; use a NEW name')
    root.mkdir(parents=True)
    env=dict(os.environ);env.pop('SUDO_PW',None)
    env.update(CUDA_VISIBLE_DEVICES='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    start=time.monotonic();deadline=start+a.max_hours*3600;pids=[];status='FAILED'
    def child(label,cmd,reserve=0):
        pids.append(invoke(cmd,root/(label+'.log'),env,deadline-reserve))
    def worker(stage):
        cmd=[sys.executable,'-B','-u',str(FOLDER/'run.py'),stage,str(root/stage),
            '--max-hours',str(a.max_hours)]+extra
        if stage in ('smoke','diagnose'):cmd+=['--inputs',str(root/'inputs')]
        child(stage,cmd,120)
    try:
        child('contracts',[sys.executable,'-B',str(FOLDER/'test_core.py')])
        if a.action=='inventory':worker('inventory');status='INVENTORY_COMPLETE'
        else:
            child('hf_contracts',[sys.executable,'-B',str(FOLDER/'test_hf.py')])
            child('inputs',[sys.executable,'-B','-u',str(FOLDER/'run.py'),'preflight',str(root/'inputs'),
                '--max-hours',str(a.max_hours)]+extra,120)
            for stage in ('smoke','diagnose'):
                worker(stage)
                child(stage+'_verify',[sys.executable,'-B',str(FOLDER/'report.py'),str(root/stage),'--inputs',str(root/'inputs')])
            status='COMPLETE'
    finally:
        atomic(root/'PIPELINE.json',dict(schema=SCHEMA,status=status,source_manifest_sha256=sha(FOLDER/'MANIFEST.json'),
            elapsed_seconds=time.monotonic()-start,owned_PIDs=pids,training_updates=0,automatic_training=False,
            semantic_success=False,facts='PENDING'))
    if status=='COMPLETE':
        child('final_verify',[sys.executable,'-B',str(FOLDER/'report.py'),str(root),'--pipeline'])
        child('collect',[sys.executable,'-B',str(FOLDER/'collect.py'),str(root),str(archive)])
        if not archive.is_file() or archive.with_name(archive.name+'.sha256').read_text().split()[0]!=sha(archive):
            raise ValueError('Actual records archive missing or invalid')
        atomic(root/'LAUNCH_FINISHED.json',dict(status='VERIFIED',archive=str(archive),
            elapsed_seconds=time.monotonic()-start,training_updates=0,semantic_success=False,facts='PENDING'))
    print('V26 '+status+'; no training or automatic expansion; keep V25 cache/weights on server',flush=True)
if __name__=='__main__':main()
