"""Stable single pipeline: shared cache audit, real smoke, paired codec adaptation."""
import argparse
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_text_channel_v18.core import FOLDER,PROJECT,ARMS,load,read,atomic,safe,snapshot,check_source

def stop_owned(child):
    if child.poll() is not None:return
    group=os.getpgid(child.pid)
    if group!=child.pid: raise RuntimeError('Refuse unexpected process-group ownership')
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if child.poll() is None and os.getpgid(child.pid)==child.pid:os.killpg(child.pid,signal.SIGKILL)
        child.wait()

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action',choices=('check','smoke','pipeline'));p.add_argument('output',nargs='?')
    p.add_argument('--features',default=str(PROJECT/'experiment_results/v14_text_pilot_01/main/text'))
    p.add_argument('--bart-path',default=str(PROJECT/'pretrained/bart-base'))
    p.add_argument('--config');p.add_argument('--max-hours',type=float,default=8)
    a=p.parse_args()
    if (a.action=='check')!=(a.output is None) or not math.isfinite(a.max_hours) or not .5<=a.max_hours<=8:p.error('NEW output required except check;0.5..8h hardcap')
    check_source();c=load(a.config)
    root=safe(a.output) if a.output else None
    # A current nohup launch log exists before Python starts; only the run root
    # can be checked here. The documented shell command guards the log itself.
    if root and root.exists(): raise FileExistsError('Existing output preserved')
    env={k:v for k,v in os.environ.items() if k!='SUDO_PW'}
    env.update(CUDA_VISIBLE_DEVICES='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',V18_REQUIRE_TORCH='1',PYTHONPATH=os.pathsep.join((str(EXT),str(PROJECT))))
    exe=os.environ.get('PYTHON_BIN') or sys.executable
    deadline=time.monotonic()+a.max_hours*3600;reserve=600 if a.action=='pipeline' else 180
    def invoke(cmd,log=None,collection=False):
        available=deadline-time.monotonic()-(0 if collection else reserve)
        if available<=0:raise TimeoutError('Global time budget reserved for collection')
        f=log.open('x',encoding='utf-8') if log else None
        child=None
        try:
            child=subprocess.Popen(cmd,cwd=str(PROJECT),env=env,stdout=f,stderr=subprocess.STDOUT if f else None,start_new_session=True)
            print('Owned PID={}{}'.format(child.pid,' log='+str(log) if log else ''),flush=True)
            if child.wait(timeout=available):raise RuntimeError('Owned stage exit nonzero; preserve logs; no automatic retry')
        finally:
            if child is not None and child.poll() is None:stop_owned(child)
            if f:f.close()
    for test in ('test_core.py','test_torch.py'):invoke([exe,'-B',str(FOLDER/test)])
    print('Contracts passed; actual saved-weight/GPU smoke still required. No install/download.',flush=True)
    if a.action=='check':return
    if shutil.disk_usage(PROJECT).free<c['minimum_disk_gib']*2**30:raise RuntimeError('Insufficient starting project disk')
    snapshot(root);atomic(root/'config.json',c)
    labels=[]
    for prefix in (['smoke_'] if a.action=='smoke' else ['smoke_','']):
        labels+=[prefix+'inputs']+[prefix+'train_'+arm for arm in ARMS]+[prefix+'eval_'+arm for arm in ('frozen',)+ARMS]
    state=dict(status='RUNNING',stages={label:'NOT_RUN' for label in labels},max_hours=a.max_hours,
        no_old_cache_regeneration=True,no_other_job_control=True,digital='NOT_RUN',FSO='NOT_RUN',image='NOT_RUN')
    atomic(root/'PIPELINE.json',state);failure=None
    try:
        for prefix in (['smoke_'] if a.action=='smoke' else ['smoke_','']):
            smoke=bool(prefix)
            for label in [x for x in labels if x.startswith(prefix) and (prefix or not x.startswith('smoke_'))]:
                state['stages'][label]='RUNNING';atomic(root/'PIPELINE.json',state)
                action='audit' if label==prefix+'inputs' else 'train' if label.startswith(prefix+'train_') else 'evaluate'
                remaining=(deadline-time.monotonic()-reserve)/3600
                cmd=[exe,'-B','-u',str(FOLDER/'run.py'),action,str(root/label),'--features',str(Path(a.features).resolve()),
                    '--bart-path',a.bart_path,'--config',str(root/'config.json'),'--max-hours',str(remaining)]
                if smoke:cmd+=['--smoke']
                if action!='audit':
                    arm=label[len(prefix)+len('train_' if action=='train' else 'eval_'):]
                    cmd+=['--inputs',str(root/(prefix+'inputs')),'--arm',arm]
                    if action=='evaluate' and arm!='frozen':cmd+=['--checkpoint',str(root/(prefix+'train_'+arm)/'last.pth')]
                invoke(cmd,root/(label+'.log'))
                # Inputs were array-checked by the producer. Recheck all input arrays
                # independently once at final collection; here validate the sealed metadata.
                report=[exe,'-B','-u',str(FOLDER/'report.py'),str(root),'--stage',label]
                if action=='train':report+=['--full']
                invoke(report,root/(label+'_verify.log'))
                state['stages'][label]='VERIFIED';atomic(root/'PIPELINE.json',state)
        state['status']='SMOKE_COMPLETE' if a.action=='smoke' else 'COMPLETE'
    except (TimeoutError,subprocess.TimeoutExpired) as exc:state['status']='PARTIAL_BUDGET';failure=exc
    except BaseException as exc:state['status']='FAILED';failure=exc
    for k,v in state['stages'].items():
        if v=='RUNNING':state['stages'][k]='UNVERIFIED_PARTIAL'
    if failure:state['exception']=dict(type=type(failure).__name__,message=str(failure))
    atomic(root/'PIPELINE.json',state)
    try:
        invoke([exe,'-B','-u',str(FOLDER/'collect.py'),str(root),str(root.with_name(root.name+'_records.tar.gz'))],root/'collection.log',True)
        print('V18 '+state['status']+'; send PRIVATE records archive and SHA256. No new job started automatically.',flush=True)
    except BaseException as exc:
        atomic(root/'COLLECTION_FAILED.json',dict(type=type(exc).__name__,message=str(exc)));raise
    if failure:raise RuntimeError('Partial/failed pipeline preserved; no auto restart')
if __name__=='__main__':main()
