"""Config-driven CPU digital / frozen GPU comparison; no training command."""
import argparse
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
EXT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_digital_compare_v19.common import FOLDER,PROJECT,safe,load,read,atomic,progress,snapshot,check_source,sha

def stop_owned(child):
    if child.poll() is not None:return
    if os.getpgid(child.pid)!=child.pid:raise RuntimeError('Unexpected child ownership')
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if child.poll() is None and os.getpgid(child.pid)==child.pid:os.killpg(child.pid,signal.SIGKILL)
        child.wait()

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action',choices=('check','digital','pipeline','negative'));p.add_argument('output',nargs='?')
    p.add_argument('--v18-root',default=str(PROJECT/'experiment_results/v18_text_noise_01'))
    p.add_argument('--digital-root');p.add_argument('--bart-path',default=str(PROJECT/'pretrained/bart-base'))
    p.add_argument('--max-hours',type=float,default=4);a=p.parse_args()
    if (a.action=='check')!=(a.output is None) or not math.isfinite(a.max_hours) or not .5<=a.max_hours<=8:p.error('NEW output required except check;0.5..8h cap')
    if (a.action=='negative')!=(a.digital_root is not None):p.error('--digital-root is required only for negative; it reuses an existing digital run')
    check_source();c=load();root=safe(a.output) if a.output else None
    if root and root.exists():raise FileExistsError('Existing output preserved')
    env={k:v for k,v in os.environ.items() if k!='SUDO_PW'}
    env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
               PYTHONPATH=os.pathsep.join((str(EXT),str(PROJECT))),CUDA_VISIBLE_DEVICES='1' if a.action in ('pipeline','negative') else '',
               CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
    exe=os.environ.get('PYTHON_BIN') or sys.executable;deadline=time.monotonic()+a.max_hours*3600
    def stage_budget():
        if time.monotonic()>=deadline-300:raise TimeoutError('Global collection reserve reached')
    def invoke(cmd,log=None,collect=False):
        remaining=deadline-time.monotonic()-(0 if collect else 300)
        if remaining<=0:raise TimeoutError('Collection reserve reached')
        child=None;f=log.open('x',encoding='utf-8') if log else None
        try:
            child=subprocess.Popen(cmd,cwd=str(PROJECT),env=env,stdout=f,stderr=subprocess.STDOUT if f else None,start_new_session=True)
            print('Owned PID={}{}'.format(child.pid,' log='+str(log) if log else ''),flush=True)
            if child.wait(timeout=remaining):raise RuntimeError('Owned child failed; preserve logs; no automatic retry')
        finally:
            if child is not None and child.poll() is None:stop_owned(child)
            if f:f.close()
    for test in ('test_core.py','test_storage.py'):invoke([exe,'-B',str(FOLDER/test)])
    if a.action=='check':
        print('CPU digital contracts passed. Actual saved V18/BART/GPU gates remain. No training/install.',flush=True);return
    if a.action in ('pipeline','negative'):
        env['V18_REQUIRE_TORCH']='1';invoke([exe,'-B',str(EXT/'docci_text_channel_v18/test_torch.py')])
    if shutil.disk_usage(PROJECT).free<4*2**30:raise RuntimeError('Need4GiB start disk reserve')
    snapshot(root);atomic(root/'config.json',c)
    labels=['prepare','smoke_digital','digital']
    if a.action!='digital':labels=['prepare','smoke_digital','smoke_negative','digital','negative']
    state=dict(action=a.action,status='RUNNING',stages={n:'NOT_RUN' for n in labels},
               max_hours=a.max_hours,no_training=True,physical_GPU=None if a.action=='digital' else 1)
    progress(root/'PIPELINE.json',state);failure=None
    try:
        if a.action=='negative':
            from docci_digital_compare_v19.report import verify
            from docci_digital_compare_v19.inputs import inherited
            old=safe(a.digital_root);r=verify(old)
            if r['status']!='DIGITAL_COMPLETE' or read(old/'config.json')!=c:raise ValueError('Verified matching DIGITAL_COMPLETE root required')
            inherited(old/'prepared',root/'prepared')
            for n in ('smoke_digital','digital'):shutil.copytree(old/n,root/n);state['stages'][n]='REUSED'
            atomic(root/'reuse.json',dict(digital_root=str(old),report_sha256=sha(old/'REPORT.json'),no_digital_rerun=True))
            state['stages']['prepare']='COMPLETE';progress(root/'PIPELINE.json',state)
        for label in labels:
            if state['stages'][label] in ('COMPLETE','REUSED'):continue
            state['stages'][label]='RUNNING';progress(root/'PIPELINE.json',state)
            action='prepare' if label=='prepare' else 'negative' if label.endswith('negative') else 'digital'
            stage=root/'prepared' if action=='prepare' else root/label
            hours=(deadline-time.monotonic()-300)/3600
            cmd=[exe,'-B','-u',str(FOLDER/'run.py'),action,str(stage),'--max-hours',str(hours)]
            if action=='prepare':cmd+=['--v18-root',a.v18_root]
            else:cmd+=['--prepared',str(root/'prepared')]
            if label.startswith('smoke_'):cmd+=['--smoke']
            if action=='negative':cmd+=['--bart-path',a.bart_path]
            invoke(cmd,root/(label+'.log'))
            if action!='prepare':
                from docci_digital_compare_v19.report import stage as verify_stage
                # No neural rerun. Full actual digital receiver replay occurs once at collection.
                verify_stage(stage,root/'prepared',c,stage_budget)
            state['stages'][label]='COMPLETE';progress(root/'PIPELINE.json',state)
        from docci_digital_compare_v19.inputs import historical_unchanged
        historical_unchanged(root/'prepared')
        state['status']='DIGITAL_COMPLETE' if a.action=='digital' else 'COMPLETE'
    except (TimeoutError,subprocess.TimeoutExpired) as e:failure=e;state['status']='PARTIAL_BUDGET'
    except BaseException as e:failure=e;state['status']='FAILED'
    for n,v in state['stages'].items():
        if v=='RUNNING':state['stages'][n]='UNVERIFIED_PARTIAL'
    if failure:state['exception']=dict(type=type(failure).__name__,message=str(failure))
    progress(root/'PIPELINE.json',state)
    invoke([exe,'-B','-u',str(FOLDER/'collect.py'),str(root),str(root.with_name(root.name+'_records.tar.gz')),
            '--max-minutes','30','--replay-digital'],root/'collection.log',True)
    print('V19 '+state['status']+'; PRIVATE records collected; no training or follow-up job started.',flush=True)
    if failure:raise RuntimeError('Partial/failed pipeline preserved; no automatic retry')
if __name__=='__main__':main()
