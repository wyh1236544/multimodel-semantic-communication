"""Finite, sequential, smoke-first GPU1 driver; diagnosis never starts training."""
import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_text_aid_v21.core import PROJECT,FOLDER,safe,atomic,source_check
from docci_text_aid_v21.run import options
from docci_text_guided_image_v20.launch import stop_owned

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action',choices=('check','diagnose','pilot'));p.add_argument('output',nargs='?')
    p.add_argument('--max-hours',type=float,default=4);options(p);a=p.parse_args()
    if (a.action=='check') != (a.output is None) or not math.isfinite(a.max_hours) or not .5<=a.max_hours<=8:
        p.error('NEW output except check, bounded 0.5..8h budget')
    if a.config or (a.action=='diagnose' and a.profile!='nominal'):
        p.error('Use only the registered V21 profiles; no missing25 in diagnosis')
    manifest=source_check()
    root=safe(a.output) if a.output else None
    if root and (root.exists() or root.with_name(root.name+'_launch.log').is_symlink()):
        raise FileExistsError('Existing output preserved; no automatic restart')
    env={k:v for k,v in os.environ.items() if k!='SUDO_PW'}
    env.update(CUDA_VISIBLE_DEVICES='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join((str(EXT),str(PROJECT))),
        V20_LPIPS_ASSETS=str(Path(a.lpips_assets).resolve()))
    exe=os.environ.get('PYTHON_BIN') or sys.executable
    deadline=time.monotonic()+a.max_hours*3600
    reserve=300
    def invoke(command,log=None,collect=False):
        remaining=deadline-time.monotonic()-(0 if collect else reserve)
        if remaining<=0:
            raise TimeoutError('Remaining wall time reserved for collection')
        handle=log.open('x',encoding='utf-8') if log else None
        child=None
        try:
            child=subprocess.Popen(command,cwd=str(PROJECT),env=env,stdout=handle,
                stderr=subprocess.STDOUT if handle else None,start_new_session=True)
            print('Owned PID={}{}'.format(child.pid,' log='+str(log) if log else ''),flush=True)
            if child.wait(timeout=remaining):
                raise RuntimeError('Owned V21 child failed; preserve logs, no automatic retry')
        finally:
            if child is not None and child.poll() is None:
                stop_owned(child)
            if handle:
                handle.close()
    invoke([exe,'-B',str(FOLDER/'test_core.py')])
    invoke([exe,'-B',str(FOLDER/'test_report.py')])
    if a.action=='check':
        print('CPU contracts passed. Real saved-weight/CUDA smoke NOT_RUN; no training.',flush=True)
        return
    if shutil.disk_usage(PROJECT).free<8*2**30:
        raise RuntimeError('Need8GiB free project disk; old cache reused read-only')
    root.mkdir(parents=True,exist_ok=False)
    for base,files,label in ((FOLDER,list(manifest['files'])+['MANIFEST.json'],'source_snapshot'),
                             (PROJECT,list(manifest['dependencies']),'dependency_snapshot')):
        for n in files:
            target=root/label/n;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(base/n,target)
    state=dict(status='RUNNING',action=a.action,profile=a.profile,max_hours=a.max_hours,
        stages=dict(CUDA_gate='NOT_RUN',neural_contracts='NOT_RUN',smoke='NOT_RUN',main='NOT_RUN'),
        automatic_retry=False,automatic_long_training=False)
    atomic(root/'PIPELINE.json',state)
    failure=None
    try:
        gate=root/'CUDA_GATE.json'
        commands=[('CUDA_gate',[exe,'-B',str(EXT/'docci_text_guided_image_v20_fix1/entry.py'),
                  'test','--receipt',str(gate)]),
                  ('neural_contracts',[exe,'-B',str(FOLDER/'test_neural.py'),'--require-cuda'])]
        for name,command in commands:
            state['stages'][name]='RUNNING';atomic(root/'PIPELINE.json',state)
            invoke(command,root/(name+'.log'))
            state['stages'][name]='VERIFIED';atomic(root/'PIPELINE.json',state)
        for label in ('smoke','main'):
            state['stages'][label]='RUNNING';atomic(root/'PIPELINE.json',state)
            command=[exe,'-B','-u',str(FOLDER/'run.py'),a.action,str(root/label),
                '--gate',str(gate),'--profile',a.profile,'--max-hours',str((deadline-time.monotonic()-reserve)/3600)]
            for name in ('v20_root','data_root','bart_path','vae_path','longclip_path',
                         'clip_tokenizer_path','lpips_assets','image_checkpoint','text_checkpoint',
                         'image_parent_metadata','text_parent_metadata'):
                command.extend(['--'+name.replace('_','-'),str(getattr(a,name))])
            if label=='smoke':
                command.append('--smoke')
            invoke(command,root/(label+'.log'))
            invoke([exe,'-B','-u',str(FOLDER/'report.py'),str(root/label),'--full'],root/(label+'_verify.log'))
            state['stages'][label]='VERIFIED';atomic(root/'PIPELINE.json',state)
        state['status']='COMPLETE'
    except BaseException as exc:
        failure=exc;state['status']='PARTIAL_BUDGET' if isinstance(exc,(TimeoutError,subprocess.TimeoutExpired)) else 'FAILED'
        state['exception']=dict(type=type(exc).__name__,message=str(exc))
    for name,status in state['stages'].items():
        if status=='RUNNING':
            state['stages'][name]='UNVERIFIED_PARTIAL'
    atomic(root/'PIPELINE.json',state)
    invoke([exe,'-B',str(FOLDER/'collect.py'),str(root),str(root.with_name(root.name+'_records.tar.gz'))],
           root/'collection.log',collect=True)
    print('V21 '+state['status']+'; send PRIVATE light records. Arrays/weights stay on server.',flush=True)
    if failure:
        raise RuntimeError('Partial/failed run preserved; do not automatically rerun') from failure

if __name__=='__main__':
    main()
