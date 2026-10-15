"""Checks -> discarded authentic smoke -> fresh main process; finite owned caps."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
from docci_semantic_probe_v12.core import FOLDER,PROJECT,SCHEMA,load,output_path,atomic_json,digest,parent,check_sources


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action',choices=('check','smoke','run')); p.add_argument('output',nargs='?')
    p.add_argument('--task',choices=('text','image','both'),default='both'); p.add_argument('--config')
    p.add_argument('--text-parent'); p.add_argument('--image-parent'); p.add_argument('--max-hours',type=float,default=6)
    for name,path in dict(data_root=PROJECT,bart_path=PROJECT/'pretrained/bart-base',vae_path=PROJECT/'pretrained/sd-vae-ft-mse',
        longclip_path=PROJECT/'longclip-B.pt',clip_tokenizer_path=PROJECT/'pretrained/clip-vit-base-patch32').items():
        p.add_argument('--'+name.replace('_','-'),default=str(path))
    a=p.parse_args(); c=load(a.config)
    if (a.action=='check')!=(a.output is None) or not 0<a.max_hours<=24: p.error('check takes no output; smoke/run require NEW output and0..24h cap')
    exe=os.environ.get('PYTHON_BIN') or sys.executable
    if not Path(exe).is_file(): raise FileNotFoundError('Existing interpreter missing; do not install dependencies')
    env={k:v for k,v in os.environ.items() if k!='SUDO_PW'}
    env.update(CUDA_VISIBLE_DEVICES='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        PYTHONDONTWRITEBYTECODE='1',V12_REQUIRE_TORCH='1',PYTHONPATH=os.pathsep.join([str(EXT),str(PROJECT)]))
    start=time.monotonic()
    def invoke(cmd,log=None):
        cap=a.max_hours*3600-(time.monotonic()-start)
        if cap<=0: raise TimeoutError('Total launcher cap exhausted')
        stream=log.open('x') if log else None; child=None
        try:
            child=subprocess.Popen(cmd,cwd=str(PROJECT),env=env,stdout=stream,stderr=subprocess.STDOUT if stream else None)
            print('Owned PID={}{}'.format(child.pid,' log='+str(log) if log else ''),flush=True)
            code=child.wait(timeout=cap)
            if code: raise RuntimeError('Owned child exit {}; logs preserved; no automatic retry'.format(code))
        except BaseException:
            if child is not None and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            raise
        finally:
            if stream: stream.close()
    check_sources()
    for name in ('test_core.py','test_report.py','test_model.py'): invoke([exe,'-B',str(FOLDER/name)])
    print('CPU contracts passed; authentic saved-parent/GPU gates still required. No dependency changes.',flush=True)
    if a.action=='check': return
    output=output_path(a.output); tasks=['text','image'] if a.task=='both' else [a.task]
    if output.exists(): raise FileExistsError('Existing output preserved; choose NEW attempt')
    for task in tasks: parent(task,getattr(a,task+'_parent'))
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'launch_identity.json',dict(schema=SCHEMA,action=a.action,tasks=tasks,config=c,
        gpu_physical=1,max_hours=a.max_hours,smoke_weights_NEVER_reused=True,collection_is_separate_CPU_step=True))
    def run(mode,name):
        command=[exe,'-B','-u',str(FOLDER/'run.py'),'--mode',mode,'--task',a.task,'--output',str(output/name),
            '--max-hours',str(max(.001,(a.max_hours*3600-(time.monotonic()-start)-15)/3600))]
        for key in ('config','text_parent','image_parent','data_root','bart_path','vae_path','longclip_path','clip_tokenizer_path'):
            value=getattr(a,key)
            if value: command+=['--'+key.replace('_','-'),str(Path(value).resolve())]
        invoke(command,output/(name+'.log'))
    try:
        run('smoke','smoke')
        # Independent smoke record verification is deliberately before main training.
        # Verifier gets a smoke-only view without forging a completed main run.
        invoke([exe,'-B',str(FOLDER/'smoke_check.py'),str(output)])
        if a.action=='run': run('train','main')
        atomic_json(output/'LAUNCH_FINISHED.json',dict(launch_identity_sha256=digest(output/'launch_identity.json'),
            smoke_only=a.action=='smoke',independent_final_verification_pending=True))
        print('FINISHED. Collect independently (no GPU retraining):',flush=True)
        print('{} -B -u {} {} {}'.format(exe,FOLDER/'collect.py',output,output.with_name(output.name+'_records.tar.gz')),flush=True)
    except BaseException as exc:
        atomic_json(output/'FAILED.json',dict(exception=type(exc).__name__,message=str(exc))); raise


if __name__=='__main__': main()
