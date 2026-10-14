"""Physical GPU1, offline, mandatory authentic smoke, sequential independent arms."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
EXT=Path(__file__).resolve().parents[1]; PROJECT=EXT.parent
sys.path.insert(0,str(EXT)); sys.path.insert(0,str(PROJECT))
from docci_reconstruction_v10.core import load,safe_write,atomic_json,initialization,TEXT_ARMS,IMAGE_ARMS


def main():
    p=argparse.ArgumentParser(allow_abbrev=False,description=__doc__)
    p.add_argument('action',choices=('check','data-check','rate-info','smoke','diagnostic','screen','pipeline'))
    p.add_argument('output',nargs='?'); p.add_argument('--branch',choices=('both','text','image'),default='both')
    p.add_argument('--arms',nargs='+',choices=TEXT_ARMS+IMAGE_ARMS)
    p.add_argument('--max-hours',type=float,default=4); p.add_argument('--experiment-config',default=str(Path(__file__).with_name('config.json')))
    p.add_argument('--image-width',type=int,choices=(8,16,32),default=8); p.add_argument('--text-width',type=int,choices=(16,32,64),default=64)
    p.add_argument('--text-init',choices=('history','fresh'),default='history'); p.add_argument('--image-init',choices=('history','fresh'),default='history')
    defaults=dict(data_root=PROJECT,bart_path=PROJECT/'pretrained/bart-base',vae_path=PROJECT/'pretrained/sd-vae-ft-mse')
    for k,v in defaults.items(): p.add_argument('--'+k.replace('_','-'),default=str(v))
    a=p.parse_args(); c=load(a.experiment_config,512,a.image_width,a.text_width)
    checks=a.action in ('check','data-check','rate-info')
    if checks!=(a.output is None): p.error('Read-only checks take no output; execution takes a NEW output')
    if not math.isfinite(a.max_hours) or not 0<a.max_hours<=24: p.error('Finite cap0..24h required')
    eligible=list(TEXT_ARMS+IMAGE_ARMS if a.branch=='both' else TEXT_ARMS if a.branch=='text' else IMAGE_ARMS)
    arms=a.arms or eligible
    if len(set(arms))!=len(arms) or not set(arms)<=set(eligible): p.error('Arm selection must match requested branch, without duplicates')
    if a.action=='rate-info':
        from docci_reconstruction_v10.channel import resource
        print(json.dumps({arm:resource(c,arm) for arm in arms},indent=2)); return
    if a.action=='data-check':
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        from docci_reconstruction_v10.data import audit
        for branch in sorted({'text' if arm in TEXT_ARMS else 'image' for arm in arms}):
            result=audit(a,c,branch); print(json.dumps(result,ensure_ascii=False,indent=2))
            key='oversized' if branch=='text' else 'missing_count'
            if any(v[key] for v in result['splits'].values()): raise ValueError('Full original source audit failed')
        return
    output=safe_write(a.output) if a.output else None
    if output:
        if output.relative_to(PROJECT).parts[0]!='experiment_results' or not output.name.startswith('v10_'): raise ValueError('Use project experiment_results/v10_*')
        if output.exists() or output.with_name(output.name+'.log').exists(): raise FileExistsError('Existing output/log preserved; NEW attempt02/03 required')
    exe=os.environ.get('PYTHON_BIN') or sys.executable
    if not Path(exe).is_file(): raise FileNotFoundError('Existing interpreter missing; no install')
    env={k:v for k,v in os.environ.items() if k!='SUDO_PW'}
    env.update(CUDA_VISIBLE_DEVICES='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        PYTHONDONTWRITEBYTECODE='1',V10_REQUIRE_TORCH='1',PYTHONPATH=os.pathsep.join([str(EXT),str(PROJECT)]),
        HF_HOME=str(PROJECT/'.cache/v10/huggingface'),TORCH_HOME=str(PROJECT/'.cache/v10/torch'))
    folder=Path(__file__).resolve().parent; start=time.monotonic()
    def invoke(cmd,log=None,limit=None):
        cap=a.max_hours*3600-(time.monotonic()-start)
        if limit is not None: cap=min(cap,limit)
        if cap<=0: raise TimeoutError('Overall cap reached; preserve all evidence')
        stream=log.open('x',encoding='utf-8') if log else None; child=None
        try:
            child=subprocess.Popen(cmd,cwd=str(PROJECT),env=env,stdout=stream,stderr=subprocess.STDOUT if stream else None)
            print('Owned PID={}{}'.format(child.pid,' log='+str(log) if log else ''),flush=True)
            code=child.wait(timeout=cap)
            if code: raise RuntimeError('Engineering child exit {}; preserve log. No auto retry; quality-stop exits0.'.format(code))
        except BaseException:
            if child is not None and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            raise
        finally:
            if stream: stream.close()
    invoke([exe,'-B',str(folder/'install.py'),'--project-root',str(PROJECT),'--verify-only'],limit=180)
    for test in ('test_core.py','test_report.py','test_model.py'): invoke([exe,'-B',str(folder/test)],limit=600)
    print('CPU contracts passed; actual pretrained BART/VAE/GPU gates remain. No performance claim.',flush=True)
    if a.action=='check': return
    for arm in arms:
        branch='text' if arm in TEXT_ARMS else 'image'; mode='fresh' if arm=='I2_shifted' else getattr(a,branch+'_init')
        initialization(branch,mode)
        if mode=='history' and (c[branch+'_width']!=(64 if branch=='text' else 8)): raise ValueError('Changed rate requires --'+branch+'-init fresh')
        if not Path(a.bart_path if branch=='text' else a.vae_path).is_dir(): raise FileNotFoundError('Missing offline backbone')
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'launch_identity.json',dict(schema='docci-reconstruction-v10-launch',action=a.action,arms=arms,gpu_physical=1,
        max_hours=a.max_hours,smoke_weights_never_reused=True,no_automatic_promotion=True,quality_stop_keeps_other_arms_running=True))
    def run(arm,mode,size,label):
        cap=a.max_hours*3600-(time.monotonic()-start)-180
        if mode=='smoke': cap=min(cap,1800)
        if cap<=0: raise TimeoutError('Collection/time reserve insufficient')
        branch='text' if arm in TEXT_ARMS else 'image'
        cmd=[exe,'-B','-u',str(folder/'run.py'),'--arm',arm,'--mode',mode,'--image-size',str(size),
            '--image-width',str(a.image_width),'--text-width',str(a.text_width),'--output-dir',str(output/label),
            '--experiment-config',str(Path(a.experiment_config).resolve()),'--max-hours',str(cap/3600),
            '--init','fresh' if arm=='I2_shifted' else getattr(a,branch+'_init')]
        for k in defaults: cmd+=['--'+k.replace('_','-'),str(Path(getattr(a,k)).resolve())]
        invoke(cmd,output/(label+'.log'),cap); invoke([exe,'-B',str(folder/'report.py'),str(output/label)],limit=180)
        if mode=='smoke' and not (output/label/'SMOKE_COMPLETE.json').is_file(): raise RuntimeError('Authentic smoke marker missing')
    # Finish ALL requested smoke gates before the first substantive training arm.
    for arm in arms:
        if arm in IMAGE_ARMS: run(arm,'smoke',256,'smoke_'+arm+'_256')
        run(arm,'smoke',512,'smoke_'+arm+'_512')
    if a.action!='smoke':
        for arm in arms: run(arm,a.action,512,a.action+'_'+arm)
    invoke([exe,'-B',str(folder/'collect.py'),str(output),str(output/'v10_docci_reconstruction_v10_records.tar.gz')],limit=180)
    print('Send '+str(output/'v10_docci_reconstruction_v10_records.tar.gz')+'; large weights remain server.',flush=True)


if __name__=='__main__': main()
