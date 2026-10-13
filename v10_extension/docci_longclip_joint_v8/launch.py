"""Offline GPU1 launcher, mandatory independent256/512 smoke, owned children."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
EXT=Path(__file__).resolve().parents[1]; PROJECT=EXT.parent
sys.path.insert(0,str(EXT))
from docci_longclip_joint_v8.core import load,safe_write,read,atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('action',choices=('check','data-check','smoke','diagnostic','screen','pipeline')); p.add_argument('output',nargs='?')
    p.add_argument('--max-hours',type=float,default=4.)
    p.add_argument('--experiment-config',default=str(Path(__file__).with_name('config.json')))
    p.add_argument('--max-text-tokens',type=int,choices=(256,512,1024),default=512)
    p.add_argument('--text-updates',type=int); p.add_argument('--text-init')
    defaults=dict(data_root=PROJECT/'DOCCI',bart_path=PROJECT/'pretrained/bart-base',vae_path=PROJECT/'pretrained/sd-vae-ft-mse',
        clip_tokenizer_path=PROJECT/'pretrained/clip-vit-base-patch32',longclip_path=PROJECT/'pretrained/longclip-B.pt')
    for key,value in defaults.items(): p.add_argument('--'+key.replace('_','-'),default=str(value))
    a=p.parse_args(); c=load(a.experiment_config,512,a.max_text_tokens)
    if not math.isfinite(a.max_hours) or not 0<a.max_hours<=24: p.error('Finite total time cap<=24h')
    if (a.action in ('check','data-check'))!=(a.output is None): p.error('Checks take no output; runs require NEW output')
    if a.text_updates is not None and not 8<=a.text_updates<=10000: p.error('text updates8..10000')
    out=safe_write(a.output) if a.output else None
    if out:
        relative=out.relative_to(PROJECT)
        if relative.parts[0]!='experiment_results' or not out.name.startswith('v10_'): raise ValueError('Only project experiment_results/v10_*')
        if out.exists() or out.with_name(out.name+'.log').exists(): raise FileExistsError('Existing output/log preserved. Choose a NEW02/03 directory; no deletion needed')
    exe=os.environ.get('PYTHON_BIN') or sys.executable
    if not Path(exe).is_file(): raise FileNotFoundError('Existing interpreter unavailable; no environment installation')
    env=dict(os.environ); env.update(CUDA_VISIBLE_DEVICES='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',V8_REQUIRE_TORCH='1')
    env['HF_HOME']=str(PROJECT/'.cache/v10/huggingface'); env['TORCH_HOME']=str(PROJECT/'.cache/v10/torch')
    env['PYTHONPATH']=os.pathsep.join([str(EXT),str(PROJECT)]+([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    folder=Path(__file__).parent; start=time.monotonic()
    def invoke(cmd,log=None,cap=None):
        remaining=a.max_hours*3600-(time.monotonic()-start)
        if cap is not None: remaining=min(remaining,cap)
        if remaining<=0: raise TimeoutError('Overall cap exhausted; preserve logs')
        stream=log.open('x',encoding='utf-8') if log else None; child=None
        try:
            child=subprocess.Popen(cmd,cwd=str(PROJECT),env=env,stdout=stream,stderr=subprocess.STDOUT if stream else None)
            print('Owned PID={}{}'.format(child.pid,' log='+str(log) if log else ''),flush=True)
            code=child.wait(timeout=remaining)
            if code: raise RuntimeError('Child exit {}; logs preserved. No automatic retry or overwriting'.format(code))
        except BaseException:
            if child is not None and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            raise
        finally:
            if stream: stream.close()
    if a.action=='data-check':
        from docci_longclip_joint_v8.data import audit
        for key in ('data_root','bart_path','clip_tokenizer_path'):
            if not Path(getattr(a,key)).exists(): raise FileNotFoundError('Missing offline '+key)
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        result=audit(a,c); print(json.dumps(result,ensure_ascii=False,indent=2))
        if any(r['missing_count'] for r in result['splits'].values()): raise FileNotFoundError('DOCCI images incomplete; no training')
        print('Length audit only. Selected full descriptions still undergo hard no-truncation checks at run time. No test or COCO model inputs.')
        return
    invoke([exe,'-B',str(folder/'install.py'),'--project-root',str(PROJECT),'--verify-only'],cap=120)
    for name in ('test_core.py','test_report.py','test_model.py'): invoke([exe,'-B',str(folder/name)],cap=600)
    print('CPU contracts passed. Authentic BART/VAE/Long-CLIP/512 backward/GPU gates require smoke; this is not a performance result.',flush=True)
    if a.action=='check': return
    for key in defaults:
        if not Path(getattr(a,key)).exists(): raise FileNotFoundError('Missing offline '+key+': '+str(getattr(a,key)))
    if a.text_init and not Path(a.text_init).is_file(): raise FileNotFoundError('Explicit text-init missing')
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'launch_identity.json',dict(action=a.action,pid=os.getpid(),python=exe,gpu_physical=1,
        source='DOCCI',smoke_512_mandatory=True,smoke_weights_never_loaded_into_main=True,total_max_hours=a.max_hours))
    def run(mode,size,label):
        cap=a.max_hours*3600-(time.monotonic()-start)-120
        if mode=='smoke': cap=min(cap,2400.)
        if cap<=0: raise TimeoutError('Insufficient main/collection reserve')
        cmd=[exe,'-B','-u',str(folder/'run.py'),'--mode',mode,'--output-dir',str(out/label),
            '--experiment-config',str(Path(a.experiment_config).resolve()),'--image-size',str(size),
            '--max-text-tokens',str(a.max_text_tokens),'--max-hours',str(cap/3600)]
        for key in defaults: cmd+=['--'+key.replace('_','-'),str(Path(getattr(a,key)).resolve())]
        if a.text_init: cmd+=['--text-init',str(Path(a.text_init).resolve())]
        if mode!='smoke' and a.text_updates is not None: cmd+=['--text-updates',str(a.text_updates)]
        invoke(cmd,out/(label+'.log'),cap); invoke([exe,'-B',str(folder/'report.py'),str(out/label)],cap=120)
        if mode=='smoke' and not (out/label/'SMOKE_COMPLETE.json').is_file(): raise RuntimeError('Actual smoke completion absent')
        return out/label
    run('smoke',256,'v10_smoke256'); selected=run('smoke',512,'v10_smoke512')
    if a.action!='smoke': selected=run(a.action,512,'v10_'+a.action)
    invoke([exe,'-B',str(folder/'collect.py'),str(selected),str(out/'v10_docci_longclip_joint_v8_records.tar.gz')],cap=120)
    print('Send '+str(out/'v10_docci_longclip_joint_v8_records.tar.gz')+'; full checkpoints and native data remain server.',flush=True)


if __name__=='__main__': main()
