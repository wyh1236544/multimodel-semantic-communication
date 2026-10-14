"""GPU1/offline private-branch launcher; quality failure does not block sibling."""
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
from docci_modal_v9.core import load,safe_write,atomic_json,initialization


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('action',choices=('check','data-check','smoke','diagnostic','screen','pipeline')); p.add_argument('output',nargs='?')
    p.add_argument('--branch',choices=('both','text','image'),default='both'); p.add_argument('--max-hours',type=float,default=4)
    p.add_argument('--experiment-config',default=str(Path(__file__).with_name('config.json')))
    p.add_argument('--max-text-tokens',type=int,choices=(512,1024),default=512)
    defaults=dict(data_root=PROJECT,bart_path=PROJECT/'pretrained/bart-base',vae_path=PROJECT/'pretrained/sd-vae-ft-mse')
    for key,value in defaults.items(): p.add_argument('--'+key.replace('_','-'),default=str(value))
    for branch in ('text','image'):
        p.add_argument('--'+branch+'-init',default='history'); p.add_argument('--'+branch+'-init-sha256')
    a=p.parse_args(); c=load(a.experiment_config,512,a.max_text_tokens)
    branches=['text','image'] if a.branch=='both' else [a.branch]
    if not math.isfinite(a.max_hours) or not 0<a.max_hours<=24: p.error('Finite maximum time0..24h')
    if (a.action in ('check','data-check'))!=(a.output is None): p.error('Checks take no output; execution needs a NEW output')
    out=safe_write(a.output) if a.output else None
    if out:
        rel=out.relative_to(PROJECT)
        if rel.parts[0]!='experiment_results' or not out.name.startswith('v10_'): raise ValueError('Use project experiment_results/v10_*')
        if out.exists() or out.with_name(out.name+'.log').exists(): raise FileExistsError('Existing output/log preserved. Use a NEW02/03 directory; no deletion')
    exe=os.environ.get('PYTHON_BIN') or sys.executable
    if not Path(exe).is_file(): raise FileNotFoundError('Existing Python interpreter missing; no dependency install')
    env={k:v for k,v in os.environ.items() if k!='SUDO_PW'}
    env.update(CUDA_VISIBLE_DEVICES='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',V9_REQUIRE_TORCH='1')
    env['PYTHONPATH']=os.pathsep.join([str(EXT),str(PROJECT)])
    env['HF_HOME']=str(PROJECT/'.cache/v10/huggingface'); env['TORCH_HOME']=str(PROJECT/'.cache/v10/torch')
    folder=Path(__file__).resolve().parent; start=time.monotonic()
    def invoke(command,log=None,limit=None):
        cap=a.max_hours*3600-(time.monotonic()-start)
        if limit is not None: cap=min(cap,limit)
        if cap<=0: raise TimeoutError('Overall cap reached; preserve completed branches/logs')
        stream=log.open('x',encoding='utf-8') if log else None; child=None
        try:
            child=subprocess.Popen(command,cwd=str(PROJECT),env=env,stdout=stream,stderr=subprocess.STDOUT if stream else None)
            print('Owned PID={}{}'.format(child.pid,' log='+str(log) if log else ''),flush=True)
            code=child.wait(timeout=cap)
            if code: raise RuntimeError('Engineering child exit {}; preserve logs. No automatic retry; quality-stop itself exits0'.format(code))
        except BaseException:
            if child is not None and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            raise
        finally:
            if stream: stream.close()
    if a.action=='data-check':
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        from docci_modal_v9.data import audit
        for branch in branches:
            result=audit(a,c,branch); print(json.dumps(result,ensure_ascii=False,indent=2))
            key='oversized' if branch=='text' else 'missing_count'
            if any(r[key] for r in result['splits'].values()): raise ValueError('Full-source '+branch+' audit failed; no truncation/filter')
        print('DOCCI original source audit only. No training or download.'); return
    invoke([exe,'-B',str(folder/'install.py'),'--project-root',str(PROJECT),'--verify-only'],limit=120)
    for test in ('test_core.py','test_report.py','test_model.py'): invoke([exe,'-B',str(folder/test)],limit=600)
    print('CPU contracts passed; authentic offline backbone/GPU smoke still mandatory. No performance claim.',flush=True)
    if a.action=='check': return
    for branch in branches:
        initialization(branch,getattr(a,branch+'_init'),getattr(a,branch+'_init_sha256'))
        path=a.bart_path if branch=='text' else a.vae_path
        if not Path(path).is_dir(): raise FileNotFoundError('Missing existing offline '+branch+' backbone: '+path)
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'launch_identity.json',dict(schema='docci-modal-v9-launch',action=a.action,requested_branches=branches,gpu_physical=1,
        max_hours=a.max_hours,smoke_weights_never_reused=True,quality_stop_does_not_block_sibling=True,engineering_failure_stops_with_logs=True))
    def run(branch,mode,size,label):
        cap=a.max_hours*3600-(time.monotonic()-start)-120
        if mode=='smoke': cap=min(cap,1800)
        if cap<=0: raise TimeoutError('Insufficient collection/time reserve')
        cmd=[exe,'-B','-u',str(folder/'run.py'),'--branch',branch,'--mode',mode,'--image-size',str(size),
            '--output-dir',str(out/label),'--experiment-config',str(Path(a.experiment_config).resolve()),
            '--max-text-tokens',str(a.max_text_tokens),'--max-hours',str(cap/3600)]
        for key in defaults: cmd+=['--'+key.replace('_','-'),str(Path(getattr(a,key)).resolve())]
        cmd+=['--'+branch+'-init',getattr(a,branch+'_init')]
        if getattr(a,branch+'_init_sha256'): cmd+=['--'+branch+'-init-sha256',getattr(a,branch+'_init_sha256')]
        invoke(cmd,out/(label+'.log'),cap); invoke([exe,'-B',str(folder/'report.py'),str(out/label)],limit=120)
        if mode=='smoke' and not (out/label/'SMOKE_COMPLETE.json').is_file(): raise RuntimeError('Authentic '+branch+' smoke completion absent')
    for branch in branches:
        if branch=='image': run(branch,'smoke',256,'v10_smoke_image256')
        run(branch,'smoke',512,'v10_smoke_'+branch+'512')
        if a.action!='smoke': run(branch,a.action,512,'v10_'+a.action+'_'+branch)
    invoke([exe,'-B',str(folder/'collect.py'),str(out),str(out/'v10_docci_modal_v9_records.tar.gz')],limit=180)
    print('Send '+str(out/'v10_docci_modal_v9_records.tar.gz')+'; weights/data stay server. Each branch has its own quality status.',flush=True)


if __name__=='__main__': main()
