"""Offline GPU1 owned-child launcher; read-only COCO check; fresh outputs only."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
EXT=Path(__file__).resolve().parents[1]; PROJECT=EXT.parent; sys.path.insert(0,str(EXT))
from v10.paths import write_path,local_environment
from bart_coco_modal_v2.core import load,TEMPLATE,caption_view
from bart_clip_joint_v1.core import offline_snapshot
from bart_coco_modal_v2 import MODES


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('action',choices=('check','data-check','smoke','pipeline')); p.add_argument('output',nargs='?')
    p.add_argument('--max-hours',type=float,default=2.); p.add_argument('--experiment-config',default=str(TEMPLATE))
    p.add_argument('--modes',nargs='+',choices=MODES)
    defaults=dict(bart_path=PROJECT/'pretrained/bart-base',
        source_run=PROJECT/'experiment_results/v10_joint_jscc_arch_v1_01/v10_screen',
        legacy_run=PROJECT/'experiment_results/v10_fso_fluctuating_v2_01/v10_screen',
        checkpoint=PROJECT/'experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth',
        a_delta=PROJECT/'experiment_results/v10_token_guidance_quick_01/train_A/last_delta.pth',
        config=EXT/'configs/v10_multichannel.json',split_manifest=EXT/'manifests/ln_v10_splits.json',
        data_root=PROJECT/'COCO2017',vae_path=PROJECT/'pretrained/sd-vae-ft-mse',text_model_path=PROJECT/'pretrained/clip-vit-base-patch32')
    for key,path in defaults.items(): p.add_argument('--'+key.replace('_','-'),default=str(path))
    a=p.parse_args(); cfg=load(a.experiment_config); modes=a.modes or cfg['default_modes']
    if len(set(modes))!=len(modes): p.error('No repeated modes')
    if not math.isfinite(a.max_hours) or not 0<a.max_hours<=24: p.error('Finite cap >0 and <=24h')
    readonly=a.action in ('check','data-check')
    if readonly!=(a.output is None): p.error('check/data-check take no output; smoke/pipeline need NEW output')
    if a.action=='data-check':
        for part,count in [('train',cfg['train_samples']),('validation',cfg['val_samples'])]:
            records,receipt=caption_view(a.data_root,a.split_manifest,part,count,cfg['seed'])
            if any(m!='text_only' for m in modes):
                folder=Path(a.data_root)/('train2017' if part=='train' else 'val2017')
                missing=[r['file_name'] for r in records if not (folder/r['file_name']).is_file()]
                if missing: raise FileNotFoundError('Missing selected images: '+str(missing[:5]))
            print(json.dumps(receipt,ensure_ascii=False,indent=2))
        print('Official COCO captions available. No annotation/split/data changed. No GPU or model download.'); return
    out=write_path(a.output,PROJECT) if a.output else None
    if out and (out.exists() or out.with_name(out.name+'.log').exists()): raise FileExistsError('Existing output/log preserved; choose02')
    if out and (out.relative_to(PROJECT).parts[0]!='experiment_results' or not out.name.startswith('v10_')): raise ValueError('Only project experiment_results/v10_*')
    exe=os.environ.get('PYTHON_BIN') or sys.executable
    if not Path(exe).is_file(): raise FileNotFoundError('Existing interpreter missing: '+exe)
    folder=Path(__file__).parent; env=local_environment(PROJECT); env['CUDA_VISIBLE_DEVICES']='1'; start=time.monotonic()
    def invoke(command,log=None,cap=None):
        remaining=a.max_hours*3600-(time.monotonic()-start)
        if cap is not None: remaining=min(remaining,cap)
        if remaining<=0: raise TimeoutError('Launcher finite cap reached')
        stream=log.open('x',encoding='utf-8') if log else None; child=None
        try:
            child=subprocess.Popen(command,cwd=str(PROJECT),env=env,stdout=stream,stderr=subprocess.STDOUT if stream else None)
            print('Owned PID={}{}'.format(child.pid,' log='+str(log) if log else ''),flush=True)
            code=child.wait(timeout=remaining)
            if code: raise RuntimeError('Owned child exit {}; preserve records, no automatic rerun'.format(code))
        except BaseException:
            if child is not None and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            raise
        finally:
            if stream: stream.close()
    invoke([exe,'-B',str(folder/'install.py'),'--project-root',str(PROJECT),'--verify-only'],cap=600)
    for name in ('test_core.py','test_report.py','test_model.py'): invoke([exe,'-B',str(folder/name)],cap=600)
    print('CPU/Torch contracts passed. Actual pretrained GPU still needs smoke. No dependency changes.',flush=True)
    if a.action=='check': return
    for key in defaults:
        path=Path(getattr(a,key))
        if not path.exists(): raise FileNotFoundError('Missing '+key+': '+str(path)+'; offline, no automatic download')
        defaults[key]=path.resolve(strict=True)
    for part,count in [('train',cfg['train_samples']),('validation',cfg['val_samples'])]: caption_view(a.data_root,a.split_manifest,part,count,cfg['seed'])
    offline_snapshot(defaults['bart_path'],'bart'); offline_snapshot(defaults['text_model_path'],'clip')
    out.mkdir(parents=True,exist_ok=False)
    def run(name,smoke):
        remaining=a.max_hours*3600-(time.monotonic()-start)-90.
        if remaining<=0: raise TimeoutError('No time left for run plus verification')
        cap=min(1800.,remaining) if smoke else remaining
        command=[exe,'-B','-u',str(folder/'run.py'),'--output-dir',str(out/name),'--max-hours',str(cap/3600),
            '--experiment-config',str(Path(a.experiment_config).resolve()),'--modes']+list(modes)
        for key,value in defaults.items(): command+=['--'+key.replace('_','-'),str(value)]
        if smoke: command.append('--smoke')
        invoke(command,out/(name+'.log'),cap); invoke([exe,'-B',str(folder/'report.py'),str(out/name)],cap=600)
        return out/name
    selected=run('v10_smoke',True)
    if a.action=='pipeline': selected=run('v10_screen',False)
    invoke([exe,'-B',str(folder/'collect.py'),str(selected),str(out/'v10_bart_coco_modal_v2_records.tar.gz')],cap=600)
    print('Send records archive, including negative text gates. partial_complete is not all-mode success.',flush=True)


if __name__=='__main__': main()
