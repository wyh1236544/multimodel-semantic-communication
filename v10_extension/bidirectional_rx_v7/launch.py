"""Offline bounded owned-child launcher; smoke never initializes main training."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
EXT=Path(__file__).resolve().parents[1]; PROJECT=EXT.parent; sys.path.insert(0,str(EXT))
from v10.paths import write_path,local_environment
from bidirectional_rx_v7.core import load,TEMPLATE
from bart_coco_modal_v2.core import caption_view
from bart_clip_joint_v1.core import offline_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('action',choices=('check','data-check','smoke','diagnostic','screen','pipeline')); p.add_argument('output',nargs='?')
    p.add_argument('--max-hours',type=float,default=3.); p.add_argument('--experiment-config',default=str(TEMPLATE))
    defaults=dict(bart_path=PROJECT/'pretrained/bart-base',source_run=PROJECT/'experiment_results/v10_joint_jscc_arch_v1_01/v10_screen',
        legacy_run=PROJECT/'experiment_results/v10_fso_fluctuating_v2_01/v10_screen',
        parent_screen=PROJECT/'experiment_results/v10_clip_bart_joint_v6_03/v10_screen',
        checkpoint=PROJECT/'experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth',
        a_delta=PROJECT/'experiment_results/v10_token_guidance_quick_01/train_A/last_delta.pth',
        config=EXT/'configs/v10_multichannel.json',split_manifest=EXT/'manifests/ln_v10_splits.json',data_root=PROJECT/'COCO2017',
        vae_path=PROJECT/'pretrained/sd-vae-ft-mse',text_model_path=PROJECT/'pretrained/clip-vit-base-patch32')
    for k,v in defaults.items(): p.add_argument('--'+k.replace('_','-'),default=str(v))
    a=p.parse_args(); c=load(a.experiment_config)
    if not math.isfinite(a.max_hours) or not 0<a.max_hours<=24: p.error('Finite time cap required')
    if (a.action in ('check','data-check'))!=(a.output is None): p.error('Run needs NEW output; checks need none')
    out=write_path(a.output,PROJECT) if a.output else None
    if out and (out.exists() or out.with_name(out.name+'.log').exists()): raise FileExistsError('Existing output/log preserved; use new02/03')
    if out and (out.relative_to(PROJECT).parts[0]!='experiment_results' or not out.name.startswith('v10_')): raise ValueError('Only project experiment_results/v10_*')
    if a.action=='data-check':
        for split,count in [('train',c['train_samples']),('validation',c['val_samples'])]:
            records,receipt=caption_view(a.data_root,a.split_manifest,split,count,c['seed'])
            part='train2017' if split=='train' else 'val2017'
            missing=[r['file_name'] for r in records if not (Path(a.data_root)/part/r['file_name']).is_file()]
            if missing: raise FileNotFoundError('Original images unavailable: '+str(missing[:5]))
            print('Original COCO {} selected={}, annotationSHA={}'.format(split,len(records),receipt['annotation_sha256']))
        return
    exe=os.environ.get('PYTHON_BIN') or sys.executable
    if not Path(exe).is_file(): raise FileNotFoundError('Existing interpreter unavailable; no install')
    folder=Path(__file__).parent; env=local_environment(PROJECT); env['CUDA_VISIBLE_DEVICES']='1'; env['V7_REQUIRE_TORCH']='1'; start=time.monotonic()
    def invoke(cmd,log=None,cap=600):
        remaining=min(cap,a.max_hours*3600-(time.monotonic()-start))
        if remaining<=0: raise TimeoutError('Overall cap')
        stream=log.open('x',encoding='utf-8') if log else None; child=None
        try:
            child=subprocess.Popen(cmd,cwd=str(PROJECT),env=env,stdout=stream,stderr=subprocess.STDOUT if stream else None)
            print('Owned PID={}{}'.format(child.pid,' log='+str(log) if log else ''),flush=True)
            code=child.wait(timeout=remaining)
            if code: raise RuntimeError('Child exit {}; logs preserved; no automatic retry'.format(code))
        except BaseException:
            if child is not None and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            raise
        finally:
            if stream: stream.close()
    invoke([exe,'-B',str(folder/'install.py'),'--project-root',str(PROJECT),'--verify-only'])
    for name in ('test_core.py','test_report.py','test_model.py'): invoke([exe,'-B',str(folder/name)])
    print('CPU contracts passed; actual saved BART/B4/VAE/CLIP/GPU gates still require smoke.',flush=True)
    if a.action=='check': return
    for k in defaults:
        if not Path(getattr(a,k)).exists(): raise FileNotFoundError('Missing offline '+k+': '+getattr(a,k))
    offline_snapshot(a.bart_path,'bart'); offline_snapshot(a.text_model_path,'clip')
    out.mkdir(parents=True,exist_ok=False)
    def run(mode):
        cap=a.max_hours*3600-(time.monotonic()-start)-120
        if mode=='smoke': cap=min(cap,1800.)
        if cap<=0: raise TimeoutError('No collection reserve')
        cmd=[exe,'-B','-u',str(folder/'run.py'),'--mode',mode,'--output-dir',str(out/('v10_'+mode)),
            '--experiment-config',str(Path(a.experiment_config).resolve()),'--max-hours',str(cap/3600)]
        for k in defaults: cmd+=['--'+k.replace('_','-'),str(Path(getattr(a,k)).resolve())]
        invoke(cmd,out/('v10_'+mode+'.log'),cap); invoke([exe,'-B',str(folder/'report.py'),str(out/('v10_'+mode))])
        return out/('v10_'+mode)
    selected=run('smoke')
    if a.action!='smoke': selected=run(a.action)
    invoke([exe,'-B',str(folder/'collect.py'),str(selected),str(out/'v10_bidirectional_rx_v7_records.tar.gz')],cap=120)
    print('Send '+str(out/'v10_bidirectional_rx_v7_records.tar.gz')+'; weights remain server.',flush=True)


if __name__=='__main__': main()
