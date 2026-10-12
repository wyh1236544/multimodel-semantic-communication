"""Owned-child bounded launcher: check -> real smoke -> text diagnosis."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
EXT=Path(__file__).resolve().parents[1]; PROJECT=EXT.parent; sys.path.insert(0,str(EXT))
from v10.paths import write_path,local_environment
from text_recovery_v3.core import load,TEMPLATE,caption_view
from bart_clip_joint_v1.core import offline_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('action',choices=('check','data-check','smoke','diagnostic','pipeline')); p.add_argument('output',nargs='?')
    p.add_argument('--max-hours',type=float,default=2.); p.add_argument('--experiment-config',default=str(TEMPLATE))
    p.add_argument('--include-common',action='store_true')
    defaults=dict(bart_path=PROJECT/'pretrained/bart-base',text_model_path=PROJECT/'pretrained/clip-vit-base-patch32',
        data_root=PROJECT/'COCO2017',split_manifest=EXT/'manifests/ln_v10_splits.json',
        channel_source_run=PROJECT/'experiment_results/v10_joint_jscc_arch_v1_01/v10_screen')
    for key,path in defaults.items(): p.add_argument('--'+key.replace('_','-'),default=str(path))
    a=p.parse_args(); c=load(a.experiment_config)
    if not math.isfinite(a.max_hours) or not 0<a.max_hours<=24: p.error('Finite max-hours required')
    readonly=a.action in ('check','data-check')
    if readonly!=(a.output is None): p.error('check/data-check no output; other actions need NEW output')
    out=write_path(a.output,PROJECT) if a.output else None
    if out and (out.exists() or out.with_name(out.name+'.log').exists()): raise FileExistsError('Existing output/log preserved; choose02')
    if out and (out.relative_to(PROJECT).parts[0]!='experiment_results' or not out.name.startswith('v10_')): raise ValueError('Project-local experiment_results/v10_* only')
    if a.action=='data-check':
        for split,count in [('train',c['train_samples']),('validation',c['val_samples'])]:
            _,receipt=caption_view(a.data_root,a.split_manifest,split,count,c['seed']); print(receipt)
        print('Original COCO short text available. No images opened or data changes.'); return
    exe=os.environ.get('PYTHON_BIN') or sys.executable
    if not Path(exe).is_file(): raise FileNotFoundError('Missing existing interpreter '+exe)
    env=local_environment(PROJECT); env['CUDA_VISIBLE_DEVICES']='1'; env['HF_HUB_OFFLINE']='1'; env['TRANSFORMERS_OFFLINE']='1'
    folder=Path(__file__).parent; start=time.monotonic()
    def invoke(command,log=None,cap=600):
        remaining=min(cap,a.max_hours*3600-(time.monotonic()-start))
        if remaining<=0: raise TimeoutError('Overall finite launcher cap')
        stream=log.open('x',encoding='utf-8') if log else None; child=None
        try:
            child=subprocess.Popen(command,cwd=str(PROJECT),env=env,stdout=stream,stderr=subprocess.STDOUT if stream else None)
            print('Owned PID={}{}'.format(child.pid,' log='+str(log) if log else ''),flush=True)
            code=child.wait(timeout=remaining)
            if code: raise RuntimeError('Owned child exit {}; preserve logs; no auto retry'.format(code))
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
    print('CPU contracts passed. Real pretrained/GPU gates still require smoke; no dependency changes.',flush=True)
    if a.action=='check': return
    for key in ('bart_path','text_model_path','data_root','split_manifest'):
        if not Path(getattr(a,key)).exists(): raise FileNotFoundError('Missing offline input '+key)
    offline_snapshot(a.bart_path,'bart'); offline_snapshot(a.text_model_path,'clip')
    caption_view(a.data_root,a.split_manifest,'train',c['diagnostic_samples'],c['seed'])
    if a.action=='pipeline' and not (Path(a.channel_source_run)/'COMPLETE.json').exists(): raise FileNotFoundError('Completed channel configuration source required')
    out.mkdir(parents=True,exist_ok=False)
    def run(name,smoke):
        cap=min(1800.,a.max_hours*3600-(time.monotonic()-start)-90.) if smoke else a.max_hours*3600-(time.monotonic()-start)-90.
        if cap<=0: raise TimeoutError('No budget left for run/collection')
        cmd=[exe,'-B','-u',str(folder/'run.py'),'--output-dir',str(out/name),'--max-hours',str(cap/3600),
             '--experiment-config',str(Path(a.experiment_config).resolve())]
        for key in defaults: cmd+=['--'+key.replace('_','-'),str(Path(getattr(a,key)).resolve())]
        if smoke: cmd.append('--smoke')
        elif a.action=='pipeline': cmd.append('--expand')
        if a.include_common: cmd.append('--include-common')
        invoke(cmd,out/(name+'.log'),cap); invoke([exe,'-B',str(folder/'report.py'),str(out/name)])
        return out/name
    selected=run('v10_smoke',True)
    if a.action!='smoke': selected=run('v10_screen',False)
    invoke([exe,'-B',str(folder/'collect.py'),str(selected),str(out/'v10_text_recovery_v3_records.tar.gz')])
    print('Send v10_text_recovery_v3_records.tar.gz. SCREEN_COMPLETE means diagnostics finished, not successful recovery.',flush=True)


if __name__=='__main__': main()
