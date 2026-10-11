"""GPU1: checks -> short frozen-weight smoke -> 128-image evaluation -> archive."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
EXT=Path(__file__).resolve().parents[1]; PROJECT=EXT.parent
sys.path.insert(0,str(EXT))
from v10.paths import write_path, local_environment
from fso_jscc_eval_v1.core import load, settings, expected_counts, TEMPLATE


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('action',choices=('check','smoke','pipeline')); p.add_argument('output',nargs='?')
    p.add_argument('--max-hours',type=float,default=6.); p.add_argument('--experiment-config',default=str(TEMPLATE))
    p.add_argument('--source-run',default=str(PROJECT/'experiment_results/v10_fso_jscc_v1_01/v10_screen'))
    p.add_argument('--checkpoint',default=str(PROJECT/'experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth'))
    p.add_argument('--a-delta',default=str(PROJECT/'experiment_results/v10_token_guidance_quick_01/train_A/last_delta.pth'))
    a=p.parse_args(); cfg=load(a.experiment_config)
    if not math.isfinite(a.max_hours) or not 0<a.max_hours<=12: p.error('Budget must be >0 and <=12h')
    exe=os.environ.get('PYTHON_BIN') or sys.executable
    if not Path(exe).is_file(): raise FileNotFoundError('Interpreter missing: '+exe)
    started=time.monotonic(); env=local_environment(PROJECT); env['CUDA_VISIBLE_DEVICES']='1'
    folder=Path(__file__).resolve().parent
    def invoke(command,log=None,cap_seconds=None):
        remaining=a.max_hours*3600-(time.monotonic()-started)
        if cap_seconds is not None: remaining=min(remaining,cap_seconds)
        if remaining<=0: raise TimeoutError('Overall launcher budget reached')
        stream=log.open('x',encoding='utf-8') if log else None; process=None
        try:
            process=subprocess.Popen(command,cwd=str(PROJECT),env=env,stdout=stream,stderr=subprocess.STDOUT if stream else None)
            print('Owned evaluation child PID={}{}'.format(process.pid,' log='+str(log) if log else ''),flush=True)
            code=process.wait(timeout=remaining)
            if code: raise RuntimeError('Child exit {}; evidence retained{}'.format(code,': '+str(log) if log else ''))
        except BaseException:
            if process is not None and process.poll() is None:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
        finally:
            if stream: stream.close()
    if a.action in ('check','pipeline'):
        for name in ('test_core.py','test_model.py'): invoke([exe,'-B',str(folder/name)])
        for module,name in [('fso_jscc_v1','test_core.py'),('fso_jscc_v1','test_model.py'),('optical_realism_v3','test_dsp.py'),('optical_realism_v3','test_repairs.py')]:
            invoke([exe,'-B',str(EXT/module/name)])
        print('CPU contracts passed; real saved checkpoints still need smoke. No training.',flush=True)
    if a.action=='check':
        if a.output: p.error('check takes no output')
        return
    if not a.output: p.error('Specify a NEW experiment_results/v10_* output')
    output=write_path(a.output,PROJECT)
    if output.exists() or output.with_name(output.name+'.log').exists(): raise FileExistsError('Existing output/log preserved')
    if output.relative_to(PROJECT).parts[0]!='experiment_results' or not output.name.startswith('v10_'): raise ValueError('Use experiment_results/v10_*')
    source=Path(a.source_run).resolve(strict=True); source.relative_to(PROJECT)
    defaults=dict(source_run=source,checkpoint=Path(a.checkpoint).resolve(),a_delta=Path(a.a_delta).resolve(),
        config=EXT/'configs/v10_multichannel.json',split_manifest=EXT/'manifests/ln_v10_splits.json',data_root=PROJECT/'COCO2017',
        vae_path=PROJECT/'pretrained/sd-vae-ft-mse',text_model_path=PROJECT/'pretrained/clip-vit-base-patch32')
    for key,path in defaults.items():
        if not path.exists(): raise FileNotFoundError('Missing prerequisite {}: {}'.format(key,path))
    for arm in ('M1_clean_ft','M2_awgn_ft','M3_fso_ft'):
        path=source/('train_'+arm)/'last_delta.pth'
        if not path.is_file(): raise FileNotFoundError('Keep original trained delta on server: '+str(path))
    print('Expanded matrix counts {}; frozen four checkpoints; no optimizer.'.format(expected_counts(cfg)),flush=True)
    def run(target,smoke):
        remaining=a.max_hours*3600-(time.monotonic()-started)
        limit=min(remaining,900.) if smoke else remaining
        command=[exe,'-B','-u',str(folder/'run.py'),'--output-dir',str(target),'--experiment-config',str(Path(a.experiment_config).resolve()),'--max-hours',str(limit/3600)]
        for key,path in defaults.items(): command+=['--'+key.replace('_','-'),str(path)]
        if smoke: command+=['--smoke']
        target.parent.mkdir(parents=True,exist_ok=True)
        invoke(command,target.with_name(target.name+'.log'),limit)
        invoke([exe,'-B',str(folder/'report.py'),str(target),'--verify-only'])
    if a.action=='pipeline':
        output.mkdir(parents=True,exist_ok=False); run(output/'v10_smoke',True)
        measured=json.loads((output/'v10_smoke/timing.json').read_text())['engines']
        counts=expected_counts(cfg)
        projection=sum(counts[e]*t['seconds']/t['rows'] for e,t in measured.items())
        print('Frozen smoke passed. Rough measured projection {:.1f} min (excludes model load/report; not a promise). Now expanded evaluation.'.format(projection/60),flush=True)
        run(output/'v10_expanded',False); selected=output/'v10_expanded'; archive=output/'v10_fso_jscc_eval_v1_records.tar.gz'
    else:
        run(output,True); selected=output; archive=output.with_name(output.name+'_records.tar.gz')
    invoke([exe,'-B',str(folder/'collect.py'),str(selected),str(archive)])
    print('Return this record archive, no weights: '+str(archive),flush=True)


if __name__=='__main__': main()
