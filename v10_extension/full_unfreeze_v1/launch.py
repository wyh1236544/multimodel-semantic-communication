"""GPU1: CPU contracts -> real64-pair memory/gradient smoke -> paired ablation; no calibration rerun."""
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
from v10.paths import write_path,local_environment
from full_unfreeze_v1 import ARMS
from full_unfreeze_v1.core import load,TEMPLATE


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('action',choices=('check','smoke','pipeline'))
    p.add_argument('output',nargs='?'); p.add_argument('--max-hours',type=float,default=4.)
    p.add_argument('--experiment-config',default=str(TEMPLATE))
    p.add_argument('--source-run',default=str(PROJECT/'experiment_results/v10_fso_fluctuating_v2_01/v10_screen'))
    p.add_argument('--calibration',default=None)
    p.add_argument('--checkpoint',default=str(PROJECT/'experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth'))
    p.add_argument('--a-delta',default=str(PROJECT/'experiment_results/v10_token_guidance_quick_01/train_A/last_delta.pth'))
    a=p.parse_args(); cfg=load(a.experiment_config)
    # Fail BEFORE CPU tests or creating output when an output/log already exists.
    if a.action!='check' and a.output:
        proposed=write_path(a.output,PROJECT)
        if proposed.exists() or proposed.with_name(proposed.name+'.log').exists():
            raise FileExistsError('Existing output/log preserved; use a NEW attempt number')
    if not math.isfinite(a.max_hours) or not 0<a.max_hours<=24: p.error('Cap >0 and <=24h; not ETA')
    exe=os.environ.get('PYTHON_BIN') or sys.executable
    if not Path(exe).is_file(): raise FileNotFoundError('Interpreter missing: '+exe)
    folder=Path(__file__).resolve().parent; started=time.monotonic()
    env=local_environment(PROJECT); env['CUDA_VISIBLE_DEVICES']='1'
    def invoke(command,log=None,cap=None):
        remaining=a.max_hours*3600-(time.monotonic()-started)
        if cap is not None: remaining=min(remaining,cap)
        if remaining<=0: raise TimeoutError('Launcher cap reached; preserve partial evidence')
        stream=log.open('x',encoding='utf-8') if log else None; proc=None
        try:
            proc=subprocess.Popen(command,cwd=str(PROJECT),env=env,stdout=stream,stderr=subprocess.STDOUT if stream else None)
            print('Owned child PID={}{}'.format(proc.pid,' log='+str(log) if log else ''),flush=True)
            code=proc.wait(timeout=remaining)
            if code: raise RuntimeError('Child exit {}; preserve diagnostics {}'.format(code,log))
        except BaseException:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            raise
        finally:
            if stream: stream.close()
    if a.action in ('check','pipeline'):
        for name in ('test_core.py','test_report.py','test_model.py'): invoke([exe,'-B',str(folder/name)],cap=600)
        for package,name in [('fso_fluctuating_v2','test_model.py'),('optical_realism_v3','test_dsp.py'),('optical_realism_v3','test_repairs.py')]:
            invoke([exe,'-B',str(EXT/package/name)],cap=600)
        print('CPU contracts passed; saved-checkpoint/GPU smoke still required. No environment changes.',flush=True)
    if a.action=='check':
        if a.output: p.error('check takes no output')
        return
    if not a.output: p.error('Specify a NEW experiment_results/v10_* directory')
    output=write_path(a.output,PROJECT)
    if output.exists() or output.with_name(output.name+'.log').exists(): raise FileExistsError('Existing run/log preserved; use a new attempt number')
    if output.relative_to(PROJECT).parts[0]!='experiment_results' or not output.name.startswith('v10_'): raise ValueError('Output outside experiment_results/v10_*')
    source=Path(a.source_run).resolve(strict=True)
    calibration=Path(a.calibration).resolve(strict=True) if a.calibration else source.parent/'calibration'
    defaults=dict(source_run=source,calibration=calibration,checkpoint=Path(a.checkpoint).resolve(),a_delta=Path(a.a_delta).resolve(),
        config=EXT/'configs/v10_multichannel.json',split_manifest=EXT/'manifests/ln_v10_splits.json',
        data_root=PROJECT/'COCO2017',vae_path=PROJECT/'pretrained/sd-vae-ft-mse',text_model_path=PROJECT/'pretrained/clip-vit-base-patch32')
    for key,path in defaults.items():
        if not path.exists(): raise FileNotFoundError('Missing prerequisite {}: {}'.format(key,path))
    for path in (source/'COMPLETE.json',source/'train_B2_fso_ft/last_delta.pth',calibration/'bank.npz',calibration/'COMPLETE.json'):
        if not path.is_file(): raise FileNotFoundError('Keep completed V2 weights/bank on server: '+str(path))
    output.mkdir(parents=True,exist_ok=False)
    def run(name,smoke):
        remaining=a.max_hours*3600-(time.monotonic()-started)
        cap=min(1800.,remaining) if smoke else remaining
        command=[exe,'-B','-u',str(folder/'run.py'),'--output-dir',str(output/name),'--max-hours',str(cap/3600),
            '--experiment-config',str(Path(a.experiment_config).resolve())]
        for key,path in defaults.items(): command+=['--'+key.replace('_','-'),str(path)]
        if smoke: command+=['--smoke']
        invoke(command,output/(name+'.log'),cap)
        invoke([exe,'-B',str(folder/'report.py'),str(output/name),'--verify-only'])
        return output/name
    selected=run('v10_smoke',True)
    if a.action=='pipeline':
        projected=0.
        for arm in ARMS[1:]:
            record=json.loads((selected/('train_'+arm)/'COMPLETE.json').read_text())
            projected+=record['wall_seconds']*cfg['epochs']*cfg['train_samples']/record['steps']
        print('Smoke passed; rough measured TRAIN-only projection {:.1f}min. 64 steps include warm-up, optimizer and checkpoint costs; first100-200 main steps refine ETA. Cache/load/evaluation excluded.'.format(projected/60),flush=True)
        for arm in ARMS[1:]:
            rec=json.loads((selected/('train_'+arm)/'COMPLETE.json').read_text())
            print('MEMORY {} peak allocated={:.2f}GiB reserved={:.2f}GiB eligible={} observed_nonzero_grad_parameters={}'.format(
                arm,rec['training_peak_allocated_gib'],rec['training_peak_reserved_gib'],rec['eligible_parameters'],rec['observed_nonzero_gradient_parameters']),flush=True)
        selected=run('v10_screen',False)
    archive=output/'v10_full_unfreeze_v1_records.tar.gz'
    invoke([exe,'-B',str(folder/'collect.py'),str(selected),str(archive)])
    print('Return '+str(archive)+'; keep new weights and V2 bank on server.',flush=True)


if __name__=='__main__': main()
