"""User-started GPU1 pipeline; no installs/upgrades or unrelated process control."""
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
from fso_fluctuating_v2.core import load, TEMPLATE


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('action',choices=('check','calibrate','smoke','pipeline')); p.add_argument('output',nargs='?')
    p.add_argument('--max-hours',type=float,default=12.)
    p.add_argument('--experiment-config',default=str(TEMPLATE))
    p.add_argument('--source-run',default=str(PROJECT/'experiment_results/v10_fso_jscc_v1_01/v10_screen'))
    p.add_argument('--checkpoint',default=str(PROJECT/'experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth'))
    p.add_argument('--a-delta',default=str(PROJECT/'experiment_results/v10_token_guidance_quick_01/train_A/last_delta.pth'))
    a=p.parse_args(); cfg=load(a.experiment_config)
    if not math.isfinite(a.max_hours) or not 0<a.max_hours<=24: p.error('Budget >0 and <=24h, NOT ETA')
    exe=os.environ.get('PYTHON_BIN') or sys.executable
    if not Path(exe).is_file(): raise FileNotFoundError('Interpreter missing: '+exe)
    folder=Path(__file__).resolve().parent; started=time.monotonic()
    env=local_environment(PROJECT); env['CUDA_VISIBLE_DEVICES']='1'
    def invoke(command,log=None,cap=None):
        remaining=a.max_hours*3600-(time.monotonic()-started)
        if cap is not None: remaining=min(remaining,cap)
        if remaining<=0: raise TimeoutError('Overall launcher cap reached')
        stream=log.open('x',encoding='utf-8') if log else None; proc=None
        try:
            proc=subprocess.Popen(command,cwd=str(PROJECT),env=env,stdout=stream,stderr=subprocess.STDOUT if stream else None)
            print('Owned child PID={}{}'.format(proc.pid,' log='+str(log) if log else ''),flush=True)
            code=proc.wait(timeout=remaining)
            if code: raise RuntimeError('Child exit {}; preserve diagnostics{}'.format(code,': '+str(log) if log else ''))
        except BaseException:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            raise
        finally:
            if stream: stream.close()
    if a.action in ('check','pipeline'):
        for name in ('test_core.py','test_report.py','test_model.py'): invoke([exe,'-B',str(folder/name)])
        for package,name in [('fso_jscc_v1','test_model.py'),('optical_realism_v3','test_dsp.py'),('optical_realism_v3','test_repairs.py')]:
            invoke([exe,'-B',str(EXT/package/name)])
        print('CPU synthetic contracts passed; real checkpoint/data smoke still required.',flush=True)
    if a.action=='check':
        if a.output: p.error('check takes no output')
        return
    if not a.output: p.error('Specify a NEW experiment_results/v10_* output')
    output=write_path(a.output,PROJECT)
    if output.exists() or output.with_name(output.name+'.log').exists(): raise FileExistsError('Existing run/log preserved; choose a new attempt number')
    if output.relative_to(PROJECT).parts[0]!='experiment_results' or not output.name.startswith('v10_'): raise ValueError('Output outside experiment_results/v10_*')
    defaults=dict(source_run=Path(a.source_run).resolve(),checkpoint=Path(a.checkpoint).resolve(),a_delta=Path(a.a_delta).resolve(),
        config=EXT/'configs/v10_multichannel.json',split_manifest=EXT/'manifests/ln_v10_splits.json',data_root=PROJECT/'COCO2017',
        vae_path=PROJECT/'pretrained/sd-vae-ft-mse',text_model_path=PROJECT/'pretrained/clip-vit-base-patch32')
    if a.action!='calibrate':
        for key,path in defaults.items():
            if not path.exists(): raise FileNotFoundError('Missing prerequisite {}: {}'.format(key,path))
        if not (defaults['source_run']/'train_M2_awgn_ft/last_delta.pth').is_file(): raise FileNotFoundError('Completed parent M2 delta missing')
    output.mkdir(parents=True,exist_ok=False)
    invoke([exe,'-B','-u',str(folder/'calibrate.py'),str(output/'calibration'),'--experiment-config',str(Path(a.experiment_config).resolve())],
        output/'calibration.log',1800)
    if a.action=='calibrate':
        print('Public simulator calibration complete; NO neural training. Return calibration/bank.json and calibration.log.',flush=True); return
    def run(name,smoke):
        target=output/name
        remaining=a.max_hours-(time.monotonic()-started)/3600
        cap=min(900.,remaining*3600) if smoke else remaining*3600
        command=[exe,'-B','-u',str(folder/'run.py'),'--output-dir',str(target),'--max-hours',str(cap/3600),
            '--calibration',str(output/'calibration'),'--experiment-config',str(Path(a.experiment_config).resolve())]
        for key,path in defaults.items(): command+=['--'+key.replace('_','-'),str(path)]
        if smoke: command+=['--smoke']
        invoke(command,output/(name+'.log'),cap)
        invoke([exe,'-B',str(folder/'report.py'),str(target),'--verify-only'])
        return target
    selected=run('v10_smoke',True)
    if a.action=='pipeline':
        projected=0.
        for arm in ('B1_awgn_ft','B2_fso_ft'):
            r=json.loads((selected/('train_'+arm)/'COMPLETE.json').read_text())
            projected+=r['wall_seconds']*cfg['epochs']*cfg['train_samples']/max(r['steps'],1)
        print('Smoke passed. Very rough measured TRAIN-only projection {:.1f}min; load/cache, waveform evaluation excluded. First 100 steps refine ETA. No extra seed sweep.'.format(projected/60),flush=True)
        selected=run('v10_screen',False)
    archive=output/'v10_fso_fluctuating_v2_records.tar.gz'
    invoke([exe,'-B',str(folder/'collect.py'),str(selected),str(archive)])
    print('Return '+str(archive)+' (includes calibration summary; no weights).',flush=True)


if __name__=='__main__': main()
