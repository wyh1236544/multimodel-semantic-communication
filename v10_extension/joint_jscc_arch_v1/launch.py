"""GPU1: required CPU tests -> bounded real smoke -> matched architecture screen."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
EXT=Path(__file__).resolve().parents[1]; PROJECT=EXT.parent; sys.path.insert(0,str(EXT))
from v10.paths import write_path,local_environment
from joint_jscc_arch_v1.core import load,TEMPLATE
from joint_jscc_arch_v1 import TRAINED_ARMS


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('action',choices=('check','smoke','pipeline'))
    p.add_argument('output',nargs='?'); p.add_argument('--max-hours',type=float,default=4.)
    p.add_argument('--experiment-config',default=str(TEMPLATE))
    p.add_argument('--source-run',default=str(PROJECT/'experiment_results/v10_fso_fluctuating_v2_01/v10_screen'))
    p.add_argument('--calibration')
    p.add_argument('--checkpoint',default=str(PROJECT/'experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth'))
    p.add_argument('--a-delta',default=str(PROJECT/'experiment_results/v10_token_guidance_quick_01/train_A/last_delta.pth'))
    a=p.parse_args(); cfg,_=load(a.experiment_config)
    if not math.isfinite(a.max_hours) or not 0<a.max_hours<=24: p.error('Time cap >0 and <=24h; not ETA')
    if a.action=='check' and a.output: p.error('check does not accept output')
    if a.action!='check' and not a.output: p.error('Specify a NEW experiment_results/v10_* directory')
    proposed=write_path(a.output,PROJECT) if a.output else None
    if proposed and (proposed.exists() or proposed.with_name(proposed.name+'.log').exists()):
        raise FileExistsError('Existing output/log preserved; use a NEW attempt suffix')
    exe=os.environ.get('PYTHON_BIN') or sys.executable
    if not Path(exe).is_file(): raise FileNotFoundError('Interpreter missing: '+exe)
    started=time.monotonic(); folder=Path(__file__).resolve().parent
    env=local_environment(PROJECT); env['CUDA_VISIBLE_DEVICES']='1'
    def invoke(command,log=None,cap=None):
        remaining=a.max_hours*3600-(time.monotonic()-started)
        if cap is not None: remaining=min(remaining,cap)
        if remaining<=0: raise TimeoutError('Launcher cap reached; preserve evidence')
        stream=log.open('x',encoding='utf-8') if log else None; child=None
        try:
            child=subprocess.Popen(command,cwd=str(PROJECT),env=env,stdout=stream,stderr=subprocess.STDOUT if stream else None)
            print('Owned child PID={}{}'.format(child.pid,' log='+str(log) if log else ''),flush=True)
            code=child.wait(timeout=remaining)
            if code: raise RuntimeError('Child exit {}; preserve diagnostics {}'.format(code,log))
        except BaseException:
            if child is not None and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            raise
        finally:
            if stream: stream.close()
    if a.action in ('check','pipeline','smoke'):
        for name in ('test_core.py','test_report.py','test_model.py'):
            invoke([exe,'-B',str(folder/name)],cap=600)
        for package,name in [('fso_fluctuating_v2','test_model.py'),('optical_realism_v3','test_dsp.py'),('optical_realism_v3','test_repairs.py')]:
            invoke([exe,'-B',str(EXT/package/name)],cap=600)
        print('CPU contracts passed; real weights/GPU/greedy text/fidelity still require smoke. No environment changes.',flush=True)
    if a.action=='check': return
    output=proposed
    if output.relative_to(PROJECT).parts[0]!='experiment_results' or not output.name.startswith('v10_'):
        raise ValueError('Output outside experiment_results/v10_*')
    source=Path(a.source_run).resolve(strict=True)
    calibration=Path(a.calibration).resolve(strict=True) if a.calibration else source.parent/'calibration'
    defaults=dict(source_run=source,calibration=calibration,checkpoint=Path(a.checkpoint).resolve(),a_delta=Path(a.a_delta).resolve(),
        config=EXT/'configs/v10_multichannel.json',split_manifest=EXT/'manifests/ln_v10_splits.json',
        data_root=PROJECT/'COCO2017',vae_path=PROJECT/'pretrained/sd-vae-ft-mse',text_model_path=PROJECT/'pretrained/clip-vit-base-patch32')
    for key,file in defaults.items():
        if not file.exists(): raise FileNotFoundError('Missing prerequisite {}: {}'.format(key,file))
    for file in (source/'COMPLETE.json',source/'train_B2_fso_ft/last_delta.pth',calibration/'bank.npz',calibration/'COMPLETE.json'):
        if not file.is_file(): raise FileNotFoundError('Keep completed V2 weights and calibration on server: '+str(file))
    output.mkdir(parents=True,exist_ok=False)
    def run(name,smoke):
        remaining=a.max_hours*3600-(time.monotonic()-started); cap=min(2700.,remaining) if smoke else remaining
        command=[exe,'-B','-u',str(folder/'run.py'),'--output-dir',str(output/name),'--max-hours',str(cap/3600),
                 '--experiment-config',str(Path(a.experiment_config).resolve())]
        for key,file in defaults.items(): command+=['--'+key.replace('_','-'),str(file)]
        if smoke: command+=['--smoke']
        invoke(command,output/(name+'.log'),cap)
        invoke([exe,'-B',str(folder/'report.py'),str(output/name),'--verify-only'])
        return output/name
    selected=run('v10_smoke',True)
    if a.action=='pipeline':
        projection=0.
        for arm in TRAINED_ARMS:
            rec=json.loads((selected/('train_'+arm)/'COMPLETE.json').read_text())
            projection+=rec['wall_seconds']*cfg['train_samples']*(cfg['clean_epochs']+cfg['noise_epochs'])/rec['steps']
            print('MEMORY {} allocated={:.2f}GiB reserved={:.2f}GiB trainable={} observed_grad={}'.format(
                arm,rec['training_peak_allocated_gib'],rec['training_peak_reserved_gib'],rec['trainable_parameters'],rec['observed_nonzero_gradient_parameters']),flush=True)
        print('Smoke passed. ROUGH scaled arm-time projection {:.1f}min; includes fixed smoke fidelity/checkpoint costs and may overestimate training. AR/full-waveform validation/loading are additional; first100-200 main steps refine ETA. 4h default is a STOP budget, not ETA.'.format(projection/60),flush=True)
        selected=run('v10_screen',False)
    archive=output/'v10_joint_jscc_arch_v1_records.tar.gz'
    invoke([exe,'-B',str(folder/'collect.py'),str(selected),str(archive)])
    print('Return '+str(archive)+'; keep new weights and calibration bank on server.',flush=True)


if __name__=='__main__': main()

