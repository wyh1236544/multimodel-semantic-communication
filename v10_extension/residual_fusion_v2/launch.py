"""GPU1, new output only, owned-child stop budgets, automatic diagnostic collection."""
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
from residual_fusion_v2.core import load,TEMPLATE


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('action',choices=('check','smoke','pipeline')); p.add_argument('output',nargs='?')
    p.add_argument('--max-hours',type=float,default=3.); p.add_argument('--experiment-config',default=str(TEMPLATE))
    p.add_argument('--source-run',default=str(PROJECT/'experiment_results/v10_joint_jscc_arch_v1_01/v10_screen'))
    p.add_argument('--legacy-run',default=str(PROJECT/'experiment_results/v10_fso_fluctuating_v2_01/v10_screen'))
    p.add_argument('--calibration',default=str(PROJECT/'experiment_results/v10_fso_fluctuating_v2_01/calibration'))
    p.add_argument('--checkpoint',default=str(PROJECT/'experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth'))
    p.add_argument('--a-delta',default=str(PROJECT/'experiment_results/v10_token_guidance_quick_01/train_A/last_delta.pth'))
    a=p.parse_args(); load(a.experiment_config)
    if not math.isfinite(a.max_hours) or not 0<a.max_hours<=24: p.error('Stop budget >0 and <=24h; not ETA')
    if a.action=='check' and a.output: p.error('check does not accept output')
    if a.action!='check' and not a.output: p.error('Specify a NEW experiment_results/v10_* directory')
    output=write_path(a.output,PROJECT) if a.output else None
    if output and (output.exists() or output.with_name(output.name+'.log').exists()): raise FileExistsError('Existing output/log preserved; use a new suffix')
    if output and (output.relative_to(PROJECT).parts[0]!='experiment_results' or not output.name.startswith('v10_')): raise ValueError('Project-local experiment_results/v10_* required')
    exe=os.environ.get('PYTHON_BIN') or sys.executable
    if not Path(exe).is_file(): raise FileNotFoundError('Interpreter missing: '+exe)
    env=local_environment(PROJECT); env['CUDA_VISIBLE_DEVICES']='1'
    folder=Path(__file__).resolve().parent; started=time.monotonic()
    def invoke(command,log=None,cap=None):
        remaining=a.max_hours*3600-(time.monotonic()-started)
        if cap is not None: remaining=min(remaining,cap)
        if remaining<=0: raise TimeoutError('Launcher time budget reached')
        stream=log.open('x',encoding='utf-8') if log else None; child=None
        try:
            child=subprocess.Popen(command,cwd=str(PROJECT),env=env,stdout=stream,stderr=subprocess.STDOUT if stream else None)
            print('Owned child PID={}{}'.format(child.pid,' log='+str(log) if log else ''),flush=True)
            code=child.wait(timeout=remaining)
            if code: raise RuntimeError('Child exit {}; preserve {}'.format(code,log))
        except BaseException:
            if child is not None and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            raise
        finally:
            if stream: stream.close()
    invoke([exe,'-B',str(folder/'install.py'),'--project-root',str(PROJECT),'--verify-only'],cap=600)
    for name in ('test_core.py','test_report.py','test_integration.py','test_model.py'):
        invoke([exe,'-B',str(folder/name)],cap=600)
    for name in ('test_dsp.py','test_repairs.py'):
        invoke([exe,'-B',str(EXT/'optical_realism_v3'/name)],cap=600)
    print('CPU contracts passed. Real B4 weights/GPU/current payload/text gate still require smoke/main. No environment changes.',flush=True)
    if a.action=='check': return
    defaults=dict(source_run=Path(a.source_run).resolve(strict=True),legacy_run=Path(a.legacy_run).resolve(strict=True),
        calibration=Path(a.calibration).resolve(strict=True),checkpoint=Path(a.checkpoint).resolve(strict=True),
        a_delta=Path(a.a_delta).resolve(strict=True),config=EXT/'configs/v10_multichannel.json',
        split_manifest=EXT/'manifests/ln_v10_splits.json',data_root=PROJECT/'COCO2017',
        vae_path=PROJECT/'pretrained/sd-vae-ft-mse',text_model_path=PROJECT/'pretrained/clip-vit-base-patch32')
    for key,file in defaults.items():
        if not file.exists(): raise FileNotFoundError('Missing prerequisite {}: {}'.format(key,file))
    for file in (defaults['source_run']/'COMPLETE.json',defaults['source_run']/'train_B4_separate/fso_delta.pth',
            defaults['legacy_run']/'train_B2_fso_ft/last_delta.pth',defaults['calibration']/'bank.npz'):
        if not file.is_file(): raise FileNotFoundError('Keep completed weights/bank on server: '+str(file))
    output.mkdir(parents=True,exist_ok=False)
    def run(name,smoke):
        remaining=a.max_hours*3600-(time.monotonic()-started); cap=min(2700.,remaining) if smoke else remaining
        command=[exe,'-B','-u',str(folder/'run.py'),'--output-dir',str(output/name),
            '--max-hours',str(cap/3600),'--experiment-config',str(Path(a.experiment_config).resolve())]
        for key,file in defaults.items(): command+=['--'+key.replace('_','-'),str(file)]
        if smoke: command.append('--smoke')
        invoke(command,output/(name+'.log'),cap)
        invoke([exe,'-B',str(folder/'report.py'),str(output/name)],cap=600)
        return output/name
    selected=run('v10_smoke',True)
    if a.action=='pipeline':
        print('Smoke validates execution, NOT the formal text-quality gate. Main uses32 training texts/1000updates per diagnostic, then at most TWO4000-forward residual arms. Budget is a cap, not ETA.',flush=True)
        selected=run('v10_screen',False)
    archive=output/'v10_residual_fusion_v2_records.tar.gz'
    invoke([exe,'-B',str(folder/'collect.py'),str(selected),str(archive)],cap=600)
    print('Return '+str(archive)+'; a diagnostic_stop archive means NO completed main fusion experiment. Keep all weights on server.',flush=True)


if __name__=='__main__': main()
