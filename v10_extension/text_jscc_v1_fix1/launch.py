"""Owned-child GPU1 launcher; fresh project-local output and finite stop budgets."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
EXT=Path(__file__).resolve().parents[1]; PROJECT=EXT.parent; sys.path.insert(0,str(EXT))
from v10.paths import write_path,local_environment
from text_jscc_v1_fix1.core import load,TEMPLATE


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('action',choices=('check','smoke','pipeline')); p.add_argument('output',nargs='?')
    p.add_argument('--max-hours',type=float,default=4.); p.add_argument('--experiment-config',default=str(TEMPLATE))
    p.add_argument('--source-run',default=str(PROJECT/'experiment_results/v10_joint_jscc_arch_v1_01/v10_screen'))
    p.add_argument('--legacy-run',default=str(PROJECT/'experiment_results/v10_fso_fluctuating_v2_01/v10_screen'))
    p.add_argument('--checkpoint',default=str(PROJECT/'experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth'))
    p.add_argument('--a-delta',default=str(PROJECT/'experiment_results/v10_token_guidance_quick_01/train_A/last_delta.pth'))
    a=p.parse_args(); load(a.experiment_config)
    if not math.isfinite(a.max_hours) or not 0<a.max_hours<=24: p.error('Stop budget must be >0 and <=24h')
    if (a.action=='check') != (a.output is None): p.error('check has no output; smoke/pipeline need a NEW output')
    out=write_path(a.output,PROJECT) if a.output else None
    if out and (out.exists() or out.with_name(out.name+'.log').exists()): raise FileExistsError('Existing output/log preserved; use suffix02')
    if out and (out.relative_to(PROJECT).parts[0]!='experiment_results' or not out.name.startswith('v10_')): raise ValueError('Only project-local experiment_results/v10_*')
    exe=os.environ.get('PYTHON_BIN') or sys.executable
    if not Path(exe).is_file(): raise FileNotFoundError('Interpreter missing: '+exe)
    env=local_environment(PROJECT); env['CUDA_VISIBLE_DEVICES']='1'
    folder=Path(__file__).parent; start=time.monotonic()
    def invoke(command,log=None,cap=None):
        remaining=a.max_hours*3600-(time.monotonic()-start)
        if cap is not None: remaining=min(remaining,cap)
        if remaining<=0: raise TimeoutError('Launcher stop budget reached')
        stream=log.open('x',encoding='utf-8') if log else None; child=None
        try:
            child=subprocess.Popen(command,cwd=str(PROJECT),env=env,stdout=stream,stderr=subprocess.STDOUT if stream else None)
            print('Owned PID={}{}'.format(child.pid,' log='+str(log) if log else ''),flush=True)
            code=child.wait(timeout=remaining)
            if code: raise RuntimeError('Child exit {}; preserve logs'.format(code))
        except BaseException:
            if child is not None and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            raise
        finally:
            if stream: stream.close()
    invoke([exe,'-B',str(folder/'install.py'),'--project-root',str(PROJECT),'--verify-only'],cap=600)
    for name in ('test_core.py','test_channel.py','test_report.py','test_model.py'):
        invoke([exe,'-B',str(folder/name)],cap=600)
    print('CPU contracts passed; actual savedB4/tokenizer/GPU gates still require smoke. No dependency changes.',flush=True)
    if a.action=='check': return
    defaults=dict(source_run=Path(a.source_run).resolve(strict=True),legacy_run=Path(a.legacy_run).resolve(strict=True),
        checkpoint=Path(a.checkpoint).resolve(strict=True),a_delta=Path(a.a_delta).resolve(strict=True),
        config=EXT/'configs/v10_multichannel.json',split_manifest=EXT/'manifests/ln_v10_splits.json',
        data_root=PROJECT/'COCO2017',vae_path=PROJECT/'pretrained/sd-vae-ft-mse',
        text_model_path=PROJECT/'pretrained/clip-vit-base-patch32')
    for key,path in defaults.items():
        if not path.exists(): raise FileNotFoundError('Missing prerequisite '+key+': '+str(path))
    out.mkdir(parents=True,exist_ok=False)
    def run(name,smoke):
        remaining=a.max_hours*3600-(time.monotonic()-start); cap=min(2700.,remaining) if smoke else remaining
        command=[exe,'-B','-u',str(folder/'run.py'),'--output-dir',str(out/name),'--max-hours',str(cap/3600),
            '--experiment-config',str(Path(a.experiment_config).resolve())]
        for key,path in defaults.items(): command+=['--'+key.replace('_','-'),str(path)]
        if smoke: command.append('--smoke')
        invoke(command,out/(name+'.log'),cap)
        invoke([exe,'-B',str(folder/'report.py'),str(out/name)],cap=600)
        return out/name
    selected=run('v10_smoke',True)
    if a.action=='pipeline':
        print('Real smoke passed execution only. Main quality gates can STOP without FSO; all records retained. Max4h is not ETA.',flush=True)
        selected=run('v10_screen',False)
    archive=out/'v10_text_jscc_v1_fix1_records.tar.gz'
    invoke([exe,'-B',str(folder/'collect.py'),str(selected),str(archive)],cap=600)
    print('Return '+str(archive)+'. Diagnostic-stop is not completed FSO training.',flush=True)


if __name__=='__main__': main()
