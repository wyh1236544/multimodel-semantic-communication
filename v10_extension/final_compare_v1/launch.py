"""Check -> smoke -> validation -> verify/collect. GPU1, zero training, new outputs."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
EXT=Path(__file__).resolve().parents[1]
PROJECT=EXT.parent
sys.path.insert(0,str(EXT))
from v10.paths import local_environment,write_path


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('action',choices=('check','smoke','validation','pipeline'))
    p.add_argument('output',nargs='?')
    p.add_argument('--max-hours',type=float,default=4.)
    p.add_argument('--plan',default=str(Path(__file__).with_name('plan.json')))
    p.add_argument('--optical-config',default=str(EXT/'optical_link_v1/optical_config.json'))
    a=p.parse_args()
    if not math.isfinite(a.max_hours) or not 0<a.max_hours<=6: p.error('Cap must be >0 and <=6h')
    exe=os.environ.get('PYTHON_BIN') or sys.executable
    if not Path(exe).is_file(): raise FileNotFoundError('Python interpreter missing: '+exe)
    env=local_environment(PROJECT); env['CUDA_VISIBLE_DEVICES']='1'
    folder=Path(__file__).resolve().parent
    started=time.monotonic()
    def remaining():
        value=a.max_hours*3600-(time.monotonic()-started)
        if value<=0: raise TimeoutError('Overall cap reached; partial results/logs preserved')
        return value
    def invoke(command,log=None):
        stream=log.open('x',encoding='utf-8') if log else None
        process=None
        try:
            process=subprocess.Popen(command,cwd=str(PROJECT),env=env,stdout=stream,stderr=subprocess.STDOUT if stream else None)
            print('Owned child PID={}{}'.format(process.pid,' | log='+str(log) if log else ''),flush=True)
            code=process.wait(timeout=remaining())
            if code: raise RuntimeError('Child exit {}. {}'.format(code,'Inspect '+str(log) if log else 'No experiment started.'))
        except BaseException:
            # ONLY our child; no GPU-user process inspection/killing.
            if process is not None and process.poll() is None:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
        finally:
            if stream: stream.close()
    if a.action in ('check','pipeline'):
        for name in ('test_core.py','test_model.py'):
            invoke([exe,'-B',str(folder/name)])
        print('CPU contract tests passed. Real data/weights/GPU still require smoke.',flush=True)
    if a.action=='check':
        if a.output: p.error('check has no output directory')
        return
    if not a.output: p.error('Specify a NEW experiment_results directory')
    output=write_path(a.output,PROJECT)
    if output.exists() or output.with_name(output.name+'.log').exists():
        raise FileExistsError('Existing output/log preserved: '+str(output))
    defaults=dict(checkpoint=PROJECT/'experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth',
        a_delta=PROJECT/'experiment_results/v10_token_guidance_quick_01/train_A/last_delta.pth',
        config=EXT/'configs/v10_multichannel.json',split_manifest=EXT/'manifests/ln_v10_splits.json',
        data_root=PROJECT/'COCO2017',vae_path=PROJECT/'pretrained/sd-vae-ft-mse',
        text_model_path=PROJECT/'pretrained/clip-vit-base-patch32')
    for key,value in defaults.items():
        if not value.exists(): raise FileNotFoundError('Prerequisite {} missing: {}'.format(key,value))
    def evaluate(target,smoke):
        command=[exe,'-B','-u',str(folder/'run.py'),'--output-dir',str(target),'--plan',str(Path(a.plan).resolve()),
            '--optical-config',str(Path(a.optical_config).resolve()),'--val-samples','2' if smoke else '32',
            '--draws','1' if smoke else '3','--save-images','2' if smoke else '8','--max-hours',str(remaining()/3600)]
        if smoke: command.append('--smoke')
        for key,value in defaults.items(): command+=['--'+key.replace('_','-'),str(value)]
        log=target.with_name(target.name+'.log'); log.parent.mkdir(parents=True,exist_ok=True)
        print('GPU1 | {} | no training | remaining {:.1f}h'.format('smoke' if smoke else 'validation',remaining()/3600),flush=True)
        invoke(command,log)
        invoke([exe,'-B',str(folder/'report.py'),str(target),'--verify-only'])
    if a.action=='pipeline':
        output.mkdir(parents=True,exist_ok=False)
        evaluate(output/'smoke',True)
        smoke=json.loads((output/'smoke/COMPLETE.json').read_text(encoding='utf-8'))
        # Diagnostic projection only: startup and waveform lengths differ.
        print('Smoke wall {:.1f}s. Main timing is estimated, not guaranteed; inspect progress.json during validation.'.format(smoke['wall_seconds']),flush=True)
        evaluate(output/'validation',False)
        invoke([exe,'-B',str(folder/'collect.py'),str(output/'validation'),str(output/'v10_final_compare_records.tar.gz')])
        print('DONE: '+str(output/'validation/report.html'),flush=True)
    else:
        evaluate(output,a.action=='smoke')
        print('DONE: '+str(output/'report.html'),flush=True)


if __name__=='__main__': main()
