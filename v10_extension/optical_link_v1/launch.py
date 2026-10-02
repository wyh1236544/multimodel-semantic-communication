"""Safe project-local launcher; GPU1, no training, logs and overall time cap."""
import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
EXT=Path(__file__).resolve().parents[1]
PROJECT=EXT.parent
sys.path.insert(0,str(EXT))
from v10.paths import local_environment,write_path


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('action',choices=('check','link','smoke','quick'))
    p.add_argument('output',nargs='?')
    p.add_argument('--max-hours',type=float)
    args,rest=p.parse_known_args()
    exe=os.environ.get('PYTHON_BIN') or sys.executable
    if not Path(exe).is_file(): raise FileNotFoundError('Explicit Python interpreter not found: '+exe)
    env=local_environment(PROJECT)
    env['CUDA_VISIBLE_DEVICES']='1'
    folder=Path(__file__).resolve().parent
    if args.action=='check':
        if args.output or rest or args.max_hours is not None: p.error('check accepts no other arguments')
        for name in ('test_core.py','test_model.py'):
            subprocess.run([exe,'-B',str(folder/name)],cwd=str(PROJECT),env=env,check=True)
        print('CPU contracts passed; actual A weights/GPU/data still require smoke.',flush=True)
        return
    if not args.output: p.error('Specify a NEW experiment_results directory')
    output=write_path(args.output,PROJECT)
    if output.exists(): raise FileExistsError('Existing output preserved: '+str(output))
    hours=args.max_hours if args.max_hours is not None else (.25 if args.action=='link' else .5 if args.action=='smoke' else 2.)
    if not math.isfinite(hours) or not 0<hours<=4: p.error('Budget must be >0 and <=4h')
    command=[exe,'-B','-u',str(folder/('diagnostic.py' if args.action=='link' else 'run.py'))]
    if args.action!='link':
        defaults=dict(checkpoint=PROJECT/'experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth',
            a_delta=PROJECT/'experiment_results/v10_token_guidance_quick_01/train_A/last_delta.pth',
            config=EXT/'configs/v10_multichannel.json',split_manifest=EXT/'manifests/ln_v10_splits.json',
            data_root=PROJECT/'COCO2017',vae_path=PROJECT/'pretrained/sd-vae-ft-mse',
            text_model_path=PROJECT/'pretrained/clip-vit-base-patch32')
        for key,value in defaults.items(): command+=['--'+key.replace('_','-'),str(value)]
        command+=['--val-samples','2' if args.action=='smoke' else '8','--save-images','2' if args.action=='smoke' else '8','--max-hours',str(hours)]
    if any(v=='--output-dir' or v.startswith('--output-dir=') or v=='--max-hours' or v.startswith('--max-hours=') for v in rest):
        p.error('Output and overall budget are controlled by launch')
    command+=['--output-dir',str(output)]+rest
    log=output.with_name(output.name+'.log')
    if log.exists(): raise FileExistsError('Existing log preserved: '+str(log))
    log.parent.mkdir(parents=True,exist_ok=True)
    print('GPU1 | {} | no training | cap={}h\nLog: {}'.format(args.action,hours,log),flush=True)
    with log.open('x',encoding='utf-8') as stream:
        process=subprocess.Popen(command,cwd=str(PROJECT),env=env,stdout=stream,stderr=subprocess.STDOUT)
        print('PID='+str(process.pid),flush=True)
        try:
            code=process.wait(timeout=hours*3600)
        except BaseException:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
    if code: raise SystemExit('Exit {}. Failure log preserved: {}'.format(code,log))
    if not (output/'COMPLETE.json').is_file(): raise RuntimeError('Exit zero without COMPLETE')
    print('Finished: '+str(output/'REPORT.md'),flush=True)


if __name__=='__main__': main()
