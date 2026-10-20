"""One bounded read-only worker; original server environment, no training."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_rx_diagnose_v26.core import (PROJECT,SCHEMA,read,sha,safe,atomic,config,
    source_check,snapshot,seal,verify_files,Budget,PINS)

def parser():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('stage',choices=('inventory','preflight','smoke','diagnose'))
    p.add_argument('output');p.add_argument('--max-hours',type=float,default=2)
    p.add_argument('--parent',default=str(PROJECT/'experiment_results/v25_blip_feature_pilot_01'))
    p.add_argument('--bart-path',default=str(PROJECT/'pretrained/bart-base'))
    p.add_argument('--inputs')
    return p

def main():
    a=parser().parse_args();source_check();budget=Budget(a.max_hours)
    root=safe(a.output)
    parent=Path(a.parent).resolve()
    if root==parent or parent in root.parents or root in parent.parents:raise ValueError('Diagnostic output must not overlap frozen parent')
    if root.exists():raise FileExistsError('Existing output preserved; use a NEW name')
    root.mkdir(parents=True);snapshot(root)
    atomic(root/'RUN.json',dict(schema=SCHEMA,stage=a.stage,arguments=vars(a),
        interpreter=sys.executable,python=sys.version,training_updates=0))
    try:
        if a.stage=='inventory':
            packages={}
            for n in ('torch','transformers','numpy'):
                try:packages[n]=importlib.metadata.version(n)
                except importlib.metadata.PackageNotFoundError:packages[n]=None
            atomic(root/'ASSETS.json',dict(packages=packages,parent_exists=Path(a.parent).exists(),
                bart_exists=Path(a.bart_path).exists(),weights_loaded=False,GPU_allocated=False,
                downloads=False,training_updates=0))
        elif a.stage=='preflight':
            from docci_rx_diagnose_v26.parents import verify
            print('V26 PRECHECK pinned V25 cache/weights/results read-only; no models loaded',flush=True)
            result=verify(a.parent,True,True);budget()
            shutil.copyfile(Path(a.parent)/'cache/index.json',root/'PARENT_INDEX.json')
            shutil.copyfile(Path(a.parent)/'LAUNCH_FINISHED.json',root/'PARENT_LAUNCH.json')
            shutil.copyfile(Path(a.parent)/'evaluation/rows.jsonl',root/'V25_EVALUATION_ROWS.jsonl')
            shutil.copyfile(Path(a.parent)/'cache/caption_only.jsonl',root/'CAPTION_ONLY.jsonl')
            shutil.copyfile(Path(a.parent)/'cache/COMPLETE.json',root/'V25_CACHE_COMPLETE.json')
            shutil.copyfile(Path(a.parent)/'cache/gray.json',root/'PARENT_GRAY.json')
            atomic(root/'PARENTS.json',result)
        else:
            if not a.inputs:raise ValueError('Explicit completed preflight --inputs required')
            verify_files(a.inputs)
            if sha(Path(a.inputs)/'PARENT_INDEX.json')!=PINS['cache/index.json']:
                raise ValueError('Completed parent index differs')
            if read(Path(a.inputs)/'PARENTS.json')['heavy_arrays_and_weights_verified'] is not True:
                raise ValueError('Actual parent heavy preflight required')
            if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('Use physical GPU1 via launcher')
            import torch
            import transformers
            if torch.__version__.split('+')[0]!='1.12.0' or transformers.__version__!='4.30.2':
                raise RuntimeError('Use original Torch1.12/HF4.30.2, no upgrades')
            if not torch.cuda.is_available():raise RuntimeError('Actual GPU required')
            c=config(a.stage)
            free,total=torch.cuda.mem_get_info()
            print('GPU1 free={:.2f}GiB total={:.2f}GiB; BART/adapters only'.format(free/2**30,total/2**30),flush=True)
            if free<c['minimum_free_gib']*2**30:raise RuntimeError('Need 4GiB free; do not stop other jobs')
            if shutil.disk_usage(PROJECT).free<c['minimum_disk_gib']*2**30:raise RuntimeError('Need 2GiB free project disk')
            torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
            torch.cuda.reset_peak_memory_stats();budget.gpu=True
            atomic(root/'config.json',c)
            from docci_rx_diagnose_v26.neural import run
            run(a,c,root,torch.device('cuda:0'),budget)
        seal(root)
        print('V26 '+a.stage+' FINISHED; independent verification pending; training=0',flush=True)
    except BaseException as error:
        if not (root/'FAILED.json').exists():
            atomic(root/'FAILED.json',dict(error_type=type(error).__name__,message=str(error),training_updates=0,no_retry=True))
        raise
if __name__=='__main__':main()
