"""Explicit one-stage worker. No automatic download, environment change or retry."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT),str(EXT.parent)]
from docci_blip_feature_rx_v25.core import (PROJECT,SCHEMA,read,safe,atomic,config,
    source_check,snapshot,Budget)

def parser():
    from docci_text_guided_image_v20.run import arguments
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('stage',choices=('inventory','cache','train','evaluation'))
    p.add_argument('output'); p.add_argument('--profile',choices=('smoke','pilot'),default='smoke')
    p.add_argument('--max-hours',type=float,default=2.)
    p.add_argument('--v20-root',default=str(PROJECT/'experiment_results/v20_text_guided_image_fix1_01'))
    p.add_argument('--caption-root',default=str(PROJECT/'experiment_results/v24_caption_calibration_fix2_01'))
    p.add_argument('--asset-lock',default=str(PROJECT/'experiment_results/v23_caption_lock_01/ASSET_LOCK.json'))
    p.add_argument('--cache'); p.add_argument('--training-root'); p.add_argument('--arm',choices=('self','visual'))
    arguments(p)
    return p

def inventory(a,c,root):
    packages = {}
    for name in ('torch','transformers','diffusers','numpy','Pillow'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    paths = {k:getattr(a,k) for k in ('v20_root','caption_root','asset_lock','bart_path','vae_path','lpips_assets','text_checkpoint','image_checkpoint')}
    value = dict(packages=packages,paths={k:dict(path=v,exists=Path(v).exists()) for k,v in paths.items()},
        training=False,weights_loaded=False,GPU_allocated=False,downloads=False,
        note='Existence is not identity, completeness or available VRAM')
    atomic(root/'ASSETS.json',value); print(json.dumps(value,ensure_ascii=False,indent=2))

def main():
    a = parser().parse_args(); source_check(); c = config(a.profile)
    if a.config: raise ValueError('V25 fixed protocol; do not reuse a V20 config')
    root = safe(a.output)
    if root.exists(): raise FileExistsError('Existing output preserved; use a NEW name')
    if a.stage in ('train','evaluation') and not a.cache: raise ValueError('Explicit verified cache required')
    if a.stage=='train' and not a.arm: raise ValueError('Explicit adapter arm required')
    if a.stage=='evaluation' and not a.training_root: raise ValueError('Two completed train arms required')
    budget = Budget(a.max_hours)
    root.mkdir(parents=True); snapshot(root)
    atomic(root/'RUN.json',dict(schema=SCHEMA,stage=a.stage,arguments=vars(a),
        interpreter=sys.executable,python=sys.version,no_automatic_download_or_install=True))
    try:
        if a.stage=='inventory':
            inventory(a,c,root)
            from docci_blip_feature_rx_v25.core import seal
            seal(root,'inventory'); return
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='1': raise ValueError('Use physical GPU1 via launcher')
        import torch
        import transformers
        if torch.__version__.split('+')[0]!='1.12.0' or transformers.__version__!='4.30.2':
            raise RuntimeError('Use original Torch1.12/HF4.30.2 environment; no upgrades')
        if not torch.cuda.is_available(): raise RuntimeError('Actual GPU required; no CPU fallback')
        torch.set_num_threads(4); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
        free,total = torch.cuda.mem_get_info()
        print('GPU1 free={:.2f}GiB total={:.2f}GiB'.format(free/2**30,total/2**30),flush=True)
        if free<c['minimum_free_gib']*2**30: raise RuntimeError('Insufficient startup reserve; do not stop other jobs')
        if shutil.disk_usage(PROJECT).free<c['minimum_disk_gib']*2**30: raise RuntimeError('Need 16GiB free project disk')
        budget.gpu = True; torch.cuda.reset_peak_memory_stats(); device = torch.device('cuda:0')
        if a.stage=='cache':
            from docci_blip_feature_rx_v25.prepare import create
            create(a,c,root,device,budget)
        else:
            from docci_blip_feature_rx_v25.report import cache as verify_cache
            verify_cache(a.cache,True)
            if read(Path(a.cache)/'config.json')!=c: raise ValueError('Cache/profile identity differs')
            if a.stage=='train':
                from docci_blip_feature_rx_v25.engine import fit
                fit(a,c,root,device,budget)
            else:
                from docci_blip_feature_rx_v25.engine import evaluate
                evaluate(a,c,root,device,budget)
        print('V25 stage finished; independent CPU verification still required',flush=True)
    except BaseException as error:
        if not (root/'FAILED.json').exists():
            atomic(root/'FAILED.json',dict(error_type=type(error).__name__,message=str(error),partial_preserved=True,automatic_retry=False))
        raise
if __name__=='__main__': main()
