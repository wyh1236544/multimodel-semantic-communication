"""One bounded stage; actual saved models and selected cache bytes are mandatory."""
import argparse
import os
from pathlib import Path
import sys
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_text_aid_v21.core import (PROJECT,FOLDER,CACHE_SHA,PARENTS,config,
    choose,source_check,safe,read,sha,atomic)
from docci_text_guided_image_v20.run import arguments

def options(p):
    arguments(p)
    p.add_argument('--v20-root',default=str(PROJECT/'experiment_results/v20_text_guided_image_fix1_01'))
    p.add_argument('--profile',choices=('nominal','missing25'),default='nominal')

def inputs(a,c):
    old=safe(a.v20_root)
    if read(old/'PIPELINE.json')['status']!='COMPLETE':
        raise ValueError('Completed V20 Fix1 run required')
    if sha(old/'cache/index.json')!=CACHE_SHA:
        raise ValueError('Pinned V20 cache index differs')
    # Exact original provenance before unpickling; no new interpretation of old results.
    fix=read(old/'V20_FIX1.json')
    from docci_text_guided_image_v20_fix1.compat import source_check as fix_source
    from docci_text_guided_image_v20_fix1.compat import validate_gate,FOLDER as FIX
    if fix['repair_manifest_sha256']!=sha(FIX/'MANIFEST.json'):
        raise ValueError('Wrong old repair provenance')
    fix_source();validate_gate(read(old/'FIX1_GPU_GATE.json'))
    for arm,h in PARENTS.items():
        p=old/('train_'+arm)/'last.pth'
        if sha(p)!=h or read(p.parent/'TRAIN_COMPLETE.json')['checkpoint_sha256']!=h:
            raise ValueError('Actual V20 saved checkpoint differs: '+arm)
    selected=choose(read(old/'cache/index.json'),c)
    # Selected arrays independently checked by Items, and all original array hashes
    # checked again at each load. Do not copy/recompute the 8192-example cache.
    from docci_text_guided_image_v20.data import Items
    for g,group in selected.items():
        items=Items(old/'cache',g)
        by={r['example_id']:i for i,r in enumerate(items.entries)}
        for r in group:
            item=items[by[r['example_id']]]
            import hashlib
            if hashlib.sha256(item['text'].encode('utf-8')).hexdigest()!=item['text_sha256']:
                raise ValueError('Selected source text hash differs')
            if sha(Path(item['source_image_path']))!=item['image_sha256']:
                raise ValueError('Original selected image changed')
    a.cache=str(old/'cache')
    return selected

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action',choices=('diagnose','pilot'));p.add_argument('output')
    p.add_argument('--smoke',action='store_true');p.add_argument('--max-hours',type=float,required=True)
    p.add_argument('--gate',required=True);options(p);a=p.parse_args()
    source_check()
    if a.config:
        p.error('Use fixed V21 profiles, not the historical V20 --config option')
    root=safe(a.output)
    if root.exists():
        raise FileExistsError('Existing output preserved')
    c=config(a.action,a.profile,a.smoke)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':
        raise RuntimeError('Physical GPU1 required')
    from docci_text_guided_image_v20_fix1.compat import validate_gate,pooling_patch
    validate_gate(read(a.gate))
    selected=inputs(a,c)
    c['selected']=selected
    import torch
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():
        raise RuntimeError('Real saved-weight CUDA smoke required')
    device=torch.device('cuda:0')
    free,total=torch.cuda.mem_get_info(device)
    print('GPU1 free={:.2f}GiB total={:.2f}GiB'.format(free/2**30,total/2**30),flush=True)
    if free<20*2**30:
        raise RuntimeError('Need 20GiB starting free; no automatic batch or job changes')
    from docci_text_guided_image_v20.core import Budget
    budget=Budget(a.max_hours,device)
    root.mkdir(parents=True,exist_ok=False)
    atomic(root/'config.json',c)
    atomic(root/'lineage.json',dict(cache_index_sha256=CACHE_SHA,parents=PARENTS,
        old_root=str(Path(a.v20_root).resolve()),cache_regenerated=False,
        actual_selected_cache_arrays_verified=True,strict_determinism=True,
        CUDA_gate=read(a.gate),source_manifest_sha256=sha(FOLDER/'MANIFEST.json')))
    from docci_text_guided_image_v20 import model
    from docci_text_aid_v21.runtime import diagnostics,train_pilot
    with pooling_patch(model):
        (diagnostics if a.action=='diagnose' else train_pilot)(a,c,root,device,budget)
    atomic(root/'RUN_FINISHED.json',dict(training_and_evaluation_finished=True,
        independent_verification_pending=True))
    print('V21 TRAIN_AND_EVAL_FINISHED; independent verification remains',flush=True)

if __name__=='__main__':
    main()
