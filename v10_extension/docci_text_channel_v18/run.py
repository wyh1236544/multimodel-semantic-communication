"""Single bounded stage. Invoked by launch.py; no automatic checkpoint retries."""
import argparse
import os
from pathlib import Path
import sys
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_text_channel_v18.core import load,check_source,safe,read,Budget,atomic

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action',choices=('audit','train','evaluate'));p.add_argument('output')
    p.add_argument('--features',required=True);p.add_argument('--inputs')
    p.add_argument('--bart-path',default=str(EXT.parent/'pretrained/bart-base'))
    p.add_argument('--config');p.add_argument('--arm',choices=('frozen','clean_continue','noise_replay'))
    p.add_argument('--checkpoint');p.add_argument('--smoke',action='store_true');p.add_argument('--max-hours',type=float,required=True)
    a=p.parse_args();check_source();c=load(a.config,a.smoke);root=safe(a.output)
    if root.exists(): raise FileExistsError('Existing stage preserved')
    if a.action=='audit':
        from docci_text_channel_v18.data import audit
        audit(a.features,root,a.smoke,Budget(a.max_hours));return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1': raise ValueError('PhysicalGPU1 only')
    import torch
    torch.set_num_threads(4);torch.manual_seed(c['seed'])
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available(): raise RuntimeError('Actual saved-model GPU stage required')
    device=torch.device('cuda:0');free,total=torch.cuda.mem_get_info(device)
    print('GPU1 free={:.2f}GiB total={:.2f}GiB'.format(free/2**30,total/2**30),flush=True)
    if free<c['minimum_free_gib']*2**30: raise RuntimeError('Insufficient starting GPU reserve')
    if not a.inputs: p.error('Verified input reference required')
    from docci_text_channel_v18.data import prepared
    items=prepared(a.inputs,c)
    from docci_text_channel_v18.engine import fit,evaluate
    if a.action=='train': fit(a,c,items['train'],root,device,Budget(a.max_hours,device,c['reserve_free_gib']),a.smoke)
    else:
        if (a.arm=='frozen')!=(a.checkpoint is None): p.error('Only frozen evaluation omits checkpoint')
        evaluate(a,c,items['qual_dev'],root,device,Budget(a.max_hours,device,c['reserve_free_gib']),a.smoke)
    print('V18 stage finished; independent CPU verification remains',flush=True)
if __name__=='__main__': main()
