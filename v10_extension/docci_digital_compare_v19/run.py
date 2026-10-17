"""A single bounded stage; this module contains no training action."""
import argparse
import os
from pathlib import Path
import sys
EXT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_digital_compare_v19.common import safe,load,check_source,Budget,read,unseal

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action',choices=('prepare','digital','negative'));p.add_argument('output')
    p.add_argument('--v18-root');p.add_argument('--prepared');p.add_argument('--smoke',action='store_true')
    p.add_argument('--bart-path',default=str(EXT.parent/'pretrained/bart-base'))
    p.add_argument('--max-hours',type=float,required=True);a=p.parse_args()
    check_source();c=load();root=safe(a.output)
    if root.exists():raise FileExistsError('Existing stage preserved')
    if a.action=='prepare':
        if not a.v18_root:p.error('--v18-root required')
        from docci_digital_compare_v19.inputs import prepare
        prepare(a.v18_root,root,Budget(a.max_hours));return
    if not a.prepared:p.error('--prepared required')
    unseal(a.prepared)
    from docci_digital_compare_v19.experiment import digital,negative
    if a.action=='digital':digital(a.prepared,root,c,Budget(a.max_hours),a.smoke);return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('Physical GPU1 only')
    import torch
    torch.set_num_threads(4);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise RuntimeError('Real GPU inference required')
    free,total=torch.cuda.mem_get_info()
    print('GPU1 free={:.2f}GiB total={:.2f}GiB'.format(free/2**30,total/2**30),flush=True)
    if free<c['minimum_free_gib']*2**30:raise RuntimeError('Insufficient start reserve; no job control')
    negative(a.prepared,root,c,Budget(a.max_hours,torch.device('cuda:0')),a.smoke,a.bart_path)
if __name__=='__main__':main()
