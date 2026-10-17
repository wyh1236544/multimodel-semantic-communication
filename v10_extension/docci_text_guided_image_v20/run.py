"""One bounded GPU stage. Never downloads, installs, retries or resumes silently."""
import argparse
import os
from pathlib import Path
import sys
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT),str(EXT.parent)]
from docci_text_guided_image_v20.core import load,safe,check_source,Budget,ARMS

def arguments(parser):
    project = EXT.parent
    parser.add_argument('--data-root',default=str(project))
    parser.add_argument('--bart-path',default=str(project/'pretrained/bart-base'))
    parser.add_argument('--vae-path',default=str(project/'pretrained/sd-vae-ft-mse'))
    parser.add_argument('--longclip-path',default=str(project/'longclip-B.pt'))
    parser.add_argument('--clip-tokenizer-path',default=str(project/'pretrained/clip-vit-base-patch32'))
    parser.add_argument('--lpips-assets',default=str(project/'_v20_lpips_assets'))
    parser.add_argument('--image-checkpoint',default=str(project/'experiment_results/v14_image_pilot_02/main/image/expanded/train/last.pth'))
    parser.add_argument('--text-checkpoint',default=str(project/'experiment_results/v18_text_noise_01/train_noise_replay/last.pth'))
    parser.add_argument('--image-parent-metadata',default=str(project/'experiment_results/v14_image_pilot_02/main/image/cache_train_identity.json'))
    parser.add_argument('--text-parent-metadata',default=str(project/'experiment_results/v14_text_pilot_01/main/text/cache_train_identity.json'))
    parser.add_argument('--config')

def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action',choices=('cache','train','evaluate'))
    p.add_argument('output')
    p.add_argument('--cache')
    p.add_argument('--arm',choices=ARMS)
    p.add_argument('--checkpoint')
    p.add_argument('--max-hours',type=float,required=True)
    p.add_argument('--smoke',action='store_true')
    arguments(p)
    a = p.parse_args()
    check_source()
    c = load(a.config,a.smoke)
    root = safe(a.output)
    if root.exists():
        raise FileExistsError('Existing stage preserved; use a NEW output')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':
        raise ValueError('Physical GPU1 required; do not change global settings')
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(c['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
        raise RuntimeError('Authentic GPU smoke required')
    free,total = torch.cuda.mem_get_info(device)
    print('GPU1 free={:.2f}GiB total={:.2f}GiB'.format(free/2**30,total/2**30),flush=True)
    if free<c['minimum_free_gib']*2**30:
        raise RuntimeError('Insufficient starting free GPU memory; no automatic batch change')
    budget = Budget(a.max_hours,device)
    if a.action=='cache':
        from docci_text_guided_image_v20.data import create
        create(a,c,root,device,budget,a.smoke)
    else:
        if not a.cache or not a.arm:
            p.error('Verified cache and explicit arm required')
        from docci_text_guided_image_v20.engine import fit,evaluate
        if a.action=='train':
            fit(a,c,root,device,budget,a.smoke)
        else:
            if not a.checkpoint:
                p.error('Actual saved final checkpoint required')
            evaluate(a,c,root,device,budget,a.smoke)
    print('V20 GPU stage finished; independent verification remains',flush=True)

if __name__=='__main__':
    main()
