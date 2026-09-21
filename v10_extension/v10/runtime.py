from __future__ import annotations
import copy
import json
import random
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from .data import build_loader

def capture_rng(device):
    import numpy as np
    state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state(device)
    return state

def restore_rng(state, device):
    import numpy as np
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state:
        torch.cuda.set_rng_state(state['cuda'], device)

def load_config(args):
    from utils.config import load_config as load, prepare_config
    root = Path(__file__).resolve().parents[1]
    spec = load(args.config)
    config = prepare_config(load(root/spec['base_config']), preset='original_4608',
                            data_root=args.data_root, vae_path=args.vae_path,
                            text_model_path=args.text_model_path)
    config['v10'] = copy.deepcopy(spec['v10'])
    config['train']['batch_size'] = args.batch_size
    config['train']['num_workers'] = args.num_workers
    if getattr(args, 'regime', 'mixed') != 'mixed':
        config['v10']['train_profiles'] = [args.regime]
    for attr in ('blocks', 'csi', 'equalizer', 'fixed_mode'):
        if getattr(args, attr, None) is not None:
            config['v10'][attr] = getattr(args, attr)
    for attr, key in (('no_uep', 'uep'), ('no_adapter', 'adapter'), ('no_interleave', 'interleave')):
        if getattr(args, attr, False):
            config['v10'][key] = False
    if getattr(args, 'keep_early_fusion', False):
        config['v10']['early_fusion'] = True
    if getattr(args, 'no_common_probe', False):
        config['v10']['common_probe_weight'] = 0.
    if getattr(args, 'no_anchor', False):
        config['v10']['anchor_weight'] = 0.
    return config

def add_shared_args(p):
    p.add_argument('--config', default='configs/v10_multichannel.json')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split-manifest', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--vae-path', required=True)
    p.add_argument('--text-model-path', required=True)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--blocks', type=int, choices=(1, 2, 4, 8))
    p.add_argument('--csi', choices=('pilot', 'perfect'))
    p.add_argument('--equalizer', choices=('mmse', 'zf'))
    p.add_argument('--fixed-mode', type=int, choices=(0, 1, 2))
    p.add_argument('--no-uep', action='store_true')
    p.add_argument('--no-adapter', action='store_true')
    p.add_argument('--no-interleave', action='store_true')
    p.add_argument('--keep-early-fusion', action='store_true')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--min-free-gb', type=float, default=8.)
    p.add_argument('--min-free-disk-gb', type=float, default=8.)
    p.add_argument('--max-hours', type=float, default=6.)
    return p

def preflight(args):
    import shutil
    from .paths import write_path
    write_path(args.output_dir)
    from utils.experiment import seed_everything, select_device
    if min(args.batch_size, args.max_hours) <= 0 or args.num_workers < 0:
        raise ValueError('Invalid resource settings')
    ancestor = Path(args.output_dir).resolve().parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    disk_free = shutil.disk_usage(ancestor).free/2**30
    if disk_free < args.min_free_disk_gb:
        raise RuntimeError('Insufficient free disk for new checkpoints: {:.2f} GiB'.format(disk_free))
    device = select_device(args.device)
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info(device)
        print('GPU free={:.2f}GiB total={:.2f}GiB'.format(free/2**30, total/2**30), flush=True)
        if free/2**30 < args.min_free_gb:
            raise RuntimeError('Insufficient free GPU budget; do not stop other users\' processes')
    seed_everything(args.seed)
    return device

def common_loss(outputs):
    return (1-F.cosine_similarity(outputs['common_receiver_semantic'].float(),
                                 outputs['common_text_semantic_target'].float(), dim=-1)).mean()

@torch.no_grad()
def evaluate(model, loader, device, profiles, snrs, seed=42, amp=True, deadline=None):
    from utils.experiment import seed_everything
    from utils.metrics import batch_image_metrics, token_accuracy
    result = {}
    model.eval()
    for profile in profiles:
        for snr in snrs:
            # Fixed draws for each model/epoch on a fixed list of images.
            seed_everything(seed+int((float(snr)+50)*100))
            sums = dict(psnr=0., ssim=0., common_cosine=0., token_correct=0., token_count=0., header_rejected=0.)
            n = 0
            for batch in loader:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError('Validation resource budget exhausted')
                images = batch['images'].to(device)
                with torch.cuda.amp.autocast(enabled=amp and device.type == 'cuda'):
                    out = model(images, batch['texts'], profile, snr)
                metrics = batch_image_metrics(out['image'], images)
                b = len(images)
                n += b
                for key in ('psnr', 'ssim'):
                    sums[key] += float(metrics[key].sum())
                sums['common_cosine'] += float((1-common_loss(out))*b)
                correct, count = token_accuracy(out['text_logits'], out['text_input_ids'], out['text_attention_mask'])
                sums['token_correct'] += correct
                sums['token_count'] += count
                sums['header_rejected'] += float(out['link_stats']['header_rejected'].float().sum())
            if n == 0:
                raise ValueError('Empty evaluation')
            token_acc = sums.pop('token_correct')/max(sums.pop('token_count'), 1)
            row = {k: v/n for k, v in sums.items()}
            row.update(samples=n, token_acc=token_acc)
            # Fixed, documented selection proxy, not a calibrated utility.
            row['score'] = row['psnr']+2*row['common_cosine']+token_acc
            result[profile+'/'+str(snr)] = row
    return result
