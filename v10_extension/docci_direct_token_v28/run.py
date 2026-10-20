"""One offline stage. Fresh outputs; no silent restart or dependency changes."""
import argparse
import math
import os
from pathlib import Path
import shutil
import sys
import time
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_direct_token_v28.core import PROJECT, load, atomic, safe, check_source
from docci_direct_token_v28.data import arguments, audit, items


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action', choices=('audit', 'train', 'evaluate'))
    p.add_argument('output'); p.add_argument('--task', choices=('image', 'text'), required=True)
    p.add_argument('--arm', choices=('frozen', 'normal_range', 'low_snr'))
    p.add_argument('--inputs'); p.add_argument('--checkpoint'); p.add_argument('--config')
    p.add_argument('--profile', choices=('smoke', 'pilot', 'full'), required=True)
    p.add_argument('--max-hours', type=float, required=True)
    p.add_argument('--gpu', choices=('0', '1'), default='1'); arguments(p)
    a = p.parse_args(); check_source(); c = load(a.config, a.profile)
    if not math.isfinite(a.max_hours) or not 0 < a.max_hours <= 48:
        p.error('Finite positive time budget <=48 hours required')
    root = safe(a.output)
    if root.exists(): raise FileExistsError('Existing output preserved')
    started = time.monotonic()
    def budget():
        if time.monotonic()-started > a.max_hours*3600: raise TimeoutError('Stage budget exhausted')
        if shutil.disk_usage(PROJECT).free < 4*2**30: raise RuntimeError('Disk reserve crossed')
    if a.action == 'audit':
        audit(a, c, a.task, root, budget)
        return
    if not a.inputs or not a.arm or (a.action == 'train' and a.arm == 'frozen'):
        p.error('Verified inputs and explicit arm required')
    if a.action == 'evaluate' and (a.arm != 'frozen') != bool(a.checkpoint):
        p.error('Only adapted arms require a saved V28 checkpoint')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != a.gpu:
        raise ValueError('Selected physical GPU does not match isolated environment')
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('Actual saved-weight CUDA smoke required')
    torch.set_num_threads(4); torch.manual_seed(c['seed'])
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    free, total = torch.cuda.mem_get_info(device)
    print('GPU{} free={:.2f}GiB total={:.2f}GiB'.format(a.gpu, free/2**30, total/2**30), flush=True)
    if free < c['minimum_free_gib']*2**30:
        raise RuntimeError('Insufficient starting free VRAM; do not stop other jobs')
    base_budget = budget
    def budget():
        base_budget()
        if torch.cuda.mem_get_info(device)[0] < c['reserve_free_gib']*2**30:
            raise RuntimeError('Free GPU reserve crossed; no automatic batch change')
    inputs = safe(a.inputs)
    from docci_direct_token_v28.engine import fit, evaluate
    train = items(inputs, 'train', c); val = items(inputs, 'qual_dev', c)
    if a.action == 'train': fit(a, c, a.task, a.arm, train, val, inputs, root, device, budget)
    else: evaluate(a, c, a.task, a.arm, val, inputs, root, device, budget, a.checkpoint)
    check_source()
    print('V28 {} {} {} finished; independent verification remains'.format(a.task, a.arm, a.action), flush=True)


if __name__ == '__main__': main()
