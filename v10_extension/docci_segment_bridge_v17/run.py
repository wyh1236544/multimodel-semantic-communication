"""One finite GPU stage; no dependency changes or automatic restarts."""
import argparse
import os
from pathlib import Path
import shutil
import sys
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_segment_bridge_v17.core import (atomic, read, sha, safe, snapshot, seal, load,
                                          Budget, ARMS, CACHE_INDEX, check_source)


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action', choices=('prepare', 'train')); p.add_argument('output')
    p.add_argument('--cache'); p.add_argument('--cache-receipt'); p.add_argument('--inputs')
    p.add_argument('--arm', choices=ARMS); p.add_argument('--smoke', action='store_true')
    p.add_argument('--bart-path', default=str(EXT.parent/'pretrained/bart-base'))
    p.add_argument('--text-checkpoint'); p.add_argument('--max-hours', type=float, required=True)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or not 0 < a.max_hours <= 8:
        p.error('Explicit physical GPU1 and0..8h stage cap required')
    if a.action == 'prepare' and not (a.cache and a.cache_receipt) or a.action == 'train' and not (a.inputs and a.arm):
        p.error('prepare needs approved cache/receipt; train needs prepared inputs/arm')
    check_source(); c = load(a.smoke)
    if shutil.disk_usage(EXT.parent).free < (8 if a.smoke else c['min_disk_gib'])*2**30:
        raise RuntimeError('Project disk startup threshold not met')
    root = safe(a.output)
    if root.exists():
        raise FileExistsError('Existing stage preserved; choose a new output')
    a.output = str(root)
    import torch
    import transformers
    if torch.__version__.split('+')[0] != '1.12.0' or transformers.__version__ != '4.30.2' or not torch.cuda.is_available():
        raise RuntimeError('Use existing offline torch1.12.0/transformers4.30.2 GPU environment')
    device = torch.device('cuda:0'); torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    free, total = torch.cuda.mem_get_info(device)
    print('GPU1 free={:.2f}GiB total={:.2f}GiB'.format(free/2**30, total/2**30), flush=True)
    if free < c['min_free_gib']*2**30:
        raise RuntimeError('Startup free GPU reserve not met; do not stop unrelated jobs')
    budget = Budget(a.max_hours*3600, device)
    from docci_rx_bridge_v16.cache import text_model
    model, tokenizer, backbone = text_model(a, device)
    if a.action == 'prepare':
        from docci_segment_bridge_v17.prepare import prepare
        prepare(a, model, tokenizer, device, budget)
    else:
        from docci_segment_bridge_v17.prepare import Prepared
        from docci_segment_bridge_v17.train import fit
        from docci_segment_bridge_v17.evaluate import evaluate
        inputs = Path(a.inputs).resolve(strict=True)
        receipt = read(inputs/'VERIFIED.json')
        if not receipt['arrays_verified'] or receipt['smoke'] != a.smoke or read(inputs/'config.json') != c:
            raise ValueError('Prepared input verification/config differs')
        snapshot(root, dict(stage='arm', arm=a.arm, smoke=a.smoke, inputs=str(inputs),
                            inputs_light_sha256=sha(inputs/'LIGHT_MANIFEST.json'), physical_gpu=1,
                            python=sys.version, torch=torch.__version__, transformers=transformers.__version__,
                            FSO=False, Common=False, diffusion=False, fixed_final=True))
        atomic(root/'config.json', c)
        prepared = Prepared(inputs, device)
        bridge = fit(model, prepared, a.arm, root, device, budget, a.smoke)
        evaluate(model, bridge, tokenizer, prepared, a.arm, root, device, budget, a.smoke)
        seal(root)
    print('V17 stage train/evaluation finished; independent CPU verification still required', flush=True)


if __name__ == '__main__':
    main()
