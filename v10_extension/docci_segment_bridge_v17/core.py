"""Frozen protocol, deterministic common random numbers; no neural imports."""
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

from docci_rx_bridge_v16.core import PROJECT, safe, sha, read, atomic, lines, batches

FOLDER = Path(__file__).resolve().parent
ARMS = ('A_self_clean', 'B_oracle_clean', 'C_self_mixed', 'D_oracle_mixed')
RUN_ORDER = ('C_self_mixed', 'D_oracle_mixed', 'A_self_clean', 'B_oracle_clean')
CONDITIONS = ('clean', 'awgn_10', 'awgn_0')
TRAIN_CONDITIONS = ('clean', 'awgn_20', 'awgn_10', 'awgn_5', 'awgn_0')
CONTROLS = ('matched', 'donor1', 'donor17', 'donor43', 'donor61', 'donor83', 'zero', 'off')
CACHE_INDEX = 'ea78989f97a3db56b21510324ee2555245e59c9e8a346bb675cac3ce2a083d23'


def load(smoke=False):
    c = read(FOLDER/'config.json')
    if c['schema'] != 'docci-segment-bridge-v17' or (c['train_samples'], c['val_samples'], c['width'], c['slots'], c['heads']) != (9647, 100, 256, 64, 4):
        raise ValueError('Fixed V17 protocol changed')
    if any(not math.isfinite(v) or v <= 0 for k, v in c.items() if k != 'schema'):
        raise ValueError('Finite positive settings required')
    if smoke:
        c.update(train_samples=8, val_samples=2, alignment_epochs=1, alignment_batch=4,
                 ce_epochs=1, ce_accumulation=1)
    return c


def kind(arm):
    if arm not in ARMS:
        raise ValueError('Unknown predeclared arm')
    return 'self' if '_self_' in arm else 'oracle_clip'


def seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).digest()[:8], 'big') % (2**31-1)


def condition(arm, phase, epoch, eid):
    kind(arm)
    return TRAIN_CONDITIONS[seed('v17-condition', phase, epoch, eid) % len(TRAIN_CONDITIONS)] if arm.endswith('_mixed') else 'clean'


def noise_seed(phase, epoch, eid, channel):
    return seed('v17-wire-noise', phase, epoch, eid, channel)


def donor_index(index, n, control):
    if not control.startswith('donor') or n < 2:
        raise ValueError('Real different donor required')
    offset = int(control[5:])
    if offset % n == 0:
        raise ValueError('Donor would equal source')
    return (index+offset) % n


def eval_conditions(smoke):
    return ('clean', 'awgn_0') if smoke else CONDITIONS


def controls(smoke):
    return ('matched', 'donor1', 'zero', 'off') if smoke else CONTROLS


def expected_parameters(dim=768, width=256, slots=64):
    return 52*width*width+5*dim*width+(slots+67)*width+7*dim+2


def parameter_groups(dim=768, width=256, slots=64):
    d, w, s = dim, width, slots
    return dict(resampler=24*w*w+d*w+(s+27)*w+2*d,
                repair=24*w*w+d*w+36*w+2*d, memory_output=w*d+2*w+d,
                residual_gate=1, matching=4*w*w+2*w+1,
                semantic_image=w*d+d, semantic_segment=w*d+d)


def geometry(arm):
    return dict(payload_reals=26624, text_reals=18432, image_reals=8192,
                complex_symbol_count=13312, repeats=1, Common=False, waveform=False,
                oracle=kind(arm) == 'oracle_clip', image_channel='ideal fixed diagnostic',
                text_channel='unit-power real AWGN or clean; not FSO',
                teacher_or_source_mask_at_receiver=False)


def check_source():
    from docci_rx_bridge_v16_fix1.compat import source_check
    source_check()
    m = read(FOLDER/'MANIFEST.json')
    for name, digest in m['files'].items():
        p = FOLDER/name
        if p.is_symlink() or sha(p) != digest:
            raise ValueError('V17 source changed: '+name)
    for name, digest in m['dependencies'].items():
        p = PROJECT/name
        if p.is_symlink() or sha(p) != digest:
            raise ValueError('Historical dependency changed: '+name)
    return m


def snapshot(root, identity):
    m = check_source()
    root.mkdir(parents=True, exist_ok=False)
    for name in list(m['files'])+['MANIFEST.json']:
        target = root/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FOLDER/name, target)
    for name in m['dependencies']:
        target = root/'dependency_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT/name, target)
    atomic(root/'identity.json', dict(identity, source_manifest_sha256=sha(FOLDER/'MANIFEST.json')))


def seal(root):
    excluded = ('FINISHED.json', 'LIGHT_MANIFEST.json', 'VERIFIED.json')
    hashes = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*'))
              if p.is_file() and p.name not in excluded and p.suffix not in ('.npz', '.pth')}
    atomic(root/'LIGHT_MANIFEST.json', hashes)
    atomic(root/'FINISHED.json', dict(light_manifest_sha256=sha(root/'LIGHT_MANIFEST.json'),
                                     independent_verification_pending=True))


def inside(root, name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('Unsafe artifact path')
    q = root/p
    if q.is_symlink() or any(v.is_symlink() for v in q.parents):
        raise ValueError('Symlink artifact')
    q.resolve(strict=True).relative_to(root.resolve())
    return q


def verify_seal(root, budget):
    if read(root/'FINISHED.json')['light_manifest_sha256'] != sha(root/'LIGHT_MANIFEST.json'):
        raise ValueError('Stage seal differs')
    for name, digest in read(root/'LIGHT_MANIFEST.json').items():
        budget()
        if sha(inside(root, name)) != digest:
            raise ValueError('Sealed artifact changed: '+name)
    m = read(root/'source_snapshot/MANIFEST.json')
    if sha(root/'source_snapshot/MANIFEST.json') != sha(FOLDER/'MANIFEST.json'):
        raise ValueError('Different V17 source release')
    for name, digest in m['files'].items():
        if sha(inside(root, 'source_snapshot/'+name)) != digest:
            raise ValueError('Source snapshot changed')
    for name, digest in m['dependencies'].items():
        if sha(inside(root, 'dependency_snapshot/'+name)) != digest:
            raise ValueError('Dependency snapshot changed')


class Budget:
    def __init__(self, seconds, device=None):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Finite positive budget')
        self.deadline, self.device = time.monotonic()+seconds, device

    def __call__(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Owned stage budget exhausted; preserve partial run')
        if shutil.disk_usage(PROJECT).free < 4*2**30:
            raise RuntimeError('Disk reserve crossed; no deletion')
        if self.device is not None:
            import torch
            if torch.cuda.mem_get_info(self.device)[0] < 4*2**30:
                raise RuntimeError('GPU reserve crossed; no automatic batch change')
