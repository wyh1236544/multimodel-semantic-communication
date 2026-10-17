"""Pure experiment contracts; no Torch, downloads, or checkpoint loading."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np

FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parents[1]
ARMS = ('A_self', 'B_tx_text', 'C_tx_rx_text')
IMAGE_SHA = '95c05e3b374deae2a3bd10603b8e95ebbd464456f7581a054f2b80597be5923b'
TEXT_SHA = '96b27347b62c918a931ba221640139ce3e162b8ff8fbcaebde89f6b13cc5312b'
DOCCI_SHA = 'c9df4819963883af35ddd2cf257949892fd8c6d88b33a012094352df60719800'
GEOMETRY = dict(image_reals=8192, text_reals=18432, payload_complex_symbols=13312,
                analog_repeats=1, common_reals=0, pilot_complex_symbols=64,
                total_complex_uses=13376, waveform=False, FSO=False, BER=None)

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def rows(p):
    with Path(p).open(encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def safe(p):
    p = Path(p)
    p = p if p.is_absolute() else PROJECT/p
    if '..' in p.parts:
        raise ValueError('Traversal rejected')
    rel = p.absolute().relative_to(PROJECT.resolve())
    if not rel.parts:
        raise ValueError('Project root is not an output')
    at = PROJECT
    for part in rel.parts:
        at = at/part
        if at.is_symlink():
            raise ValueError('Symlink write rejected')
    return p

def atomic(p, value):
    p = safe(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.part-'+str(os.getpid()))
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(p))

def seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], 'big') % (2**31-1)

def load(path=None, smoke=False):
    default = read(FOLDER/'config.json')
    c = read(path) if path else default.copy()
    if set(c) != set(default) or c['schema'] != default['schema']:
        raise ValueError('Unknown config schema/keys')
    for k in ('seed','train_samples','holdout_candidates','dev_samples','epochs','accumulation',
              'evaluation_repeats','log_every','save_examples','bootstrap_draws'):
        if type(c[k]) is not int or c[k] <= 0:
            raise ValueError('Positive integer required: '+k)
    for k in ('learning_rate','weight_decay','clean_fraction','latent_weight','lpips_weight',
              'minimum_free_gib','reserve_free_gib','minimum_disk_gib'):
        if type(c[k]) not in (int,float) or not math.isfinite(c[k]) or c[k] <= 0:
            raise ValueError('Finite positive value required: '+k)
    if not (c['seed']==42 and 2048 <= c['train_samples'] <= 9147 and c['holdout_candidates'] == 500
            and c['dev_samples'] == 100 and 1 <= c['epochs'] <= 32
            and c['accumulation'] in (2,4,8) and c['evaluation_repeats'] == 3
            and c['bootstrap_draws'] == 10000 and c['save_examples'] <= 8):
        raise ValueError('Fixed cohort/protocol bounds')
    if (c['training_snr_db'] != [0,5,10,15,20]
            or c['evaluation_snr_db'] != [None,10,5,0] or c['clean_fraction'] != .25):
        raise ValueError('Fixed channel matrix')
    if not (c['learning_rate'] <= .0001 and c['weight_decay'] < 1
            and c['lpips_weight'] <= .1 and c['latent_weight'] == .01
            and 20 <= c['minimum_free_gib'] <= 48 and c['reserve_free_gib'] == 4
            and c['minimum_disk_gib'] >= 64):
        raise ValueError('Learning/resource safeguards')
    if smoke:
        c.update(train_samples=8, dev_samples=2, epochs=1, accumulation=2,
                 evaluation_repeats=1, save_examples=2)
    return c

def split_records(train, validation, used_ids, c, smoke=False):
    if len(train) != 9647 or len(validation) != 100:
        raise ValueError('Complete official train/qual_dev required')
    if len({r['example_id'] for r in train+validation}) != 9747:
        raise ValueError('Duplicate source ID')
    # Text exact dedup plus actual image-byte dedup is completed before caching.
    val_hashes = {r['text_sha256'] for r in validation}
    pool = [r for r in train if r['example_id'] not in used_ids and r['text_sha256'] not in val_hashes]
    pool.sort(key=lambda r: seed(c['seed'], 'new-dev-v20', r['example_id']))
    holdout = pool[:c['holdout_candidates']]
    if len(holdout) != c['holdout_candidates']:
        raise ValueError('Insufficient known-parent-unseen development candidates')
    blocked = {r['text_sha256'] for r in holdout} | val_hashes
    candidates = [r for r in train if r['text_sha256'] not in blocked]
    selected = candidates[:c['train_samples']]
    if len(selected) != c['train_samples']:
        raise ValueError('Not enough training pairs after protected split')
    return dict(train=selected, dev=holdout[:c['dev_samples']],
                qual_dev=validation[:2 if smoke else 100], heldout=holdout)

def batches(n, c):
    for epoch in range(c['epochs']):
        order = list(range(n))
        random.Random(seed(c['seed'], 'order', epoch)).shuffle(order)
        for start in range(0, n, c['accumulation']):
            yield epoch, order[start:start+c['accumulation']]

def train_snr(c, epoch, eid):
    n = seed(c['seed'], 'train-snr', epoch, eid)
    return None if n % 4 == 0 else c['training_snr_db'][(n//4) % 5]

def trial_matrix(c, arm, smoke=False):
    for s in ([None,0] if smoke else c['evaluation_snr_db']):
        for repeat in range(1 if s is None else c['evaluation_repeats']):
            yield s, repeat, 'matched'
    if arm == 'C_tx_rx_text':
        for s in (None,0):
            for control in ('rx_shuffled','rx_zero','rx_oracle'):
                yield s, 0, control

def bootstrap(values, draws=10000, level=.95):
    x = np.asarray(values, np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('At least two finite independent-description values required')
    rng = np.random.default_rng(42)
    means = np.empty(draws)
    for start in range(0, draws, 200):
        n = min(200, draws-start)
        means[start:start+n] = x[rng.integers(0,len(x),(n,len(x)))].mean(1)
    q = (1-level)/2
    return dict(mean=float(x.mean()), ci=np.quantile(means,[q,1-q]).tolist(), descriptions=len(x))

def check_source():
    m = read(FOLDER/'MANIFEST.json')
    for base, files in ((FOLDER,m['files']),(PROJECT,m['dependencies'])):
        for n,h in files.items():
            if (base/n).is_symlink() or sha(base/n) != h:
                raise ValueError('Source/dependency changed: '+n)
    return m

class Budget:
    def __init__(self, hours, device=None):
        if not math.isfinite(hours) or not 0 < hours <= 48:
            raise ValueError('Bounded 0..48h budget required')
        self.deadline = time.monotonic()+hours*3600
        self.device = device

    def __call__(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Budget exhausted; preserve partial run')
        if shutil.disk_usage(PROJECT).free < 4*2**30:
            raise RuntimeError('Project disk reserve crossed')
        if self.device is not None:
            import torch
            if torch.cuda.mem_get_info(self.device)[0] < 4*2**30:
                raise RuntimeError('GPU reserve crossed; do not stop other jobs')
