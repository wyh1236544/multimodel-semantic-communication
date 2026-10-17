"""Reusable configuration-driven text channel experiment; no neural imports."""
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
ARMS = ('clean_continue', 'noise_replay')
PARENT_SHA = '3e6743eb1bcdbe4bed5db14ba9c9d5cb6d30cb2676066fc41c63fbb09ee3b000'
PARAMETERS = 25618432

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()
def safe(p):
    p = Path(p)
    p = p if p.is_absolute() else PROJECT/p
    if '..' in p.parts: raise ValueError('Traversal rejected')
    rel = p.absolute().relative_to(PROJECT.resolve())
    if not rel.parts: raise ValueError('Project root is not an output')
    at = PROJECT
    for part in rel.parts:
        at = at/part
        if at.is_symlink(): raise ValueError('Symlink output rejected')
    return p
def atomic(p, value):
    p = safe(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.part-'+str(os.getpid()))
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
        f.flush(); os.fsync(f.fileno())
    os.replace(str(tmp), str(p))
def seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], 'big') % (2**31-1)
def batches(n, accumulation, epochs, value):
    for epoch in range(epochs):
        ix = list(range(n)); random.Random(seed(value, 'order', epoch)).shuffle(ix)
        for start in range(0, n, accumulation): yield epoch, ix[start:start+accumulation]
def channel_for(c, epoch, eid):
    return c['training_snr_db'][seed(c['seed'], 'train-channel', epoch, eid) % len(c['training_snr_db'])]
def noise(width, snr, value):
    if snr is None: return np.zeros(width, dtype=np.float32)
    if not math.isfinite(snr) or not -10 <= snr <= 40: raise ValueError('Bounded real SNR required')
    x = np.random.Generator(np.random.PCG64(value)).standard_normal(width).astype(np.float32)
    return x*np.float32(10.**(-snr/20.))
def trial_seed(c, phase, epoch, eid, snr, repeat=0):
    return seed(c['seed'], 'v18-epsilon', phase, epoch, eid, snr, repeat)
def load(path=None, smoke=False):
    c = read(path or FOLDER/'config.json'); default = read(FOLDER/'config.json')
    if set(c) != set(default) or c['schema'] != default['schema']: raise ValueError('Unknown config schema/keys')
    ints = ('seed','train_samples','validation_samples','epochs','accumulation','evaluation_repeats','lru','log_every','bootstrap_draws')
    if any(type(c[k]) is not int or c[k] <= 0 for k in ints): raise ValueError('Positive integer settings required')
    for k in ('learning_rate','weight_decay','clean_weight','minimum_free_gib','reserve_free_gib','minimum_disk_gib'):
        if type(c[k]) not in (int,float) or not math.isfinite(c[k]) or c[k] <= 0: raise ValueError('Invalid '+k)
    if (c['train_samples'],c['validation_samples']) != (2048,100): raise ValueError('Pinned V14 cohort; no silent dataset change')
    if c['training_snr_db'] != [0,5,10,15,20] or c['evaluation_snr_db'] != [20,15,10,5,0]: raise ValueError('Predeclared SNR matrix')
    if not (c['epochs'] <= 30 and c['accumulation'] <= 16 and c['learning_rate'] <= .0001 and c['weight_decay'] < 1 and c['clean_weight'] == .25 and c['evaluation_repeats'] == 3 and c['lru'] <= 64 and c['bootstrap_draws'] == 10000): raise ValueError('Protocol bounds')
    if not 4 <= c['reserve_free_gib'] < c['minimum_free_gib'] <= 48 or c['minimum_disk_gib'] < 16: raise ValueError('Resource floors cannot be weakened')
    if smoke: c.update(train_samples=8,validation_samples=2,epochs=1,accumulation=2,evaluation_repeats=1)
    return c
def trials(c, smoke=False):
    yield None, 0, 'matched'
    for s in ([0] if smoke else c['evaluation_snr_db']):
        for repeat in range(c['evaluation_repeats']): yield s,repeat,'matched'
    for s in (None,0):
        for control in ('shuffled','zero'): yield s,0,control
def snapshot(root):
    root = safe(root); root.mkdir(parents=True, exist_ok=False)
    m = check_source()
    for name in list(m['files'])+['MANIFEST.json']:
        out = root/'source_snapshot'/name; out.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(FOLDER/name,out)
    for name in m['dependencies']:
        out = root/'dependency_snapshot'/name; out.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(PROJECT/name,out)
def check_source():
    m = read(FOLDER/'MANIFEST.json')
    for base, spec in ((FOLDER,m['files']),(PROJECT,m['dependencies'])):
        for n,h in spec.items():
            p=base/n
            if p.is_symlink() or not p.is_file() or sha(p)!=h: raise ValueError('Source missing/changed: '+n)
    return m
def seal(root):
    excluded={'SEALED.json','LIGHT_MANIFEST.json','VERIFIED.json'}
    files={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in excluded and p.suffix not in ('.pth','.npz')}
    atomic(root/'LIGHT_MANIFEST.json',files)
    atomic(root/'SEALED.json',dict(manifest_sha256=sha(root/'LIGHT_MANIFEST.json')))
class Budget:
    def __init__(self,hours,device=None,reserve=4):
        if not math.isfinite(hours) or not 0 < hours <= 12: raise ValueError('Finite bounded hours')
        self.deadline=time.monotonic()+hours*3600; self.device=device; self.reserve=reserve
    def __call__(self):
        if time.monotonic() >= self.deadline: raise TimeoutError('Stage budget exhausted')
        if shutil.disk_usage(PROJECT).free < 4*2**30: raise RuntimeError('Disk reserve crossed')
        if self.device is not None:
            import torch
            if torch.cuda.mem_get_info(self.device)[0] < self.reserve*2**30: raise RuntimeError('Free GPU reserve crossed; no auto retry')
