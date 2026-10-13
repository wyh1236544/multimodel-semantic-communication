"""Public experiment protocol; no Torch import in record verification."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from bidirectional_rx_v7 import ARMS, CONTROLS
from clip_bart_joint_v5.core import geometry, metrics, summarize

TEMPLATE = Path(__file__).with_name('config.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    return hashlib.sha256(np.asarray(array, dtype='<f4').tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def load(path=TEMPLATE):
    c = read(path); t = read(TEMPLATE)
    if set(c) != set(t) or c['schema'] != t['schema']: raise ValueError('Config schema differs')
    bounds = dict(seed=(0, 2**31-1), hidden=(128,128), heads=(4,4), depth=(2,2),
        train_samples=(8,2000), val_samples=(2,128), epochs=(1,12), diagnostic_samples=(8,32),
        diagnostic_val=(2,16), diagnostic_epochs=(2,16), grad_accum=(1,8), save_images=(0,8))
    for k,(lo,hi) in bounds.items():
        if type(c[k]) is not int or not lo <= c[k] <= hi: raise ValueError('Invalid '+k)
    for k in set(c)-set(bounds)-{'schema'}:
        if type(c[k]) not in (int,float) or not math.isfinite(c[k]) or c[k] <= 0: raise ValueError('Invalid '+k)
    if c['learning_rate'] > .001 or c['forward_tolerance'] > .001: raise ValueError('Unsafe numeric policy')
    if c['reserve_free_gib'] >= c['min_free_gib'] or c['private_drop_fraction'] >= 1: raise ValueError('Memory/drop budget')
    if c['diagnostic_samples'] > c['train_samples'] or c['save_images'] > c['val_samples']: raise ValueError('Subset budget')
    return c


def plan(c, mode):
    if mode == 'smoke': return dict(train_samples=8, val_samples=2, epochs=1, phases=['clean','fso'])
    if mode == 'diagnostic': return dict(train_samples=c['diagnostic_samples'], val_samples=c['diagnostic_val'], epochs=c['diagnostic_epochs'], phases=['clean','fso'])
    if mode in ('screen','pipeline'):
        return dict(train_samples=c['train_samples'], val_samples=c['val_samples'], epochs=c['epochs'], phases=['fso'] if mode=='screen' else ['clean','fso'])
    raise ValueError('Unknown run mode')


def mask_policy(c, step, image_id):
    rng = np.random.default_rng(c['seed']+1000003*step+int(image_id))
    # Offline TRAIN feature corruption, NOT packet loss or saved airtime.
    pattern = step%4
    return dict(image_slots=sorted(rng.choice(224,round(224*c['private_drop_fraction']),replace=False).tolist()) if pattern in (1,3) else [],
        text_slots=sorted(rng.choice(32,round(32*c['private_drop_fraction']),replace=False).tolist()) if pattern in (2,3) else [])


def variants(phases):
    out = [('B0_frozen','frozen','normal')]
    for phase in phases:
        for arm in ARMS:
            for control in (CONTROLS if arm == ARMS[-1] else ('normal',)):
                out.append((arm,phase,control))
    return out


def seal(root):
    return {str(f.relative_to(root)):digest(f) for f in sorted(root.rglob('*'))
        if f.is_file() and f.suffix!='.pth' and f.name not in ('COMPLETE.json','FAILED.json')}


def paired_ci(rows, arm, phase, target, seed=42):
    # Conditional on ONE trained checkpoint and this fixed validation set.
    base={(r['condition'],r['image_id']):r for r in rows if r['arm']=='R0_self' and r['phase']==phase and r['control']=='normal'}
    selected=[r for r in rows if r['arm']==arm and r['phase']==phase and r['control']=='normal' and r['condition'] in target]
    values={}
    for r in selected:
        b=base[(r['condition'],r['image_id'])]
        values.setdefault(r['image_id'],[]).append([r['psnr']-b['psnr'],r['normalized_wer']-b['normalized_wer']])
    if not values: return None
    x=np.asarray([np.mean(v,axis=0) for v in values.values()]); rng=np.random.default_rng(seed)
    boot=np.mean(x[rng.integers(0,len(x),(2000,len(x)))],axis=1)
    return dict(images=len(x), conditions=target, delta_psnr_db=float(x[:,0].mean()), delta_wer=float(x[:,1].mean()),
        psnr_ci95=np.percentile(boot[:,0],[2.5,97.5]).tolist(), wer_ci95=np.percentile(boot[:,1],[2.5,97.5]).tolist(),
        scope='paired image-cluster bootstrap; conditional one checkpoint; NOT multi-seed or independent test')
