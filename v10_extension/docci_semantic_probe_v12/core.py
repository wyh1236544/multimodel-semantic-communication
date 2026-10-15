"""Strict experiment/identity contracts; no Torch, model downloads or pickle."""
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np

FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parents[1]
SCHEMA = 'docci-semantic-probe-v12'
TEXT_ARMS = ('B0_ce', 'B1_ce_kd')
IMAGE_ARMS = ('A0_image_only', 'A1_oracle_text')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1048576), b''): h.update(data)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def safe(path, root=PROJECT):
    root = Path(root).resolve(strict=True)
    path = Path(path)
    if not path.is_absolute(): path = root / path
    try: parts = path.absolute().relative_to(root).parts
    except ValueError: raise ValueError('Write outside project rejected')
    if not parts or any(x in ('.', '..') for x in parts): raise ValueError('Unsafe write path')
    at = root
    for part in parts:
        at = at / part
        if at.is_symlink(): raise ValueError('Symlink write path rejected: ' + str(at))
    return path


def output_path(path):
    path = safe(path)
    rel = path.relative_to(PROJECT)
    if len(rel.parts) != 2 or rel.parts[0] != 'experiment_results' or not rel.name.startswith('v10_'):
        raise ValueError('Use a NEW direct experiment_results/v10_* directory')
    return path


def atomic_json(path, value):
    path = safe(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.part-' + str(os.getpid()))
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def jsonlines(path):
    with Path(path).open(encoding='utf-8') as f: return [json.loads(s) for s in f if s.strip()]


def load(path=None):
    template = read(FOLDER / 'config.json'); c = read(path or FOLDER / 'config.json')
    if set(c) != set(template) or c['schema'] != SCHEMA: raise ValueError('Configuration keys/schema differ')
    bounds = dict(seed=(0, 2**31-1), train_samples=(32, 2048), val_samples=(2, 100),
        text_updates=(2, 20000), text_batch_size=(1, 2), image_updates=(2, 10000), image_grad_accum=(1, 4),
        warmup_updates=(1, 500), diagnostic_samples=(2, 16), train_eval_samples=(2, 32), save_images=(0, 8))
    for k, (lo, hi) in bounds.items():
        if type(c[k]) is not int or not lo <= c[k] <= hi: raise ValueError('Invalid ' + k)
    if c['seed'] != 42: raise ValueError('This matched approved-source probe fixes seed42; no multi-seed sweep')
    for k in ('text_learning_rate', 'image_learning_rate'):
        if type(c[k]) not in (int, float) or not math.isfinite(c[k]) or not 0 < c[k] <= .001: raise ValueError('Invalid LR')
    for k in ('kd_weight', 'image_latent_weight', 'weight_decay', 'final_lr_fraction'):
        if type(c[k]) not in (int, float) or not math.isfinite(c[k]) or not 0 < c[k] < 1: raise ValueError('Invalid ' + k)
    if type(c['kd_temperature']) not in (int, float) or not 1 <= c['kd_temperature'] <= 4: raise ValueError('Invalid temperature')
    if not 0 < c['reserve_free_gib'] < c['min_free_gib'] <= 48: raise ValueError('Invalid GPU limits')
    if c['save_images'] > c['val_samples'] or max(c['diagnostic_samples'], c['train_eval_samples']) > c['train_samples']:
        raise ValueError('Sample limits')
    for key in ('image_train_conditions', 'image_eval_conditions'):
        if c[key] != ['clean', 'mask25']: raise ValueError('Fixed two diagnostic conditions required')
    if c['image_primary_condition'] != 'mask25': raise ValueError('Primary condition fixed before experiment')
    return c


def lr(c, task, update, updates):
    warm = min(c['warmup_updates'], max(1, updates // 4))
    decay = .5 * (1 + math.cos(math.pi * update / max(1, updates - 1)))
    return c[task + '_learning_rate'] * min(1., (update + 1) / warm) * (c['final_lr_fraction'] + (1-c['final_lr_fraction']) * decay)


def seed_for(*items):
    return int(hashlib.sha256(json.dumps(items, separators=(',', ':')).encode()).hexdigest()[:8], 16)


def indices(count, size, update, seed):
    result = []
    for cursor in range(update * size, (update + 1) * size):
        order = np.random.default_rng(seed_for(seed, 'v12-order', cursor // count)).permutation(count)
        result.append(int(order[cursor % count]))
    return result


def damage_numpy(wire, condition, seed):
    """Source-independent receiver erasure diagnostic, NOT an FSO simulator."""
    x = np.asarray(wire, dtype=np.float32)
    if x.ndim != 1 or not np.isfinite(x).all(): raise ValueError('Finite flat wire required')
    if condition == 'clean': return x.copy()
    if condition != 'mask25': raise ValueError('Unknown diagnostic condition')
    positions = np.random.default_rng(seed).permutation(len(x))[:len(x)//4]
    out = x.copy(); out[positions] = 0.
    return out


def check_sources():
    m = read(FOLDER / 'MANIFEST.json')
    if m['schema'] != SCHEMA + '-source': raise ValueError('Wrong release manifest')
    for name, sha in m['files'].items():
        if Path(name).name != name or digest(FOLDER / name) != sha: raise ValueError('New source drift: ' + name)
    for name, sha in m['required_sources'].items():
        if digest(safe(name)) != sha: raise ValueError('Historical source drift: ' + name)
    return m


def parent(kind, override=None):
    spec = read(FOLDER / 'parents.json')[kind]
    root = Path(override) if override else PROJECT / spec['default_run']
    root = root.resolve(strict=True)
    for name, sha in spec['files'].items():
        p = root / name
        if p.is_symlink() or not p.is_file() or digest(p) != sha: raise ValueError('Approved ' + kind + ' record differs: ' + name)
    checkpoint = root / spec['checkpoint']
    if checkpoint.is_symlink() or not checkpoint.is_file() or digest(checkpoint) != spec['checkpoint_sha256']:
        raise ValueError('Approved checkpoint hash differs BEFORE pickle loading: ' + str(checkpoint))
    identity = read(root / 'run_identity.json'); config = read(root / 'config.json')
    return dict(root=root, checkpoint=checkpoint, spec=spec, identity=identity, config=config,data=read(root/'data_identity.json'))


def compare_pairs(a, b, key):
    """Difference a-b, description-level conditional bootstrap; not seed uncertainty."""
    if set(a) != set(b) or len(a) < 2: raise ValueError('Unmatched/empty comparison')
    d = np.asarray([a[k][key]-b[k][key] for k in sorted(a)], dtype=np.float64)
    if not np.isfinite(d).all(): raise ValueError('Nonfinite comparison')
    rng = np.random.default_rng(42)
    means = np.asarray([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    return dict(n=len(d),mean=float(d.mean()),CI95=np.quantile(means, [.025, .975]).tolist(),
        positive=int((d>0).sum()),negative=int((d<0).sum()),tied=int((d==0).sum()),
        scope='paired reused qual_dev descriptions, one trained seed; NOT independent-test or training-seed CI')
