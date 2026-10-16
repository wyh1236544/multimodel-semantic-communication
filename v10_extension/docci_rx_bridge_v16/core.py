"""Immutable contracts. This module imports neither Torch nor a model loader."""
import hashlib
import json
import math
import os
from pathlib import Path
import random

FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parents[1]
ARMS = ('self', 'oracle_clip', 'rx_clip', 'rx_i0')
CONTROLS = ('matched', 'shuffled', 'zero', 'off')
DOCCI_SHA256 = 'c9df4819963883af35ddd2cf257949892fd8c6d88b33a012094352df60719800'
PARENTS = {
    'text': dict(path='experiment_results/v14_text_pilot_01/main/text/released64/train/last.pth',
                 sha256='3e6743eb1bcdbe4bed5db14ba9c9d5cb6d30cb2676066fc41c63fbb09ee3b000'),
    'image': dict(path='experiment_results/v14_image_pilot_02/main/image/expanded/train/last.pth',
                  sha256='95c05e3b374deae2a3bd10603b8e95ebbd464456f7581a054f2b80597be5923b')}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1048576), b''):
            h.update(data)
    return h.hexdigest()


def safe(path):
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT / p
    if '..' in p.parts:
        raise ValueError('Traversal rejected')
    rel = p.absolute().relative_to(PROJECT.resolve())
    if not rel.parts:
        raise ValueError('Project root is not an artifact')
    at = PROJECT
    for part in rel.parts:
        at = at / part
        if at.is_symlink():
            raise ValueError('Symlink write rejected')
    return p


def atomic(path, value):
    p = safe(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.part-'+str(os.getpid()))
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(p))


def lines(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load(path=None, smoke=False):
    default = read(FOLDER/'config.json')
    c = path.copy() if isinstance(path, dict) else read(path) if path else default.copy()
    if set(c) != set(default) or c['schema'] != default['schema']:
        raise ValueError('Configuration schema/keys differ')
    for k, v in c.items():
        if k != 'schema' and (type(v) not in (int, float) or not math.isfinite(v) or v <= 0):
            raise ValueError('Finite positive configuration required: '+k)
    integers = ('seed', 'train_samples', 'val_samples', 'width', 'slots', 'heads',
                'alignment_epochs', 'alignment_batch', 'ce_epochs', 'ce_accumulation',
                'memory_lru', 'log_every', 'save_examples', 'bootstrap_draws')
    if any(type(c[k]) is not int for k in integers):
        raise ValueError('Integer configuration required')
    if (c['train_samples'], c['val_samples'], c['width'], c['slots'], c['heads']) != (9647, 100, 256, 64, 4):
        raise ValueError('Full TRAIN and fixed bridge geometry required; smoke is explicit')
    if not (c['alignment_batch'] in (32, 64, 128, 256) and 1 <= c['ce_accumulation'] <= 16
            and 1 <= c['ce_epochs'] <= 20 and 1 <= c['alignment_epochs'] <= 10
            and 1 <= c['memory_lru'] <= 64 and 10000 <= c['bootstrap_draws'] <= 100000
            and c['seed'] <= 2**31-1 and c['log_every'] <= 1000):
        raise ValueError('Training/RAM/time bounds')
    if not 0 < c['reserve_free_gib'] < c['min_free_gib'] <= 48 or c['reserve_free_gib'] < 4:
        raise ValueError('GPU reserve cannot be weakened below4GiB')
    if c['min_disk_gib'] < 64 or max(c['alignment_lr'], c['ce_lr']) > .001:
        raise ValueError('Disk/LR bounds')
    if c['save_examples'] > 8 or c['temperature'] > 1 or c['weight_decay'] >= 1:
        raise ValueError('Fraction/output bounds')
    if (c['minimum_wer_gain_pp'], c['minimum_donor_gain_pp']) != (2., 1.):
        raise ValueError('Predeclared screen thresholds changed; version protocol first')
    if smoke:
        c.update(train_samples=8, val_samples=2, alignment_epochs=1,
                 alignment_batch=4, ce_epochs=1, ce_accumulation=1, save_examples=2)
    return c


def geometry(arm):
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    return dict(image_reals=8192, text_reals=18432, payload_reals=26624,
                payload_complex_symbols=13312, repeats=1, common_reals=0,
                condition='ideal_clean', snr_db=None, waveform_simulated=False,
                auxiliary_is_privileged=arm == 'oracle_clip',
                equal_resource_communication=arm != 'oracle_clip',
                original_source_length_or_mask_at_RX=False,
                physical_overheads_BER_and_rate=None)


def parameter_groups(c, dim=768):
    """Independent closed-form count; Torch runtime must agree, not a measurement."""
    w, s = c['width'], c['slots']
    resampler = 2*dim+dim*w+w+s*w+24*w*w+26*w
    cross = 4*dim*dim+w*dim+7*dim+1
    return dict(resampler=resampler, cross0=cross, cross1=cross)


def order(n, seed, epoch):
    ix = list(range(n))
    random.Random('{}:v16:{}'.format(seed, epoch)).shuffle(ix)
    return ix


def batches(n, batch, epochs, seed):
    for epoch in range(epochs):
        ix = order(n, seed, epoch)
        for start in range(0, n, batch):
            yield epoch, ix[start:start+batch]


def check_source():
    m = read(FOLDER/'MANIFEST.json')
    for name, digest in m['files'].items():
        p = FOLDER/name
        if p.is_symlink() or not p.is_file() or sha(p) != digest:
            raise ValueError('V16 source changed: '+name)
    for name, digest in m['dependencies'].items():
        p = PROJECT/name
        if p.is_symlink() or not p.is_file() or sha(p) != digest:
            raise ValueError('Historical dependency missing/changed: '+name)
    return m


def seal(root, name='FINISHED.json'):
    excluded = {name, 'LIGHT_MANIFEST.json', 'VERIFIED.json'}
    light = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*'))
             if p.is_file() and p.name not in excluded and p.suffix not in ('.pth', '.npz')}
    atomic(root/'LIGHT_MANIFEST.json', light)
    atomic(root/name, dict(manifest_sha256=sha(root/'LIGHT_MANIFEST.json'),
                          independent_verification_pending=True, quality_success=False))
