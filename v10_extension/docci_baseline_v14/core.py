"""Public clean contracts, bounded data expansion and immutable artifacts; stdlib."""
import hashlib
import json
import math
import os
from pathlib import Path
import random

FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parents[1]
SCHEMA = 'docci-baseline-v14'
ARMS = ('legacy64', 'released64', 'wide768')
DECODE = {
    'greedy': dict(num_beams=1, no_repeat_ngram_size=0, repetition_penalty=1.),
    'deployment': dict(num_beams=4, no_repeat_ngram_size=3, repetition_penalty=1.2),
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def safe(path, project=PROJECT):
    root = Path(project).resolve(strict=True)
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    # Check lexical traversal before resolution, then every existing ancestor.
    if '..' in p.parts:
        raise ValueError('Traversal rejected')
    try:
        rel = p.absolute().relative_to(root)
    except ValueError:
        raise ValueError('All writes must remain inside multimodel')
    if not rel.parts:
        raise ValueError('Project root is not a writable artifact target')
    at = root
    for name in rel.parts:
        at = at / name
        if at.is_symlink():
            raise ValueError('Symlink write rejected: ' + str(at))
    return p


def atomic(path, value):
    p = safe(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.part-' + str(os.getpid()))
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(p))


def load(path=None, profile='pilot'):
    c = read(path or FOLDER / 'config.json')
    if set(c) != set(read(FOLDER / 'config.json')) or c['schema'] != SCHEMA:
        raise ValueError('Unknown configuration fields/schema')
    bounds = dict(seed=(0, 2**31-1), text_train_samples=(8, 9647),
                  image_train_samples=(8, 9647), val_samples=(2, 100),
                  text_epochs=(1, 100), image_epochs=(1, 100),
                  text_microbatch=(1, 4), text_accumulation=(1, 64),
                  image_accumulation=(1, 16), warmup_updates=(1, 5000),
                  cache_lru=(1, 128), train_eval_samples=(2, 32),
                  save_images=(0, 8), log_every=(1, 1000))
    for k, (lo, hi) in bounds.items():
        if type(c[k]) is not int or not lo <= c[k] <= hi:
            raise ValueError('Out of bound: ' + k)
    for k in set(c) - set(bounds) - {'schema'}:
        if type(c[k]) not in (int, float) or not math.isfinite(c[k]) or c[k] <= 0:
            raise ValueError('Finite positive number required: ' + k)
    if max(c['text_learning_rate'], c['image_learning_rate']) > .001:
        raise ValueError('Learning rate cap')
    if not 0 < c['reserve_free_gib'] < c['min_free_gib'] <= 48:
        raise ValueError('GPU resource bounds')
    if not 0 < c['min_lr_fraction'] <= 1 or not 0 < c['weight_decay'] < 1:
        raise ValueError('Fraction bounds')
    if c['min_disk_gib'] < 8 or c['save_images'] > c['val_samples']:
        raise ValueError('Disk/sample bounds')
    if profile == 'full':
        c.update(text_train_samples=9647, image_train_samples=9647)
    elif profile == 'smoke':
        c.update(text_train_samples=8, image_train_samples=8, val_samples=2,
                 text_epochs=1, image_epochs=1, text_microbatch=1,
                 text_accumulation=1, image_accumulation=1, train_eval_samples=2, save_images=2)
    elif profile != 'pilot':
        raise ValueError('Unknown profile')
    return c


def geometry(task):
    if task not in ('text', 'image'):
        raise ValueError('Independent modality required')
    n = 18432 if task == 'text' else 8192
    return dict(payload_reals=n, payload_complex_symbols=n//2,
                analog_repeats=1, common_reals=0, opposite_modality_reals=0,
                raw_utf8_bytes=0, channel='ideal_clean', snr_db=None,
                SNR_is_not_40dB=True, waveform_simulated=False,
                fixed_chunks=9 if task == 'text' else None,
                all_empty_chunks_charged=task == 'text',
                source_mask_or_length_at_RX=False, FSO=False, diffusion=False,
                physical_overheads_measured=False, bit_rate_and_BER=None,
                image_complex_uses_per_RGB_scalar=1/192 if task == 'image' else None)


def order(count, seed, epoch):
    ids = list(range(count))
    random.Random('{}:{}:{}'.format(seed, 'v14', epoch)).shuffle(ids)
    return ids


def batches(count, size, epochs, seed):
    for epoch in range(epochs):
        ids = order(count, seed, epoch)
        for start in range(0, count, size):
            yield epoch, ids[start:start+size]


def updates(count, size, epochs):
    return math.ceil(count / size) * epochs


def source_check():
    manifest = read(FOLDER / 'MANIFEST.json')
    for name, value in manifest['files'].items():
        p = FOLDER / name
        if p.is_symlink() or not p.is_file() or sha(p) != value:
            raise ValueError('V14 source changed: ' + name)
    for name, value in manifest['dependencies'].items():
        p = PROJECT / name
        if p.is_symlink() or not p.is_file() or sha(p) != value:
            raise ValueError('Historical dependency missing/changed: ' + name)
    return manifest


def array_hash(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def parameter_hash(model):
    return {n: array_hash(p.detach().cpu().contiguous().numpy())
            for n, p in model.named_parameters()}
