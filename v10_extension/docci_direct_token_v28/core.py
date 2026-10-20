"""V28 experiment contracts. Standard library only; no model/data imports."""
import hashlib
import json
import math
import os
from pathlib import Path
import random

FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parents[1]
ARMS = ('normal_range', 'low_snr')
WIDTH = dict(image=8192, text=18432)
PARENTS = {
    'image': dict(checkpoint='experiment_results/v14_image_pilot_02/main/image/expanded/train/last.pth',
        sha256='95c05e3b374deae2a3bd10603b8e95ebbd464456f7581a054f2b80597be5923b',
        features='experiment_results/v14_image_pilot_02/main/image',
        metadata='76eb1cf1a774c2c0c93288cab6f783934b80a213be5b6ce1038e8e15dc643b3a',
        validation_metadata='103f44c547acfea97573c847c4035776e2e00cb82592ff86c5acd44e7dcd55d7',
        backbone_metadata='1b48260ddd6cad536c92a6044f1341b7ed43713847a116191a2c829a1020c000',
        reference_sha256='31d0b4c1157b080fe38d00f8eea5303b37ec41eb722e373afcffc8b94f95dad6',
        reference='experiment_results/v14_image_pilot_02/main/image/expanded/validation/rows.jsonl'),
    'text': dict(checkpoint='experiment_results/v18_text_noise_01/train_noise_replay/last.pth',
        sha256='96b27347b62c918a931ba221640139ce3e162b8ff8fbcaebde89f6b13cc5312b',
        features='experiment_results/v14_text_pilot_01/main/text',
        metadata='ae088842a4fa02c5b5a1df7a19e54cdca3e961c59ecc97675406cfb4a28ec142',
        validation_metadata='193927b32554dccc7e7361f92cc2a2acd8a950dae2ef340134f9228c4e086ddf',
        backbone_metadata='6f5d6bf6abd532f5f502e55cc46df217809433885179c5773948b8c16d406903',
        reference_sha256='1580d3d9b86a6cff176c3cac24efaecd3033ed556d38d96acd3fa3fd788f43be',
        reference='experiment_results/v18_text_noise_01/eval_noise_replay/rows.jsonl')}
PROFILES = dict(smoke=(8, 2, 1, 1), pilot=(512, 16, 4, 1), full=(2048, 100, 10, 3))


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def rows(p):
    with Path(p).open(encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda: f.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], 'big') % (2**31-1)


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
            raise ValueError('Symlink output rejected')
    return p


def atomic(p, value):
    p = safe(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.part-'+str(os.getpid()))
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(str(tmp), str(p))


def load(path=None, profile='pilot'):
    default = read(FOLDER/'config.json')
    c = read(path) if path else default.copy()
    if set(c) != set(default) or c['schema'] != default['schema'] or profile not in PROFILES:
        raise ValueError('Unknown configuration/profile')
    # Freeze the causal comparison; changing the channel matrix is a new protocol.
    for k in ('seed', 'normal_snr_db', 'low_snr_db', 'eval_snr_db', 'clean_weight', 'bootstrap_draws'):
        if c[k] != default[k]:
            raise ValueError('Fixed protocol field: '+k)
    for k in ('text_lr', 'image_lr', 'weight_decay', 'minimum_free_gib', 'reserve_free_gib', 'minimum_disk_gib'):
        if type(c[k]) not in (int, float) or not math.isfinite(c[k]) or c[k] <= 0:
            raise ValueError('Finite positive field: '+k)
    if max(c['text_lr'], c['image_lr']) > .0001 or c['weight_decay'] >= 1:
        raise ValueError('Learning safeguards')
    if not 0 < c['reserve_free_gib'] < c['minimum_free_gib'] <= 48 or c['minimum_disk_gib'] < 12:
        raise ValueError('Resource safeguards')
    for k in ('text_accumulation', 'image_accumulation', 'log_every', 'save_images'):
        if type(c[k]) is not int or not 1 <= c[k] <= (8 if k == 'save_images' else 64):
            raise ValueError('Integer field: '+k)
    n, v, epochs, repeats = PROFILES[profile]
    c.update(profile=profile, train_samples=n, val_samples=v, epochs=epochs, repeats=repeats)
    return c


def schedule(c, task):
    n = c['train_samples']; batch = c[task+'_accumulation']
    if c['profile'] == 'smoke':
        return [(0, list(range(min(batch, n)))) for _ in range(2)]
    result = []
    for epoch in range(c['epochs']):
        order = list(range(n)); random.Random(seed(c['seed'], 'order', epoch)).shuffle(order)
        result.extend((epoch, order[i:i+batch]) for i in range(0, n, batch))
    return result


def training_snr(c, arm, epoch, eid):
    values = c['normal_snr_db'] if arm == 'normal_range' else c['low_snr_db'] if arm == 'low_snr' else None
    if values is None:
        raise ValueError('Unknown training arm')
    return values[seed(c['seed'], 'snr-choice', epoch, eid) % len(values)]


def trials(c):
    result = [(None, 0, 'matched')]
    snrs = [-5, 0] if c['profile'] == 'smoke' else c['eval_snr_db']
    result.extend((s, r, 'matched') for s in snrs for r in range(c['repeats']))
    result.extend((s, 0, control) for s in (None, -5) for control in ('shuffled', 'zero'))
    return result


def geometry(task):
    return dict(payload_reals=WIDTH[task], payload_complex_uses=WIDTH[task]//2,
        pilot_complex_uses=32, total_complex_uses=WIDTH[task]//2+32,
        repetitions=1, common_reals=0, AWGN=True, FSO=False,
        QAM=False, FEC=False, bit_rate=None, BER=None)


def check_source():
    m = read(FOLDER/'MANIFEST.json')
    for base, files in ((FOLDER, m['files']), (PROJECT, m['dependencies'])):
        for name, h in files.items():
            if sha(base/name) != h:
                raise ValueError('Source differs: '+name)
    return m
