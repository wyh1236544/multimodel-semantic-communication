"""V22: immutable, receiver-only feasibility experiments; no training entry."""
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parents[1]
SCHEMA = 'docci-pretrained-checks-v22'
STAGES = ('prepare', 'digital', 'caption', 'enhance', 'score')
CONFIG = dict(schema=SCHEMA, seed=42, snr_db=[None, 0, -5],
    profiles={'smoke': [2, 1], 'quick': [8, 1], 'pilot': [32, 3]},
    image_payload_symbols=4096, image_pilots=32, text_payload_symbols=9216,
    text_pilots=32, repeats_are_independent_trials=True, transmissions_per_trial=1,
    diffusion_steps=30, diffusion_strength=.25, guidance_scale=5.,
    max_prompt_chunks=16, caption_max_new_tokens=512,
    bootstrap_draws=10000, minimum_relative_lpips_gain=.02,
    psnr_noninferiority_db=-.5, facts_status='PENDING', semantic_success=False)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)

def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]

def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.writing')
    with temporary.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    temporary.replace(path)

def safe(path, project=PROJECT):
    p = Path(path).absolute()
    base = Path(project).resolve()
    if '..' in p.parts or base not in p.resolve().parents or any(x.is_symlink() for x in [p]+list(p.parents) if x != base.parent):
        raise ValueError('Output must be a non-symlink path inside this project')
    return p

def local(root, name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('Unsafe relative artifact')
    out = Path(root)/p
    if out.is_symlink() or Path(root).resolve() not in out.resolve().parents:
        raise ValueError('Escaped artifact')
    return out

def seed(*parts):
    return int(digest(json.dumps(parts, separators=(',', ':')).encode())[:16], 16)

def noise(eid, snr, repeat, kind, n):
    if snr not in CONFIG['snr_db'] or kind not in ('image', 'text', 'image_pilot', 'text_pilot'):
        raise ValueError('Unregistered channel condition')
    if snr is None:
        return np.zeros(n, np.float32)
    value = seed(42, SCHEMA, eid, snr, repeat, kind)
    return (np.random.Generator(np.random.PCG64(value)).standard_normal(n).astype(np.float32)
            * np.float32(10.**(-snr/20.)))

def donor_map(ids):
    if len(ids) < 2 or len(ids) != len(set(ids)):
        raise ValueError('At least two unique examples for real mismatches')
    return {x: ids[(i+1) % len(ids)] for i, x in enumerate(ids)}

def case_key(row):
    return row['example_id'], row['snr_db'], row['repeat']

def case_name(eid, snr, repeat):
    if not eid or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in eid):
        raise ValueError('Unsafe example identity')
    return '{}__{}__r{}'.format(eid, 'clean' if snr is None else 'snr'+str(snr), repeat)

def clip_blocks(body, bos, eos, pad):
    blocks = [body[i:i+75] for i in range(0, len(body), 75)] or [[]]
    if len(blocks) > CONFIG['max_prompt_chunks']:
        raise ValueError('Prompt exceeds explicit full-coverage chunk budget; not truncated')
    ids = [[bos]+list(b)+[eos]+[pad]*(75-len(b)) for b in blocks]
    return ids, dict(content_tokens=len(body), chunks=len(blocks), truncated=False)

def metric(source, output, rectangle):
    a, b = np.asarray(source), np.asarray(output)
    if a.dtype != np.uint8 or b.dtype != np.uint8 or a.shape != (512, 512, 3) or b.shape != a.shape:
        raise ValueError('Exact 512 RGB uint8 comparison required; no evaluation resizing')
    l, t, r, bot = map(int, rectangle)
    if not (0 <= l < r <= 512 and 0 <= t < bot <= 512):
        raise ValueError('Invalid evaluation-only rectangle')
    mse = float(np.mean(((a[t:bot, l:r].astype(float)-b[t:bot, l:r])/255.)**2))
    # Finite JSON: identical images report null PSNR + explicit identity, not 0dB.
    return dict(MSE=mse, psnr=None if mse == 0 else -10*math.log10(mse), identical=mse == 0)

def seal_tree(folder):
    folder = Path(folder).resolve(strict=True)
    files = {}
    for p in sorted(folder.rglob('*')):
        if p.is_file():
            if p.suffix in ('.py', '.so'):
                raise ValueError('Model directory must not contain executable custom code')
            files[p.relative_to(folder).as_posix()] = sha(p)
    if not files:
        raise ValueError('Empty asset directory')
    return dict(path=str(folder), files=files)

def verify_tree(value):
    folder = Path(value['path']).resolve(strict=True)
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    if actual != set(value['files']):
        raise ValueError('Asset file set changed')
    for n, h in value['files'].items():
        if sha(folder/n) != h:
            raise ValueError('Asset bytes changed: '+n)
    return folder

class Budget:
    def __init__(self, hours, gpu=False):
        if not math.isfinite(hours) or not 0 < hours <= 12:
            raise ValueError('Explicit stage budget must be in (0,12] hours')
        self.start = time.monotonic()
        self.deadline = self.start+hours*3600
        self.gpu = gpu
    def __call__(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Stage budget exceeded; preserve partial output')
        if self.gpu:
            import torch
            if torch.cuda.mem_get_info()[0] < 4*2**30:
                raise RuntimeError('Free GPU reserve crossed; no automatic changes')

def complete(root, stage, metadata):
    root = Path(root)
    files = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError('Symlink in result')
        if p.is_file() and p.name not in ('COMPLETE.json', 'LAUNCH.json'):
            files[p.relative_to(root).as_posix()] = sha(p)
    atomic(root/'COMPLETE.json', dict(schema=SCHEMA, stage=stage, files=files,
        source_manifest_sha256=sha(FOLDER/'MANIFEST.json') if (FOLDER/'MANIFEST.json').is_file() else None,
        metadata=metadata, training_updates=0, semantic_success=False, facts_status='PENDING'))

def verify_files(root, stage=None):
    root = Path(root).resolve(strict=True)
    receipt = read(root/'COMPLETE.json')
    if receipt['schema'] != SCHEMA or (stage and receipt['stage'] != stage) or receipt['training_updates'] != 0:
        raise ValueError('Wrong stage receipt')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and p.name not in ('COMPLETE.json', 'LAUNCH.json')}
    if actual != set(receipt['files']):
        raise ValueError('Missing or unexpected result file')
    for n, h in receipt['files'].items():
        if sha(local(root, n)) != h:
            raise ValueError('Result file changed: '+n)
    return receipt

def prepared(root):
    verify_files(root, 'prepare')
    index = read(Path(root)/'index.json')
    rs = rows(Path(root)/'rows.jsonl')
    ids = [r['example_id'] for r in index['examples']]
    wanted = {(x, s, rep) for x in ids for s in CONFIG['snr_db'] for rep in range(index['repeats'])}
    if len(rs) != len(wanted) or {case_key(r) for r in rs} != wanted:
        raise ValueError('Missing/duplicate prepared cases')
    by = {r['example_id']: r for r in index['examples']}
    for r in rs:
        if (r['source_text_at_RX'] or r['post_noise_renormalization'] or
                r['image_payload_symbols'] != 4096 or r['text_payload_symbols'] != 9216):
            raise ValueError('Invalid receiver/resource contract')
        if sha(local(root, r['received_image'])) != r['received_image_sha256']:
            raise ValueError('Received image identity differs')
        if digest(r['received_caption'].encode()) != r['received_caption_sha256']:
            raise ValueError('Received text identity differs')
        if r['donor_id'] != donor_map(ids)[r['example_id']]:
            raise ValueError('Wrong donor pairing')
    return index, rs, by

def source_check():
    m = read(FOLDER/'MANIFEST.json')
    for base, names in ((FOLDER, m['files']), (PROJECT, m['dependencies'])):
        for n, h in names.items():
            if sha(local(base, n)) != h:
                raise ValueError('Pinned source changed: '+n)
    return m
