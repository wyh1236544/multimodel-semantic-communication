"""Stdlib-only immutable evidence-review contracts. No model/training dependencies."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid

FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parents[1]
SCHEMA = 'docci-fact-bridge-v27'
CATEGORIES = ('objects', 'attributes', 'colors', 'counts', 'relations', 'actions', 'other')
VISIBILITY = ('potentially_visual', 'not_visual', 'uncertain')
LABELS = ('retained', 'contradicted', 'missing', 'uncertain')
METHODS = ('base', 'self', 'visual_matched', 'visual_shuffled', 'caption')
MODES = {'base': ('base', 'matched'), 'self': ('self', 'matched'),
         'visual_matched': ('visual', 'matched'), 'visual_shuffled': ('visual', 'shuffled')}


def sha(path):
    if not Path(path).is_file() or Path(path).is_symlink():
        raise ValueError('Expected a regular, unlinked file')
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def reject_pairs(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError('Duplicate JSON key: ' + k)
        out[k] = v
    return out


def read(path):
    if Path(path).stat().st_size > 64 * 1024 ** 2:
        raise ValueError('Unexpectedly large JSON input')
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=reject_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def rows(path):
    if Path(path).stat().st_size > 64 * 1024 ** 2:
        raise ValueError('Unexpectedly large JSONL input')
    return [json.loads(s, object_pairs_hook=reject_pairs) for s in Path(path).read_text(encoding='utf-8').splitlines() if s]


def atomic(path, obj):
    path = Path(path)
    data = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('x', encoding='utf-8') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        # Atomic no-replace publication. A racing/existing destination is preserved.
        os.link(str(temp), str(path))
    finally:
        if temp.exists():
            temp.unlink()


def write_text(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        f.write(value)


def local(root, name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or '\\' in str(name):
        raise ValueError('Unsafe relative path')
    root = Path(root).resolve(strict=True)
    target = root / p
    for q in [target] + list(target.parents):
        if q == root:
            break
        if q.is_symlink():
            raise ValueError('Linked evidence/output forbidden')
    if root not in target.resolve().parents:
        raise ValueError('Outside evidence root')
    return target


def new_output(path, protected=()):
    raw = Path(path).absolute()
    if any(q.is_symlink() for q in [raw] + list(raw.parents)):
        raise ValueError('Linked output path forbidden')
    dest = raw.resolve()
    if PROJECT.resolve() not in dest.parents or dest.exists():
        raise FileExistsError('Use a NEW directory inside this multimodel project')
    for p in protected:
        p = Path(p).resolve()
        if p == dest or p in dest.parents or dest in p.parents:
            raise ValueError('Input/output overlap forbidden')
    if shutil.disk_usage(PROJECT).free < 256 * 1024 ** 2:
        raise RuntimeError('Project disk reserve below 256 MiB')
    dest.mkdir(parents=True)
    return dest


def config():
    c = read(FOLDER / 'config.json')
    if c['schema'] != SCHEMA or c['training_updates'] or c['semantic_success'] or c['GPU_allocated']:
        raise ValueError('Review-only contract changed')
    return c


def source_check():
    m = read(FOLDER / 'MANIFEST.json')
    for n, h in m['files'].items():
        if sha(local(FOLDER, n)) != h:
            raise ValueError('Frozen V27 source differs: ' + n)
    return m


def snapshot(root):
    m = source_check()
    folder = Path(root) / 'source_snapshot'
    folder.mkdir()
    for n in list(m['files']) + ['MANIFEST.json']:
        shutil.copyfile(FOLDER / n, folder / n)


def seal(root, stage, status):
    root = Path(root)
    files = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    atomic(root / 'COMPLETE.json', dict(schema=SCHEMA, stage=stage, status=status, files=files,
        source_manifest_sha256=sha(FOLDER / 'MANIFEST.json'), training_updates=0,
        automatic_training_approval=False, semantic_success=False, private=True))


def verify_files(root):
    root = Path(root).resolve(strict=True)
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Linked records forbidden')
    c = read(root / 'COMPLETE.json')
    if (c['schema'] != SCHEMA or type(c['training_updates']) is not int or c['training_updates'] != 0
            or c['automatic_training_approval'] is not False or c['semantic_success'] is not False):
        raise ValueError('Not a read-only review receipt')
    if c['source_manifest_sha256'] != sha(FOLDER / 'MANIFEST.json'):
        raise ValueError('Executed V27 manifest differs')
    found = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != root / 'COMPLETE.json'}
    if found != set(c['files']):
        raise ValueError('Missing or unexpected stage artifacts')
    for n, h in c['files'].items():
        if sha(local(root, n)) != h:
            raise ValueError('Artifact hash differs: ' + n)
    for n in list(source_check()['files']) + ['MANIFEST.json']:
        if sha(root / 'source_snapshot' / n) != sha(FOLDER / n):
            raise ValueError('Executed source snapshot differs')
    return c


class Budget:
    def __init__(self, minutes):
        if not math.isfinite(minutes) or not 0 < minutes <= 60:
            raise ValueError('max-minutes must be in (0,60]')
        self.deadline = time.monotonic() + minutes * 60

    def __call__(self):
        if time.monotonic() > self.deadline:
            raise TimeoutError('CPU review time budget exhausted; preserve output')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def require_keys(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError('Unexpected or missing document fields')


def short(value, name, maximum=400):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError('Missing/oversized ' + name)
    return value.strip()
