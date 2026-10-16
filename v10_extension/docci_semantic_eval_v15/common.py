"""V15: immutable, project-local, stdlib-only evaluation infrastructure."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import random
import statistics

FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_rows(path, values):
    with Path(path).open('x', encoding='utf-8') as f:
        for r in values:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')


def inside(root, name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('Invalid relative path')
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or str(p) != name:
        raise ValueError('Unsafe relative path: ' + name)
    value = Path(root)
    for part in p.parts:
        value = value / part
        if value.is_symlink():
            raise ValueError('Symlink artifact forbidden')
    return value


def fresh(path):
    p = Path(path).absolute()
    if p == PROJECT or PROJECT not in p.parents:
        raise ValueError('Write only below the existing project root')
    for parent in (p,) + tuple(p.parents):
        if parent.is_symlink():
            raise ValueError('Output symlink forbidden')
    p.mkdir(parents=True, exist_ok=False)
    return p


def finite(x):
    return type(x) in (float, int) and math.isfinite(x)


def verify_source():
    m = read(FOLDER / 'MANIFEST.json')
    for root, mapping in ((FOLDER, m['files']), (PROJECT, m['dependencies'])):
        for name, expected in mapping.items():
            if sha(inside(root, name)) != expected:
                raise ValueError('Source/dependency changed: ' + name)
    return m


def legacy():
    """Load ONLY trusted, hash-pinned local V13 code; never code from records."""
    verify_source()
    path = PROJECT / 'v10_extension/docci_semantic_eval_v13/core.py'
    spec = importlib.util.spec_from_file_location('_v15_trusted_v13_core', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def matrix(cases, expected_n=None):
    groups, refs, seen = {}, {}, set()
    lex = legacy().lexical
    for c in cases:
        key = c['case_id']
        if key in seen or key != c['arm'] + '/' + c['control'] + '/' + c['example_id']:
            raise ValueError('Duplicate/inconsistent case identity')
        seen.add(key)
        if not isinstance(c['reference'], str) or not c['reference'] or not isinstance(c['candidate'], str):
            raise ValueError('Invalid reference/candidate')
        for k in ('reference', 'candidate'):
            if text_sha(c[k]) != c[k + '_sha256']:
                raise ValueError('Text hash mismatch')
        eid = c['example_id']
        if eid in refs and refs[eid] != c['reference']:
            raise ValueError('Cross-group source mismatch')
        refs[eid] = c['reference']
        if c['lexical'] != lex(c['reference'], c['candidate']):
            raise ValueError('Independent lexical recomputation differs')
        groups.setdefault(c['arm'] + '/' + c['control'], []).append(c)
    if len(refs) < 2 or (expected_n is not None and len(refs) != expected_n):
        raise ValueError('Wrong source cohort size')
    for g, rr in groups.items():
        if len(rr) != len(refs) or {r['example_id'] for r in rr} != set(refs):
            raise ValueError('Incomplete group: ' + g)
    return groups, refs


def seal(root, marker, metadata):
    files = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    write(root / marker, dict(metadata, files=files, source_manifest_sha256=sha(FOLDER / 'MANIFEST.json')))


def verify_seal(root, marker):
    verify_source()
    root = Path(root)
    m = read(root / marker)
    if m['source_manifest_sha256'] != sha(FOLDER / 'MANIFEST.json'):
        raise ValueError('Use the exact V15 source version')
    for name, expected in m['files'].items():
        if sha(inside(root, name)) != expected:
            raise ValueError('Frozen artifact changed: ' + name)
    return m


def stats(values, replicates=2000, seed=42):
    if not values or not all(finite(x) for x in values):
        raise ValueError('Empty/nonfinite statistic')
    rng, n = random.Random(seed), len(values)
    samples = sorted(statistics.mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(replicates))
    def q(p):
        x = (len(samples) - 1) * p
        lo, hi = math.floor(x), math.ceil(x)
        return samples[lo] + (samples[hi] - samples[lo]) * (x - lo)
    return dict(n=n, mean=statistics.mean(values), SD=statistics.stdev(values) if n > 1 else None,
                description_bootstrap_CI95=[q(.025), q(.975)])


def anchor_position(matched, shuffled, native):
    """Ratio of COHORT MEANS, not mean of per-example ratios or accuracy."""
    if not (len(matched) == len(shuffled) == len(native)) or not matched:
        raise ValueError('Anchors must be paired')
    denominator = statistics.mean(native) - statistics.mean(shuffled)
    if denominator <= 1e-8:
        return dict(value=None, status='UNDEFINED_NONPOSITIVE_OR_TINY_ANCHOR_GAP')
    return dict(value=(statistics.mean(matched) - statistics.mean(shuffled)) / denominator,
                status='EXPLORATORY_RELATIVE_POSITION_NOT_ACCURACY', clipped=False)
