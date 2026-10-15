"""Read-only historical text adapter and dependency-free evaluation utilities."""
import hashlib
import html
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import statistics
import tarfile
import unicodedata

FOLDER = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def save(path, value):
    with Path(path).open('xb') as handle:
        handle.write(encode(value))


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonlines(path, rows):
    with Path(path).open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def readlines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def fresh(path):
    path = Path(path).absolute()
    for parent in (path,) + tuple(path.parents):
        if parent.is_symlink():
            raise ValueError('Output symlink forbidden: ' + str(parent))
    path.mkdir(parents=True, exist_ok=False)
    return path


class Source:
    """Read known JSONL suffixes in a directory or tar. Never extract or execute."""
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self.receipts = []
        self.tar = None
        if self.path.is_file():
            self.tar = tarfile.open(self.path, 'r:*')
            members = self.tar.getmembers()
            names = [m.name for m in members]
            if len(names) != len(set(names)):
                self.tar.close()
                raise ValueError('Duplicate archive member')
            for member in members:
                parts = PurePosixPath(member.name).parts
                if not parts or member.name.startswith('/') or '..' in parts or '\\' in member.name:
                    self.tar.close()
                    raise ValueError('Unsafe archive member')
                if not (member.isfile() or member.isdir()):
                    self.tar.close()
                    raise ValueError('Links/special archive members forbidden')
                if member.size > 512 * 1024 * 1024:
                    self.tar.close()
                    raise ValueError('Oversized archive member')
            self.inventory = {m.name: m for m in members if m.isfile()}
            self.identity = dict(kind='tar', name=self.path.name, sha256=file_hash(self.path))
        else:
            self.inventory = None
            self.identity = dict(kind='directory', name=self.path.name,
                                 scope='Only consumed JSONL bytes hashed; NOT whole-run verification')

    def close(self):
        if self.tar:
            self.tar.close()

    def rows(self, suffix):
        if self.tar:
            found = [name for name in self.inventory if name == suffix or name.endswith('/' + suffix)]
            if len(found) != 1:
                raise ValueError('Need exactly one historical member: ' + suffix)
            member = self.inventory[found[0]]
            if member.size > 32 * 1024 * 1024:
                raise ValueError('JSONL too large')
            data = self.tar.extractfile(member).read()
            name = found[0]
        else:
            file = self.path / suffix
            if not file.is_file() or file.is_symlink():
                raise ValueError('Pass the actual extracted records root: ' + str(file))
            if file.stat().st_size > 32 * 1024 * 1024:
                raise ValueError('JSONL too large')
            data = file.read_bytes()
            name = suffix
        self.receipts.append(dict(member=name, bytes=len(data), sha256=digest(data)))
        return [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(part)
    return hasher.hexdigest()


def words(text):
    return re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold(), re.UNICODE)


def distance(a, b):
    # Exact unit-cost Myers algorithm; same historical metric, no source edits.
    if len(a) > len(b):
        a, b = b, a
    if not a:
        return len(b)
    n = len(a)
    eq = {}
    for i, item in enumerate(a):
        eq[item] = eq.get(item, 0) | (1 << i)
    mask = (1 << n) - 1
    high = 1 << (n - 1)
    vp, vn, score = mask, 0, n
    for item in b:
        e = eq.get(item, 0)
        xv = e | vn
        xh = (((e & vp) + vp) ^ vp) | e
        ph, mh = vn | ~(xh | vp), vp & xh
        score += int(bool(ph & high)) - int(bool(mh & high))
        ph, mh = ((ph << 1) | 1) & mask, (mh << 1) & mask
        vp, vn = (mh | ~(xv | ph)) & mask, ph & xv
    return score


def lexical(source, candidate):
    a, b = words(source), words(candidate)
    errors = distance(a, b)
    return dict(normalized_WER=errors / max(1, len(a)), word_errors=errors,
                source_words=len(a), raw_CER=distance(source, candidate) / max(1, len(source)),
                exact_text=source == candidate, candidate_words=len(b),
                adjacent_repeat=sum(x == y for x, y in zip(b, b[1:])) / max(1, len(b) - 1),
                unique_word_fraction=len(set(b)) / max(1, len(b)))


def cases_from_sources(v11, v12, expected):
    cases, references = [], {}
    specifications = [
        (v11, 'g20480_validation/rows.jsonl', 'V11_g20480', False),
        (v12, 'main/text/B0_ce/validation/rows.jsonl', 'V12_CE', False),
        (v12, 'main/text/B1_ce_kd/validation/rows.jsonl', 'V12_CE_KD', False),
        (v11, 'parent_reference/native_validation/rows.jsonl', 'native_BART', True)]
    cohort = None
    for src, suffix, arm, native in specifications:
        rows = src.rows(suffix)
        seen, ids = set(), set()
        required = {'matched'} if native else {'matched', 'shuffled', 'zero'}
        for row in rows:
            control = 'matched' if native else row['control']
            example = row['example_id']
            if (row['official_split'] != 'qual_dev' or row['cohort'] != 'validation'
                    or row.get('condition', 'clean') != 'clean' or row['input_truncated']):
                raise ValueError('Wrong split/condition or truncated input')
            if control not in required or (example, control) in seen:
                raise ValueError('Duplicate/unknown historical control')
            seen.add((example, control))
            ids.add(example)
            source, candidate = row['source_text'], row['received_caption']
            if not isinstance(source, str) or not source or not isinstance(candidate, str):
                raise ValueError('Invalid text field; empty candidate is allowed, empty reference is not')
            if digest(source.encode('utf-8')) != row['source_text_sha256']:
                raise ValueError('Source text digest differs')
            if example in references and references[example] != source:
                raise ValueError('Cross-arm source differs')
            references[example] = source
            if not native:
                resources = row['stats']['resources']
                if (resources['payload_complex_symbols'] != 9216 or resources['analog_repeats'] != 1
                        or resources['raw_utf8_wire_bytes'] != 0
                        or row['stats'].get('clean_sentinel_not_measured_snr') is not True):
                    raise ValueError('Historical clean resource contract differs')
            metric = lexical(source, candidate)
            if not math.isclose(metric['normalized_WER'], row['normalized_wer'], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Historical WER differs from independent string recomputation')
            cases.append(dict(case_id=arm + '/' + control + '/' + example, arm=arm, control=control,
                              example_id=example, reference=source, candidate=candidate,
                              reference_sha256=digest(source.encode('utf-8')),
                              candidate_sha256=digest(candidate.encode('utf-8')),
                              receiver_failure=row['receiver_failure'], lexical=metric,
                              communication_resources=None if native else dict(payload_complex_symbols=9216,
                                  analog_repeats=1, raw_utf8_wire_bytes=0, condition='clean; 40dB sentinel NOT measured optical SNR'),
                              scope='privileged no-compression diagnostic' if native else 'saved clean continuous receiver'))
        if len(ids) != expected or seen != {(i, c) for i in ids for c in required}:
            raise ValueError('Incomplete historical matrix')
        if cohort is not None and ids != cohort:
            raise ValueError('Cohort differs')
        cohort = ids
    ids = sorted(cohort)
    if len(ids) < 2:
        raise ValueError('At least two examples needed for derangement')
    # A fixed cyclic permutation has no self-pairs, unlike unconstrained shuffle.
    for i, example in enumerate(ids):
        candidate = references[ids[(i + 1) % len(ids)]]
        source = references[example]
        cases.append(dict(case_id='unrelated_description/deranged/' + example,
                          arm='unrelated_description', control='deranged', example_id=example,
                          reference=source, candidate=candidate,
                          reference_sha256=digest(source.encode('utf-8')),
                          candidate_sha256=digest(candidate.encode('utf-8')),
                          receiver_failure=False, lexical=lexical(source, candidate),
                          scope='evaluator calibration, NOT a transmitted method', donor_example_id=ids[(i + 1) % len(ids)]))
    return sorted(cases, key=lambda r: r['case_id']), references


def select_audit(references, count, seed):
    ranked = sorted(references, key=lambda i: (len(words(references[i])), i))
    bins = [ranked[j * len(ranked) // 3:(j + 1) * len(ranked) // 3] for j in range(3)]
    rng = random.Random(seed)
    selected = []
    for j, group in enumerate(bins):
        n = count // 3 + int(j < count % 3)
        selected.extend(rng.sample(group, min(n, len(group))))
    if len(selected) != count:
        raise ValueError('Not enough examples in length strata')
    return sorted(selected)


def percentile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] * (hi - x) + values[hi] * (x - lo) if hi != lo else values[lo]


def interval(values, seed=42, replicates=2000):
    rng = random.Random(seed)
    n = len(values)
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(replicates)]
    return [percentile(means, .025), percentile(means, .975)]


def summarize(cases):
    groups = {}
    for row in cases:
        groups.setdefault(row['arm'] + '/' + row['control'], []).append(row)
    result = {}
    for key, rows in groups.items():
        vals = [r['lexical']['normalized_WER'] for r in rows]
        result[key] = dict(n=len(rows), macro_normalized_WER=statistics.mean(vals),
                           description_bootstrap_CI95=interval(vals),
                           corpus_normalized_WER=sum(r['lexical']['word_errors'] for r in rows) / sum(r['lexical']['source_words'] for r in rows),
                           exact=sum(r['lexical']['exact_text'] for r in rows),
                           macro_raw_CER=statistics.mean(r['lexical']['raw_CER'] for r in rows),
                           macro_whole_text_adjacent_repeat=statistics.mean(r['lexical']['adjacent_repeat'] for r in rows))
    return result


def html_table(cases, summary):
    e = html.escape
    parts = ['<!doctype html><meta charset="utf-8"><title>V13 离线文本核查</title>',
             '<style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:15px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:10px;vertical-align:top}td{white-space:pre-wrap}h2{margin-top:35px}</style>',
             '<h1>V13：完整文本与对照</h1><p>事后探索性评价；不是独立 test。事实审核与 BERTScore 尚未测量。旧 QUALITY_STOP 不变。数字基线 NOT_RUN。</p>',
             '<h2>独立重算的字面指标</h2><pre>' + e(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre>']
    by_id = {}
    for row in cases:
        by_id.setdefault(row['example_id'], []).append(row)
    for example, rows in sorted(by_id.items()):
        parts += ['<h2>' + e(example) + '</h2><p>' + e(rows[0]['reference']) + '</p><table><tr><th>组别</th><th>完整恢复文本</th><th>WER</th></tr>']
        for row in rows:
            parts.append('<tr><td>' + e(row['arm'] + '/' + row['control']) + '</td><td>' + e(row['candidate']) + '</td><td>{:.2%}</td></tr>'.format(row['lexical']['normalized_WER']))
        parts.append('</table>')
    return '\n'.join(parts)


def verify_prepared(root):
    root = Path(root)
    manifest = load(root / 'PREPARED.json')
    for name, value in manifest['files'].items():
        if file_hash(root / name) != value:
            raise ValueError('Prepared file changed: ' + name)
    if manifest['protocol_sha256'] != file_hash(FOLDER / 'protocol.json'):
        raise ValueError('Use the original frozen V13 protocol')
    for name, expected in manifest.get('source_code', {}).items():
        if file_hash(FOLDER / name) != expected:
            raise ValueError('Use original frozen evaluator code: ' + name)
    return manifest, readlines(root / 'cases.jsonl')
