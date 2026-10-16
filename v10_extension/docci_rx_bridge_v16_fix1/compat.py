"""Narrow, recorded reader compatibility; original files and model code unchanged."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import shutil
from docci_rx_bridge_v16.core import FOLDER as ORIGINAL, atomic, read, sha, check_source

FOLDER = Path(__file__).resolve().parent
ORIGINAL_MANIFEST = '7ea2874353f748c4f7f9bbae8d95ef6475def1804e388d505976bcb1732c89eb'


def source_check():
    if sha(ORIGINAL/'MANIFEST.json') != ORIGINAL_MANIFEST:
        raise ValueError('Expected original immutable V16 release; do not edit its manifest')
    check_source()
    m = read(FOLDER/'MANIFEST.json')
    for name, expected in m['files'].items():
        p = FOLDER/name
        if p.is_symlink() or not p.is_file() or sha(p) != expected:
            raise ValueError('Fix1 source missing/changed: '+name)
    return m


def snapshot(root):
    m = source_check()
    dest = root/'compatibility_snapshot/docci_rx_bridge_v16_fix1'
    dest.mkdir(parents=True, exist_ok=False)
    for name in list(m['files'])+['MANIFEST.json']:
        shutil.copyfile(FOLDER/name, dest/name)
    atomic(root/'V16_FIX1.json', dict(schema='v16-missing-source-tokens-reader-fix1',
           original_manifest_sha256=ORIGINAL_MANIFEST,
           repair_manifest_sha256=sha(FOLDER/'MANIFEST.json'),
           change='baseline source_tokens absent: use hash-bound source index count',
           source_length_used_at_receiver=False, predictions_or_metrics_rewritten=False,
           original_source_and_raw_files_rewritten=False))


@contextmanager
def installed():
    """Patch ONLY baseline JSONL reading in legacy CPU report, restore on exit.

    Legacy index_check verifies source9x64 IDs and count BEFORE reading baseline
    when arrays=True. arrays=False explicitly retains only sealed-index evidence.
    No default for wrong/present/null counts, no generated-text token estimation.
    """
    source_check()
    from docci_rx_bridge_v16 import report, run, collect
    old_lines, old_verify = report.lines, report.verify
    old_snapshot, old_collect_verify, old_copy = run.snapshot, collect.verify, collect.copy_light
    events = {}
    def repaired_lines(path):
        result = old_lines(path)
        path = Path(path)
        if path.name != 'baseline.jsonl' or not (path.parent/'index.json').is_file():
            return result
        index = read(path.parent/'index.json')['records']
        mapping = {r['example_id']: r for r in index}
        if len(mapping) != len(index):
            raise ValueError('Duplicate original source index')
        repaired = []
        count = 0
        for row in result:
            value = dict(row)
            if 'source_tokens' not in row:
                origin = mapping.get(row['example_id'])
                if origin is None or row['source_text'] != origin['text'] or row['source_text_sha256'] != origin['text_sha256'] or row['official_split'] != origin['official_split']:
                    raise ValueError('Cannot normalize baseline with different source identity')
                if hashlib.sha256(row['source_text'].encode()).hexdigest() != origin['text_sha256']:
                    raise ValueError('Original source text hash differs')
                n = origin['source_tokens']
                if type(n) is not int or not 2 <= n <= 512:
                    raise ValueError('Invalid source-index length; no guessed default')
                value['source_tokens'] = n
                count += 1
            repaired.append(value)
        if count:
            events[str(path.resolve())] = dict(normalized_rows=count,
                    baseline_sha256=sha(path), index_sha256=sha(path.parent/'index.json'),
                    original_baseline_bytes_changed=False)
        return repaired
    def verified(*args, **kwargs):
        result = old_verify(*args, **kwargs)
        return dict(result, compatibility_fix1=dict(repair_manifest_sha256=sha(FOLDER/'MANIFEST.json'),
             original_manifest_sha256=ORIGINAL_MANIFEST, derived_from='sealed original source index',
             normalizations=list(events.values()), original_files_rewritten=False))
    def run_snapshot(root, manifest):
        old_snapshot(root, manifest)
        snapshot(root)
    def copy_light(source, target, budget):
        old_copy(source, target, budget)
        if not (target/'V16_FIX1.json').exists():
            snapshot(target)
    report.lines, report.verify = repaired_lines, verified
    run.snapshot, collect.verify, collect.copy_light = run_snapshot, verified, copy_light
    try:
        yield verified
    finally:
        report.lines, report.verify = old_lines, old_verify
        run.snapshot, collect.verify, collect.copy_light = old_snapshot, old_collect_verify, old_copy


def write_receipt(root, value):
    """Only a NEW derived CPU receipt; never overwrite even an old failed receipt."""
    path = Path(root)/'VERIFIED.json'
    if path.exists():
        if read(path) != value:
            raise FileExistsError('Existing different VERIFIED.json preserved; inspect it, no overwrite')
        return
    # Exclusive claim prevents concurrent verifiers replacing one another.
    import json
    from docci_rx_bridge_v16.core import safe
    path = safe(path)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
