"""Import already-verified fixed V26 bytes, never execute attached source or load weights."""
import shutil
from pathlib import Path
from core import read, rows, sha, local, config, digest, text_sha, METHODS, MODES

# Public hashes only; no raw DOCCI text or private content is bundled with source.
PINS = {
    'PIPELINE.json': '9aa394e8f61221d73de6f3994dc1f9f5bd7ee83ffeaf26d17797cb7505a5307e',
    'inputs/COMPLETE.json': '6457d406a71496f329fcccfa187d9a404f0fd28626413fb2e4cf06fd8b1dee8b',
    'smoke/COMPLETE.json': '0ed2e1b7a9af75b147938cd47cbf9416ae98b1e9ad8730087c8a5b0b003dcc13',
    'diagnose/COMPLETE.json': 'b1394a3afcd23fd1ba2ec3df02023bfdb3b3a4735978840ec6cbd784403ecb3a',
    'diagnose/REFERENCES.json': '101d57ed4fbbc7ac0663ddf430213e0f1f9a8f70803dd8ed0d5f112f7c543936',
    'diagnose/generations.jsonl': 'ddb2c2d3047f75e26af869ec713a35da9cf55460305360631846b20b70e9be30',
    'inputs/CAPTION_ONLY.jsonl': '440d876509ac1325bed700ae1d428593208d69d2a89b4df91a515134186983d3',
    'diagnose/CHECKPOINTS.json': '1116592f7ab5fc5d44a71b0de53d683e555e3e805a3a97feab565a444d17c844'
}
COPIES = ('PIPELINE.json', 'diagnose/REFERENCES.json', 'diagnose/generations.jsonl',
          'inputs/CAPTION_ONLY.jsonl', 'diagnose/CHECKPOINTS.json')


def check_parent(root, records, budget):
    root = Path(root).resolve(strict=True)
    if records:
        coll = read(root / 'COLLECTION.json')
        if not coll['private'] or coll['training_updates'] or coll['weights_or_NPZ_included']:
            raise ValueError('Expected original PRIVATE V26 records')
        found = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != root / 'COLLECTION.json'}
        if found != set(coll['files']):
            raise ValueError('Original V26 collection file set differs')
        for n, h in coll['files'].items():
            budget()
            if sha(local(root, n)) != h:
                raise ValueError('V26 collection bytes differ')
        root = root / 'data'
    for n, h in PINS.items():
        budget()
        if sha(local(root, n)) != h:
            raise ValueError('Not the pinned actual V26 run: ' + n)
    for stage in ('inputs', 'smoke', 'diagnose'):
        p = root / stage
        complete = read(p / 'COMPLETE.json')
        found = {x.relative_to(p).as_posix() for x in p.rglob('*') if x.is_file() and x != p / 'COMPLETE.json'}
        if found != set(complete['files']):
            raise ValueError('V26 sealed stage file set differs')
        for n, h in complete['files'].items():
            budget()
            if sha(local(p, n)) != h:
                raise ValueError('V26 completed file changed')
    state = read(root / 'PIPELINE.json')
    if state['status'] != 'COMPLETE' or state['training_updates']:
        raise ValueError('V26 diagnosis incomplete')
    # The final launcher marker is post-collection. Its absence in records is expected.
    launcher = root / 'LAUNCH_FINISHED.json'
    if launcher.exists() and read(launcher)['status'] != 'VERIFIED':
        raise ValueError('Existing final V26 launcher marker is not verified')
    return root, dict(pinned_files=PINS, mode='collected_records' if records else 'server_run',
        launcher_marker_present=launcher.exists(),
        boundary='Exact import of previously independently verified V26; no new GPU/weight replay.')


def copy_evidence(source, dest):
    dest = Path(dest)
    dest.mkdir()
    for n in COPIES:
        p = dest / n
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(source) / n, p)
    verify_evidence(dest)


def verify_evidence(root):
    root = Path(root)
    found = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if found != set(COPIES):
        raise ValueError('Copied V26 evidence set differs')
    for n in COPIES:
        if sha(local(root, n)) != PINS[n]:
            raise ValueError('Copied V26 evidence changed')


def materialize(evidence):
    evidence = Path(evidence)
    verify_evidence(evidence)
    return assemble(read(evidence / 'diagnose/REFERENCES.json')['records'],
                    rows(evidence / 'diagnose/generations.jsonl'),
                    rows(evidence / 'inputs/CAPTION_ONLY.jsonl'))


def assemble(references, generations, captions):
    """Pure deterministic indexing; also exercised with explicitly TEST_ONLY fixtures."""
    refs = [r for r in references if r['cohort'] == 'dev']
    if len(refs) != 32 or len({r['example_id'] for r in refs}) != 32:
        raise ValueError('All fixed 32 DEV references required')
    wanted = {r['example_id'] for r in refs}
    for r in refs:
        if r['official_split'] != 'train' or text_sha(r['text']) != r['text_sha256']:
            raise ValueError('Original source reference differs')
    relevant = [r for r in generations if r['cohort'] == 'dev' and r['regime'] == 'both0'
                and (r['arm'], r['control']) in MODES.values()]
    by = {(r['example_id'], r['arm'], r['control']): r for r in relevant}
    expected = {(e, arm, ctrl) for e in wanted for arm, ctrl in MODES.values()}
    if len(by) != len(relevant) or set(by) != expected:
        raise ValueError('Primary generation matrix missing/duplicated')
    cap = [r for r in captions if r['image_snr_db'] == 0 and r['example_id'] in wanted]
    cb = {r['example_id']: r for r in cap}
    if len(cb) != 32 or len(cap) != 32:
        raise ValueError('Primary native-caption matrix differs')
    sources = [dict(reference_id=r['example_id'], source_text=r['text'],
                    source_text_sha256=r['text_sha256']) for r in refs]
    candidates = []
    for r in refs:
        for method in METHODS:
            if method == 'caption':
                text = cb[r['example_id']]['output_text']
                wer = None  # Short caption is not a long-source reconstruction WER baseline.
            else:
                g = by[r['example_id'], *MODES[method]]
                if (g['source_text'] != r['text'] or g['source_text_sha256'] != r['text_sha256']
                        or g['source_text_at_RX'] or g['source_mask_at_RX'] or g['source_image_at_RX']
                        or g['training_updates'] or g['generation_provenance'] != 'V25_VERIFIED_DEV_REUSE'
                        or (g['image_snr_db'], g['text_snr_db']) != (0, 0)):
                    raise ValueError('Frozen candidate/source boundary differs')
                text, wer = g['received_caption'], g['normalized_wer']
            candidates.append(dict(reference_id=r['example_id'], method=method, text=text,
                                   historical_normalized_WER=wer))
    candidates.sort(key=lambda r: digest(['V27-blind-order-42', r['reference_id'], r['method']]))
    for i, r in enumerate(candidates):
        r['case_id'] = 'case_{:04d}'.format(i + 1)
        r['candidate_sha256'] = text_sha(r['text'])
    return dict(references=sources, candidates=candidates, channel=config()['transmission'],
                cohort='reused_dev_from_official_train', independent_test=False)
