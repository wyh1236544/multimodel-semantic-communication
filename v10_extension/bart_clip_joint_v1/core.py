"""Public fixed budget, strict configuration, offline files and result metrics."""
import hashlib
import json
import math
from pathlib import Path
from text_jscc_v1_fix1.core import metrics, summarize, file_under
from fso_jscc_v1.core import digest, seed_for

TEMPLATE = Path(__file__).with_name('config.json')
BART_REVISION = 'aadd2ab0ae0c8268c7c9693540e9904811f36177'


def load(path=TEMPLATE):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    template = json.loads(TEMPLATE.read_text(encoding='utf-8'))
    if set(c) != set(template) or c['schema'] != 'bart-clip-joint-v1':
        raise ValueError('Unknown/missing configuration key')
    fixed = dict(max_tokens=128, common_slots=32, image_slots=224, text_slots=32,
                 width=16, hidden=256, layers=2, heads=4)
    for k, v in fixed.items():
        if type(c[k]) is not int or c[k] != v: raise ValueError('Public fixed layout: '+k)
    bounds = dict(seed=(0, 2**31-1), diagnostic_samples=(2, 32), diagnostic_updates=(8, 5000),
                  feature_warmup_updates=(0, 1000), train_samples=(8, 10000), val_samples=(2, 128),
                  clean_epochs=(1, 20), continue_epochs=(1, 8), grad_accum=(1, 8),
                  fidelity_frames_per_condition=(4, 8), save_images=(0, 8))
    for k, (lo, hi) in bounds.items():
        if type(c[k]) is not int or not lo <= c[k] <= hi: raise ValueError('Invalid integer '+k)
    for k in set(c)-set(fixed)-set(bounds)-{'schema'}:
        v = c[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError('Invalid positive finite '+k)
    if max(c['learning_rate'], c['fine_learning_rate']) > .001: raise ValueError('LR cap')
    if c['feature_warmup_updates'] >= c['diagnostic_updates']: raise ValueError('No CE updates left')
    if c['diagnostic_samples'] > c['train_samples'] or c['save_images'] > c['val_samples']:
        raise ValueError('Subset exceeds parent')
    if c['reserve_free_gib'] >= c['min_free_gib']: raise ValueError('Reserve must be smaller')
    for k in ('overfit_wer_gate', 'memory_margin_gate', 'clean_validation_wer_gate'):
        if not 0 < c[k] < 1: raise ValueError('Invalid gate '+k)
    return c


def offline_snapshot(directory, kind):
    root = Path(directory).resolve(strict=True)
    names = ['config.json', 'pytorch_model.bin', 'vocab.json', 'merges.txt']
    if kind == 'clip': names += ['preprocessor_config.json']
    elif kind != 'bart': raise ValueError('Unknown backbone')
    for n in names:
        f = root/n
        if not f.is_file() or f.is_symlink(): raise FileNotFoundError('Missing local '+kind+' file: '+str(f))
    config = json.loads((root/'config.json').read_text())
    if config.get('model_type') != kind: raise ValueError('Wrong model_type at '+str(root))
    if kind == 'bart':
        expected = dict(d_model=768, vocab_size=50265, encoder_layers=6, decoder_layers=6,
                        bos_token_id=0, pad_token_id=1, eos_token_id=2, decoder_start_token_id=2)
        if any(config.get(k) != v for k, v in expected.items()): raise ValueError('facebook/bart-base required')
    selected = names + [n for n in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'README.md') if (root/n).is_file()]
    hashes = {n: digest(root/n) for n in selected}
    # User's download receipt, if present, is checked, not silently ignored.
    sums = root/'SHA256SUMS'
    if sums.is_file():
        for line in sums.read_text().splitlines():
            if not line.strip(): continue
            sha, name = line.split(None, 1); name = name.strip().lstrip('*')
            if Path(name).name != name or digest(root/name) != sha: raise ValueError('Download checksum mismatch '+name)
    return dict(path=str(root), files_sha256=hashes, offline_only=True,
                expected_bart_download_revision=BART_REVISION if kind == 'bart' else None,
                revision_provenance='expected download pin; local hashes recorded, not remotely attested')


def quality_gate(groups, cfg, diagnostic=True):
    wer = groups['matched']['covered_wer']
    margin = {k: groups[k]['covered_wer']-wer for k in ('zero', 'shuffled')}
    limit = cfg['overfit_wer_gate'] if diagnostic else cfg['clean_validation_wer_gate']
    return dict(passed=bool(wer <= limit and min(margin.values()) >= cfg['memory_margin_gate']),
                matched_wer=wer, margins=margin, quality_limit=limit,
                memory_margin_limit=cfg['memory_margin_gate'], groups=groups,
                scope='TRAIN memorization gate' if diagnostic else 'exploratory validation, not held-out test')


def make_row(item, arm, condition, received, **extra):
    return dict(image_id=item['image_id'], source_text=item['text'], covered_text=item['covered_text'],
                received_caption=received, arm=arm, condition=condition,
                input_truncated=item['truncated'], stopped_by_capacity=extra.pop('stopped_by_capacity', False),
                source_sha256=item['source_sha256'], source_utf8_bytes=item['source_utf8_bytes'],
                **metrics(item['text'], item['covered_text'], received), **extra)


def tensor_signature(tensor):
    a = tensor.detach().cpu().contiguous().numpy()
    header = json.dumps(dict(shape=list(a.shape), dtype=str(a.dtype)), sort_keys=True).encode()
    return hashlib.sha256(header+a.tobytes()).hexdigest()
