"""Text-only source records, fixed public budget and independently checked gates."""
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
import numpy as np
from v10.data import validate_manifest, sha256
from final_compare_v1.metrics import caption_metrics
from joint_jscc_arch_v1.core import chunks
from fso_fluctuating_v2.core import seed_for

TEMPLATE = Path(__file__).with_name('config.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path=TEMPLATE):
    c = json.loads(Path(path).read_text())
    template = json.loads(TEMPLATE.read_text())
    if set(c) != set(template) or c['schema'] != 'text-only-jscc-v1':
        raise ValueError('Unknown/missing text configuration')
    bounds = {'seed': (0, 2**31-1), 'train_samples': (8, 10000), 'val_samples': (2, 256),
              'batch_size': (1, 8), 'max_tokens': (128, 128), 'slots': (48, 48),
              'width': (16, 16), 'hidden': (256, 256), 'layers': (2, 2), 'heads': (4, 4),
              'diagnostic_samples': (2, 128), 'diagnostic_updates': (8, 5000),
              'clean_epochs': (1, 20), 'continue_epochs': (1, 8),
              'bank_train_frames': (8, 48), 'bank_holdout_frames': (4, 16),
              'fidelity_frames': (4, 8), 'bootstrap_resamples': (100, 10000)}
    for key, (lo, hi) in bounds.items():
        if type(c[key]) is not int or not lo <= c[key] <= hi:
            raise ValueError('Invalid integer '+key)
    for key in set(c)-set(bounds)-{'schema'}:
        if isinstance(c[key], bool) or not isinstance(c[key], (int, float)) or not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError('Invalid finite positive '+key)
    if max(c['learning_rate'], c['fine_learning_rate'], c['diagnostic_lr']) > .001:
        raise ValueError('Learning rate cap')
    if c['diagnostic_samples'] > c['train_samples'] or c['fidelity_frames'] > c['bank_holdout_frames']:
        raise ValueError('Insufficient source/holdout samples')
    if c['reserve_free_gib'] >= c['min_free_gib']:
        raise ValueError('GPU reserve exceeds startup threshold')
    for key in ('overfit_wer_gate', 'memory_margin_gate', 'clean_validation_wer_gate', 'clean_anchor_fraction'):
        if not 0 < c[key] < 1: raise ValueError('Invalid fraction '+key)
    return c


def text_records(data_root, manifest_file, split, count, seed, tokenizer, maximum):
    if split not in ('train', 'validation'):
        raise ValueError('No test/calibration source access in this screen')
    manifest = json.loads(Path(manifest_file).read_text()); validate_manifest(manifest)
    part = 'train' if split == 'train' else 'val'
    source = Path(data_root)/'annotations/localized_narratives'/('coco_'+part+'_captions.jsonl')
    if sha256(source) != manifest['annotation_sha256'][part]:
        raise ValueError('Text annotation identity differs from fixed split')
    ids = list(manifest[split])
    if split == 'train': random.Random(seed).shuffle(ids)
    if not 0 < count <= len(ids): raise ValueError('Not enough split records')
    ids = ids[:count]; wanted = set(ids); found = {}
    with source.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip(): continue
            row = json.loads(line); iid = int(row['image_id'])
            # First nonempty caption per image, paired with old first-caption policy.
            if iid in wanted and iid not in found and str(row.get('caption', '')).strip():
                text = str(row['caption']).strip()
                raw = tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
                info = chunks(raw, maximum, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
                covered = tokenizer.decode(info['target_ids'][:info['target_length']], skip_special_tokens=True)
                found[iid] = dict(image_id=iid, text=text, covered_text=covered, split=split,
                    source_sha256=hashlib.sha256(text.encode()).hexdigest(), source_utf8_bytes=len(text.encode()),
                    target_ids=info['target_ids'], target_length=info['target_length'],
                    source_content_tokens=info['source_content_tokens'], covered_content_tokens=info['covered_content_tokens'],
                    truncated=info['truncated'])
    if set(found) != wanted: raise ValueError('Missing text records; no silent reduction')
    return [found[i] for i in ids]


def metrics(source, covered, received):
    full = caption_metrics(source, received); cut = caption_metrics(covered, received)
    a=source.lower().strip().split(); b=received.lower().strip().split()
    precision=[]
    for n in (1,2):
        aa=Counter(tuple(a[i:i+n]) for i in range(max(0,len(a)-n+1)))
        bb=Counter(tuple(b[i:i+n]) for i in range(max(0,len(b)-n+1)))
        precision.append((sum(min(v,aa[k]) for k,v in bb.items())+1.)/(sum(bb.values())+1.))
    bleu=0. if not a or not b else min(1.,math.exp(1.-len(a)/len(b)))*math.sqrt(precision[0]*precision[1])
    return dict(full, bleu2=bleu, covered_wer=cut['normalized_wer'], covered_cer=cut['text_cer'],
                covered_word_errors=cut['normalized_word_errors'], covered_source_words=cut['normalized_source_words'])


def summarize(rows):
    if not rows: raise ValueError('No empty successful group')
    values = np.asarray([r['covered_wer'] for r in rows], float)
    errors = sum(r['covered_word_errors'] for r in rows)
    words = sum(r['covered_source_words'] for r in rows)
    return dict(entries=len(rows), covered_wer=float(values.mean()), covered_wer_sd=float(values.std(ddof=1)) if len(values)>1 else None,
                covered_corpus_wer=errors/max(words,1), full_wer=float(np.mean([r['normalized_wer'] for r in rows])),
                full_exact_fraction=float(np.mean([r['text_exact'] for r in rows])),
                covered_cer=float(np.mean([r['covered_cer'] for r in rows])),
                bleu2=float(np.mean([r['bleu2'] for r in rows])),
                truncated_fraction=float(np.mean([r['input_truncated'] for r in rows])),
                capacity_stop_fraction=float(np.mean([r['stopped_by_capacity'] for r in rows])))


def overfit_gate(groups, config):
    matched = groups['matched']['covered_wer']
    margins = {k: groups[k]['covered_wer']-matched for k in ('zero', 'shuffled')}
    return dict(passed=matched <= config['overfit_wer_gate'] and min(margins.values()) >= config['memory_margin_gate'],
                matched_wer=matched, memory_margins=margins, source_split='train',
                quality_limit=config['overfit_wer_gate'], memory_margin_limit=config['memory_margin_gate'],
                is_engineering_gate_not_generalization_or_paper_standard=True)


def bootstrap(values, seed, count):
    x = np.asarray(values, float)
    if not len(x) or not np.isfinite(x).all(): raise ValueError('Invalid bootstrap values')
    rng = np.random.default_rng(seed)
    v = x[rng.integers(len(x), size=(count, len(x)))].mean(1)
    return dict(mean=float(x.mean()), ci95=np.quantile(v, [.025, .975]).tolist(), images=len(x),
                scope='image clusters; one trained replicate and fixed weather paths, exploratory validation')


def file_under(root, name):
    p = Path(name)
    if p.is_absolute() or not p.parts or any(x in ('.', '..') for x in p.parts): raise ValueError('Unsafe record path')
    cursor = Path(root)
    for part in p.parts:
        cursor = cursor/part
        if cursor.is_symlink(): raise ValueError('Symlink records refused')
    return cursor
