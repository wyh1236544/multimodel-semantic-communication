"""No neural execution: strict budgets, fixed-memory protocol and metrics."""
import hashlib
import json
import math
from pathlib import Path
from bart_coco_modal_v2.core import caption_view
from text_jscc_v1_fix1.core import metrics, summarize

TEMPLATE = Path(__file__).with_name('config.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path=TEMPLATE):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    t = json.loads(TEMPLATE.read_text(encoding='utf-8'))
    if set(c) != set(t) or c['schema'] != t['schema']:
        raise ValueError('Unknown/missing configuration')
    fixed = dict(slots=32, depth=3, head_dim=64, max_tokens=128)
    for k, v in fixed.items():
        if type(c[k]) is not int or c[k] != v: raise ValueError('Fixed architecture '+k)
    bounds = dict(seed=(0, 2**31-1), batch_size=(1, 8), diagnostic_samples=(2, 32),
        diagnostic_updates=(8, 5000), train_samples=(8, 10000), val_samples=(2, 128),
        clean_updates=(8, 50000), continue_updates=(8, 10000), lr_warmup_updates=(0, 1000),
        snapshot_every=(20, 1000), proxy_pilots=(8, 256))
    for k, (lo, hi) in bounds.items():
        if type(c[k]) is not int or not lo <= c[k] <= hi: raise ValueError('Invalid '+k)
    for k in set(c)-set(fixed)-set(bounds)-{'schema', 'proxy_profiles', 'proxy_snr_db'}:
        v = c[k]
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0: raise ValueError('Invalid '+k)
    if c['diagnostic_samples'] > c['train_samples'] or c['batch_size'] > c['diagnostic_samples']:
        raise ValueError('Batch/subset budget inconsistent')
    if max(c['learning_rate'], c['continue_lr']) > .001 or c['reserve_free_gib'] >= c['min_free_gib']:
        raise ValueError('Unsafe LR/memory policy')
    if c['forward_tolerance'] > .001 or not .5 <= c['kd_temperature'] <= 4:
        raise ValueError('Invalid numerical policy')
    for k in ('overfit_wer_gate', 'clean_validation_wer_gate', 'memory_margin_gate', 'clean_anchor_weight'):
        if not 0 < c[k] < 1: raise ValueError('Invalid gate/weight '+k)
    if not isinstance(c['proxy_snr_db'], list) or not c['proxy_snr_db'] or len(set(c['proxy_snr_db'])) != len(c['proxy_snr_db']):
        raise ValueError('Invalid SNR grid')
    if any(type(v) not in (int, float) or not math.isfinite(v) or not -20 <= v <= 40 for v in c['proxy_snr_db']):
        raise ValueError('Invalid SNR')
    if set(c['proxy_profiles']) != {'fso_moderate', 'fso_strong'}: raise ValueError('Predeclared proxy profiles')
    for profile in c['proxy_profiles'].values():
        if set(profile) != {'alpha', 'beta', 'xi', 'a0', 'loss_db'}: raise ValueError('Profile keys')
        for k, v in profile.items():
            if type(v) not in (int, float) or not math.isfinite(v) or (v < 0 if k == 'loss_db' else v <= 0):
                raise ValueError('Invalid profile '+k)
        if profile['a0'] > 1: raise ValueError('Pointing aperture gain <=1')
    return c


def geometry(width):
    if type(width) is not int or width not in (16, 32, 64): raise ValueError('Width16/32/64 only')
    return dict(text_slots=32, width=width, total_reals=32*width,
        payload_complex_symbols=16*width, analog_repeats=1, length_header_reals=0,
        length_header_complex_symbols=0, raw_utf8_wire_bytes=0, common_slots=0, image_slots=0,
        receiver_memory_slots=32, receiver_mask='fixed public all32; output length predicted by EOS')


def gate(groups, c, diagnostic=True):
    w = groups['matched']['covered_wer']
    margins = {k: groups[k]['covered_wer']-w for k in ('zero', 'shuffled')}
    limit = c['overfit_wer_gate'] if diagnostic else c['clean_validation_wer_gate']
    ended = groups['matched']['capacity_stop_fraction'] == 0
    return dict(passed=bool(w <= limit and min(margins.values()) >= c['memory_margin_gate'] and ended),
        matched_wer=w, quality_limit=limit, margins=margins, memory_margin_limit=c['memory_margin_gate'],
        natural_eos_all=ended, groups=groups, scope='TRAIN memorization' if diagnostic else 'exploratory validation')


def row(item, arm, control, result, width, **extra):
    return dict(image_id=item['image_id'], annotation_id=item['annotation_id'], source_split=item['split'],
        source_text=item['text'], covered_text=item['covered_text'], covered_tokens=item['covered_tokens'],
        source_sha256=item['source_sha256'], source_utf8_bytes=item['source_utf8_bytes'],
        input_truncated=item['truncated'], received_caption=result['text'],
        stopped_by_capacity=result['stopped_by_capacity'], receiver_failure=result['receiver_failure'],
        raw_text_exact=item['text'] == result['text'], arm=arm, control=control, geometry=geometry(width),
        **metrics(item['text'], item['covered_text'], result['text']), **extra)


def batch_indices(size, batch, update, seed):
    # Fixed shuffled epochs; all arms see the same records in the same order.
    import random
    indices = []
    for j in range(update*batch, (update+1)*batch):
        epoch, offset = divmod(j, size)
        order = list(range(size)); random.Random(seed+epoch).shuffle(order)
        indices.append(order[offset])
    return indices
