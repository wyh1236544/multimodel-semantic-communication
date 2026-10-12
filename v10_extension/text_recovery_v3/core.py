"""Pure protocol/config contracts. Original length is never an RX argument."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from text_recovery_v3 import ARMS, OPTIONAL_ARMS
from bart_coco_modal_v2.core import caption_view, metrics, summarize, tensor_signature

TEMPLATE = Path(__file__).with_name('config.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path=TEMPLATE):
    c = json.loads(Path(path).read_text())
    t = json.loads(TEMPLATE.read_text())
    if set(c) != set(t) or c['schema'] != 'text-recovery-v3':
        raise ValueError('Unknown/missing configuration')
    bounds = dict(seed=(0, 2**31-1), hidden=(256, 256), heads=(4, 4), layers=(2, 2),
        max_tokens=(128, 128), diagnostic_samples=(2, 32), diagnostic_steps=(8, 5000),
        native_steps=(8, 2000), warmup_steps=(0, 1000), ramp_steps=(1, 1000),
        train_samples=(8, 10000), val_samples=(2, 128), clean_epochs=(1, 20),
        grad_accum=(1, 8), rank=(8, 8), lora_layers=(2, 2), snapshot_every=(20, 1000))
    for k, (lo, hi) in bounds.items():
        if type(c[k]) is not int or not lo <= c[k] <= hi: raise ValueError('Invalid '+k)
    for k in set(c)-set(bounds)-{'schema'}:
        if isinstance(c[k], bool) or not isinstance(c[k], (int, float)) or not math.isfinite(c[k]) or c[k] <= 0:
            raise ValueError('Invalid finite positive '+k)
    if c['warmup_steps'] >= c['diagnostic_steps'] or c['diagnostic_samples'] > c['train_samples']:
        raise ValueError('Inconsistent diagnostic budget')
    if max(c[k] for k in ('legacy_lr', 'bridge_lr', 'lora_lr')) > .001:
        raise ValueError('Unsafe learning rate')
    if c['reserve_free_gib'] >= c['min_free_gib'] or c['forward_tolerance'] > .001:
        raise ValueError('Unsafe memory/forward policy')
    if any(not 0 < c[k] < 1 for k in ('overfit_wer_gate', 'clean_validation_wer_gate', 'memory_margin_gate')):
        raise ValueError('Quality gate out of range')
    return c


def spec(arm):
    if arm not in ARMS+OPTIONAL_ARMS: raise ValueError('Unknown arm')
    return dict(arm=arm, width=64 if arm == 'A4_wide' else 16,
        common=arm == 'A5_common', length_mask=arm != 'A0_legacy',
        ramp=arm not in ('A0_legacy', 'A1_length_mask'),
        lora=arm in ('A3_lora', 'A4_wide', 'A5_common'))


def geometry(arm):
    s = spec(arm); feature_reals = 32*s['width']+(512 if s['common'] else 0)
    header = 16 if s['length_mask'] else 0
    return dict(feature_reals=feature_reals, length_header_reals=header,
        length_header_complex_symbols=header//2, total_reals=feature_reals+header,
        payload_complex_symbols=(feature_reals+header)//2, raw_utf8_wire_bytes=0,
        image_slots=0, text_slots=32, common_slots=32 if s['common'] else 0,
        width=s['width'], analog_repeats=1, length_information_bits=7 if header else 0,
        length_code='7 bits of (length-1), each twice, 2 public padding reals; no CRC' if header else None)


def encode_header(length):
    if type(length) is not int or not 2 <= length <= 128: raise ValueError('BOS/EOS-inclusive length 2..128')
    bits = ((length-1) >> np.arange(7)) & 1
    return np.concatenate((np.repeat(bits*2.-1., 2), np.ones(2))).astype(np.float32)


def decode_header(observed):
    x = np.asarray(observed)
    if x.shape != (16,) or not np.isfinite(x).all(): raise ValueError('Observed finite header only')
    bits = x[:14].reshape(7, 2).sum(1) >= 0
    length = 1+int(np.sum(bits*(1 << np.arange(7))))
    return length, bool(2 <= length <= 128)


def schedule(step, total, c, arm, diagnostic=True):
    s = spec(arm)
    warm = c['warmup_steps'] if diagnostic else min(c['warmup_steps'], total//4)
    ce = 0. if step < warm else min(1., (step-warm+1)/c['ramp_steps']) if s['ramp'] else 1.
    return dict(ce_weight=ce, feature_weight=c['feature_weight'] if s['ramp'] else .2,
        pad_weight=c['pad_feature_weight'] if s['ramp'] else 0.)


def gate(groups, c, diagnostic=True):
    w = groups['matched']['covered_wer']
    limit = c['overfit_wer_gate'] if diagnostic else c['clean_validation_wer_gate']
    margins = {k: groups[k]['covered_wer']-w for k in ('zero', 'shuffled')}
    no_cap = groups['matched']['capacity_stop_fraction'] == 0
    return dict(passed=bool(w <= limit and min(margins.values()) >= c['memory_margin_gate'] and no_cap),
        quality_limit=limit, matched_wer=w, margins=margins, no_capacity_stops=no_cap,
        memory_margin_limit=c['memory_margin_gate'], scope='TRAIN memorization' if diagnostic else 'exploratory validation')


def row(item, arm, control, result, **extra):
    return dict(image_id=item['image_id'], annotation_id=item['annotation_id'], arm=arm, control=control,
        source_split=item['split'], source_text=item['text'], covered_text=item['covered_text'],
        source_sha256=item['source_sha256'], source_utf8_bytes=item['source_utf8_bytes'],
        covered_tokens=item['covered_tokens'],
        received_caption=result['text'], input_truncated=item['truncated'],
        stopped_by_capacity=result['stopped_by_capacity'], receiver_failure=result['receiver_failure'],
        **metrics(item['text'], item['covered_text'], result['text']), **extra)
