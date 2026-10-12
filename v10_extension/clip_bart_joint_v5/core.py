"""Explicit equal wire budget; source-separation mask and strict run settings."""
import json
import math
from pathlib import Path
import numpy as np
from clip_bart_joint_v5 import ARMS
from bart_latent_codec_v4.core import digest, caption_view, metrics, summarize, batch_indices

TEMPLATE = Path(__file__).with_name('config.json')


def load(path=TEMPLATE):
    c = json.loads(Path(path).read_text()); template = json.loads(TEMPLATE.read_text())
    if set(c) != set(template) or c['schema'] != template['schema']: raise ValueError('Unknown/missing config')
    for k, v in dict(hidden=256, heads=4, text_width=64).items():
        if type(c[k]) is not int or c[k] != v: raise ValueError('Fixed geometry '+k)
    bounds = dict(seed=(0, 2**31-1), text_train_samples=(8, 10000), text_updates=(8, 50000),
        train_samples=(8, 2000), val_samples=(2, 128), clean_epochs=(1, 12), continue_epochs=(1, 8),
        grad_accum=(1, 8), joint_warm_steps=(0, 1000), clean_anchor_period=(2, 8),
        snapshot_every=(20, 1000), save_images=(0, 8))
    for k, (lo, hi) in bounds.items():
        if type(c[k]) is not int or not lo <= c[k] <= hi: raise ValueError('Invalid integer '+k)
    for k in set(c)-set(bounds)-{'schema', 'hidden', 'heads', 'text_width'}:
        if type(c[k]) not in (int, float) or not math.isfinite(c[k]) or c[k] <= 0: raise ValueError('Invalid '+k)
    if c['train_samples'] > c['text_train_samples'] or c['joint_warm_steps'] >= c['train_samples']*c['clean_epochs']:
        raise ValueError('Inconsistent warm/data budget')
    if max(c[k] for k in ('joint_lr', 'text_lr', 'continue_lr')) > .001 or c['reserve_free_gib'] >= c['min_free_gib']:
        raise ValueError('Unsafe LR/memory')
    if c['save_images'] > c['val_samples'] or c['forward_tolerance'] > .001: raise ValueError('Invalid subset/tolerance')
    if not all(0 < c[k] < 1 for k in ('memory_margin_gate', 'validation_wer_gate')): raise ValueError('Invalid quality gates')
    return c


def geometry():
    return dict(common_slots=32, common_width=16, image_slots=224, image_width=16,
        text_slots=32, text_width=64, common_reals=512, image_reals=3584, text_reals=2048,
        total_reals=6144, payload_complex_symbols=3072, analog_repeats=1,
        raw_utf8_wire_bytes=0, source_length_header_reals=0, reverse_feedback_symbols=0,
        public_routing='C[0:512],I[512:4096],T[4096:6144]; fixed32 BART memory')


def source_mask(arm):
    if arm not in ARMS: raise ValueError('Unknown arm')
    # Columns: 224 image tokens, CLIP image, CLIP text. True means blocked.
    mask = np.zeros((32, 226), dtype=bool)
    if arm == 'J0_separate':
        mask[:16, 225] = True; mask[16:, :225] = True
    return mask


def text_gate(groups, c):
    w = groups['matched']['covered_wer']
    margins = {k: groups[k]['covered_wer']-w for k in ('zero', 'shuffled')}
    eos = groups['matched']['capacity_stop_fraction'] == 0
    return dict(passed=bool(w <= c['validation_wer_gate'] and min(margins.values()) >= c['memory_margin_gate'] and eos),
        groups=groups, matched_wer=w, margins=margins, natural_eos_all=eos,
        quality_limit=c['validation_wer_gate'], memory_margin_limit=c['memory_margin_gate'])


def make_row(item, arm, phase, condition, result, image_metrics, **extra):
    return dict(image_id=item['image_id'], annotation_id=item['annotation_id'], source_split=item['split'],
        arm=arm, phase=phase, condition=condition, source_text=item['text'], covered_text=item['covered_text'],
        source_sha256=item['source_sha256'], source_utf8_bytes=item['source_utf8_bytes'],
        input_truncated=item['truncated'], received_caption=result['text'],
        stopped_by_capacity=result['stopped_by_capacity'], receiver_failure=result['receiver_failure'],
        geometry=geometry(), **metrics(item['text'], item['covered_text'], result['text']), **image_metrics, **extra)
