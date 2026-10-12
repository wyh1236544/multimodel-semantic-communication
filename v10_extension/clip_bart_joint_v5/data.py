"""Reuse exact original COCO image/cache policy, V4 codec loss anchors only."""
import torch
from bart_coco_modal_v2.data import cache as original_cache


@torch.no_grad()
def cache(args, c, old, reference, text_model, tok, clip, processor, split, count, device, budget):
    items, receipt = original_cache(args, c, 'joint', old, reference, text_model.bart, tok,
        clip, processor, split, count, device, budget)
    for item in items:
        budget(); x = text_model.transmit(item['bart_hidden'].to(device), item['mask'].to(device))
        item['text_anchor'] = text_model.receive_text(x)['memory'].cpu()
    receipt.update(anchor_scope='frozen generalized text codec, TRAIN loss only; never supplied at receiver',
        clip_text_caption_same_as_bart=True)
    return items, receipt


def source(model, item, device, donor_text=None):
    # Optional source-mismatch diagnostic, distinct from trained no-CLIP ablation.
    text = item if donor_text is None else donor_text
    return model.source_encode(text['bart_hidden'].to(device), text['mask'].to(device),
        item['image_wire'].to(device), item['clip_image'].to(device), text['clip_text'].to(device))


def clean_args(wire):
    b = wire.shape[0]
    return wire, torch.ones(b, device=wire.device, dtype=torch.bool), wire.new_full((b,), 40.), wire.new_ones(b)
