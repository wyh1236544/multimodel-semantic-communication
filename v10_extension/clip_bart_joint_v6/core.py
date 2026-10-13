"""V6 policy is explicit; wire budget unchanged; masking is a TRAIN surrogate."""
import json
import hashlib
from pathlib import Path
import numpy as np
from clip_bart_joint_v5.core import geometry, digest, metrics, summarize, text_gate, make_row
from bart_latent_codec_v4.core import caption_view
from fso_fluctuating_v2.core import seed_for

TEMPLATE = Path(__file__).with_name('config.json')


def load(path=TEMPLATE):
    c = json.loads(Path(path).read_text())
    baseline = json.loads(TEMPLATE.read_text())
    # Deliberately fixed first experiment, no hidden hyperparameter sweep.
    if c != baseline:
        raise ValueError('Use published V6 config; changes require a new declared experiment')
    return c


def private_mask(c, step, image_id, phase):
    image, text = [], []
    if phase != 'common':
        rng = np.random.default_rng(seed_for(c['seed'], 'v6-private-mask', step, image_id))
        if step % 4 in (1, 3):
            image = sorted(int(v) for v in rng.choice(224, int(224*c['private_drop_fraction']), replace=False))
        if step % 4 in (2, 3):
            text = sorted(int(v) for v in rng.choice(32, int(32*c['private_drop_fraction']), replace=False))
    return dict(image_slots=image, text_slots=text, public_payload_complex_symbols=3072,
        scope='TRAIN receiver-feature dropout only; not FSO/packet loss or bandwidth saving')


def source_mask(arm):
    from clip_bart_joint_v6 import ARMS
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    blocked = np.zeros((32, 32), dtype=bool)
    if arm == 'J0_separate':
        blocked[:16, 16:] = True
        blocked[16:, :16] = True
    return blocked


def attach_text_source_hashes(items, joint_receipt, text_receipt):
    """Add text-only identity metadata; preserve all original joint identities.

    Cross-check the earlier independently cached BART caption view. No tensor,
    model input, tokenizer output, or receiver metadata is changed.
    """
    ids=[item['image_id'] for item in items]
    if ids!=joint_receipt['image_ids'] or len(set(ids))!=len(ids):
        raise ValueError('Joint source IDs/order differ')
    split=joint_receipt['split']
    if split not in ('train','validation') or text_receipt['split']!=split:
        raise ValueError('Joint/text receipt split differs')
    if joint_receipt.get('mode')!='joint' or joint_receipt['annotation_sha256']!=text_receipt['annotation_sha256']:
        raise ValueError('Joint/text original annotation identity differs')
    text_ids=set(text_receipt['image_ids'])
    result={}
    for item in items:
        key=str(item['image_id']); text=item['text']
        if item['split']!=split or item['image_id'] not in text_ids:
            raise ValueError('Joint text not in declared BART split')
        if not isinstance(text,str) or len(text.encode('utf-8'))!=item['source_utf8_bytes']:
            raise ValueError('Joint original text/UTF8 length differs')
        if item['annotation_id']!=joint_receipt['annotation_ids'].get(key) or item['annotation_id']!=text_receipt['annotation_ids'].get(key):
            raise ValueError('Joint/text selected annotation differs')
        if item['source_sha256']!=joint_receipt['source_sha256'].get(key):
            raise ValueError('Joint image/text identity differs')
        text_sha=hashlib.sha256(text.encode('utf-8')).hexdigest()
        if text_sha!=text_receipt['source_sha256'].get(key):
            raise ValueError('Joint original text differs from independent BART view')
        result[key]=text_sha
    return dict(joint_receipt, text_source_sha256=result,
        text_source_sha256_scope='normalized original COCO caption UTF8; audit metadata only, never receiver input')
