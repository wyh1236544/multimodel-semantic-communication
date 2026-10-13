"""V6 policy is explicit; wire budget unchanged; masking is a TRAIN surrogate."""
import json
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
