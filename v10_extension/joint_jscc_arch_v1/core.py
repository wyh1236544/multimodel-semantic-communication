"""Public fixed layout, strict configuration, text boundaries and paired design."""
import json
import math
from pathlib import Path
import numpy as np
from fso_fluctuating_v2.core import load as load_channel, seed_for, digest

EXT = Path(__file__).resolve().parents[1]
PROJECT = EXT.parent
TEMPLATE = Path(__file__).with_name('config.json')
WIRE_GROUPS = (0,)*16 + (1,)*16 + (0,)*224 + (1,)*32
DECODE_GROUPS = (0,)*16 + (1,)*16 + (0,)*256 + (1,)*32
IMAGE_INDICES = ((np.arange(256)*137) % 256)[:224].tolist()


def load(path=TEMPLATE):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = json.loads(TEMPLATE.read_text())
    if set(c) != set(expected) or c['schema'] != expected['schema']:
        raise ValueError('Unknown/missing architecture configuration')
    for key, lo, hi in [('seed',0,2**31-1),('train_samples',8,10000),
            ('clean_epochs',1,8),('noise_epochs',1,8),('grad_accum',1,8),
            ('val_samples',2,128),('draws',1,4),('save_images',0,8),
            ('max_text_tokens',77,192),('common_text_tokens',4,48),
            ('codec_hidden',16,256),('codec_layers',1,4),
            ('bootstrap_resamples',100,10000),('fidelity_frames_per_condition',4,12)]:
        if type(c[key]) is not int or not lo <= c[key] <= hi:
            raise ValueError('Invalid integer '+key)
    if c['codec_hidden'] % 4 or c['common_text_tokens'] > c['max_text_tokens']:
        raise ValueError('Invalid attention/text dimensions')
    if c['save_images'] > c['val_samples'] or c['reserve_free_gib'] >= c['min_free_gib']:
        raise ValueError('Invalid image/memory limits')
    for key in ('old_lr','new_lr','weight_decay','latent_weight','pixel_weight',
                'ssim_weight','text_weight','common_latent_weight','common_text_weight',
                'min_free_gib','reserve_free_gib','arm_max_hours'):
        if not isinstance(c[key],(int,float)) or isinstance(c[key],bool) or not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError('Invalid positive finite '+key)
    if max(c['old_lr'],c['new_lr']) > .001 or c['arm_max_hours'] > 6:
        raise ValueError('Learning rate/time cap out of range')
    name = Path(c['channel_config'])
    if name.is_absolute() or '..' in name.parts or name.parts[:1] != ('v10_extension',):
        raise ValueError('Project-local channel configuration required')
    channel = load_channel(PROJECT/name)
    if c['fidelity_frames_per_condition'] > channel['bank_holdout_frames']:
        raise ValueError('Not enough public holdout calibration frames')
    return c, channel


def chunks(content_ids, maximum, bos, eos, pad, chunk_length=75):
    """Source-only CLIP chunks plus an EOS-terminated AR target. No RX length."""
    if maximum < 3 or not 1 <= chunk_length <= 75:
        raise ValueError('Invalid source capacity')
    content = list(map(int,content_ids))[:maximum-2]
    target = [bos]+content+[eos]
    length = len(target)
    target += [pad]*(maximum-length)
    pieces = []
    for start in range(0,max(1,len(content)),chunk_length):
        part = content[start:start+chunk_length]
        ids = [bos]+part+[eos]
        mask = [1]*len(ids)+[0]*(77-len(ids))
        ids += [pad]*(77-len(ids))
        pieces.append((ids,mask,len(part)))
    return dict(target_ids=target,target_length=length,chunks=pieces,
                source_content_tokens=len(content_ids),covered_content_tokens=len(content),
                truncated=len(content_ids)>maximum-2)


def arm_options(arm):
    from joint_jscc_arch_v1 import ARMS
    if arm not in ARMS[1:]: raise ValueError('New-architecture arm required')
    return dict(modern=arm!='B1_text',auxiliary=arm in ('B3_common','B4_separate'),
                separate=arm=='B4_separate')


def source_groups(text_length):
    return (0,)*16+(1,)*16+(0,)*256+(1,)*text_length


def check_pair_histories(histories):
    fields = ('phase','epoch','image_id','bank_index','anchor','task_loss_applied','noise_seed')
    reference = None
    for records in histories:
        identity = [tuple(r[k] for k in fields) for r in records]
        if reference is None: reference = identity
        elif identity != reference: raise ValueError('Unpaired training schedule')


def expected_keys(arms, ids, conditions, draws):
    return {(a,i,c['id'],r) for a in arms for i in ids for c in conditions for r in range(draws)} | {
        (a,i,'clean',0) for a in arms for i in ids}


def bootstrap(values, identity, resamples):
    x = np.asarray(values,dtype=float)
    if not len(x): return dict(mean=None,ci95=None,images=0)
    rng = np.random.default_rng(seed_for('architecture-image-bootstrap',identity))
    sample = x[rng.integers(len(x),size=(resamples,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(sample,[.025,.975]).tolist(),images=len(x),
                confidence_scope='image clusters conditional on fixed weather and one training replicate')

