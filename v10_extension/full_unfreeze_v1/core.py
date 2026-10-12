"""Bounded full-unfreeze screening, immutable channel and common references."""
import json
import math
from pathlib import Path
from fso_fluctuating_v2.core import (load as load_channel, digest, seed_for,
    paths, transport, schedule, environment)

TEMPLATE = Path(__file__).with_name('config.json')
TAIL_ROOT = 'base.vae_wrapper.vae.decoder.'
CHANNEL_KEYS = ('seed', 'fs_hz', 'sps', 'rolloff', 'span', 'pilots',
    'bank_train_frames', 'bank_holdout_frames', 'sequence_window_frames',
    'frame_spacing_s', 'coherence_time_s', 'clean_anchor_fraction',
    'papr_weight', 'papr_target_db', 'distance_m', 'attenuation_db_per_km',
    'wavelength_m', 'beam_divergence_half_angle_rad',
    'receiver_aperture_diameter_m', 'weather', 'conditions', 'primary_conditions')


def tail_prefixes(blocks):
    if type(blocks) is not int or not 1 <= blocks <= 8:
        raise ValueError('Unsupported VAE decoder up_blocks')
    return (TAIL_ROOT+'up_blocks.'+str(blocks-1)+'.',
            TAIL_ROOT+'conv_norm_out.', TAIL_ROOT+'conv_out.')


def load(path=TEMPLATE):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = json.loads(TEMPLATE.read_text(encoding='utf-8'))
    if set(value) != set(expected) or value['schema'] != 'full-unfreeze-v1':
        raise ValueError('Unknown/missing experiment configuration')
    for key, lo, hi in [('train_samples',8,10000),('epochs',1,8),
            ('grad_accum',1,8),('warmup_optimizer_steps',0,500),
            ('val_samples',2,128),('primary_validation_offset',0,127),
            ('draws',1,4),('save_images',0,8),('bootstrap_resamples',100,10000)]:
        if type(value[key]) is not int or not lo <= value[key] <= hi:
            raise ValueError('Invalid integer '+key)
    if (value['primary_validation_offset'] >= value['val_samples'] or
            value['save_images'] > value['val_samples']):
        raise ValueError('Invalid validation subset')
    for key, maximum in [('tail_lr',1e-4),('backbone_lr',1e-4),('auxiliary_lr',1e-4),
            ('min_free_gib',48.),('reserve_free_gib',16.),('arm_max_hours',6.)]:
        if not isinstance(value[key],(int,float)) or not math.isfinite(value[key]) or not 0 < value[key] <= maximum:
            raise ValueError('Invalid '+key)
    if value['warmup_optimizer_steps'] >= value['epochs']*math.ceil(value['train_samples']/value['grad_accum']):
        raise ValueError('Encoder must actually be jointly trained')
    if value['reserve_free_gib'] >= value['min_free_gib']:
        raise ValueError('GPU startup gate must exceed sampled free reserve')
    channel = load_channel()
    channel.update(value)
    return channel


def parameter_group(name, prefixes):
    # First three rules reproduce the existing semantic/tail optimizer exactly.
    from_semantic_encoder = ('base.multimodal_encoder.', 'base.bottleneck.')
    from_semantic_decoder = ('base.joint_decoder.', 'base.image_head.', 'base.text_head.',
                            'base.text_direct_adapter.', 'base.text_direct_gate_logit')
    if name.startswith(prefixes): return 'tail'
    if name.startswith(from_semantic_encoder): return 'encoder'
    if name.startswith('base.channel.'): return 'mapping'
    if name.startswith(from_semantic_decoder): return 'decoder'
    if name.startswith(('base.vae_wrapper.', 'base.text_encoder.')): return 'backbone'
    return 'auxiliary'


def check_bank(config, bank):
    if bank.get('smoke'):
        raise ValueError('Full completed V2 calibration required, not a smoke bank')
    for key in CHANNEL_KEYS:
        if config[key] != bank['config'][key]:
            raise ValueError('Reused channel configuration differs: '+key)


def check_parent(payload, record, design, identity, hashes, bank_hash):
    if (payload.get('version') != 'V10-FSO-fluctuating-v2' or
            payload.get('arm') != 'B2_fso_ft' or design.get('smoke')):
        raise ValueError('Completed full V2 B2 FSO delta required')
    if record.get('arm') != 'B2_fso_ft':
        raise ValueError('Wrong parent arm')
    for key, value in hashes.items():
        if payload.get(key) != value or identity.get(key) != value:
            raise ValueError('Parent initialization identity differs: '+key)
    for key, value in [('calibration_sha256',bank_hash),
            ('init_awgn_delta_sha256',identity['init_awgn_delta_sha256'])]:
        if payload.get(key) != value or identity.get(key) != value:
            raise ValueError('Parent calibration/initialization differs: '+key)
    if (payload.get('steps') != design['train_forward_steps'] or
            payload.get('optimizer_steps') != design['optimizer_steps'] or
            record.get('steps') != payload['steps'] or
            record.get('optimizer_steps') != payload['optimizer_steps']):
        raise ValueError('Incomplete parent training budget')


def check_history_pair(a, b):
    fields = ('epoch','image_id','bank_index','anchor','task_loss_applied',
              'network_seed','noise_seed','reference_latent_sha256','posterior_noise_sha256')
    if len(a) != len(b) or any(tuple(x[k] for k in fields) != tuple(y[k] for k in fields)
                              for x,y in zip(a,b)):
        raise ValueError('Unequal paired data/channel/network/task schedule')


def visual_positions(config):
    n, count, offset = config['val_samples'],config['save_images'],config['primary_validation_offset']
    if offset == 0:
        return list(range(count))
    first = min((count+1)//2,offset)
    positions = list(range(first))+list(range(offset,min(n,offset+count-first)))
    positions += [i for i in range(n) if i not in positions][:count-len(positions)]
    return positions
