"""Exactly the old independent I2/released64; no new neural parameters."""
from pathlib import Path
import torch
import torch.nn.functional as F
from docci_direct_token_v28.core import read, sha, PARENTS
from docci_direct_token_v28.data import parent, features


def make(args, task, device, checkpoint=None):
    from docci_modal_v9.backbones import load, historical_backbone
    backbone, tok, identity = load(task, args, device)
    historical_backbone(task, identity['files'])
    if identity['files'] != read(features(args, task)/'backbone.json')['files']:
        raise ValueError('Backbone differs from native feature cache')
    path = parent(args, task)
    record = torch.load(str(path), map_location='cpu')
    if task == 'image':
        from docci_reconstruction_v10.model import ImageBranch
        if (record.get('schema'), record.get('task'), record.get('arm')) != ('v14-checkpoint', 'image', 'I2_shifted'):
            raise ValueError('Expected completed pure V14 I2 spatial codec')
        model = ImageBranch(dict(image_size=512, image_width=8), 'I2_shifted').to(device)
    else:
        from docci_baseline_v14.model import TextModel
        if (record.get('schema'), record.get('arm'), record.get('updates')) != ('v18-text-checkpoint', 'noise_replay', 2560):
            raise ValueError('Expected V18 noise-replay text codec')
        model = TextModel(backbone, 'released64').to(device)
    model.restore(record['state']); del record
    if checkpoint:
        checkpoint = Path(checkpoint)
        done = read(checkpoint.parent/'TRAIN_COMPLETE.json')
        if sha(checkpoint) != done['checkpoint_sha256']:
            raise ValueError('V28 final checkpoint changed')
        record = torch.load(str(checkpoint), map_location='cpu')
        if (record['schema'], record['task'], record['parent_sha256']) != (
                'v28-checkpoint', task, PARENTS[task]['sha256']):
            raise ValueError('Wrong V28 checkpoint identity')
        model.restore(record['state']); del record
    model.eval(); backbone.eval().requires_grad_(False)
    if task == 'text' and sum(p.numel() for p in model.parameters() if p.requires_grad) != 25618432:
        raise ValueError('Text parameter geometry changed')
    return model, backbone, tok, identity


def source(item, task, device):
    keys = ('latent',) if task == 'image' else ('chunk_ids', 'chunk_hidden')
    return {k: torch.as_tensor(item[k].copy(), device=device)[None] for k in keys}


def encode(model, src, task):
    return model.encode(src) if task == 'image' else model.encode(src['chunk_ids'], src['chunk_hidden'])


def receive(model, observed, task, stats):
    if task == 'text':
        return model.receive(observed)  # NO source IDs/mask/length or SNR truth.
    return model.receive(observed, torch.ones(1, dtype=torch.bool, device=observed.device),
        observed.new_tensor([stats['conditioning_snr']]), observed.new_tensor([stats['reliability']]))


def text_objective(model, ids, rx):
    from bart_latent_codec_v4.model import prediction
    flat = ids.flatten(0, 1)
    logits, labels = prediction(model.bart, flat, flat.ne(1), rx['memory'].flatten(0, 1),
                                rx['receiver_mask'].flatten(0, 1))
    summed = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction='sum')
    return summed, int(labels.ne(-100).sum())


def pixels(backbone, latent):
    return (backbone.decode(latent/float(backbone.config.scaling_factor)).sample+1)/2


def image_objective(backbone, latent, src, item, device):
    rgb = pixels(backbone, latent)  # frozen weights, but preserve decoder input gradient
    target = torch.as_tensor(item['rgb'].copy(), device=device).permute(2, 0, 1)[None].float()/255.
    l, t, r, b = map(int, item['rectangle'])
    return F.mse_loss(rgb[:, :, t:b, l:r], target[:, :, t:b, l:r])+.01*F.mse_loss(latent, src['latent'].float())
