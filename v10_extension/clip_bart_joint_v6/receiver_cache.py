"""Train-only received conditions for the separate semantic prompt adapter."""
import numpy as np
import torch
from clip_bart_joint_v5.data import source, clean_args
from bart_latent_codec_v4.channel import proxy_tensor
from bart_latent_codec_v4.model import generate


@torch.no_grad()
def save_training_conditions(model, items, c, tc, tok, device, budget, path):
    common, captions, targets, ids, noisy, snrs = [], [], [], [], [], []
    for i,item in enumerate(items[:c['generation_train_samples']]):
        budget()
        if item['split']!='train':
            raise ValueError('Adapter training cache must be TRAIN only')
        wire=source(model,item,device); args=clean_args(wire)
        if i%3:
            observed,stats=proxy_tensor(wire,tc,i,[item['image_id']])
            args=(observed,args[1],wire.new_tensor([stats[0]['pilot_snr_db']]),wire.new_tensor([.5]))
        out=model.receive(*args); result=generate(model.text.bart,out['text'],tok)
        common.append(out['common_wire'][0].cpu().numpy())
        captions.append(result['text']); targets.append(item['text']); ids.append(item['image_id']); noisy.append(bool(i%3))
        snrs.append(float(args[2][0]))
    np.savez_compressed(path,common=np.stack(common),received_caption=np.asarray(captions),
        source_text_train_teacher=np.asarray(targets),image_id=np.asarray(ids),proxy_noisy=np.asarray(noisy),
        source_split=np.asarray(['train']*len(ids)),pilot_snr_db=np.asarray(snrs))
