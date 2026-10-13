"""Receiver-local frozen visual semantics, never clean transmitter features."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def receiver_features(own,vae,scaling,clip,processor):
    pixel=((vae.decode(own['latent']/scaling).sample+1)/2).clamp(0,1)
    size=int(clip.config.vision_config.image_size)
    pixel=F.interpolate(pixel,size=(size,size),mode='bicubic',align_corners=False).clamp(0,1)
    cfg=processor.image_processor
    mean=pixel.new_tensor(cfg.image_mean)[None,:,None,None]; std=pixel.new_tensor(cfg.image_std)[None,:,None,None]
    features=F.normalize(clip.get_image_features(pixel_values=(pixel-mean)/std),dim=-1)
    own=dict(own,received_clip=features*own['text']['available'].to(features)[:,None])
    return own
