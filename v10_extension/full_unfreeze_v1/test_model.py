"""Server-only Torch toy contracts; no pretrained models/data/downloads."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from full_unfreeze_v1.model import candidates,stage,install_live_encoder,gradients,eligible_names,optimizer_groups
from full_unfreeze_v1.core import load
from vae_tail_ablation_v1.model import apply_parent,verify_frozen


class VAE(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=nn.Linear(2,2); self.quant_conv=nn.Linear(2,2)
        self.post_quant_conv=nn.Linear(2,2); self.decoder=nn.Module()
        self.decoder.up_blocks=nn.ModuleList([nn.Linear(2,2),nn.Linear(2,2)])
        self.decoder.conv_norm_out=nn.LayerNorm(2); self.decoder.conv_out=nn.Linear(2,2)
    def encode(self,x):
        mean=self.quant_conv(torch.tanh(self.encoder(x)))
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=mean,std=torch.ones_like(mean)*.2))
    def decode(self,x):
        x=self.post_quant_conv(x)
        for block in self.decoder.up_blocks: x=torch.tanh(block(x))
        return self.decoder.conv_out(self.decoder.conv_norm_out(x))


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__(); self.vae=VAE(); self.freeze=True; self.scaling_factor=.2
        self.detach_latents=True; self.use_fp16=False
    def _vae_dtype(self): return next(self.vae.parameters()).dtype
    def decode_latents(self,x): return self.vae.decode(x/self.scaling_factor)


class Text(nn.Module):
    def __init__(self):
        super().__init__(); self.freeze=True; self.backbone=nn.Linear(2,2); self.projection=nn.Linear(2,2)
    def forward(self,x):
        if self.freeze:
            with torch.no_grad(): y=self.backbone(x)
        else: y=self.backbone(x)
        return self.projection(y)


class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.base=nn.Module(); b=self.base
        b.vae_wrapper=Wrapper(); b.text_encoder=Text(); b.vae_tokenizer=nn.Linear(2,2)
        b.multimodal_encoder=nn.Linear(2,2); b.bottleneck=nn.Linear(2,2)
        b.channel=nn.Module(); b.channel.tx_map=nn.Linear(2,2); b.channel.rx_map=nn.Linear(2,2)
        b.joint_decoder=nn.Linear(2,2); b.image_head=nn.Linear(2,2); b.text_head=nn.Linear(2,2)
        b.inactive=nn.Linear(2,2)
    def forward(self,x,t):
        b=self.base; latent=b.vae_wrapper.encode_images(x)
        z=b.multimodal_encoder(b.vae_tokenizer(latent)+b.text_encoder(t)); z=b.bottleneck(z)
        z=b.channel.rx_map(b.channel.tx_map(z)); z=b.joint_decoder(z)
        return b.vae_wrapper.decode_latents(b.image_head(z)),b.text_head(z),latent


class Contracts(unittest.TestCase):
    def test_true_live_path_and_frozen_control(self):
        for full in (False,True):
            torch.manual_seed(10); m=Model(); named,prefixes=candidates(m); install_live_encoder(m.base.vae_wrapper)
            m.base.vae_wrapper.posterior_noise=torch.randn(4,2)
            before={n:p.detach().clone() for n,p in named}; stage(m,False,full,prefixes)
            allowed=eligible_names(m,full,prefixes)
            optimizer=torch.optim.AdamW(optimizer_groups(named,load(),full,prefixes))
            for group in optimizer.param_groups: group['lr']=.01
            x=torch.rand(4,2); t=torch.randn(4,2)
            image,text,latent=m(x,t); self.assertEqual(latent.requires_grad,full)
            loss=(image-.3).square().mean()+text.square().mean(); loss.backward()
            grads=gradients(named,prefixes)
            self.assertGreater(grads['vae_tail'],0.)
            for key in ('clip','vae_encoder','text_projection'):
                self.assertEqual(grads[key]>0,full)
            optimizer.step(); verify_frozen(m,before,allowed)
            for scope in ('base.vae_wrapper.vae.encoder.','base.vae_wrapper.vae.decoder.up_blocks.0.','base.text_encoder.backbone.'):
                self.assertEqual(sum(float((p.detach()-before[n]).square().sum()) for n,p in named if n.startswith(scope))>0,full)
            # Full eligibility must NOT secretly activate unused branches.
            self.assertTrue(all(p.grad is None for p in m.base.inactive.parameters()))
            self.assertTrue(all(torch.equal(p.detach(),before[n]) for n,p in named if n.startswith('base.inactive.')))

    def test_same_epsilon_same_encoder_and_gradient(self):
        m=Model(); named,p=candidates(m); install_live_encoder(m.base.vae_wrapper)
        m.base.vae_wrapper.posterior_noise=torch.zeros(3,2); x=torch.rand(3,2)
        stage(m,False,False,p); a=m.base.vae_wrapper.encode_images(x)
        stage(m,False,True,p); b=m.base.vae_wrapper.encode_images(x)
        self.assertTrue(torch.equal(a,b)); b.square().mean().backward()
        self.assertGreater(float(m.base.vae_wrapper.vae.encoder.weight.grad.square().sum()),0.)
        m.base.vae_wrapper.posterior_noise=torch.zeros(1,2)
        with self.assertRaises(ValueError): m.base.vae_wrapper.encode_images(x)

    def test_full_delta_roundtrip_and_bad_keys(self):
        m=Model(); named,_=candidates(m); data={n:p.detach().clone() for n,p in named}
        apply_parent(m,named,dict(delta=data))
        with self.assertRaises(ValueError): apply_parent(m,named,dict(delta={}))
        data[named[0][0]]=torch.full_like(named[0][1],float('nan'))
        with self.assertRaises(ValueError): apply_parent(m,named,dict(delta=data))


if __name__=='__main__': unittest.main(verbosity=2)
