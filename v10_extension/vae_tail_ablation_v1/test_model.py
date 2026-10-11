"""Server Torch CPU: exact eligibility, real backprop/update, strict delta replay."""
import sys
import unittest
from pathlib import Path
import torch
from torch import nn
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from vae_tail_ablation_v1.model import candidates,stage,apply_parent,verify_frozen
from fso_jscc_v1.model import parameters as semantic_parameters


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__(); self.freeze=True
        self.vae=nn.Module(); self.vae.encoder=nn.Linear(2,2)
        decoder=nn.Module(); decoder.up_blocks=nn.ModuleList([nn.Linear(2,2),nn.Linear(2,2)])
        decoder.conv_norm_out=nn.LayerNorm(2); decoder.conv_out=nn.Linear(2,2)
        self.vae.decoder=decoder
    def decode_latents(self,x):
        d=self.vae.decoder
        for block in d.up_blocks: x=torch.tanh(block(x))
        return d.conv_out(d.conv_norm_out(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.base=nn.Module()
        self.base.multimodal_encoder=nn.Linear(2,2)
        self.base.joint_decoder=nn.Linear(2,2)
        self.base.text_encoder=nn.Linear(2,2)
        self.base.vae_wrapper=Wrapper()
    def forward(self,x):
        return self.base.vae_wrapper.decode_latents(self.base.joint_decoder(self.base.multimodal_encoder(x)))


class TorchContracts(unittest.TestCase):
    def test_frozen_and_partial_real_updates(self):
        for unfreeze in (False,True):
            torch.manual_seed(3); model=Model(); named,prefixes=candidates(model)
            before={n:p.detach().clone() for n,p in model.named_parameters()}
            stage(model,False,unfreeze,prefixes)
            allowed={n for n,p in model.named_parameters() if p.requires_grad}
            self.assertFalse(any(n.startswith('base.text_encoder.') or n.startswith('base.vae_wrapper.vae.encoder.') for n in allowed))
            self.assertFalse(any(n.startswith('base.vae_wrapper.vae.decoder.up_blocks.0.') for n in allowed))
            optimizer=torch.optim.AdamW([p for _,p in named if p.requires_grad],lr=.01)
            model(torch.randn(4,2)).square().mean().backward()
            tail_grad=sum(float(p.grad.square().sum()) for n,p in named if n.startswith(prefixes) and p.grad is not None)
            self.assertEqual(tail_grad>0,unfreeze); optimizer.step()
            verify_frozen(model,before,allowed)
            change=sum(float((p.detach()-before[n]).square().sum()) for n,p in named if n.startswith(prefixes))
            self.assertEqual(change>0,unfreeze)

    def test_frozen_decoder_still_passes_gradient(self):
        model=Model(); _,prefixes=candidates(model); stage(model,False,False,prefixes)
        x=torch.randn(4,2,requires_grad=True)
        model.base.vae_wrapper.decode_latents(x).square().mean().backward()
        self.assertGreater(float(x.grad.square().sum()),0.)

    def test_strict_replay_and_nonfinite_rejection(self):
        model=Model(); named=semantic_parameters(model)
        data={n:torch.ones_like(p) for n,p in named}; apply_parent(model,named,dict(delta=data))
        self.assertTrue(all(torch.equal(p,data[n]) for n,p in named))
        with self.assertRaises(ValueError): apply_parent(model,named,dict(delta={}))
        data[named[0][0]]=torch.full_like(named[0][1],float('nan'))
        with self.assertRaises(ValueError): apply_parent(model,named,dict(delta=data))


if __name__=='__main__': unittest.main(verbosity=2)
