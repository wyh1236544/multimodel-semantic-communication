"""Synthetic CPU contracts; actual pretrained CUDA gates are launch smoke."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
HAS_TORCH=importlib.util.find_spec('torch') is not None
if os.environ.get('V7_REQUIRE_TORCH')=='1' and not HAS_TORCH: raise RuntimeError('Server Torch contracts cannot be skipped')
if HAS_TORCH:
    import torch
    from bidirectional_rx_v7.model import ReceiverAdapter
    from bidirectional_rx_v7.core import load
    torch.set_num_threads(1)


@unittest.skipUnless(HAS_TORCH,'Torch unavailable: NOT executed locally; required on server')
class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42); self.c=load(); self.m=ReceiverAdapter(64,self.c)
        self.own=dict(latent=torch.randn(2,4,16,16),received_clip=torch.randn(2,512),text=dict(memory=torch.randn(2,32,64),receiver_mask=torch.ones(2,32,dtype=torch.bool),available=torch.tensor([True,False])))
        self.pilot=torch.tensor([1.,2.]); self.rel=torch.tensor([.5,.6])

    def active(self):
        for l in (self.m.image_output,self.m.text_output): torch.nn.init.normal_(l.weight,std=.01)

    def changed(self):
        import copy
        v=copy.deepcopy(self.own); v['latent']+=2; v['text']['memory']+=1
        return v

    def test_initial_B0_and_batch_erasure(self):
        out=self.m(self.own,self.pilot,self.rel)
        self.assertTrue(torch.equal(out['latent'][0],self.own['latent'][0])); self.assertTrue(torch.equal(out['text']['memory'][0],self.own['text']['memory'][0]))
        self.assertEqual(float(out['latent'][1].abs().sum()),0); self.assertEqual(float(out['text']['memory'][1].abs().sum()),0)

    def test_R0_no_cross_information(self):
        self.active(); self.m.configure('R0_self'); out=self.m(self.own,self.pilot,self.rel); other=self.changed()
        textchange=dict(self.own,text=other['text']); imagechange=dict(self.own,latent=other['latent'])
        self.assertTrue(torch.equal(out['latent'],self.m(textchange,self.pilot,self.rel)['latent']))
        self.assertTrue(torch.equal(out['text']['memory'],self.m(imagechange,self.pilot,self.rel)['text']['memory']))

    def test_directional_causality_and_no_control_leak(self):
        self.active(); other=self.changed()
        self.m.configure('R1_text_to_image'); out=self.m(self.own,self.pilot,self.rel); changed=self.m(dict(self.own,text=other['text']),self.pilot,self.rel)
        self.assertGreater(float((out['latent'][0]-changed['latent'][0]).abs().max()),0)
        self.assertTrue(torch.equal(out['text']['memory'],self.m(self.own,self.pilot,self.rel,'zero_text_companion')['text']['memory']))
        self.m.configure('R2_image_to_text'); out=self.m(self.own,self.pilot,self.rel); changed=self.m(dict(self.own,latent=other['latent']),self.pilot,self.rel)
        self.assertGreater(float((out['text']['memory'][0]-changed['text']['memory'][0]).abs().max()),0)
        self.assertTrue(torch.equal(out['latent'],self.m(self.own,self.pilot,self.rel,'zero_image_companion')['latent']))

    def test_R3_off_zero_shuffle_and_no_source_input(self):
        self.active(); self.m.configure('R3_bidirectional'); out=self.m(self.own,self.pilot,self.rel)
        for ctrl in ('cross_off','zero_text_companion','shuffle_text_companion'):
            other=self.m(self.own,self.pilot,self.rel,ctrl,self.changed())
            self.assertGreater(float((out['latent'][0]-other['latent'][0]).abs().max()),0)
        with self.assertRaises(TypeError): self.m(self.own,self.pilot,self.rel,source_text='secret')

    def test_all_groups_learn_and_checkpoint(self):
        self.m.configure('R3_bidirectional'); initial=self.m.saved(); opt=torch.optim.AdamW(self.m.parameters(),lr=.001)
        seen=set()
        for _ in range(4):
            opt.zero_grad(); out=self.m(self.own,self.pilot,self.rel); loss=(out['latent'][0]-1).square().mean()+(out['text']['memory'][0]-1).square().mean(); loss.backward()
            for n,p in self.m.named_parameters():
                if p.grad is not None and p.grad.abs().sum()>0: seen.add(n)
            opt.step()
        saved=self.m.saved()
        for prefix in self.m.groups().values():
            self.assertTrue(any(n.startswith(prefix) for n in seen)); self.assertTrue(any(n.startswith(prefix) and not torch.equal(saved[n],initial[n]) for n in saved))
        self.m.restore(saved)
        with torch.no_grad(): a=self.m(self.own,self.pilot,self.rel)
        b=self.m(self.own,self.pilot,self.rel); self.assertTrue(torch.equal(a['latent'],b['latent']))
        self.assertTrue(torch.equal(a['text']['memory'],b['text']['memory']))


if __name__=='__main__': unittest.main(verbosity=2)
