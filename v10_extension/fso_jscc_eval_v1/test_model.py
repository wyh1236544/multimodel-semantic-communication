"""Torch CPU tests; fail honestly if Torch is not available (never silently skip)."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fso_jscc_v1 import ARMS, VERSION as PARENT_VERSION
from fso_jscc_eval_v1.checkpoint import apply_delta, advance_parent_vae_rng
from fso_jscc_eval_v1.core import state_fingerprint


class FrozenContracts(unittest.TestCase):
    def setUp(self):
        self.model=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.Linear(3,2)).float()
        self.named=list(self.model.named_parameters()); self.source=dict(config=dict(synthetic=True))
        self.record=dict(arm=ARMS[3],steps=2,optimizer_steps=1)
        self.payload=dict(version=PARENT_VERSION,arm=ARMS[3],steps=2,optimizer_steps=1,config=self.source['config'],
            base_checkpoint_sha256='base',a_delta_sha256='A',split_sha256='split',delta={n:torch.full_like(q,.125) for n,q in self.named})
    def apply(self,payload): apply_delta(self.model,self.named,payload,self.record,self.source,'base','A','split')
    def test_exact_apply_freeze_and_forward_immutability(self):
        self.apply(self.payload); self.assertFalse(self.model.training)
        self.assertTrue(all(not p.requires_grad for p in self.model.parameters()))
        self.assertTrue(all(torch.equal(p,self.payload['delta'][n]) for n,p in self.named))
        before=state_fingerprint(self.model)
        with torch.no_grad(): self.model(torch.ones(1,2))
        self.assertEqual(before,state_fingerprint(self.model))
    def test_reject_every_bad_tensor_before_any_mutation(self):
        last=self.named[-1][0]
        for kind in ('missing','extra','shape','dtype','nan','wrong_arm'):
            with self.subTest(kind=kind):
                payload=copy.deepcopy(self.payload)
                if kind=='missing': del payload['delta'][last]
                elif kind=='extra': payload['delta']['other']=torch.ones(1)
                elif kind=='shape': payload['delta'][last]=torch.ones(11)
                elif kind=='dtype': payload['delta'][last]=payload['delta'][last].double()
                elif kind=='nan': payload['delta'][last].fill_(float('nan'))
                else: payload['arm']=ARMS[2]
                before=state_fingerprint(self.model)
                with self.assertRaises(ValueError): self.apply(payload)
                self.assertEqual(before,state_fingerprint(self.model))
    def test_fingerprint_detects_buffers_and_weights(self):
        self.model.register_buffer('probe',torch.ones(1)); before=state_fingerprint(self.model)
        self.model.probe.add_(1); self.assertNotEqual(before,state_fingerprint(self.model))
        before=state_fingerprint(self.model)
        with torch.no_grad(): self.named[0][1].add_(1)
        self.assertNotEqual(before,state_fingerprint(self.model))
    def test_omitted_cache_rng_matches_actual_posterior_draws(self):
        class SampleVAE:
            def encode_images(self,image): return torch.ones(1,4,2,2)+torch.randn(1,4,2,2)*.5
        wrapper=SampleVAE(); image=torch.zeros(1,3,16,16)
        torch.manual_seed(42)
        for _ in range(7): wrapper.encode_images(image)
        expected=wrapper.encode_images(image)
        torch.manual_seed(42); receipt=advance_parent_vae_rng(wrapper,image,7)
        self.assertTrue(torch.equal(expected,wrapper.encode_images(image)))
        self.assertFalse(receipt['training_images_read'])
        class ExtraRandomVAE:
            def encode_images(self,image):
                torch.rand(3); return wrapper.encode_images(image)
        before=torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError): advance_parent_vae_rng(ExtraRandomVAE(),image,7)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))


if __name__=='__main__': unittest.main(verbosity=2)
