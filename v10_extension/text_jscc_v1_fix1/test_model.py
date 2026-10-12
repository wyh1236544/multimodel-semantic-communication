"""Mandatory server Torch contracts. Missing Torch is error, never a skipped pass."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch
from torch import nn
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from text_jscc_v1_fix1.model import TextJSCC,legacy_receive,grad_norms
from models.text_reconstruction_head import TextReconstructionHead
from joint_jscc_arch_v1.test_model import toy as b4toy


def legacy_toy():
    old=nn.Module(); old.text_encoder=nn.Module()
    old.text_encoder.tokenizer=SimpleNamespace(vocab_size=64,bos_token_id=1,eos_token_id=2,pad_token_id=2)
    old.text_encoder.backbone=nn.Module(); old.text_encoder.backbone.embeddings=nn.Module()
    old.text_encoder.backbone.embeddings.token_embedding=nn.Embedding(64,16)
    old.text_encoder.projection=nn.Linear(16,16)
    old.text_head=TextReconstructionHead(16,64,77,4,1)
    old.requires_grad_(False)
    return old


def toy():
    old=legacy_toy()
    return TextJSCC(old,dict(hidden=16,slots=48,max_tokens=128,heads=4,layers=1)).eval()


class TorchContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)
    def test_frozen_reference_copy_explicitly_trainable_and_unaliased(self):
        old=legacy_toy(); before={n:p.detach().clone() for n,p in old.named_parameters()}
        m=TextJSCC(old,dict(hidden=16,slots=48,max_tokens=128,heads=4,layers=1)).eval()
        self.assertTrue(all(p.requires_grad for p in m.parameters()))
        self.assertTrue(all(not p.requires_grad for p in old.parameters()))
        for new,reference in zip(m.language.decoder.parameters(),old.text_head.decoder.parameters()):
            self.assertNotEqual(new.data_ptr(),reference.data_ptr())
        ids=torch.tensor([[1,3,4,2]+[2]*124]); lens=torch.tensor([4]); opt=torch.optim.Adam(m.parameters(),lr=.001)
        x=m.encode(ids,lens); h=m.receive(x,torch.tensor([1.]),torch.tensor([40.]),torch.tensor([True]))
        m.language.teacher_loss(ids,lens,h).backward(); opt.step()
        self.assertTrue(all(p.grad is None and not p.requires_grad and torch.equal(p,before[n]) for n,p in old.named_parameters()))
    def test_copied_frozen_decoder_has_gradients_and_actual_updates(self):
        torch.manual_seed(19); m=toy(); before={n:p.detach().clone() for n,p in m.language.decoder.named_parameters()}
        ids=torch.tensor([[1,3,4,2]+[2]*124]); lens=torch.tensor([4]); opt=torch.optim.Adam(m.parameters(),lr=.001)
        for _ in range(3):
            opt.zero_grad(); wire=m.encode(ids,lens)
            h=m.receive(wire,torch.tensor([1.]),torch.tensor([40.]),torch.tensor([True]))
            m.language.teacher_loss(ids,lens,h).backward()
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().max()>0) for p in m.language.decoder.parameters()))
            opt.step()
        self.assertTrue(any(not torch.equal(p,before[n]) for n,p in m.language.decoder.named_parameters()))
    def test_budget_and_unit_power(self):
        m=toy(); ids=torch.tensor([[1,3,4,2]+[2]*124]); x=m.encode(ids,torch.tensor([4]))
        self.assertEqual(x.shape,(1,48,16)); self.assertAlmostEqual(float(x.square().mean()),1.,places=5)
    def test_tied_identity_embedding_and_output(self):
        m=toy(); self.assertIs(m.embedding.weight,m.language.embedding.weight)
        self.assertIs(m.embedding.weight,m.language.output.weight)
    def test_source_padding_cannot_leak(self):
        m=toy(); ids=torch.tensor([[1,3,4,2]+[2]*124]); other=ids.clone(); other[:,4:]=9
        self.assertTrue(torch.allclose(m.encode(ids,torch.tensor([4])),m.encode(other,torch.tensor([4])),atol=1e-6))
    def test_all_source_codec_receiver_decoder_groups_get_gradients(self):
        torch.manual_seed(3); m=toy(); ids=torch.tensor([[1,3,4,2]+[2]*124]); lens=torch.tensor([4]); before={n:p.detach().clone() for n,p in m.named_parameters()}
        opt=torch.optim.Adam(m.parameters(),lr=.001)
        for _ in range(3):
            opt.zero_grad(); x=m.encode(ids,lens); h=m.receive(x+torch.randn_like(x)*.05,torch.tensor([.8]),torch.tensor([2.]),torch.tensor([True]))
            loss=m.language.teacher_loss(ids,lens,h); loss.backward()
            self.assertTrue(all(v>0 for v in grad_norms(m).values())); opt.step()
        self.assertTrue(any(not torch.equal(p,before[n]) for n,p in m.named_parameters()))
    def test_public_receiver_failure_zero_memory(self):
        m=toy(); h=m.receive(torch.randn(1,48,16),torch.tensor([0.]),torch.tensor([-10.]),torch.tensor([False]))
        self.assertTrue(torch.equal(h,torch.zeros_like(h)))
    def test_ar_causal_and_eos_pad(self):
        m=toy(); memory=torch.randn(1,48,16); a=torch.tensor([[1,3,4,5]]); b=a.clone(); b[:,2:]=7
        self.assertTrue(torch.allclose(m.language.logits(a,memory)[:,:2],m.language.logits(b,memory)[:,:2],atol=1e-6))
        self.assertTrue(torch.isfinite(m.language.teacher_loss(torch.tensor([[1,3,2,2]]),torch.tensor([3]),memory)))
    def test_b4_text_independent_of_image_and_equivalent_short_receiver(self):
        torch.manual_seed(4); m=b4toy(); m.configure('B4_separate'); m.eval()
        text=torch.randn(1,128,32); mask=torch.ones(1,128,dtype=torch.bool)
        a=m.source_encode(torch.randn(1,4,16,16),text,mask); b=m.source_encode(torch.randn(1,4,16,16)*10,text,mask)
        indices=list(range(16,32))+list(range(256,288))
        self.assertTrue(torch.allclose(a[:,indices],b[:,indices],atol=1e-5))
        r=torch.randn(1,288,16); snr=torch.tensor([0.]); ok=torch.tensor([True])
        full=m.receive(r,torch.ones(1,288,1),snr,ok)['text_memory']
        short=legacy_receive(m,r[:,indices],1.,snr,ok)
        self.assertTrue(torch.allclose(full,short,atol=1e-5))
    def test_checkpoint_roundtrip(self):
        a=toy(); b=toy(); b.load_state_dict(a.state_dict(),strict=True)
        self.assertTrue(all(torch.equal(v,b.state_dict()[n]) for n,v in a.state_dict().items()))


if __name__=='__main__': unittest.main(verbosity=2)
