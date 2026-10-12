"""Mandatory server Torch/HF tests. Missing dependencies fail, never silently skip."""
import inspect
from pathlib import Path
import sys
import unittest
import torch
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from bart_clip_joint_v1.test_model import toy,inputs
from bart_coco_modal_v2.model import ModalBridge
from bart_coco_modal_v2.channel import clean


def modal():
    old=toy(); return ModalBridge(old.bart,old.image_reference,dict(hidden=16,heads=4,layers=1),clip_width=16)


def encode(m,a):
    if m.modality=='image_only': return m.payload(m.source_encode(image_wire=a[2],clip_image=a[3]))
    if m.modality=='text_only': return m.payload(m.source_encode(a[0],a[1],clip_text=a[4]))
    return m.payload(m.source_encode(*a))


class TorchContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)
    def test_same_registered_architecture(self):
        a,b=toy(),modal(); self.assertEqual(set(a.state_dict()),set(b.state_dict()))
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
    def test_compact_and_unit_power(self):
        for mode,n in [('text_only',64),('image_only',256),('joint',288)]:
            m=modal(); m.configure_mode(mode); x=encode(m,inputs()); self.assertEqual(tuple(x.shape),(1,n,16))
            self.assertAlmostEqual(float(x.square().mean()),1.,places=5)
    def test_reject_opposite_source(self):
        m=modal(); a=inputs(); m.configure_mode('text_only')
        with self.assertRaises(ValueError): m.source_encode(*a)
        m.configure_mode('image_only')
        with self.assertRaises(ValueError): m.source_encode(*a)
    def test_private_has_no_common_source(self):
        m=modal(); m.configure_mode('text_only',True); x=encode(m,inputs()); self.assertEqual(float(x[:,:32].abs().sum()),0.)
        self.assertFalse(m.common_tx.weight.requires_grad); self.assertTrue(m.text_tx.weight.requires_grad)
    def test_text_source_padding_not_leaked(self):
        m=modal(); m.configure_mode('text_only'); a=inputs(); b=[v.clone() for v in a]; b[0][:,7:]=10000
        self.assertTrue(torch.allclose(encode(m,a),encode(m,b),atol=1e-5))
    def test_all_used_learning_groups_update_and_backbones_frozen(self):
        for mode in ('text_only','image_only','joint'):
            torch.manual_seed(3); m=modal(); m.configure_mode(mode); a=inputs(); before=m.bridge_state()
            frozen={n:p.clone() for n,p in m.named_parameters() if not p.requires_grad}
            x=encode(m,a); out=m.receive_payload(*clean(x)); loss=out['common_alignment'].square().mean()
            if mode!='image_only': loss=loss+m.teacher_loss(torch.tensor([[0,4,5,6,7,8,2]+[1]*121]),a[1],out['text_memory'])
            if mode!='text_only': loss=loss+out['latent'].square().mean()
            opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.001); loss.backward()
            for name,prefix in m.learning_groups().items():
                self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and bool(p.grad.abs().sum()>0) for n,p in m.named_parameters() if n.startswith(prefix)),name)
            opt.step(); after=m.bridge_state()
            for name,prefix in m.learning_groups().items(): self.assertTrue(any(not torch.equal(before[n],after[n]) for n in before if n.startswith(prefix)),name)
            self.assertTrue(all(p.grad is None and torch.equal(p,frozen[n]) for n,p in m.named_parameters() if n in frozen))
    def test_receiver_failure_erases_task_outputs(self):
        for mode in ('text_only','image_only','joint'):
            m=modal(); m.configure_mode(mode); x=encode(m,inputs())
            out=m.receive_payload(x,torch.zeros(1),torch.tensor([-10.]),torch.tensor([False]))
            for key in ('latent','text_memory'):
                if key in out: self.assertEqual(float(out[key].abs().sum()),0.)
    def test_receiver_has_no_source_and_image_does_not_decode_text(self):
        m=modal(); self.assertEqual(list(inspect.signature(m.receive_payload).parameters),['payload','reliability','pilot_snr','available'])
        m.configure_mode('image_only'); out=m.receive_payload(*clean(encode(m,inputs())))
        self.assertNotIn('text_memory',out)
    def test_full_state_survives_mode_switch(self):
        a,b=modal(),modal(); a.configure_mode('text_only'); state=a.bridge_state(); b.configure_mode('image_only'); b.load_bridge(state)
        self.assertTrue(all(torch.equal(v,b.bridge_state()[n]) for n,v in state.items()))
        self.assertFalse(any(n.startswith(('bart.','image_reference.')) for n in state))
        with self.assertRaises(ValueError): b.load_bridge({})
    def test_joint_exact_parent_forward(self):
        a,b=toy(),modal(); b.load_state_dict(a.state_dict()); x=inputs()
        self.assertTrue(torch.allclose(a.source_encode(*x),b.source_encode(*x),atol=1e-6))
    def test_grad_and_no_grad_consistency(self):
        for mode in ('image_only','text_only','joint'):
            m=modal(); m.configure_mode(mode); a=inputs(); x=encode(m,a); out=m.receive_payload(*clean(x))
            with torch.no_grad(): xx=encode(m,a); other=m.receive_payload(*clean(xx))
            self.assertLessEqual(float((x-xx).abs().max()),.0002)
            for key in ('text_memory','latent'):
                if key in out: self.assertLessEqual(float((out[key]-other[key]).abs().max()),.0002)


if __name__=='__main__': unittest.main(verbosity=2)
