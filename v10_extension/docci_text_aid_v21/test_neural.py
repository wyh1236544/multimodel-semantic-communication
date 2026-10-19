"""Mandatory server CUDA contracts; explicit local skips without existing Torch."""
import argparse
import importlib.util
from pathlib import Path
import sys
import unittest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[2])]
READY=all(importlib.util.find_spec(n) is not None for n in ('torch','transformers'))
REQUIRE='--require-cuda' in sys.argv

@unittest.skipUnless(READY,'Existing Torch/HF absent locally; mandatory server check')
class Neural(unittest.TestCase):
    def setUp(self):
        import torch
        torch.manual_seed(42);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        if REQUIRE and not torch.cuda.is_available():raise RuntimeError('Real GPU required')
        self.device=torch.device('cuda:0' if REQUIRE else 'cpu')
    def test_receiver_only_training_keeps_exact_TX_and_equal_parameter_counts(self):
        import torch
        from docci_text_guided_image_v20 import model as mod
        from docci_text_guided_image_v20_fix1.compat import pooling_patch
        with pooling_patch(mod):
            counts=[]
            for arm in ('self','correct','donor'):
                torch.manual_seed(42)
                m=mod.GuidedImage().to(self.device).eval().requires_grad_(False)
                before=m.saved()
                active={n:p for n,p in m.named_parameters() if n.startswith(('rx32.','rx64.'))}
                for p in active.values():p.requires_grad_(True)
                counts.append(sum(p.numel() for p in active.values()))
                opt=torch.optim.AdamW(list(active.values()),lr=.001)
                z=torch.randn(1,4,64,64,device=self.device)
                with torch.no_grad():wire=m.encode(z,None)
                memory=None if arm=='self' else torch.randn(1,288,768,device=self.device)
                out=m.receive(wire,memory,wire.new_tensor([0]),wire.new_tensor([.5]))
                (out-z).square().mean().backward()
                self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in active.values()))
                opt.step()
                with torch.no_grad():self.assertTrue(torch.equal(wire,m.encode(z,None)))
                after=m.saved()
                self.assertTrue(all(torch.equal(before[n],after[n]) for n in before if n not in active))
                self.assertTrue(any(not torch.equal(before[n],after[n]) for n in active))
            self.assertEqual(counts,[514018]*3)
    def test_trace_has_no_forward_effect_and_RX_memory_cannot_change_TX(self):
        import torch
        from docci_text_guided_image_v20 import model as mod
        from docci_text_guided_image_v20_fix1.compat import pooling_patch
        from docci_text_aid_v21.runtime import Trace
        with pooling_patch(mod),torch.no_grad():
            m=mod.GuidedImage().to(self.device).eval()
            z=torch.randn(1,4,64,64,device=self.device)
            s=torch.randn(1,32,512,device=self.device)
            memory=torch.randn(1,288,768,device=self.device)
            wire=m.encode(z,s);base=m.receive(wire,memory,wire.new_tensor([0]),wire.new_tensor([.5]))
            trace=Trace(m)
            wire2=m.encode(z,s);out=m.receive(wire2,memory,wire.new_tensor([0]),wire.new_tensor([.5]))
            self.assertTrue(torch.equal(wire,wire2));self.assertTrue(torch.equal(base,out))
            self.assertEqual(set(trace.values),{'tx32','tx16','rx32','rx64'})
            self.assertTrue(all(v['residual_relative_RMS']>=0 for v in trace.values.values()))
            trace.close()
    def test_TX_text_intervention_preserves_unit_power_and_can_change_payload(self):
        import torch
        from docci_text_guided_image_v20 import model as mod
        from docci_text_guided_image_v20_fix1.compat import pooling_patch
        with pooling_patch(mod),torch.no_grad():
            m=mod.GuidedImage().to(self.device).eval()
            z=torch.randn(1,4,64,64,device=self.device)
            s=torch.randn(1,32,512,device=self.device)
            a,b=m.encode(z,s),m.encode(z,torch.zeros_like(s))
            self.assertFalse(torch.equal(a,b))
            self.assertAlmostEqual(float(a.square().mean()),1.,places=4)
            self.assertAlmostEqual(float(b.square().mean()),1.,places=4)

if __name__=='__main__':
    if REQUIRE:
        sys.argv.remove('--require-cuda')
        if not READY:raise RuntimeError('Existing Torch/HF required; no automatic installation')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Neural))
    if not result.wasSuccessful() or (REQUIRE and result.skipped):sys.exit(1)
