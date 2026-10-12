"""Required server Torch contracts; missing Torch is an ERROR, never a skip/pass."""
import copy
from pathlib import Path
import sys
import unittest
import torch
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from residual_fusion_v2.model import ResidualFusion,image_side,frozen_receive
from residual_fusion_v2.core import parameter_counts
from joint_jscc_arch_v1.test_model import toy


class TorchContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)
    def inputs(self):
        torch.manual_seed(3)
        return (torch.randn(1,4,16,16),torch.randn(1,272,32),torch.randn(1,48,32),torch.ones(1,288,1)*.5,torch.tensor([-2.]),torch.tensor([True]))
    def test_zero_initialization_exact_baseline(self):
        m=ResidualFusion(32,16); inp=self.inputs(); out,g=m(*inp)
        self.assertTrue(torch.equal(out,inp[0])); self.assertGreater(float(g[0]),0.)
    def test_disabled_exact_after_training(self):
        m=ResidualFusion(32,16); torch.nn.init.normal_(m.output.weight); inp=self.inputs(); out,g=m(*inp,disabled=True)
        self.assertTrue(torch.equal(out,inp[0])); self.assertEqual(float(g[0]),0.)
    def test_unavailable_zero_gate(self):
        m=ResidualFusion(32,16); torch.nn.init.normal_(m.output.weight); inp=list(self.inputs()); inp[-1]=torch.tensor([False]); out,g=m(*inp)
        self.assertTrue(torch.equal(out,inp[0])); self.assertEqual(float(g[0]),0.)
    def test_zero_text_is_exact_baseline(self):
        m=ResidualFusion(32,16); torch.nn.init.normal_(m.output.weight); inp=list(self.inputs()); inp[2]=torch.zeros_like(inp[2]); out,g=m(*inp)
        self.assertTrue(torch.equal(out,inp[0]))
    def test_shape_and_parameter_scale(self):
        m=ResidualFusion(256,64)
        self.assertEqual({n:p.numel() for n,p in m.named_parameters()},parameter_counts(256,64))
        self.assertEqual(image_side(torch.zeros(1,272,256)).shape,(1,48,256))
    def test_all_required_groups_learn(self):
        m=ResidualFusion(32,16); before={n:p.detach().clone() for n,p in m.named_parameters()}; opt=torch.optim.Adam(m.parameters(),lr=.01)
        inp=self.inputs()
        for _ in range(8):
            opt.zero_grad(); out,g=m(*inp); loss=(out-.5).square().mean(); loss.backward(); opt.step()
        for prefix in ('query.','memory.','attention.','gate.','output.'):
            self.assertGreater(sum(float((p.detach()-before[n]).square().sum()) for n,p in m.named_parameters() if n.startswith(prefix)),0.)
    def test_equal_initialization_and_exact_reload(self):
        a=ResidualFusion(32,16); b=ResidualFusion(32,16); b.load_state_dict(a.state_dict(),strict=True)
        self.assertTrue(all(torch.equal(v,b.state_dict()[n]) for n,v in a.state_dict().items()))
    def test_frozen_B4_capture_no_text_leakage(self):
        torch.manual_seed(2); core=toy(); core.configure('B4_separate'); core.requires_grad_(False); core.eval()
        r=torch.randn(1,288,16); rel=torch.ones(1,288,1); snr=torch.tensor([0.]); ok=torch.tensor([True])
        a=frozen_receive(core,r,rel,snr,ok); changed=r.clone(); changed[:,16:32]+=2; changed[:,256:]+=2
        b=frozen_receive(core,changed,rel,snr,ok)
        self.assertEqual(a['image_memory'].shape,(1,272,32)); self.assertEqual(a['text_memory'].shape,(1,48,32))
        self.assertTrue(torch.allclose(a['image_memory'],b['image_memory'],atol=1e-5))
        m=ResidualFusion(32,16); out,g=m(a['latent'],a['image_memory'],a['text_memory'],rel,snr,ok); out.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in core.parameters()))


if __name__=='__main__': unittest.main(verbosity=2)
