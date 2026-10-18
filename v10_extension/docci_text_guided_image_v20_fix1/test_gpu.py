"""Actual CUDA determinism contracts. SKIP locally, mandatory before any smoke."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
READY=all(importlib.util.find_spec(n) is not None for n in ('torch','transformers'))

def require_cuda():
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or not torch.cuda.is_available():
        raise RuntimeError('Physical GPU1 required for the actual CUDA repair gate')
    free,_=torch.cuda.mem_get_info(0)
    if free<20*2**30:
        raise RuntimeError('CUDA gate needs 20GiB free; do not stop unrelated jobs')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.set_num_threads(4)

@unittest.skipUnless(READY,'Local Torch/HF absent; actual CUDA gate NOT EXECUTED')
class CUDAContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_cuda()

    def test_pool_forward_gradient_equivalence_and_bitwise_repeat(self):
        import torch
        import torch.nn.functional as F
        from docci_text_guided_image_v20_fix1.compat import fixed_pool
        for h,w in ((16,16),(32,32),(64,64),(16,32)):
            torch.manual_seed(42)
            reference=torch.randn(2,3,h,w,requires_grad=True)
            weight=torch.randn(2,3,4,8)
            expected=F.adaptive_avg_pool2d(reference,(4,8))
            (expected*weight).sum().backward()
            x=reference.detach().cuda().requires_grad_()
            y=fixed_pool(x)
            (y*weight.cuda()).sum().backward()
            self.assertTrue(torch.allclose(y.cpu(),expected,rtol=1e-5,atol=1e-6))
            self.assertTrue(torch.allclose(x.grad.cpu(),reference.grad,rtol=1e-5,atol=1e-6))
            grad=x.grad.clone();value=y.detach().clone();x.grad=None
            again=fixed_pool(x);(again*weight.cuda()).sum().backward()
            self.assertTrue(torch.equal(value,again));self.assertTrue(torch.equal(grad,x.grad))
            self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_all_arms_cuda_backward_update_with_unchanged_parameter_count(self):
        import torch
        from docci_text_guided_image_v20.model import GuidedImage,select_sides
        from docci_text_guided_image_v20_fix1.compat import pooling_patch
        from docci_text_guided_image_v20 import model as module
        counts=[]
        with pooling_patch(module):
            for arm in ('A_self','B_tx_text','C_tx_rx_text'):
                torch.manual_seed(42)
                m=GuidedImage().cuda()
                z=torch.randn(1,4,64,64,device='cuda')
                tx,rx=select_sides(arm,torch.randn(1,32,512,device='cuda'),torch.randn(1,288,768,device='cuda'))
                before={n:p.detach().clone() for n,p in m.named_parameters()}
                optimizer=torch.optim.AdamW(m.parameters(),lr=1e-4)
                wire=m.encode(z,tx)
                out=m.receive(wire,rx,wire.new_tensor([10]),wire.new_tensor([.8]))
                (out-z).square().mean().backward()
                for group,prefixes in m.groups().items():
                    p=[v for n,v in m.named_parameters() if n.startswith(prefixes)]
                    self.assertTrue(any(v.grad is not None and torch.isfinite(v.grad).all() and bool(v.grad.abs().sum()>0) for v in p),group)
                optimizer.step()
                for group,prefixes in m.groups().items():
                    self.assertTrue(any(not torch.equal(v,before[n]) for n,v in m.named_parameters() if n.startswith(prefixes)),group)
                counts.append(sum(p.numel() for p in m.parameters()))
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                del optimizer,m,before,z,wire,out,tx,rx
        self.assertEqual(counts,[3879760]*3)

    def test_lpips_cuda_backward_keeps_frozen_weights_and_determinism(self):
        import torch
        from docci_text_guided_image_v20.perceptual import Perceptual
        assets=os.environ.get('V20_LPIPS_ASSETS')
        if not assets:
            raise RuntimeError('Existing offline LPIPS assets required, not an optional skip')
        metric=Perceptual(assets).cuda()
        torch.manual_seed(42)
        x=torch.rand(1,3,64,64,device='cuda',requires_grad=True)
        loss=metric(x,torch.zeros_like(x)).mean();loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all() and bool(x.grad.abs().sum()>0))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in metric.parameters()))
        self.assertTrue(torch.are_deterministic_algorithms_enabled())

if __name__=='__main__': unittest.main(verbosity=2)
