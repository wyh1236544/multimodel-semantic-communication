"""Required on server; explicit skips on hosts lacking existing Torch/HF."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
READY = all(importlib.util.find_spec(x) is not None for x in ('torch','transformers'))
if os.environ.get('V20_REQUIRE_TORCH')=='1' and not READY:
    raise RuntimeError('Existing Torch/Transformers required; do not install automatically')

@unittest.skipUnless(READY,'Torch/HF unavailable locally; mandatory server gate')
class ModelContracts(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_equal_parameters_and_all_groups_real_gradients(self):
        import torch
        from docci_text_guided_image_v20.model import GuidedImage,select_sides
        counts = []
        for arm in ('A_self','B_tx_text','C_tx_rx_text'):
            torch.manual_seed(42)
            model = GuidedImage()
            z = torch.randn(1,4,64,64)
            s,m = select_sides(arm,torch.randn(1,32,512),torch.randn(1,288,768))
            wire = model.encode(z,s)
            self.assertAlmostEqual(float(wire.square().mean()),1.,places=4)
            out = model.receive(wire,m,wire.new_tensor([10]),wire.new_tensor([.9]))
            loss = (out-z).square().mean()
            loss.backward()
            for group,prefixes in model.groups().items():
                parameters = [p for n,p in model.named_parameters() if n.startswith(prefixes)]
                self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in parameters),group)
            counts.append(sum(p.numel() for p in model.parameters()))
        self.assertEqual(len(set(counts)),1)

    def test_received_side_cannot_modify_tx_and_self_ignores_text(self):
        import torch
        from docci_text_guided_image_v20.model import GuidedImage,select_sides
        m = GuidedImage().eval()
        z = torch.randn(1,4,64,64)
        with torch.no_grad():
            a = m.encode(z,None)
            b = m.encode(z,None)
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(select_sides('B_tx_text','source','rx1'),('source',None))
        self.assertEqual(select_sides('A_self','source','rx2'),(None,None))

    def test_checkpoint_exact_roundtrip_and_wrong_schema(self):
        import torch
        from docci_text_guided_image_v20.model import GuidedImage
        m = GuidedImage()
        state = m.saved()
        m.restore(state)
        self.assertTrue(all(torch.equal(state[k],v) for k,v in m.saved().items()))
        state.pop(next(iter(state)))
        with self.assertRaises(ValueError):
            m.restore(state)

    def test_zero_fusion_recovers_original_spatial_codec(self):
        import torch
        from docci_text_guided_image_v20.model import GuidedImage
        m = GuidedImage().eval()
        for fusion in (m.tx32,m.tx16,m.rx32,m.rx64):
            fusion.alpha.data.zero_()
        z = torch.randn(1,4,64,64)
        with torch.no_grad():
            wire = m.encode(z,None)
            original = m.base.encode(dict(latent=z))
            self.assertTrue(torch.equal(wire,original))
            snr,rel = wire.new_tensor([10]),wire.new_tensor([.8])
            new = m.receive(wire,None,snr,rel)
            old = m.base.receive(wire,torch.ones(1,dtype=torch.bool),snr,rel)['latent']
            self.assertTrue(torch.equal(new,old))

    def test_awgn_gradient_no_rx_renormalization_pilot_accounting(self):
        import torch
        from docci_text_guided_image_v20.channel import channel
        wire = torch.ones(1,8192,requires_grad=True)
        received,estimate,reliability,s = channel(wire,0,0,'test','image','test')
        received.sum().backward()
        self.assertTrue(torch.equal(wire.grad,torch.ones_like(wire)))
        self.assertFalse(s['post_noise_renormalization'])
        self.assertGreater(float(received.square().mean()),1.5)
        self.assertIsNotNone(s['estimated_snr_db'])

    def test_real_assets_lpips_identity_gradient_and_optional_upstream_parity(self):
        import torch
        from docci_text_guided_image_v20.perceptual import Perceptual
        assets = os.environ.get('V20_LPIPS_ASSETS')
        if not assets:
            self.skipTest('Real offline assets tested by saved-backbone smoke')
        metric = Perceptual(assets)
        x = torch.rand(1,3,64,64,requires_grad=True)
        self.assertLess(abs(float(metric(x,x).sum())),1e-9)
        value = metric(x,torch.zeros_like(x)).mean()
        value.backward()
        self.assertTrue(bool(x.grad.abs().sum()>0))
        self.assertTrue(all(p.grad is None for p in metric.parameters()))
        # A second forward using functional convolutions verifies layer placement,
        # normalization, scale and calibration without requiring the lpips package.
        import torch.nn.functional as F
        def features(image):
            image = (image*2-1-metric.shift)/metric.scale
            result = []
            for i,conv in enumerate(metric.convs):
                if i in (1,2):
                    image = F.max_pool2d(image,kernel_size=3,stride=2)
                image = F.relu(F.conv2d(image,conv.weight,conv.bias,
                    stride=4 if i==0 else 1,padding=2 if i<2 else 1))
                result.append(image/(image.square().sum(1,keepdim=True).sqrt()+1e-10))
            return result
        expected = sum(((a-b).square()*getattr(metric,'weight%d'%i)).sum(1).mean()
            for i,(a,b) in enumerate(zip(features(x.detach()),features(torch.zeros_like(x)))))
        self.assertAlmostEqual(float(value),float(expected),places=6)

if __name__=='__main__':
    unittest.main(verbosity=2)
