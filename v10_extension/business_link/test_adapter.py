"""Server CPU tests: real torch, small tensors; no COCO/CLIP/VAE download."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from business_link.adapter import BusinessFrame, receiver_text_hook
from business_link.engine import identity_seed


class AdapterTests(unittest.TestCase):
    def test_digital_text_features_and_failure(self):
        context = {'policy': '16qam_conv', 'source_text': 'a real received text',
                   'field_seed': 1, 'noise_seed': 2}
        frame = BusinessFrame({'family': 'awgn'}, 40, 1, torch.device('cpu'),
                              settings={}, context=context, fixed_mode=0)
        x = torch.linspace(-1, 1, 32*16).reshape(1, 32, 16)
        restored, reliability = frame.payload(x, torch.ones(1, 32), torch.ones(1, 32))
        self.assertEqual(frame.received_text, context['source_text'])
        self.assertTrue(frame.text_ok)
        self.assertTrue(bool(frame.header_ok[0]))
        self.assertTrue((reliability == 1).all())
        self.assertEqual(tuple(restored.shape), tuple(x.shape))
        bad = BusinessFrame({'family': 'awgn'}, -30, 1, torch.device('cpu'),
                            settings={}, context=context, fixed_mode=0)
        erased, _ = bad.payload(x, torch.ones(1, 32), torch.ones(1, 32))
        self.assertFalse(bad.text_ok)
        self.assertEqual(bad.received_text, '')
        self.assertFalse(bool(bad.header_ok[0]))
        self.assertTrue((erased == 0).all())

    def test_received_text_hook_no_source_leak_and_failure_fallback(self):
        from types import SimpleNamespace
        calls = []
        def encoder(texts, device):
            calls.extend(texts)
            return {'tokens': torch.ones(1, 3, 4), 'attention_mask': torch.tensor([[1, 1, 0]])}
        model = SimpleNamespace(base=SimpleNamespace(text_encoder=encoder),
            last_frame=SimpleNamespace(text_ok=False, received_text='', fusion_used=False))
        def refiner(image, memory, reliability=None):
            return image+memory.mean(1, keepdim=True), torch.tensor(1.)
        hook = receiver_text_hook(model, refiner, .25)
        image, memory = torch.zeros(1, 2, 4), torch.zeros(1, 3, 4)
        original, _ = refiner(image, memory)
        fallback, _ = hook(image, memory)
        self.assertTrue(torch.equal(fallback, original)); self.assertEqual(calls, [])
        model.last_frame.text_ok, model.last_frame.received_text = True, 'CRC-recovered only'
        assisted, _ = hook(image, memory)
        self.assertEqual(calls, ['CRC-recovered only'])
        self.assertTrue(torch.all(assisted == .25))
        self.assertTrue(model.last_frame.fusion_used)

    def test_paired_receiver_and_policy_probes(self):
        probes = []
        for policy in ('analog', '16qam_conv', '16qam_repeat2'):
            frame = BusinessFrame({'family': 'fso', 'alpha': 4., 'beta': 2., 'xi': 1.8, 'a0': .9},
                20, 1, torch.device('cpu'), fixed_mode=0, settings={},
                context={'policy': policy, 'source_text': 'same', 'field_seed': 16, 'noise_seed': 18})
            probes.append((frame.field_identity, float(frame.routing_snr[0])))
        self.assertEqual(len(set(probes)), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
