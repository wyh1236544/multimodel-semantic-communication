"""Server Torch CPU tests, no Transformers/data/checkpoints/GPU required."""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fso_jscc_v1.core import load, numpy_channel
from fso_jscc_v1.model import ContinuousChannel, Frame, parameters, stage, gradient_groups


class DummyTransport:
    def probe(self):
        return 10.

    def continuous(self, values, repeats, equalizer):
        return np.full((288, 16), .25, np.float32), .5, {'analog_receiver_available': True}


class TorchTests(unittest.TestCase):
    def channel(self, profile='fso_moderate'):
        old = nn.Module(); old.adapter = nn.Identity(); old.use_adapter = False
        cfg = load(); channel = ContinuousChannel(old, cfg)
        frame = Frame({}, 0., 1, torch.device('cpu'), config=cfg, context=dict(profile=profile, snr=0., field_seed=4, noise_seed=7), fixed_mode=0)
        channel.frame = frame
        return channel, frame, cfg

    def test_numpy_matches_torch_and_backward(self):
        channel, frame, cfg = self.channel()
        x = torch.randn(1, 288, 16, requires_grad=True)
        y = channel(x)
        expected, _, _ = numpy_channel(frame.captured, cfg, 'fso_moderate', 0., 4, 7)
        self.assertTrue(np.allclose(y.detach().numpy()[0], expected, atol=3e-6))
        y.square().mean().backward()
        self.assertGreater(float(x.grad.square().sum()), 0)
        for mapping in (channel.tx_map, channel.rx_map):
            self.assertGreater(float(mapping.last.weight.grad.square().sum()), 0)

    def test_identity_initialization_clean_and_group_power(self):
        channel, frame, _ = self.channel('clean')
        x = torch.randn(1, 288, 16)
        y = channel(x)
        self.assertTrue(np.allclose(y.detach().numpy()[0], frame.captured, atol=1e-6))
        for start, end in ((0, 32), (32, 256), (256, 288)):
            self.assertAlmostEqual(float(y[:, start:end].square().mean()), 1., places=5)
        self.assertEqual(channel.waveform_papr_db.numel(), 1)

    def test_no_source_bypass_in_waveform_receiver(self):
        channel, _, cfg = self.channel()
        with torch.no_grad():
            def receive(x):
                channel.frame = Frame({}, 0., 1, torch.device('cpu'), context={'transport': DummyTransport()}, config=cfg, fixed_mode=0)
                return channel(x)
            a = receive(torch.randn(1, 288, 16))
            b = receive(torch.randn(1, 288, 16)*3)
        self.assertTrue(torch.equal(a, b))

    def test_waveform_forbids_fake_straight_through_gradient(self):
        channel, _, cfg = self.channel()
        channel.frame = Frame({}, 0., 1, torch.device('cpu'), context={'transport': DummyTransport()}, config=cfg, fixed_mode=0)
        with self.assertRaises(ValueError):
            channel(torch.randn(1, 288, 16, requires_grad=True))

    def test_joint_encoder_is_trainable_but_backbones_are_frozen(self):
        model = nn.Module(); model.base = nn.Module()
        for name in ('multimodal_encoder', 'bottleneck', 'joint_decoder', 'image_head', 'text_head', 'vae_wrapper', 'text_encoder'):
            setattr(model.base, name, nn.Linear(16, 16))
        model.base.channel = self.channel()[0]
        stage(model, True)
        self.assertFalse(model.base.multimodal_encoder.weight.requires_grad)
        stage(model, False)
        self.assertTrue(model.base.multimodal_encoder.weight.requires_grad)
        self.assertFalse(model.base.vae_wrapper.weight.requires_grad)
        self.assertFalse(model.base.text_encoder.weight.requires_grad)
        self.assertTrue(any(n.startswith('base.channel.tx_map.') for n, _ in parameters(model)))

    def test_receiver_erasure_cannot_become_bias_evidence(self):
        channel, frame, _ = self.channel()
        channel.rx_map.last.bias.data.fill_(10.)
        frame.header_ok.fill_(False)
        self.assertEqual(int(torch.count_nonzero(channel(torch.randn(1, 288, 16)))), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
