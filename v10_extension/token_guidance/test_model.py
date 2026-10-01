"""Torch CPU hooks and surrogate-backward checks. No Transformers/VAE needed."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from token_guidance.model import Guidance, TokenFrame


class Early(nn.Module):
    def forward(self, image, text, mask=None):
        return image+text.mean(1, keepdim=True), image.new_tensor(1.)


class Late(nn.Module):
    def forward(self, image, text, reliability=None):
        return image+text.mean(1, keepdim=True), image.new_tensor(1.)


class Stub(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Module()
        self.base.text_to_image_fusion = Early()
        self.base.decoder_text_to_image_refiner = Late()
        self.base.image_head = nn.Linear(2, 2)
        self.base.text_encoder = nn.Linear(2, 2)


class Checks(unittest.TestCase):
    def test_factorial_and_sender_only_mismatch(self):
        m = Stub()
        g = Guidance(m)
        image, text = torch.zeros(1, 2, 2), torch.ones(1, 3, 2)
        for v, expected in [('A', False), ('B', True), ('C', False), ('D', True)]:
            g.set(v)
            self.assertEqual(not m.base.v10_disable_early_fusion, expected)
        g.set('A')
        restored, _ = m.base.decoder_text_to_image_refiner(image, text)
        torch.testing.assert_close(restored, image)
        g.set('D', 'tx_wrong', sender=(text*3, None))
        guided, _ = m.base.text_to_image_fusion(image, text)
        torch.testing.assert_close(guided, image+3)
        # Sender override must not affect receiver memory.
        restored, _ = m.base.decoder_text_to_image_refiner(image, text)
        torch.testing.assert_close(restored, image+1)

    def test_received_memory_only_and_erased_donor(self):
        m = Stub()
        g = Guidance(m)
        x, memory = torch.zeros(1, 2, 2), torch.ones(1, 3, 2)
        g.set('D', 'rx_wrong', receiver=memory*4)
        y, _ = m.base.decoder_text_to_image_refiner(x, memory)
        torch.testing.assert_close(g.memory, memory)
        torch.testing.assert_close(y, x+4)
        g.set('D', 'rx_wrong', receiver=memory*4, receiver_ok=False)
        y, _ = m.base.decoder_text_to_image_refiner(x, memory)
        torch.testing.assert_close(y, x)
        g.trainable('A')
        self.assertTrue(m.base.image_head.weight.requires_grad)
        self.assertFalse(m.base.text_encoder.weight.requires_grad)

    def test_ste_forward_exact_and_erasure_zero_gradient(self):
        context = dict(clean=True, field_seed=1, noise_seed=2)
        x = torch.ones(1, 288, 16, requires_grad=True)
        f = TokenFrame({'family': 'awgn'}, 40, 1, torch.device('cpu'), context=context, fixed_mode=0)
        y, _ = f.payload(x, None, None)
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        x.grad = None
        def erased(*args):
            return np.zeros((288, 16), np.float32), {'feature_object_crc_accepted': False}, None
        with patch('token_guidance.model.send', erased):
            y, _ = f.payload(x, None, None)
            self.assertFalse(y.requires_grad)
            self.assertFalse(bool(f.header_ok[0]))
            self.assertTrue(torch.all(y == 0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
