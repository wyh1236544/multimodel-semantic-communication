"""Mandatory server Torch/HF tiny-model tests. Missing dependencies FAIL, not skip."""
import inspect
from pathlib import Path
import sys
import unittest
import torch
from torch import nn
from transformers import BartConfig, BartForConditionalGeneration
EXT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXT))
from bart_clip_joint_v1.model import JointBridge
from bart_clip_joint_v1.channel import clean
from bart_clip_joint_v1.channel import noisy
from unittest.mock import patch
import numpy as np


class ImageReference(nn.Module):
    def __init__(self):
        super().__init__(); self.projection = nn.Linear(16, 4)
    def receive(self, wire, rel, snr, available):
        x = self.projection(wire[:, 32:256].mean(1))
        return dict(latent=x[:, :, None, None].expand(-1, -1, 16, 16))


def toy():
    config = BartConfig(vocab_size=32, d_model=32, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=4, decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64,
        max_position_embeddings=256, dropout=0., attention_dropout=0., activation_dropout=0.,
        bos_token_id=0, eos_token_id=2, pad_token_id=1, decoder_start_token_id=2,
        forced_bos_token_id=0, forced_eos_token_id=None)
    cfg = dict(hidden=16, heads=4, layers=1)
    return JointBridge(BartForConditionalGeneration(config), ImageReference(), cfg, clip_width=16)


def inputs():
    return [torch.randn(1, 128, 32), torch.arange(128)[None]<7,
            torch.randn(1, 224, 16), torch.randn(1, 16), torch.randn(1, 16)]


class TorchContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)
    def test_fixed_budget_and_group_power(self):
        m = toy(); x = m.source_encode(*inputs()); self.assertEqual(tuple(x.shape), (1, 288, 16))
        for a, b in ((0, 16), (16, 32), (32, 256), (256, 288)):
            self.assertAlmostEqual(float(x[:, a:b].square().mean()), 1., places=5)
    def test_source_padding_not_leaked(self):
        m = toy(); a = inputs(); b = [v.clone() for v in a]; b[0][:, 7:] = 10000
        self.assertTrue(torch.allclose(m.source_encode(*a), m.source_encode(*b), atol=1e-5))
    def test_frozen_decoder_input_gradient_actual_update(self):
        torch.manual_seed(2); m = toy(); a = inputs(); saved = {n: p.clone() for n, p in m.bart.named_parameters()}
        before = m.text_tx.weight.clone(); x = m.source_encode(*a); out = m.receive(*clean(x))
        ids = torch.tensor([[0, 4, 5, 6, 7, 8, 2]+[1]*121]); opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=.001)
        loss = m.teacher_loss(ids, a[1], out['text_memory'])+out['latent'].square().mean()+out['common_alignment'].square().mean()
        loss.backward(); self.assertTrue(m.text_tx.weight.grad.abs().sum()>0)
        opt.step(); self.assertFalse(torch.equal(before, m.text_tx.weight))
        self.assertTrue(all(p.grad is None and not p.requires_grad and torch.equal(saved[n], p) for n, p in m.bart.named_parameters()))
    def test_bart_pretrained_tied_head_not_random_head(self):
        m = toy(); self.assertEqual(m.bart.lm_head.weight.data_ptr(), m.bart.model.shared.weight.data_ptr())
        self.assertFalse(any(p.requires_grad for p in m.bart.parameters()))
    def test_separate_no_cross_modal_source_path(self):
        m = toy(); m.configure('J2_separate'); a = inputs(); b = [v.clone() for v in a]
        b[2] *= 100; b[3] *= 10
        x, y = m.source_encode(*a), m.source_encode(*b)
        self.assertTrue(torch.allclose(x[:, 16:32], y[:, 16:32], atol=1e-5))
        self.assertTrue(torch.allclose(x[:, 256:], y[:, 256:], atol=1e-5))
    def test_separate_receiver_text_independent_image(self):
        m = toy(); m.configure('J2_separate'); x = torch.randn(1, 288, 16); y = x.clone(); y[:, :16] *= 50; y[:, 32:256] *= 30
        a = m.receive(*clean(x))['text_memory']; b = m.receive(*clean(y))['text_memory']
        self.assertTrue(torch.allclose(a, b, atol=1e-5))
    def test_joint_receiver_responds_common_not_image_private(self):
        m = toy(); x = torch.randn(1, 288, 16); y = x.clone(); y[:, 32:256] += 30
        a = m.receive(*clean(x))['text_memory']; b = m.receive(*clean(y))['text_memory']
        self.assertTrue(torch.allclose(a, b, atol=1e-5))
        y[:, :32] += 10
        self.assertFalse(torch.allclose(a, m.receive(*clean(y))['text_memory'], atol=1e-5))
    def test_receiver_failure_zero_memory_latent(self):
        m = toy(); x = torch.randn(1, 288, 16)
        out = m.receive(x, torch.zeros(1), torch.tensor([-10.]), torch.tensor([False]))
        self.assertEqual(float(out['text_memory'].abs().sum()), 0.)
        self.assertEqual(float(out['latent'].abs().sum()), 0.)
    def test_roundtrip_no_backbone_or_source_tensors_saved(self):
        a, b = toy(), toy(); state = a.trainable_state(); self.assertFalse(any(k.startswith(('bart.', 'image_reference.')) for k in state))
        b.load_trainable(state)
        for n, p in b.named_parameters():
            if p.requires_grad: self.assertTrue(torch.equal(p, state[n]))
        with self.assertRaises(ValueError): b.load_trainable({})
    def test_causal_decoder_and_public_receive_signature(self):
        m = toy(); memory = torch.randn(1, 128, 32)
        from transformers.modeling_outputs import BaseModelOutput
        a = torch.tensor([[2, 0, 4, 5]]); b = a.clone(); b[:, 2:] = 9
        options = dict(encoder_outputs=BaseModelOutput(last_hidden_state=memory), attention_mask=torch.ones(1, 128, dtype=torch.long), use_cache=False)
        x = m.bart(decoder_input_ids=a, **options).logits; y = m.bart(decoder_input_ids=b, **options).logits
        self.assertTrue(torch.allclose(x[:, :2], y[:, :2], atol=1e-5))
        self.assertEqual(list(inspect.signature(m.receive).parameters), ['wire', 'reliability', 'pilot_snr', 'available', 'text_only'])
    def test_eos_start_is_not_predicted_end_and_generation_overrides(self):
        m = toy(); seen = {}
        def fake_generate(**kw):
            seen.update(kw); return torch.tensor([[2, 0, 5, 2]])
        m.bart.generate = fake_generate
        class Tok:
            def decode(self, ids, skip_special_tokens=True): return 'word'
        out = m.generate(torch.randn(1, 128, 32), Tok())
        self.assertFalse(out['stopped_by_capacity']); self.assertEqual(seen['num_beams'], 1)
        self.assertEqual(seen['no_repeat_ngram_size'], 0); self.assertTrue(seen['attention_mask'].all())
        m.bart.generate = lambda **kw: torch.tensor([[2, 0, 5]])
        self.assertTrue(m.generate(torch.randn(1, 128, 32), Tok())['stopped_by_capacity'])
    def test_surrogate_matches_numpy_dtype_and_gradient(self):
        from fso_fluctuating_v2.core import apply_operator, as_complex, seed_for
        x = torch.randn(1, 288, 16, requires_grad=True)
        coefficient = np.asarray([.02]*5+[.001j]*5+[0.], dtype=np.complex64)
        residual = np.random.default_rng(5).standard_normal(2304).astype(np.complex64)*.01
        bank = dict(records=[dict(partition='train', reliability=.8, probe_snr_db=2., available=True)],
                    coefficients=np.asarray([coefficient]), residuals=np.asarray([residual]))
        cfg = dict(seed=42)
        with patch('bart_clip_joint_v1.channel.schedule', return_value=0): y, rel, snr, ok = noisy(x, bank, cfg, 1, 2)
        rng = np.random.default_rng(seed_for(42, 1, 2, 'new-bart-residual'))
        rr = np.roll(residual, int(rng.integers(2304)))*np.exp(1j*rng.uniform(-np.pi, np.pi))
        expected = apply_operator(as_complex(x.detach().numpy()), coefficient, rr)
        actual = as_complex(y.detach().numpy())
        self.assertTrue(np.allclose(expected, actual, atol=1e-6)); self.assertEqual(y.dtype, x.dtype)
        y.square().sum().backward(); self.assertTrue(x.grad.abs().sum()>0)
    def test_holdout_cannot_train(self):
        bank = dict(records=[dict(partition='holdout')])
        with patch('bart_clip_joint_v1.channel.schedule', return_value=0):
            with self.assertRaises(ValueError): noisy(torch.randn(1, 288, 16), bank, {}, 0, 0)


if __name__ == '__main__': unittest.main(verbosity=2)
