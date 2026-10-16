"""Tiny authentic Torch/HF contracts; server check refuses missing dependencies."""
import os
from pathlib import Path
import sys
import unittest
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
try:
    import torch
    from transformers import BartConfig, BartForConditionalGeneration
    from docci_baseline_v14.model import TextModel, loss, AlignedCodec
    from docci_reconstruction_v10.model import ImageBranch
    AVAILABLE = True
except ImportError:
    AVAILABLE = False
    if os.environ.get('V14_REQUIRE_TORCH') == '1':
        raise RuntimeError('Torch/Transformers contracts mandatory on server; do not claim skipped checks passed')


@unittest.skipUnless(AVAILABLE, 'Torch/HF unavailable locally; mandatory server gates')
class Models(unittest.TestCase):
    def bart(self):
        c = BartConfig(vocab_size=32, d_model=128, encoder_layers=1, decoder_layers=1,
            encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=256,
            decoder_ffn_dim=256, max_position_embeddings=1024, pad_token_id=1,
            bos_token_id=0, eos_token_id=2, decoder_start_token_id=2,
            dropout=0., attention_dropout=0., activation_dropout=0.)
        return BartForConditionalGeneration(c).requires_grad_(False).eval()

    def test_all_text_arms_unit_power_updates_frozen_bart_and_roundtrip(self):
        torch.set_num_threads(2)
        for arm in ('legacy64', 'released64', 'wide768'):
            torch.manual_seed(42)
            bart = self.bart()
            frozen = {n:p.detach().clone() for n,p in bart.named_parameters()}
            model = TextModel(bart, arm)
            ids = torch.full((1,9,64),1,dtype=torch.long)
            ids[:,:,0] = 0
            ids[:,:,1] = 2
            ids[0,0,1:4] = torch.tensor([4,5,2])
            hidden = torch.randn(1,9,64,128)
            before = model.saved()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
            summed, n = loss(model, ids, hidden)
            (summed/n).backward()
            for name,p in model.named_parameters():
                if p.grad is not None:
                    self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
            optimizer.step()
            self.assertTrue(any(not torch.equal(v, model.saved()[k]) for k,v in before.items()))
            self.assertTrue(all(torch.equal(v, dict(bart.named_parameters())[k]) for k,v in frozen.items()))
            self.assertTrue(all(p.grad is None for p in bart.parameters()))
            wire = model.encode(ids,hidden)
            self.assertEqual(tuple(wire.shape), (1,18432))
            self.assertAlmostEqual(float(wire.square().mean()), 1., places=5)
            saved = model.saved()
            model.restore(saved)
            self.assertTrue(all(torch.equal(v, model.saved()[k]) for k,v in saved.items()))

    def test_padding_cannot_change_transmitted_features(self):
        torch.manual_seed(42)
        for arm in ('legacy64', 'released64', 'wide768'):
            model = TextModel(self.bart(), arm)
            ids = torch.ones(1,9,64,dtype=torch.long)
            ids[:,:,0], ids[:,:,1] = 0, 2
            x = torch.randn(1,9,64,128)
            y = x.clone()
            y[ids==1] = 10000.
            self.assertTrue(torch.allclose(model.encode(ids,x), model.encode(ids,y), atol=1e-5))

    def test_released_narrow_vs_optional_wide_are_not_confused(self):
        small = AlignedCodec(self.bart(), 'released64')
        large = AlignedCodec(self.bart(), 'wide768')
        self.assertEqual(small.queries.shape[-1],64)
        self.assertEqual(large.queries.shape[-1],128)
        self.assertEqual(small.width, large.width)
        self.assertGreater(sum(p.numel() for p in large.parameters()), sum(p.numel() for p in small.parameters()))

    def test_receiver_does_not_accept_source_length_or_mask(self):
        m = TextModel(self.bart(), 'released64')
        with self.assertRaises(TypeError):
            m.receive(torch.ones(1,18432), source_length=7)
        with self.assertRaises(ValueError):
            m.receive(torch.ones(1,18431))
        out = m.receive(torch.ones(1,18432))
        self.assertEqual(tuple(out['memory'].shape),(1,9,32,128))
        self.assertTrue(bool(out['receiver_mask'].all()))

    def test_real_I2_gradient_and_payload(self):
        torch.manual_seed(42)
        m = ImageBranch(dict(image_size=512,image_width=8), 'I2_shifted')
        wire = m.encode(dict(latent=torch.randn(1,4,64,64)))
        self.assertEqual(tuple(wire.shape),(1,8192))
        rx = m.receive(wire,torch.ones(1,dtype=torch.bool),torch.tensor([40.]),torch.ones(1))
        rx['latent'].square().mean().backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in m.image_encoder.parameters()))
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in m.image_decoder.parameters()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
