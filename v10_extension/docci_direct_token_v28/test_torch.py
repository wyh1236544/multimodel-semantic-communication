"""Real Torch/HF CPU models, no downloaded weights. GPU/saved parents still need smoke."""
import os
from pathlib import Path
import sys
import unittest
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
try:
    import torch
    from transformers import BartConfig, BartForConditionalGeneration
    AVAILABLE = True
except ImportError:
    AVAILABLE = False
    if os.environ.get('V28_REQUIRE_TORCH') == '1': raise


@unittest.skipUnless(AVAILABLE, 'Actual Torch/Transformers unavailable locally; server gate mandatory')
class NeuralContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(42)

    def test_continuous_channel_gradient_and_no_renormalization(self):
        from docci_direct_token_v28.channel import transmit
        z = torch.randn(1, 8192, requires_grad=True)
        wire = z/z.square().mean().sqrt()
        observed, stats = transmit(wire, 'image', -5, 2)
        observed.square().mean().backward()
        self.assertGreater(float(z.grad.abs().sum()), 0)
        self.assertFalse(stats['post_noise_normalization'])
        self.assertGreater(float(observed.square().mean()), 2.)

    def test_real_spatial_encoder_and_receiver_both_update(self):
        from docci_reconstruction_v10.model import ImageBranch
        from docci_direct_token_v28.backend import receive
        from docci_direct_token_v28.channel import transmit
        model = ImageBranch(dict(image_size=512, image_width=8), 'I2_shifted')
        before = model.saved(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        latent = torch.randn(1, 4, 64, 64)
        wire = model.encode(dict(latent=latent)); observed, stats = transmit(wire, 'image', -5, 42)
        value = receive(model, observed, 'image', stats)['latent']
        (value-latent).square().mean().backward()
        for group, prefixes in model.groups().items():
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0)
                for n, p in model.named_parameters() if n.startswith(prefixes)), group)
        optimizer.step()
        self.assertTrue(any(not torch.equal(before[n], p.detach()) for n, p in model.named_parameters() if n.startswith('image_encoder.')))
        saved = model.saved(); model.restore(saved)
        self.assertTrue(all(torch.equal(saved[n], p.detach()) for n, p in model.named_parameters()))

    def test_real_bart_memory_gradient_with_frozen_backbone(self):
        from docci_baseline_v14.model import TextModel
        from docci_direct_token_v28.backend import text_objective, receive
        from docci_direct_token_v28.channel import transmit
        config = BartConfig(vocab_size=100, d_model=64, encoder_layers=1, decoder_layers=1,
            encoder_attention_heads=4, decoder_attention_heads=4, encoder_ffn_dim=128, decoder_ffn_dim=128,
            max_position_embeddings=128, pad_token_id=1, bos_token_id=0, eos_token_id=2, decoder_start_token_id=2)
        bart = BartForConditionalGeneration(config).requires_grad_(False).eval()
        model = TextModel(bart, 'released64')
        ids = torch.ones(1, 9, 64, dtype=torch.long); ids[:, :, :5] = torch.tensor([0, 7, 8, 9, 2])
        with torch.no_grad(): hidden = bart.model.encoder(input_ids=ids[0], attention_mask=ids[0].ne(1)).last_hidden_state[None]
        original = {n: p.clone() for n, p in bart.named_parameters()}
        wire = model.encode(ids, hidden); observed, stats = transmit(wire, 'text', -5, 42)
        rx = receive(model, observed, 'text', stats)
        loss, n = text_objective(model, ids, rx); (loss/n).backward()
        for group, prefixes in model.groups().items():
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0)
                for name, p in model.named_parameters() if name.startswith(prefixes)), group)
        self.assertTrue(all(p.grad is None and torch.equal(original[name], p) for name, p in bart.named_parameters()))
        self.assertEqual(wire.shape, (1, 18432))


if __name__ == '__main__': unittest.main(verbosity=2)
