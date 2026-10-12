"""Mandatory server Torch/HF contracts; no download, no skipped fake pass."""
from pathlib import Path
import sys
import unittest
EXT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXT))
import torch
from transformers import BartConfig, BartForConditionalGeneration
from bart_latent_codec_v4.core import load
from bart_latent_codec_v4.model import LatentCodec, prediction, distillation
from bart_latent_codec_v4.channel import proxy_tensor, proxy_numpy
from fso_jscc_v1.core import seed_for


class TorchContracts(unittest.TestCase):
    def fixture(self, width=16):
        torch.manual_seed(4); torch.set_num_threads(1)
        cfg = BartConfig(vocab_size=31, d_model=32, encoder_layers=1, decoder_layers=1,
            encoder_attention_heads=4, decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64,
            max_position_embeddings=128, dropout=0., attention_dropout=0., activation_dropout=0.,
            bos_token_id=0, pad_token_id=1, eos_token_id=2, decoder_start_token_id=2)
        bart = BartForConditionalGeneration(cfg).eval().requires_grad_(False)
        c = load(); c.update(head_dim=8, depth=1)
        model = LatentCodec(bart, c, width).eval()
        ids = torch.ones(2, 128, dtype=torch.long); ids[:, :5] = torch.tensor([0, 4, 5, 6, 2])
        mask = ids.ne(1)
        with torch.no_grad(): hidden = bart.model.encoder(input_ids=ids, attention_mask=mask.long()).last_hidden_state
        return model, dict(input_ids=ids, mask=mask, bart_hidden=hidden), c
    def test_fixed_memory_no_length_oracle(self):
        model, b, _ = self.fixture(); wire = model.transmit(b['bart_hidden'], b['mask']); out = model.receive_text(wire)
        self.assertEqual(out['memory'].shape, (2, 32, 32)); self.assertTrue(out['receiver_mask'].all())
        with self.assertRaises(TypeError): model.receive_text(wire, source_mask=b['mask'])
        with self.assertRaises(TypeError): model.receive_text(wire, source_length=5)
    def test_pad_cannot_change_transmission(self):
        model, b, _ = self.fixture(); h = b['bart_hidden'].clone(); h[:, 5:] += 1000
        self.assertTrue(torch.equal(model.transmit(h, b['mask']), model.transmit(b['bart_hidden'], b['mask'])))
    def test_per_slot_unit_power(self):
        for width in (16, 32, 64):
            model, b, _ = self.fixture(width); z = model.encode(b['bart_hidden'], b['mask'])
            self.assertTrue(torch.allclose(z.square().mean(-1), torch.ones(2, 32), atol=1e-5))
    def test_all_codec_groups_get_gradients_with_CE_alone(self):
        model, b, _ = self.fixture(); before = model.state(); seen = set()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        frozen = {n: p.detach().clone() for n, p in model.bart.named_parameters()}
        for _ in range(3):
            out = model.receive_text(model.transmit(b['bart_hidden'], b['mask']))
            pred, labels = prediction(model.bart, b['input_ids'], b['mask'], out['memory'], out['receiver_mask'])
            torch.nn.functional.cross_entropy(pred.reshape(-1, 31), labels.reshape(-1), ignore_index=-100).backward()
            seen.update(n for n, p in model.named_parameters() if p.grad is not None and float(p.grad.abs().sum())>0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        after = model.state()
        for prefixes in model.groups().values():
            self.assertTrue(any(n.startswith(prefixes) for n in seen))
            self.assertTrue(any(n.startswith(prefixes) and not torch.equal(before[n], v) for n, v in after.items()))
        self.assertTrue(all(p.grad is None and torch.equal(frozen[n], p) for n, p in model.bart.named_parameters()))
    def test_KD_teacher_detached_PAD_ignored(self):
        s = torch.randn(2, 4, 31, requires_grad=True); t = torch.randn(2, 4, 31, requires_grad=True)
        labels = torch.tensor([[0, 4, 2, -100], [0, 5, 2, -100]])
        loss = distillation(s, t, labels, 1.); loss.backward()
        self.assertIsNone(t.grad); self.assertEqual(float(s.grad[:, 3].abs().sum()), 0.)
        self.assertGreater(float(s.grad[:, :3].abs().sum()), 0.)
    def test_label_shift_length_and_EOS(self):
        model, b, _ = self.fixture(); out = model.receive_text(model.transmit(b['bart_hidden'], b['mask']))
        _, labels = prediction(model.bart, b['input_ids'], b['mask'], out['memory'], out['receiver_mask'])
        self.assertEqual(labels.shape, (2, 5)); self.assertTrue(labels[:, -1].eq(2).all())
    def test_grad_no_grad_same_interface(self):
        model, b, _ = self.fixture(); x = model.transmit(b['bart_hidden'], b['mask']); out = model.receive_text(x)
        with torch.no_grad(): xx = model.transmit(b['bart_hidden'], b['mask']); yy = model.receive_text(xx)
        self.assertTrue(torch.equal(x.detach(), xx)); self.assertTrue(torch.equal(out['memory'].detach(), yy['memory']))
    def test_same_seed_same_initialization_and_restore(self):
        a, _, _ = self.fixture(); b, _, _ = self.fixture()
        self.assertTrue(all(torch.equal(v, b.state()[n]) for n, v in a.state().items()))
        state = a.state()
        with torch.no_grad():
            for p in a.parameters():
                if p.requires_grad: p.add_(.1)
        a.restore(state); self.assertTrue(all(torch.equal(v, a.state()[n]) for n, v in state.items()))
    def test_public_erasure_zeroes_memory(self):
        model, b, _ = self.fixture(); wire = model.transmit(b['bart_hidden'], b['mask'])
        out = model.receive_text(wire, torch.tensor([False, True]))
        self.assertEqual(float(out['memory'][0].abs().sum()), 0.); self.assertFalse(bool(out['available'][0]))
    def test_proxy_tensor_matches_numpy_and_backpropagates(self):
        model, b, _ = self.fixture(); c = load(); x = model.transmit(b['bart_hidden'], b['mask']); update = 2; ids = [7, 8]
        y, stats = proxy_tensor(x, c, update, ids)
        for j, iid in enumerate(ids):
            yy = proxy_numpy(x[j].detach().numpy(), c, stats[j]['profile'], stats[j]['nominal_snr_db'],
                seed_for(c['seed'], 'v4-field', update, iid, j), seed_for(c['seed'], 'v4-noise', update, iid, j))[0]
            self.assertTrue(torch.allclose(y[j], torch.tensor(yy), atol=2e-6))
        y.square().mean().backward(); self.assertGreater(float(model.queries.grad.abs().sum()), 0.)


if __name__ == '__main__': unittest.main(verbosity=2)
