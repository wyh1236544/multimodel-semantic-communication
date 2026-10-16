"""Real Torch/HF contracts mandatory on server; explicit local skips if absent."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
AVAILABLE = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('transformers') is not None
if os.environ.get('V17_REQUIRE_TORCH') == '1' and not AVAILABLE:
    raise RuntimeError('Server Torch/HF numerical tests are mandatory; no environment upgrades')
if AVAILABLE:
    import torch
    from transformers import BartConfig, BartForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput
    from docci_segment_bridge_v17.model import SegmentBridge, canonical, ce, content_gate, native_teacher, generate
    from docci_segment_bridge_v17.data import Receiver
    from docci_segment_bridge_v17.core import expected_parameters


@unittest.skipUnless(AVAILABLE, 'Torch/HF absent locally; neural contracts NOT_EXECUTED')
class Neural(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(42)
        c = BartConfig(vocab_size=40, d_model=32, encoder_layers=1, decoder_layers=2,
             encoder_attention_heads=4, decoder_attention_heads=4, encoder_ffn_dim=64,
             decoder_ffn_dim=64, max_position_embeddings=128, pad_token_id=1,
             bos_token_id=0, eos_token_id=2, decoder_start_token_id=2,
             dropout=0., attention_dropout=0., activation_dropout=0., forced_eos_token_id=None)
        self.bart = BartForConditionalGeneration(c).requires_grad_(False).eval()
        self.original = {n: p.clone() for n, p in self.bart.named_parameters()}
        self.bridge = SegmentBridge(32, dict(width=8, slots=4, heads=2, delta_bound=.25)).eval()
        self.memory = torch.randn(2, 9, 32, 32)
        self.aux = torch.randn(2, 196, 32)
        self.gate = content_gate(self.bart, self.memory)
        self.ids = torch.tensor([0, 4, 5, 6, 2]+[1]*59)[None, None].expand(2, 9, -1).clone()
        self.ids[1, :, 2] = 7

    def tearDown(self):
        self.assertTrue(all(torch.equal(self.original[n], p) and not p.requires_grad and p.grad is None for n, p in self.bart.named_parameters()))

    def test_zero_initialization_off_and_public_empty_gate(self):
        x, _ = self.bridge.repair(self.memory, self.aux, self.gate)
        self.assertTrue(torch.equal(x, self.memory))
        with torch.no_grad():self.bridge.alpha.fill_(.5)
        x, _ = self.bridge.repair(self.memory, self.aux, self.gate, False)
        self.assertTrue(torch.equal(x, self.memory))
        x, _ = self.bridge.repair(self.memory, self.aux, torch.zeros_like(self.gate))
        self.assertTrue(torch.equal(x, self.memory))

    def test_parameter_formula_and_all_groups_get_gradients(self):
        self.assertEqual(sum(p.numel() for p in self.bridge.parameters()), expected_parameters(32, 8, 4))
        before = {n: p.clone() for n, p in self.bridge.named_parameters()}; seen = set()
        opt = torch.optim.AdamW(self.bridge.parameters(), lr=.01)
        teacher, counts = native_teacher(self.bart, self.ids)
        for _ in range(4):
            opt.zero_grad(set_to_none=True)
            value, n, features = ce(self.bart, self.bridge, self.memory, self.aux, self.gate, self.ids, True)
            itc, segment, matching = self.bridge.auxiliary(self.memory, self.aux, self.gate, teacher, counts, self.aux.roll(1, 0), True, features=features)
            (value/n+.1*(itc+segment+matching)).backward()
            seen.update(n for n, p in self.bridge.named_parameters() if p.grad is not None and bool(p.grad.abs().sum()))
            opt.step()
        for prefixes in self.bridge.groups().values():
            names = [n for n, _ in self.bridge.named_parameters() if n.startswith(prefixes)]
            self.assertTrue(any(n in seen for n in names))
            self.assertTrue(any(not torch.equal(before[n], dict(self.bridge.named_parameters())[n]) for n in names))

    def test_causal_and_incremental_BART_unchanged(self):
        with torch.no_grad():self.bridge.alpha.fill_(.3)
        memory, _ = self.bridge.repair(self.memory[:1], self.aux[:1], self.gate[:1])
        def forward(ids, **kwargs):
            return self.bart(encoder_outputs=BaseModelOutput(last_hidden_state=memory[0]),
                attention_mask=torch.ones(9, 32, dtype=torch.long), decoder_input_ids=ids, **kwargs)
        ids = torch.tensor([2, 0, 4, 5])[None].expand(9, -1)
        full = forward(ids, use_cache=False).logits
        prefix = forward(ids[:, :3], use_cache=True)
        step = forward(ids[:, 3:], past_key_values=prefix.past_key_values, use_cache=True).logits
        self.assertTrue(torch.allclose(full[:, -1], step[:, -1], atol=2e-6, rtol=2e-5))
        other = ids.clone(); other[:, -1] = 8
        self.assertTrue(torch.allclose(full[:, :3], forward(other, use_cache=False).logits[:, :3], atol=2e-6))

    def test_wire_AWGN_common_noise_and_received_gate_recomputation(self):
        bart = self.bart
        class Frozen:
            def __init__(self):self.bart = bart
            def receive(self, observed):
                return dict(memory=observed.reshape(-1, 9, 32, 64)[..., :32])
        rx = Receiver(Frozen()); wire = torch.ones(1, 18432); before = wire.clone()
        clean, _, cs = rx(wire, 'clean', 42)
        a, gate, stats = rx(wire, 'awgn_0', 123)
        b, _, other = rx(wire, 'awgn_0', 123)
        self.assertTrue(torch.equal(a, b)); self.assertEqual(stats, other)
        self.assertTrue(torch.equal(wire, before)); self.assertFalse(torch.equal(clean, a))
        self.assertTrue(torch.equal(gate, content_gate(bart, a)))
        self.assertLess(abs(stats['noise_power']-1), .06)
        self.assertEqual(cs['noise_power'], 0); self.assertIsNone(cs['nominal_snr_db'])

    def test_self_aux_cannot_read_visual_and_segment_queries_differ(self):
        a = canonical(self.memory, self.aux, 'self')
        b = canonical(self.memory, torch.randn_like(self.aux), 'self')
        self.assertTrue(torch.equal(a, b))
        _, q, _ = self.bridge.features(self.memory, self.aux, self.gate)
        self.assertFalse(torch.equal(q[:, 0], q[:, 1]))

    def test_teacher_empty_segments_and_single_batch_contrast_refused(self):
        ids = self.ids.clone(); ids[:, 1:] = torch.tensor([0, 2]+[1]*62)
        target, counts = native_teacher(self.bart, ids)
        self.assertTrue(torch.equal(counts[:, 1:], torch.zeros_like(counts[:, 1:])))
        self.assertTrue(torch.equal(target[:, 1:9], torch.zeros_like(target[:, 1:9])))
        with self.assertRaises(ValueError):
            self.bridge.auxiliary(self.memory[:1], self.aux[:1], self.gate[:1], target[:1], counts[:1], self.aux[:1], True)

    def test_actual_saved_state_roundtrip(self):
        with tempfile.TemporaryDirectory(prefix='v17_torch_') as d:
            path = Path(d)/'checkpoint.pth'; torch.save(self.bridge.state_dict(), str(path))
            other = SegmentBridge(32, dict(width=8, slots=4, heads=2, delta_bound=.25))
            other.load_state_dict(torch.load(str(path), map_location='cpu'), strict=True)
            self.assertTrue(all(torch.equal(v, other.state_dict()[k]) for k, v in self.bridge.state_dict().items()))

    def test_real_greedy_generation_fixed_public_frames(self):
        class Tokenizer:
            bos_token_id, eos_token_id, pad_token_id = 0, 2, 1
            def decode(self, ids, **kwargs):return ' '.join(map(str, ids))
        result = generate(self.bart, self.bridge, Tokenizer(), self.memory[:1], self.aux[:1], self.gate[:1])
        self.assertEqual(len(result['chunk_generated_token_ids']), 9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
