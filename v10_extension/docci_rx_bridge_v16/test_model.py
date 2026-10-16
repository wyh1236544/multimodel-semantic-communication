"""Actual Torch/HF tiny contracts; REQUIRED on server, explicitly skipped if absent."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
AVAILABLE = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('transformers') is not None
if os.environ.get('V16_REQUIRE_TORCH') == '1' and not AVAILABLE:
    raise RuntimeError('Server Torch/HF numerical contracts MUST run; do not install/upgrade dependencies')
if AVAILABLE:
    import torch
    from transformers import BartConfig, BartForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput
    from docci_rx_bridge_v16.model import Bridge, canonical, ce, content_gate, generate, prepared


@unittest.skipUnless(AVAILABLE, 'Torch/Transformers absent locally; real CPU/GPU contracts NOT_EXECUTED')
class TorchContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cfg = BartConfig(vocab_size=40, d_model=32, encoder_layers=1, decoder_layers=2,
                encoder_attention_heads=4, decoder_attention_heads=4, encoder_ffn_dim=64,
                decoder_ffn_dim=64, max_position_embeddings=128, pad_token_id=1,
                bos_token_id=0, eos_token_id=2, decoder_start_token_id=2,
                dropout=0., attention_dropout=0., activation_dropout=0., forced_eos_token_id=None)
        self.bart = BartForConditionalGeneration(cfg).requires_grad_(False).eval()
        self.before = {n: p.detach().clone() for n, p in self.bart.named_parameters()}
        self.bridge = Bridge(self.bart, dict(width=8, slots=4, heads=2)).eval()
        self.memory = torch.randn(1, 9, 32, 32)
        self.aux = torch.randn(1, 196, 32)
        self.ids = torch.tensor([0, 4, 5, 6, 2]+[1]*59)[None, None].expand(1, 9, 64).clone()
        self.gate = content_gate(self.bart, self.memory)

    def tearDown(self):
        self.bridge.close()
        self.assertTrue(all(torch.equal(self.before[n], p) and p.grad is None and not p.requires_grad
                            for n, p in self.bart.named_parameters()))

    def forward(self, ids, **extra):
        return self.bart(encoder_outputs=BaseModelOutput(last_hidden_state=self.memory[0]),
                attention_mask=torch.ones(9, 32, dtype=torch.long), decoder_input_ids=ids,
                return_dict=True, **extra)

    def test_gate_off_and_zero_initialization_are_exact(self):
        ids = torch.tensor([2, 0, 4])[None].expand(9, -1)
        original = self.forward(ids).logits
        slots, gate = prepared(self.bridge, self.memory, self.aux, self.gate)
        with self.bridge.use(slots, gate):
            self.assertTrue(torch.equal(original, self.forward(ids).logits))
        with torch.no_grad():
            for block in self.bridge.cross:
                block.gate.fill_(.3)
        with self.bridge.use(slots, gate, False):
            self.assertTrue(torch.equal(original, self.forward(ids).logits))
        self.assertIsNone(self.bridge.context)

    def test_all_bridge_groups_learn_and_baseline_stays_frozen(self):
        optimizer = torch.optim.AdamW(self.bridge.parameters(), lr=.005)
        initial = {n: p.detach().clone() for n, p in self.bridge.named_parameters()}
        seen = set()
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            summed, count = ce(self.bart, self.bridge, self.memory, self.aux, self.gate, self.ids)
            (summed/count).backward()
            seen.update(n for n, p in self.bridge.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0)
            optimizer.step()
        for prefixes in self.bridge.groups().values():
            selected = [(n, p) for n, p in self.bridge.named_parameters() if n.startswith(prefixes)]
            self.assertTrue(any(n in seen for n, p in selected))
            self.assertTrue(any(not torch.equal(initial[n], p) for n, p in selected))
        state = self.bridge.state_dict()
        self.bridge.load_state_dict({k: v.clone() for k, v in state.items()}, strict=True)
        self.assertTrue(all(torch.equal(v, self.bridge.state_dict()[k]) for k, v in state.items()))

    def test_causal_full_vs_incremental_cache_with_open_adapter(self):
        with torch.no_grad():
            for block in self.bridge.cross:
                block.gate.fill_(.3)
        slots, gate = prepared(self.bridge, self.memory, self.aux, self.gate)
        ids = torch.tensor([2, 0, 4, 5])[None].expand(9, -1)
        with self.bridge.use(slots, gate):
            full = self.forward(ids, use_cache=False).logits
            prefix = self.forward(ids[:, :3], use_cache=True)
            step = self.forward(ids[:, 3:], past_key_values=prefix.past_key_values, use_cache=True).logits
            self.assertTrue(torch.allclose(full[:, -1], step[:, -1], atol=2e-6, rtol=2e-5))
            changed = ids.clone()
            changed[:, -1] = 7
            other = self.forward(changed, use_cache=False).logits
            self.assertTrue(torch.allclose(full[:, :3], other[:, :3], atol=2e-6))

    def test_first_free_EOS_probability_not_forced_BOS(self):
        ids = torch.tensor([2, 0])[None].expand(9, -1)
        probability = 1-self.forward(ids, use_cache=False).logits[:, -1].softmax(-1)[:, 2]
        self.assertTrue(torch.equal(self.gate.flatten(), probability))

    def test_real_generation_greedy_and_beam_expansion(self):
        class Tokenizer:
            bos_token_id, eos_token_id, pad_token_id = 0, 2, 1
            def decode(self, ids, **kwargs):
                return ' '.join(map(str, ids))
        with torch.no_grad():
            for block in self.bridge.cross:
                block.gate.fill_(.1)
        for strategy in ('greedy', 'deployment'):
            result = generate(self.bart, self.bridge, self.memory, self.aux, self.gate, Tokenizer(), strategy)
            self.assertEqual(len(result['chunk_generated_token_ids']), 9)
        self.assertIsNone(self.bridge.context)

    def test_equal_support_canonical_sources_and_zero(self):
        from docci_rx_bridge_v16.core import parameter_groups
        self.assertEqual(sum(p.numel() for p in self.bridge.parameters()),
                         sum(parameter_groups(dict(width=8, slots=4), dim=32).values()))
        for kind, raw in [('self', torch.randn(1, 9, 32, 768)), ('rx_i0', torch.randn(1, 196, 96)),
                          ('rx_clip', torch.randn(1, 196, 768)), ('oracle_clip', torch.randn(1, 196, 768))]:
            z = canonical(raw, kind)
            self.assertEqual(tuple(z.shape), (1, 196, 768))
            self.assertTrue(torch.isfinite(z).all())
        self.assertTrue(torch.equal(canonical(torch.zeros(1, 196, 768), 'rx_clip'), torch.zeros(1, 196, 768)))

    def test_context_cleared_on_error(self):
        with self.assertRaises(RuntimeError):
            with self.bridge.use(torch.zeros(9, 4, 8), torch.ones(9)):
                raise RuntimeError('synthetic failure')
        self.assertIsNone(self.bridge.context)


if __name__ == '__main__':
    unittest.main(verbosity=2)
