"""Local non-Torch contracts; not neural performance tests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
EXT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXT))
import numpy as np
from bart_latent_codec_v4.core import load, geometry, batch_indices, gate, row, summarize
from bart_latent_codec_v4.channel import proxy_numpy, resource, send
from fso_fluctuating_v2.core import load as physical_config, paths


class Contracts(unittest.TestCase):
    def test_default(self): self.assertEqual(load()['diagnostic_updates'], 1200)
    def test_width_resources(self):
        for w, n in [(16, 256), (32, 512), (64, 1024)]:
            g = geometry(w); self.assertEqual(g['payload_complex_symbols'], n)
            self.assertEqual(g['length_header_reals'], 0); self.assertEqual(g['receiver_memory_slots'], 32)
    def test_unknown_width(self):
        for w in (8, True, 64.):
            with self.assertRaises(ValueError): geometry(w)
    def bad(self, k, v):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'config.json'; c = load(); c[k] = v; p.write_text(json.dumps(c))
            with self.assertRaises(ValueError): load(p)
    def test_bad_config(self):
        for k, v in [('seed', True), ('learning_rate', float('nan')), ('oracle', 1), ('proxy_snr_db', []),
            ('batch_size', 8), ('depth', 4), ('kd_temperature', 0), ('memory_margin_gate', 1)]:
            if k == 'batch_size': continue
            self.bad(k, v)
    def test_no_split_oracle_selection(self):
        from bart_latent_codec_v4.core import caption_view
        with self.assertRaises(ValueError): caption_view('missing', 'missing', 'test', 8, 42)
    def test_fixed_order(self):
        a = sum([batch_indices(8, 4, i, 42) for i in range(2)], [])
        self.assertEqual(set(a), set(range(8))); self.assertEqual(len(a), 8)
        self.assertEqual(a, sum([batch_indices(8, 4, i, 42) for i in range(2)], []))
    def test_native_clean_proxy_identity(self):
        x = np.random.default_rng(2).standard_normal(512).astype(np.float32)
        y, coeff, variance, stats = proxy_numpy(x, load(), 'clean', 20, 1, 2)
        self.assertTrue(np.allclose(x, y, atol=1e-6)); self.assertEqual(coeff, 1); self.assertEqual(variance, 0)
    def test_noise_not_reset_after_fading(self):
        x = np.zeros(512, np.float32); c = load()
        a = proxy_numpy(x, c, 'fso_strong', 0, 3, 8)[3]
        b = proxy_numpy(x, c, 'fso_strong', 0, 9, 8)[3]
        self.assertEqual(a['noise_variance_before_fade'], b['noise_variance_before_fade'])
        self.assertNotEqual(a['true_postfade_snr_db'], b['true_postfade_snr_db'])
        self.assertEqual(a['forward_complex_symbols'], 256+c['proxy_pilots'])
    def test_fresh_waveform_compact_frames(self):
        c = physical_config(); condition = c['conditions'][0]; trace = paths(c, condition['weather'], 1, 'v4-tests')[0]
        for width in (16, 32, 64):
            x = np.random.default_rng(1).standard_normal(32*width).astype(np.float32)
            y, stats, transport = send(c, condition, trace, 19, width, x)
            self.assertEqual(x.shape, y.shape); self.assertTrue(np.isfinite(y).all())
            self.assertEqual(stats['used_samples'], resource(c, width)['scheduled_samples'])
            self.assertEqual(stats['payload_complex_symbols'], 16*width); self.assertEqual(len(transport.events), 2)
            self.assertEqual(stats['analog_repeats'], 1); self.assertIsNone(stats['pre_fec_ber'])
    def test_gate_no_beam_shortcut(self):
        good = dict(covered_wer=.1, capacity_stop_fraction=0); bad = dict(good, covered_wer=1)
        g = dict(matched=good, zero=bad, shuffled=bad)
        self.assertTrue(gate(g, load())['passed'])
        g['matched'] = dict(good, capacity_stop_fraction=.1)
        self.assertFalse(gate(g, load())['passed'])
        g['matched'] = dict(good, covered_wer=.8); g['matched_beam4'] = good
        self.assertFalse(gate(g, load())['passed'])


if __name__ == '__main__': unittest.main(verbosity=2)
