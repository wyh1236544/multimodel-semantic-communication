"""Run with the server's existing torch Python: python -B -m unittest ..."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

import torch

from v10.channels import PROFILES
from v10.protocol import amplitudes, normalize_groups
from link_compare.coherent import CoherentFSOFrame, quantize_complex
from link_compare.rate import effective_snr_db, serialization


class RateTests(unittest.TestCase):
    def test_rate_noise_scaling_and_accounting(self):
        self.assertAlmostEqual(effective_snr_db(20, 10e6, 1e6), 10)
        result = serialization(2400, 1e6, 64, 1e6)
        self.assertAlmostEqual(result['sequential_link_seconds'], .002464)
        self.assertGreater(result['ideal_sequential_pairs_per_second'], 400)

    def test_bad_rate_rejected(self):
        with self.assertRaises(ValueError):
            effective_snr_db(10, 0, 1e6)
        with self.assertRaises(ValueError):
            serialization(2400, 1e6, 64, 0)


class CoherentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(271)
        torch.set_num_threads(2)

    def frame(self, alphabet='continuous'):
        return CoherentFSOFrame(PROFILES['fso_weak'], 80, 2, torch.device('cpu'),
                                feedback_snr_db=80, constellation=alphabet)

    def test_optical_field_irradiance_and_frame_resource(self):
        frame = self.frame()
        torch.testing.assert_close(frame.true_gain.abs().square(), frame.irradiance, atol=1e-6, rtol=1e-5)
        x = normalize_groups(torch.randn(2, 288, 16), frame.tx_mode)
        a = amplitudes(frame.tx_mode, 288, False, True)
        y, _ = frame.payload(x, a, a)
        self.assertLess(float((x-y).square().mean()), 1e-4)
        self.assertEqual(frame.last_stats['forward_uses'], 2400)
        self.assertEqual(frame.last_stats['reverse_rf_real_uses'], 64)
        self.assertEqual(frame.last_stats['physical_unit'], 'coherent_optical_complex_symbol')

    def test_constellation_alphabets_and_hard_recovery(self):
        for alphabet, points in (('qpsk', 4), ('16qam', 16)):
            frame = self.frame(alphabet)
            x = normalize_groups(torch.randn(2, 288, 16), frame.tx_mode)
            a = amplitudes(frame.tx_mode, 288, False, True)
            y, _ = frame.payload(x, a, a)
            self.assertTrue(torch.isfinite(y).all())
            self.assertEqual(frame.last_stats['forward_uses'], 2400)
            self.assertLess(float(frame.last_stats['constellation_symbol_error_rate'].max()), .02)
            self.assertGreaterEqual(float(frame.last_stats['quantization_mse'].mean()), 0)
            constellation = quantize_complex(torch.complex(torch.randn(10000), torch.randn(10000)), alphabet)
            self.assertEqual(len(torch.unique(torch.view_as_real(constellation), dim=0)), points)

    def test_reject_non_fso(self):
        with self.assertRaises(ValueError):
            CoherentFSOFrame(PROFILES['awgn'], 10, 1, 'cpu')


if __name__ == '__main__':
    unittest.main(verbosity=2)
