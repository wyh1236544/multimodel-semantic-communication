"""Run with the existing SERVER torch environment before real model evaluation."""
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from v10.channels import PROFILES
from physical_link.channel import WaveformFSOFrame
from physical_link.digital import encode_source


class AdapterTests(unittest.TestCase):
    def test_digital_and_analog_cpu_payload(self):
        tokens = torch.randn(1, 32, 16, generator=torch.Generator().manual_seed(23))
        amp = torch.ones(1, 32)
        for variant in ('analog', '16qam_none', '16qam_conv', 'qpsk_conv'):
            torch.manual_seed(12)
            frame = WaveformFSOFrame(PROFILES['fso_weak'], 60., 1, torch.device('cpu'),
                                     variant=variant, fixed_mode=0, feedback_snr_db=60.)
            restored, reliability = frame.payload(tokens, amp, amp)
            self.assertEqual(tuple(restored.shape), (1, 32, 16))
            self.assertEqual(tuple(reliability.shape), (1, 32, 1))
            self.assertTrue(torch.isfinite(restored).all())
            self.assertTrue(bool(frame.header_ok[0]))
            if variant != 'analog':
                self.assertTrue(frame.last_stats['crc_accepted'])
                _, quantized, _ = encode_source(tokens.numpy())
                np.testing.assert_allclose(restored.numpy().ravel(), quantized, atol=1e-6)
                self.assertEqual(frame.last_stats['post_fec_ber'], 0.)
            else:
                self.assertLess(float((restored - tokens).square().mean()), .01)
            self.assertEqual(frame.last_stats['forward_uses'],
                frame.last_stats['payload_uses'] + frame.last_stats['pilot_uses'] + frame.last_stats['header_uses'])
            json.dumps(frame.last_stats)

    def test_rate_variant_fading_identity_and_noise_scaling(self):
        gains, seeds = [], []
        for variant, rate in [('analog', 8e9), ('16qam_conv', 8e9), ('analog', 14e9)]:
            torch.manual_seed(99)
            frame = WaveformFSOFrame(PROFILES['fso_moderate'], 20., 1, torch.device('cpu'), variant=variant, rs=rate)
            gains.append(frame.true_gain.clone())
            seeds.append(frame.wave_seed)
            self.assertAlmostEqual(float(frame.true_noise_var[0, 0]), .01 * rate / 14e9, places=7)
        self.assertTrue(all(torch.equal(gains[0], g) for g in gains))
        self.assertEqual(len(set(seeds)), 1)

    def test_header_erasure_not_repaired(self):
        torch.manual_seed(1)
        frame = WaveformFSOFrame(PROFILES['fso_weak'], 60., 1, torch.device('cpu'), variant='16qam_conv')
        frame.header_ok.fill_(False)
        x = torch.ones(1, 32, 16)
        y, r = frame.payload(x, torch.ones(1, 32), torch.ones(1, 32))
        self.assertEqual(float(y.abs().sum()), 0)
        self.assertEqual(float(r.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
