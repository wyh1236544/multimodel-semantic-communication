import unittest
from unittest.mock import patch
import torch
from v10.channels import LinkFrame, PROFILES, sample_gain, mode_codebook, detect_mode, ReliabilityAdapter
from v10.protocol import DeterministicPacker, amplitudes, metadata_from_modes, normalize_groups, unpack
from v10.data import validate_manifest

class PhysicsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def frame(self, profile='awgn', snr=80, batch=16, **kw):
        return LinkFrame(PROFILES[profile], snr, batch, torch.device('cpu'), feedback_snr_db=80, **kw)

    def test_complex_rayleigh_moments(self):
        h = sample_gain(PROFILES['rayleigh'], (100000,), 'cpu')
        self.assertLess(abs(float(h.abs().square().mean())-1), .02)
        self.assertLess(abs(float(h.real.mean())), .01)
        self.assertLess(abs(float(h.imag.mean())), .01)

    def test_optical_gamma_pointing_moments(self):
        p = PROFILES['fso_moderate']
        h = sample_gain(p, (100000,), 'cpu')
        expected = p['a0']*p['xi']**2/(1+p['xi']**2)
        self.assertLess(abs(float(h.mean())/expected-1), .035)
        self.assertTrue((h >= 0).all())
        torch.manual_seed(21)
        a = sample_gain(p, (100,), 'cpu')
        torch.manual_seed(21)
        b = sample_gain(dict(p, loss_db=10), (100,), 'cpu')
        torch.testing.assert_close(b, a*.1)

    def test_control_roundtrip_and_rejection(self):
        book = mode_codebook('cpu')
        decoded, ok, _ = detect_mode(book, book, .2)
        self.assertEqual(decoded.tolist(), [0, 1, 2])
        self.assertTrue(ok.all())
        self.assertFalse(detect_mode(torch.zeros_like(book), book, .2)[1].any())

    def test_noiseless_rf_roundtrip(self):
        for blocks in (1, 4):
            f = self.frame('rayleigh', blocks=blocks)
            x = torch.randn(16, 288, 16)
            amps = torch.ones(16, 288)
            y, _ = f.payload(x, amps, amps)
            self.assertLess(float((x-y).square().mean()), 1e-5)

    def test_optical_modulation_roundtrip_and_cost(self):
        f = self.frame('fso_weak', snr=100)
        x = torch.randn(16, 288, 16)
        amps = amplitudes(f.tx_mode, 288, True, True)
        y, _ = f.payload(x, amps, amps)
        self.assertLess(float((torch.tanh(x)-y).square().mean()), 1e-5)
        torch.testing.assert_close(f.last_stats['tx_average_power'], torch.ones(16))
        self.assertEqual(f.last_stats['payload_uses'], 9216)
        self.assertEqual(f.last_stats['forward_uses'], 9408)
        self.assertEqual(f.last_stats['reverse_rf_real_uses'], 64)

    def test_rf_uep_fixed_power(self):
        f = self.frame(batch=3)
        modes = torch.tensor([0, 1, 2])
        x = normalize_groups(torch.randn(3, 288, 16), modes)
        amp = amplitudes(modes, 288, False, True)
        f.payload(x, amp, amp)
        torch.testing.assert_close(f.last_stats['tx_average_power'], torch.ones(3), atol=1e-5, rtol=1e-5)
        self.assertEqual(f.last_stats['forward_uses'], 2400)

    def test_pilot_noise_estimation(self):
        f = self.frame(snr=10, batch=10000)
        self.assertLess(abs(float(f.receiver.noise_var.mean())-.1), .003)
        self.assertLess(abs(float((f.receiver.gain-f.true_gain).abs().square().mean())-.1/32), .0003)

    def test_deep_fade_is_finite_not_division_explosion(self):
        with patch('v10.channels.sample_gain', return_value=torch.zeros(4, 1, dtype=torch.complex64)):
            f = self.frame(snr=-5, batch=4)
        x = torch.randn(4, 32, 16, requires_grad=True)
        a = torch.ones(4, 32)
        y, _ = f.payload(x, a, a)
        self.assertTrue(torch.isfinite(y).all())
        self.assertLess(float(y.abs().max()), 20)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_all_profiles_differentiable(self):
        for name in PROFILES:
            f = self.frame(name, snr=10, batch=4)
            x = torch.randn(4, 288, 16, requires_grad=True)
            a = amplitudes(f.tx_mode, 288, f.optical, True)
            y, r = f.payload(x, a, a)
            adapter = ReliabilityAdapter()
            loss = adapter(y, r).square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(x.grad).all(), name)
            self.assertTrue(torch.isfinite(adapter.net[-1].weight.grad).all(), name)

    def test_receiver_metadata_ignores_source(self):
        f = self.frame(batch=3)
        f.tx_mode = torch.tensor([0, 1, 2])
        f.rx_mode = f.tx_mode.clone()
        packer = DeterministicPacker()
        packer.frame = f
        c, i, t = torch.randn(3, 32, 16), torch.randn(3, 256, 16), torch.randn(3, 77, 16)
        p1, m1 = packer.pack(c, i, t, text_attention_mask=torch.ones(3, 77))
        p2, m2 = packer.pack(c, i, t, text_attention_mask=torch.zeros(3, 77), snr_db=999)
        torch.testing.assert_close(p1, p2)
        for a, b in zip(m1['text_indices'], m2['text_indices']):
            torch.testing.assert_close(a, b)
        cc, ii, tt = unpack(p1, m1)
        torch.testing.assert_close(cc, c)
        for b in range(3):
            torch.testing.assert_close(ii[b, m1['image_indices'][b]], i[b, m1['image_indices'][b]])
            ids = m1['text_indices'][b].unique()
            torch.testing.assert_close(tt[b, ids], t[b, ids])
        # Simulated header corruption MUST change receiver metadata, not be healed.
        f.rx_mode = torch.tensor([2, 2, 0])
        _, wrong = packer.pack(c, i, t)
        self.assertEqual(wrong['text_counts'].tolist(), [192, 192, 32])

    def test_adapter_initial_identity(self):
        x, r = torch.randn(2, 32, 16), torch.rand(2, 32, 1)
        torch.testing.assert_close(ReliabilityAdapter()(x, r), x)

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            self.frame(pilots=1)
        with self.assertRaises(ValueError):
            sample_gain(dict(PROFILES['fso_weak'], alpha=-1), (2,), 'cpu')
        with self.assertRaises(ValueError):
            mode_codebook('cpu', 7)
        d = dict(train=[1], validation=[2], calibration=[3], test=[1], excluded_history=[])
        with self.assertRaises(ValueError):
            validate_manifest(d)

if __name__ == '__main__':
    unittest.main(verbosity=2)
