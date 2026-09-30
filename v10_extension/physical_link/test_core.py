"""Small deterministic numerical contract checks, NOT real COCO results."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physical_link.digital import *
from physical_link.waveform import *
from physical_link.report import make_report


class DigitalTests(unittest.TestCase):
    def test_source_quantization_crc_and_length(self):
        x = np.linspace(-4, 4, 4608)
        for q in (4, 6, 8):
            bits, quantized, _ = encode_source(x, q)
            restored, ok = decode_source(bits, len(x), q)
            self.assertTrue(ok)
            np.testing.assert_array_equal(restored, quantized)
            self.assertEqual(len(bits), packet_size(len(x), q))
            self.assertLessEqual(np.max(np.abs(x - restored)), 4 / ((1 << q) - 1) + 1e-12)
            damaged = bits.copy()
            damaged[150] ^= 1
            restored, ok = decode_source(damaged, len(x), q)
            self.assertFalse(ok)
            self.assertTrue(np.all(restored == 0))
            self.assertFalse(decode_source(bits[:-8], len(x), q)[1])

    def test_gray_mapping_and_bit_demapping(self):
        points, labels = alphabet(16)
        self.assertAlmostEqual(float(np.mean(np.abs(points) ** 2)), 1.)
        np.testing.assert_allclose(points[[0, 4, 12, 8]].real * np.sqrt(10), [-3, -1, 1, 3])
        for order in (4, 16):
            points, labels = alphabet(order)
            decoded = (demodulate(modulate(labels.ravel(), order), order, .01) < 0).astype(np.uint8)
            np.testing.assert_array_equal(decoded, labels.ravel())

    def test_convolutional_roundtrip_and_correction(self):
        # MSB-newest shift register, standard octal tap ordering; impulse test.
        np.testing.assert_array_equal(fec_encode(np.array([1], dtype=np.uint8), block=1),
                                      [1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        rng = np.random.default_rng(22)
        bits = rng.integers(0, 2, 3081, dtype=np.uint8)
        coded = fec_encode(bits)
        self.assertEqual(len(coded), fec_size(len(bits)))
        llr = (1. - 2. * coded) * 8
        llr[::257] *= -1
        self.assertGreater(np.mean((llr < 0) != coded), 0)
        np.testing.assert_array_equal(fec_decode(llr, len(bits)), bits)

    def test_fec_noise_benefit(self):
        rng = np.random.default_rng(32)
        bits = rng.integers(0, 2, 12000, dtype=np.uint8)
        coded = fec_encode(bits)
        received = (1. - 2. * coded) + rng.normal(0, .65, len(coded))
        raw_error = np.mean((received < 0) != coded)
        corrected_error = np.mean(fec_decode(received * 2 / .65 ** 2, len(bits)) != bits)
        self.assertLess(corrected_error, raw_error / 3)


class WaveformTests(unittest.TestCase):
    def test_rrc_special_values_and_energy(self):
        self.assertAlmostEqual(rrc(np.array([25]), .01)[0], rrc(np.array([-25]), .01)[0])
        self.assertTrue(np.isfinite(rrc(np.array([0, 25, -25]), .01)).all())
        for h in Pulse().kernels:
            self.assertAlmostEqual(float(np.sum(h * h)), 1.)

    def test_clock_and_no_noise_loopback(self):
        bits = np.random.default_rng(2).integers(0, 2, 4000, dtype=np.uint8)
        symbols = np.r_[preamble(), modulate(bits, 16)]
        for rate in (8e9, 14e9, 16e9):
            pulse = Pulse(rs=rate)
            if rate == 14e9:
                self.assertEqual((pulse.p, pulse.q), (32, 7))
            for delay in (0, 7, 23):
                observed, tx, codes = transmit(symbols, pulse, .35 - .8j, 0., np.random.default_rng(8), delay=delay)
                restored, rx = receive(observed, pulse, len(symbols), preamble())
                self.assertEqual(rx['sync_sample'], delay)
                recovered = demodulate(restored[256:], 16, rx['equalized_noise_variance']) < 0
                np.testing.assert_array_equal(recovered[:len(bits)], bits)
                self.assertEqual(len(codes), tx['awg_samples'])
                self.assertAlmostEqual(tx['waveform_seconds'], len(codes) / 64e9)
                self.assertLessEqual(np.max(np.abs(codes.real)), 1)

    def test_matched_noise_calibration(self):
        pulse = Pulse(rs=16e9)
        wave = pulse.shape(np.zeros(5000))
        rng = np.random.default_rng(49)
        noise = (rng.standard_normal(len(wave)) + 1j * rng.standard_normal(len(wave))) / np.sqrt(2)
        observed = pulse.match(noise, 5000)[100:-100]
        self.assertAlmostEqual(float(np.mean(np.abs(observed) ** 2)), 1, delta=.06)

    def test_fixed_sample_power_across_rates(self):
        bits = np.random.default_rng(34).integers(0, 2, 40000, dtype=np.uint8)
        symbols = np.r_[preamble(), modulate(bits, 16)]
        powers = []
        for rate in (8e9, 14e9, 16e9):
            _, diag, _ = transmit(symbols, Pulse(rs=rate), 1., .01 * rate / 14e9,
                                  np.random.default_rng(3))
            powers.append(diag['sample_average_power'])
        self.assertLess(max(powers) / min(powers), 1.01)

    def test_full_digital_fec_crc_waveform(self):
        original = np.random.default_rng(91).normal(0, 1, 4608)
        bits, expected, _ = encode_source(original)
        coded = fec_encode(bits ^ scramble(len(bits)))
        symbols = np.r_[preamble(), modulate(coded, 16)]
        pulse = Pulse()
        observed, tx, _ = transmit(symbols, pulse, .7 + .3j, 1e-4, np.random.default_rng(81))
        equalized, rx = receive(observed, pulse, len(symbols), preamble())
        llr = demodulate(equalized[256:], 16, rx['equalized_noise_variance'])[:len(coded)]
        decoded = fec_decode(llr, len(bits)) ^ scramble(len(bits))
        recovered, ok = decode_source(decoded, len(original))
        self.assertTrue(ok)
        np.testing.assert_array_equal(recovered, expected)
        self.assertEqual(len(bits), 36976)
        self.assertEqual(len(coded), 76220)
        self.assertEqual(len(symbols) - 256, 19055)

    def test_invalid_and_bandwidth_impairment(self):
        with self.assertRaises(ValueError):
            Pulse(rs=64e9, alpha=.1)
        with self.assertRaises(ValueError):
            Pulse(span=7)
        symbols = np.r_[preamble(), modulate(np.random.default_rng(3).integers(0, 2, 4000), 16)]
        pulse = Pulse()
        clean, _, _ = transmit(symbols, pulse, 1, 0, np.random.default_rng(5))
        impaired, _, _ = transmit(symbols, pulse, 1, 0, np.random.default_rng(5), bandwidth=2e9)
        clean_rx, _ = receive(clean, pulse, len(symbols), preamble())
        poor_rx, _ = receive(impaired, pulse, len(symbols), preamble())
        self.assertGreater(np.mean(np.abs(poor_rx[256:] - symbols[256:]) ** 2),
                           10 * np.mean(np.abs(clean_rx[256:] - symbols[256:]) ** 2))


class ReportTests(unittest.TestCase):
    def test_strict_pairing_and_missing_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            design = {'args': {'samples': 2, 'repeats': 1, 'profiles': ['fso_moderate'], 'snr_list': [20]}, 'runs': []}
            for rate in (8e9, 14e9):
                d = root / str(rate)
                d.mkdir()
                rows = [{'image_id': i, 'profile': 'fso_moderate', 'nominal_snr_db': 20,
                         'repeat': 0, 'optical_irradiance_gain': .8, 'psnr': 20., 'ssim': .7,
                         'bleu2': .4} for i in range(2)]
                (d / 'samples.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
                (d / 'COMPLETE.json').write_text(json.dumps({'rows': 2}))
                (d / 'summary.json').write_text(json.dumps({'rows': 2, 'split': 'validation',
                    'source_caption_used_by_receiver': False, 'diffusion_executed': False}))
                (d / 'run_identity.json').write_text(json.dumps({'checkpoint_sha256': 'synthetic', 'split_sha256': 'synthetic'}))
                design['runs'].append({'output': str(d), 'rate': rate, 'variant': 'analog'})
            self.assertEqual(make_report(root, design)['rows'], 4)
            rows[0]['optical_irradiance_gain'] = .7
            (d / 'samples.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with self.assertRaises(ValueError):
                make_report(root, design)
            (d / 'samples.jsonl').write_text(json.dumps(rows[1]) + '\n')
            with self.assertRaises(ValueError):
                make_report(root, design)


if __name__ == '__main__':
    unittest.main(verbosity=2)
