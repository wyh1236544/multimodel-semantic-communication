"""NumPy/stdlib transport and report checks; no real model claims."""
import copy
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from token_guidance.spec import VARIANTS, DIAGNOSTICS, conditions, donor_map, expected_rows
from token_guidance.wire import send
from token_guidance.report import analyze


class Core(unittest.TestCase):
    def test_real_donors(self):
        self.assertEqual(donor_map([1, 2, 3]), {1: 2, 2: 3, 3: 1})
        for values in ([1], [1, 1]):
            with self.assertRaises(ValueError): donor_map(values)

    def test_clean_quantization(self):
        x = np.random.default_rng(3).standard_normal((288, 16)).astype(np.float32)
        y, stats, transport = send(x, {'family': 'awgn'}, 40, 1, 2, clean=True)
        self.assertIsNone(transport)
        self.assertEqual(stats['raw_utf8_wire_bytes'], 0)
        self.assertTrue(stats['feature_object_crc_accepted'])
        self.assertLess(np.mean((x-y)**2), .002)

    def test_waveform_no_raw_text(self):
        x = np.random.default_rng(3).standard_normal((288, 16)).astype(np.float32)
        y, stats, transport = send(x, {'family': 'awgn'}, 40, 1, 2)
        self.assertTrue(stats['feature_object_crc_accepted'])
        self.assertEqual({e['purpose'] for e in transport.events}, {'probe', 'features'})
        self.assertEqual(stats['raw_utf8_wire_bytes'], 0)
        clean, _, _ = send(x, {'family': 'awgn'}, 40, 1, 2, clean=True)
        np.testing.assert_array_equal(y, clean)
        z, s, _ = send(x, {'family': 'awgn'}, -20, 1, 2)
        self.assertFalse(s['feature_object_crc_accepted'])
        self.assertTrue(np.all(z == 0))

    @staticmethod
    def fixture():
        rows = []
        design = dict(validation_ids=[1, 2], smoke=True)
        for v in VARIANTS:
            for d in (('matched',)+DIAGNOSTICS if v == 'D' else ('matched',)):
                for i in design['validation_ids']:
                    for c in conditions(True):
                        rows.append(dict(variant=v, diagnostic=d, image_id=i, channel_kind=c[0], profile=c[1], nominal_snr_db=c[2],
                            psnr=20+(v == 'D'), ssim=.7, token_acc=.4, bleu2=.3, text_cer=.8, text_wer=.9,
                            semantic_reals=4608, raw_utf8_wire_bytes=0, field_identity='same', noise_seed=i,
                            awg_samples=9000, semantic_body_bytes=4608, feature_object_wire_bytes=4624,
                            pre_errors=1, pre_bits=100, post_errors=0, post_bits=100,
                            feature_object_crc_accepted=True, scheduled_seconds=1e-6, donor_crc_accepted=True))
        return rows, design

    def test_complete_pairing(self):
        rows, design = self.fixture()
        summary = analyze(rows, design)
        self.assertEqual(summary['rows'], expected_rows(2, True))
        self.assertEqual(summary['primary_physical_20db']['D-A']['mean_delta'], 1.)
        self.assertEqual(summary['direction_status'], 'no_clear_positive_evidence_yet')

    def test_reject_incomplete_duplicate_and_cheating(self):
        rows, design = self.fixture()
        with self.assertRaises(ValueError): analyze(rows[:-1], design)
        with self.assertRaises(ValueError): analyze(rows+[rows[0]], design)
        for key, value in [('raw_utf8_wire_bytes', 40), ('awg_samples', 3), ('noise_seed', 555), ('psnr', float('nan'))]:
            changed = copy.deepcopy(rows)
            changed[0][key] = value
            with self.assertRaises(ValueError): analyze(changed, design)


if __name__ == '__main__':
    unittest.main(verbosity=2)
