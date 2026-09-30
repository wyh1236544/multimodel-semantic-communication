"""Pure-stdlib contract checks; runs locally without PyTorch."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

from link_compare.rate import effective_snr_db, serialization
from link_compare.report import make_report


class AccountingTests(unittest.TestCase):
    def test_rate_tradeoff(self):
        self.assertAlmostEqual(effective_snr_db(20, 10e6, 1e6), 10)
        self.assertAlmostEqual(serialization(2400, 1e6, 64, 1e6)['sequential_link_seconds'], .002464)
        with self.assertRaises(ValueError):
            effective_snr_db(10, 0, 1e6)

    def test_strict_report_and_pairing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runs = []
            for variant in ('coherent_continuous', 'coherent_qpsk'):
                folder = root / variant
                folder.mkdir()
                (folder / 'COMPLETE.json').write_text(json.dumps({'rows': 2, 'expected_rows': 2}))
                group = {'rows': 2, 'psnr': 20 if variant == 'coherent_continuous' else 19,
                         'psnr_p05': 18, 'ssim': .7, 'bleu2': .3, 'token_acc': .4,
                         'common_to_text_retrieval_r1': .5, 'header_rejected': 0}
                (folder / 'summary.json').write_text(json.dumps({
                    'rows': 2, 'split': 'validation', 'source_caption_used_by_receiver': False,
                    'diffusion_executed': False, 'summary': {'fso_moderate/10.0': group}}))
                with (folder / 'samples.jsonl').open('w') as stream:
                    for image_id in (101, 202):
                        row = {'image_id': image_id, 'repeat': 0, 'profile': 'fso_moderate',
                               'nominal_snr_db': 10., 'payload_uses': 2304, 'pilot_uses': 32,
                               'header_uses': 64, 'forward_uses': 2400,
                               'reverse_rf_real_uses': 64,
                               'physical_unit': 'coherent_optical_complex_symbol',
                               'tx_average_power': 1., 'psnr': group['psnr'], 'bleu2': .3}
                        stream.write(json.dumps(row) + '\n')
                runs.append({'variant': variant, 'forward_symbol_rate_sps': 1e6,
                             'reference_snr_db': [10.], 'simulated_nominal_snr_db': [10.],
                             'output': str(folder)})
            design = {'args': {'samples': 2, 'repeats': 1, 'profiles': ['fso_moderate'],
                               'snr_list': [10.], 'symbol_rates': [1e6],
                               'feedback_symbol_rate': 1e6}, 'runs': runs}
            report = make_report(root, design)
            self.assertEqual(report['rows'], 4)
            self.assertEqual(len(report['conditions']), 2)
            self.assertEqual(report['paired_coherent_constellation_differences'][0]['mean_delta_psnr_db'], -1)
            self.assertIsNone(report['conditions'][0]['real_application_bit_rate_bps'])
            self.assertTrue((root / 'rate_summary.csv').is_file())


if __name__ == '__main__':
    unittest.main(verbosity=2)
