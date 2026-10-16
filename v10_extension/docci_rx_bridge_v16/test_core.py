import ast
import math
from pathlib import Path
import sys
import tempfile
import unittest
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_rx_bridge_v16.core import FOLDER, PROJECT, load, geometry, order, safe, atomic, read, seal, sha


class Contracts(unittest.TestCase):
    def test_all_sources_parse_as_python38(self):
        for p in FOLDER.glob('*.py'):
            ast.parse(p.read_text(), filename=str(p), feature_version=8)

    def test_frozen_geometry_and_privileged_oracle(self):
        for arm in ('self', 'oracle_clip', 'rx_clip', 'rx_i0'):
            g = geometry(arm)
            self.assertEqual(g['payload_complex_symbols'], 13312)
            self.assertEqual(g['image_reals']+g['text_reals'], g['payload_reals'])
            self.assertIsNone(g['snr_db'])
            self.assertEqual(g['auxiliary_is_privileged'], arm == 'oracle_clip')
            self.assertFalse(g['original_source_length_or_mask_at_RX'])

    def test_full_data_and_bounded_smoke(self):
        self.assertEqual(load()['train_samples'], 9647)
        self.assertEqual(load(smoke=True)['train_samples'], 8)
        self.assertEqual(load(smoke=True)['val_samples'], 2)
        self.assertEqual(len(set(order(9647, 42, 0))), 9647)
        self.assertNotEqual(order(8, 42, 0), order(8, 42, 1))

    def test_atomic_seal_and_preserve_conflicts(self):
        with tempfile.TemporaryDirectory(prefix='v16_test_', dir=str(PROJECT)) as d:
            root = Path(d)
            atomic(root/'data.json', dict(value=1))
            seal(root)
            self.assertEqual(read(root/'LIGHT_MANIFEST.json')['data.json'], sha(root/'data.json'))
            self.assertFalse(read(root/'FINISHED.json')['quality_success'])
            with self.assertRaises(ValueError):
                safe(PROJECT/'../outside')
            with self.assertRaises(ValueError):
                safe(PROJECT)
            (root/'link').symlink_to(root/'data.json')
            with self.assertRaises(ValueError):
                safe(root/'link')

    def test_config_rejects_silent_threshold_or_capacity_change(self):
        with tempfile.TemporaryDirectory(prefix='v16_test_', dir=str(PROJECT)) as d:
            for key, value in [('train_samples', 2048), ('minimum_wer_gain_pp', .1), ('reserve_free_gib', 1), ('ce_epochs', 1000), ('alignment_batch', 1)]:
                c = load()
                c[key] = value
                p = Path(d)/(key+'.json')
                atomic(p, c)
                with self.assertRaises(ValueError):
                    load(p)


if __name__ == '__main__':
    unittest.main(verbosity=2)
