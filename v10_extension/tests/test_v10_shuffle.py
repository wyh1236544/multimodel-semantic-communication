"""Cross-image donor mapping has no PyTorch dependency."""
import unittest
import hashlib
import json
import tempfile
from pathlib import Path
from v10.shuffle import cyclic_donor_map
from verify_shuffled_v10 import verify


class ShuffledCommonMappingTests(unittest.TestCase):
    def test_cycle_is_a_derangement_and_covers_all_images(self):
        ids = [42, 123, 2026, 9]
        donors = cyclic_donor_map(ids)
        self.assertEqual(set(donors), set(ids))
        self.assertEqual(set(donors.values()), set(ids))
        self.assertTrue(all(target != donor for target, donor in donors.items()))
        self.assertEqual(donors, cyclic_donor_map(ids))

    def test_single_image_and_duplicates_rejected(self):
        for ids in ([], [42], [42, 42], [42, 123, 42]):
            with self.assertRaises(ValueError):
                cyclic_donor_map(ids)

    def test_verifier_rejects_the_historical_no_op(self):
        code = Path(__file__).resolve().parents[1]
        names = ('eval_v10.py', 'models/joint_image_text_model.py', 'v10/model.py', 'v10/shuffle.py')
        sources = {name: hashlib.sha256((code/name).read_bytes()).hexdigest() for name in names}
        with tempfile.TemporaryDirectory(prefix='v10_shuffle_') as temporary:
            root = Path(temporary)
            manifest = root/'splits.json'
            manifest.write_text(json.dumps({'validation': [10, 20]}), encoding='utf-8')
            identity = {'status': 'complete', 'source_sha256': sources,
                        'args': {'ablation': 'common_only_shuffled', 'split': 'validation',
                                 'split_manifest': str(manifest), 'samples': 2,
                                 'repeats': 1, 'profiles': ['awgn'], 'snr_list': [0]}}
            (root/'run_identity.json').write_text(json.dumps(identity), encoding='utf-8')
            samples = root/'samples.jsonl'
            rows = [{'image_id': 10, 'common_payload_source_image_id': 20,
                     'common_payload_l2_delta': 1.},
                    {'image_id': 20, 'common_payload_source_image_id': 10,
                     'common_payload_l2_delta': 2.}]
            samples.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
            self.assertEqual(verify(root)['images_with_distinct_payloads'], 2)
            rows[0]['common_payload_l2_delta'] = 0.
            samples.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'equals target'):
                verify(root)


if __name__ == '__main__':
    unittest.main()
