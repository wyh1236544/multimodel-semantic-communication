"""CPU contracts, dataset-scale accounting, negative safety cases. No weights."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
import numpy as np
from docci_baseline_v14.core import (PROJECT, FOLDER, load, geometry, batches,
                                    updates, safe, DECODE, read)
from docci_reconstruction_v10.core import chunk_plan
from docci_baseline_v14.report import inside, paired, expected_outputs


class Contracts(unittest.TestCase):
    def setUp(self):
        (PROJECT/'releases').mkdir(exist_ok=True)

    def test_full_9647_is_supported_without_legacy_mutation(self):
        c = load(profile='full')
        self.assertEqual(c['text_train_samples'], 9647)
        self.assertEqual(c['image_train_samples'], 9647)
        self.assertEqual(load()['text_train_samples'], 2048)
        self.assertEqual(read(EXT/'docci_reconstruction_v10/config.json')['train_samples'], 64)

    def test_config_unknown_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory(dir=str(PROJECT/'releases'), prefix='v14_contract_') as tmp:
            p = Path(tmp)/'bad.json'
            c = load()
            c['text_learning_rate'] = float('nan')
            p.write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                load(p)
            c = load()
            c['silent_truncate'] = True
            p.write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                load(p)

    def test_tail_examples_not_dropped_or_free_padded(self):
        got = list(batches(17, 8, 3, 42))
        self.assertEqual(len(got), updates(17, 8, 3))
        for epoch in range(3):
            self.assertEqual(sorted(x for e, ids in got if e==epoch for x in ids), list(range(17)))
        self.assertEqual([len(ids) for e, ids in got if e==0], [8, 8, 1])

    def test_deterministic_training_presentations(self):
        self.assertEqual(list(batches(64, 8, 2, 42)), list(batches(64, 8, 2, 42)))
        self.assertNotEqual(list(batches(64, 8, 2, 42)), list(batches(64, 8, 2, 43)))

    def test_image_data_controls_have_equal_update_budget(self):
        seq = list(batches(2048, 2, 5, 42))
        small = [[x % 64 for x in ix] for _, ix in seq]
        self.assertEqual(sum(map(len, small)), 2048*5)
        self.assertEqual(len({x for ix in small for x in ix}), 64)

    def test_resources_are_clean_not_finite_SNR(self):
        t, i = geometry('text'), geometry('image')
        self.assertEqual(t['payload_complex_symbols'], 9216)
        self.assertEqual(i['payload_complex_symbols'], 4096)
        self.assertIsNone(t['snr_db'])
        self.assertFalse(t['FSO'])
        self.assertEqual(t['analog_repeats'], 1)
        self.assertTrue(t['all_empty_chunks_charged'])
        self.assertEqual(i['image_complex_uses_per_RGB_scalar'], 4096/(512*512*3))

    def test_full_original_chunk_roundtrip_and_loss_token_count(self):
        for n in (0, 61, 62, 63, 128, 510):
            body = list(range(3, 3+n))
            chunks = chunk_plan(body, dict(max_text_tokens=512, chunk_tokens=64))
            self.assertEqual(chunks.shape, (9, 64))
            self.assertEqual([int(v) for row in chunks for v in row[1:list(row).index(2)]], body)
            self.assertEqual(int((chunks != 1).sum()), n+18)

    def test_overflow_not_truncated(self):
        with self.assertRaises(ValueError):
            chunk_plan(list(range(3, 514)), dict(max_text_tokens=512, chunk_tokens=64))

    def test_decode_protocols_separate_and_primary_unchanged(self):
        self.assertEqual(DECODE['greedy'], dict(num_beams=1, no_repeat_ngram_size=0, repetition_penalty=1.))
        self.assertEqual(DECODE['deployment']['num_beams'], 4)

    def test_unsafe_output_and_symlink_rejected(self):
        with self.assertRaises(ValueError):
            safe('/tmp/not_inside_multimodel_v14')
        with self.assertRaises(ValueError):
            safe(PROJECT/'releases/../other')
        with tempfile.TemporaryDirectory(dir=str(PROJECT/'releases'), prefix='v14_contract_') as tmp:
            root = Path(tmp)
            (root/'link').symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                safe(root/'link/result.json')
            with self.assertRaises(ValueError):
                inside(root, '../elsewhere')

    def test_calibration_has_no_training_outputs(self):
        plan = expected_outputs(dict(mode='calibrate', task='text', text_arms=['legacy64','released64']))
        self.assertEqual(plan, ['text/native_validation', 'text/historical_reference/validation'])

    def test_paired_metrics_preserve_IDs_and_effect_direction(self):
        a = [dict(example_id=str(i), score=i) for i in range(3)]
        b = [dict(example_id=str(i), score=i+2) for i in range(3)]
        self.assertEqual(paired(a, b, 'score')['right_minus_left'], 2)
        with self.assertRaises(ValueError):
            paired(a, list(reversed(b)), 'score')

    def test_python38_source_grammar(self):
        for path in FOLDER.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=path.name, feature_version=8)

    def test_lpips_is_explicit_offline_no_auto_pretrained(self):
        src = (FOLDER/'perceptual.py').read_text()
        self.assertIn('pretrained=False, pnet_rand=True', src)
        self.assertIn("sha(p) != expected", src)
        self.assertNotIn('load_state_dict_from_url', src)

    def test_disk_cache_RAM_is_bounded_and_array_integrity_checked(self):
        from docci_baseline_v14.data import Dataset
        from docci_baseline_v14.core import sha, array_hash
        with tempfile.TemporaryDirectory(dir=str(PROJECT/'releases'),prefix='v14_cache_') as tmp:
            entries=[]
            for i in range(3):
                path=Path(tmp)/(str(i)+'.npz')
                x=np.asarray([i],np.float32)
                identity=dict(example_id=str(i))
                with path.open('xb') as f:
                    np.savez_compressed(f,meta=np.asarray(json.dumps(dict(identity=identity,array_hashes={'value':array_hash(x)}))),value=x)
                entries.append(dict(cache_file=str(path),cache_sha256=sha(path),identity=identity))
            data=Dataset(entries,1)
            for i in range(3):
                self.assertEqual(data[i]['value'].tolist(),[i])
                self.assertLessEqual(len(data.cached),1)
            data.entries[0]['cache_sha256']='0'*64
            with self.assertRaises(ValueError):
                data[0]


if __name__ == '__main__':
    unittest.main(verbosity=2)
