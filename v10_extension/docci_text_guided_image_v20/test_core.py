import copy
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from docci_text_guided_image_v20 import core

class Contracts(unittest.TestCase):
    def test_resource_count(self):
        g = core.GEOMETRY
        self.assertEqual((g['image_reals']+g['text_reals'])//2,13312)
        self.assertEqual(g['total_complex_uses'],13312+64)
        self.assertEqual(g['analog_repeats'],1)

    def test_config_and_smoke(self):
        self.assertEqual(core.load()['train_samples'],8192)
        self.assertEqual(core.load(smoke=True)['train_samples'],8)
        self.assertEqual(len(list(core.batches(8,core.load(smoke=True)))),4)

    def test_parent_unseen_holdout_and_no_split_overlap(self):
        train = [dict(example_id='train_%05d'%i,text_sha256=str(i)) for i in range(9647)]
        val = [dict(example_id='qual_dev_%05d'%i,text_sha256='v'+str(i)) for i in range(100)]
        c = core.load()
        used = {r['example_id'] for r in train[:2048]}
        splits = core.split_records(train,val,used,c)
        held = {r['example_id'] for r in splits['heldout']}
        self.assertFalse(held & used)
        self.assertFalse(held & {r['example_id'] for r in splits['train']})
        self.assertEqual(len(splits['train']),8192)
        self.assertEqual(len(splits['heldout']),500)

    def test_channel_schedule_is_arm_independent_and_has_clean(self):
        c = core.load()
        values = [core.train_snr(c,e,str(i)) for e in range(2) for i in range(100)]
        self.assertEqual(set(values),{None,0,5,10,15,20})
        self.assertEqual(values,[core.train_snr(c,e,str(i)) for e in range(2) for i in range(100)])

    def test_controls_do_not_change_matched_repeats(self):
        c = core.load()
        a = list(core.trial_matrix(c,'A_self'))
        b = list(core.trial_matrix(c,'C_tx_rx_text'))
        self.assertEqual(a,[x for x in b if x[2]=='matched'])
        self.assertEqual(len(a),10)
        self.assertEqual(len(b),16)

    def test_paired_interval_and_degenerate_case(self):
        value = core.bootstrap([.1]*4,200)
        self.assertAlmostEqual(value['mean'],.1)
        self.assertEqual(value['descriptions'],4)
        with self.assertRaises(ValueError):
            core.bootstrap([1.])

    def test_output_path_guards(self):
        for p in ('../escape','.',str(core.PROJECT.parent/'outside')):
            with self.assertRaises(ValueError):
                core.safe(p)

if __name__ == '__main__':
    unittest.main(verbosity=2)
