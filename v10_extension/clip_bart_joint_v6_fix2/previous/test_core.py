"""V6 CPU contracts: masks, resource equality and strict published budgets."""
from pathlib import Path
import json
import sys
import unittest
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from clip_bart_joint_v6.core import load,geometry,private_mask,source_mask
from clip_bart_joint_v5.report import controls


class Contracts(unittest.TestCase):
    def test_extended_text_budget(self):
        c=load(); self.assertEqual((c['text_updates'],c['text_block_updates']),(9000,3000))
    def test_budget_unchanged(self):
        self.assertEqual(geometry()['payload_complex_symbols'],3072)
        self.assertEqual(geometry()['analog_repeats'],1)
    def test_balanced_separate_source(self):
        m=source_mask('J0_separate')
        self.assertEqual(m.shape,(32,32)); self.assertTrue(m[:16,16:].all()); self.assertFalse(m[:16,:16].any())
        self.assertTrue(m[16:,:16].all()); self.assertFalse(m[16:,16:].any())
    def test_joint_source_unblocked(self):
        self.assertFalse(source_mask('J1_joint').any())
    def test_no_clip_is_same_mask_not_new_budget(self):
        self.assertTrue((source_mask('J1_joint')==source_mask('J2_no_clip_text')).all())
    def test_dropout_reproducible(self):
        c=load(); self.assertEqual(private_mask(c,3,42,'clean'),private_mask(c,3,42,'fso'))
    def test_full_anchor_no_dropout(self):
        p=private_mask(load(),0,42,'clean'); self.assertEqual(p['image_slots'],[]); self.assertEqual(p['text_slots'],[])
    def test_private_dropout_not_common(self):
        p=private_mask(load(),3,42,'clean')
        self.assertEqual(len(p['image_slots']),56); self.assertEqual(len(p['text_slots']),8)
        self.assertTrue(all(0<=i<224 for i in p['image_slots']))
        self.assertTrue(all(0<=i<32 for i in p['text_slots']))
    def test_common_warm_no_private_dropout(self):
        p=private_mask(load(),3,42,'common'); self.assertEqual(p['image_slots'],[]); self.assertEqual(p['text_slots'],[])
    def test_unknown_arm_rejected(self):
        with self.assertRaises(ValueError): source_mask('anything')


if __name__=='__main__': unittest.main(verbosity=2)
