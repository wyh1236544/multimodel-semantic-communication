"""CPU exact arithmetic, framing and explicit continuation contracts."""
from pathlib import Path
import json
import random
import sys
import tempfile
import unittest
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
from docci_text_continue_v11.core import load,schedule,phase_lr,probes,choose_indices
from docci_text_continue_v11.metrics import distance,metrics,diagnostic
from final_compare_v1.metrics import distance as old_distance
from docci_longclip_joint_v8.metrics import metrics as old_metrics


class Contracts(unittest.TestCase):
    def test_exact_edit_distance_random_unicode_and_tokens(self):
        r=random.Random(42)
        for _ in range(200):
            a=''.join(r.choices('aé中🙂 ',k=r.randrange(24))); b=''.join(r.choices('b中é🙂 ',k=r.randrange(24)))
            self.assertEqual(distance(a,b),old_distance(a,b))
            self.assertEqual(distance(a.split(),b.split()),old_distance(a.split(),b.split()))
    def test_all_legacy_metric_fields_exact(self):
        for a,b in (('',''),('A cat sits.','a cat cat sits'),('中文 café 😀','中文 cafe 😀'),('Word '*90,'word '*100),('abc','')):
            self.assertEqual(metrics(a,b),old_metrics(a,a,b))
    def test_fullEOS_is_not_early_and_limit_is_not_repeat(self):
        seqs=[[2,0]+list(range(3,65))+[2]]+[[2,0,2]]*8
        row=dict(source_tokens=64,chunk_generated_token_ids=seqs)
        out=diagnostic([row]); self.assertEqual(out['slot_groups']['full62']['early_EOS'],0)
        self.assertEqual(out['slot_groups']['full62']['EOS_at_expected_length'],1)
        seqs[0]=[2,0]+list(range(3,66)); out=diagnostic([row])
        self.assertEqual(out['slot_groups']['full62']['capacity_stop'],1); self.assertEqual(out['first_segment_adjacent_repeat'],0)
    def test_partialEOS_does_not_prove_complete_content(self):
        row=dict(source_tokens=12,chunk_generated_token_ids=[[2,0,3,4,2]]+[[2,0,2]]*8)
        self.assertEqual(diagnostic([row])['slot_groups']['partial']['early_EOS'],1)
    def test_20epoch_budget_and_new_LR_stage(self):
        c=load(); s=schedule(c); self.assertEqual(s['target_updates'],20480); self.assertEqual(s['additional_updates'],10480)
        self.assertEqual(s['final_cumulative_presentations'],40960)
        self.assertEqual(phase_lr(c,0,10480),.00003); self.assertAlmostEqual(phase_lr(c,10479,10480),.00001)
    def test_global_order_does_not_restart(self):
        self.assertNotEqual(choose_indices(2048,2,10000,0,42),choose_indices(2048,2,0,0,42))
        self.assertEqual(probes(20480,10000,2000),[10000,12000,14000,16000,18000,20000,20480])
    def test_invalid_phase_bounds(self):
        with self.assertRaises(ValueError): schedule(load(),parent_updates=30000)
        with self.assertRaises(ValueError): phase_lr(load(),10480,10480)
    def test_phase_schema_finiteness_and_fixed_smoke(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'phase.json'; original=load()
            for key,value in (('target_epochs',10),('probe_samples',1),('smoke_updates',3),('phase_learning_rate',float('nan')),('phase_min_learning_rate',.0001),('extra',1)):
                c=dict(original); c[key]=value; path.write_text(json.dumps(c))
                with self.assertRaises(ValueError): load(path)


if __name__=='__main__': unittest.main(verbosity=2)
