"""Local config/mask/resource checks; no neural-performance claims."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
import numpy as np
from clip_bart_joint_v5.core import load,geometry,source_mask,text_gate,caption_view
from clip_bart_joint_v5.channel import resource,send
from fso_fluctuating_v2.core import load as physical_config,paths


class Contracts(unittest.TestCase):
    def test_default(self): self.assertEqual(load()['text_train_samples'],1024)
    def test_equal_registered_wire_contract(self):
        g=geometry(); self.assertEqual(g['total_reals'],6144); self.assertEqual(g['payload_complex_symbols'],3072)
        self.assertEqual(g['common_reals']+g['image_reals']+g['text_reals'],g['total_reals'])
        self.assertEqual(g['analog_repeats'],1); self.assertEqual(g['raw_utf8_wire_bytes'],0)
    def test_separate_source_blocks(self):
        m=source_mask('J0_separate'); self.assertEqual(m.shape,(32,226))
        self.assertTrue(m[:16,225].all()); self.assertTrue(m[16:,:225].all())
        self.assertFalse(m[:16,:225].any()); self.assertFalse(m[16:,225].any())
    def test_joint_masks_no_oracle(self):
        self.assertFalse(source_mask('J1_joint').any()); self.assertFalse(source_mask('J2_no_clip_text').any())
        with self.assertRaises(ValueError): source_mask('oracle')
    def test_bad_configuration(self):
        for key,value in [('seed',True),('text_width',16),('oracle',1),('joint_lr',float('nan')),('joint_warm_steps',1024)]:
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'config.json'; c=load(); c[key]=value; p.write_text(json.dumps(c))
                with self.assertRaises(ValueError): load(p)
    def test_no_test_source(self):
        with self.assertRaises(ValueError): caption_view('missing','missing','test',8,42)
    def test_gate_requires_dependence_and_EOS(self):
        good=dict(covered_wer=.1,capacity_stop_fraction=0); bad=dict(good,covered_wer=1.)
        groups=dict(matched=good,zero=bad,shuffled=bad)
        self.assertTrue(text_gate(groups,load())['passed'])
        groups['matched']=dict(good,capacity_stop_fraction=.1)
        self.assertFalse(text_gate(groups,load())['passed'])
    def test_new_full3072_waveform(self):
        c=physical_config(); condition=c['conditions'][0]; trace=paths(c,condition['weather'],1,'v5-frame-test')[0]
        x=np.random.default_rng(1).standard_normal(6144).astype(np.float32)
        y,stats,t=send(c,condition,trace,18,x); r=resource(c)
        self.assertEqual(y.shape,x.shape); self.assertTrue(np.isfinite(y).all())
        self.assertEqual(stats['used_samples'],r['scheduled_samples']); self.assertEqual(len(t.events),2)
        self.assertEqual(stats['payload_complex_symbols'],3072); self.assertEqual(r['symbol_rate_baud'],16e9)
        self.assertIsNone(stats['pre_fec_ber']); self.assertEqual(stats['analog_repeats'],1)


if __name__=='__main__': unittest.main(verbosity=2)
