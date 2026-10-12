"""No Torch: length/resources/schedules/quality/report negative contracts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
import numpy as np
from text_recovery_v3.core import load,spec,geometry,encode_header,decode_header,schedule,gate,row,summarize
from text_recovery_v3.report import check_text
from text_recovery_v3.channel import resource,send
from fso_fluctuating_v2.core import load as channel_config,paths


class Contracts(unittest.TestCase):
    def test_default_configuration(self): self.assertEqual(load()['schema'],'text-recovery-v3')
    def test_header_all_lengths(self):
        for n in range(2,129):
            x=encode_header(n); self.assertEqual(decode_header(x),(n,True)); self.assertEqual(float((x*x).mean()),1.)
    def test_header_invalid_range(self):
        x=np.ones(16); x[:14]=-1
        self.assertEqual(decode_header(x),(1,False))
    def test_header_corruption_not_oracle_corrected(self):
        x=encode_header(12); x[:2]*=-1
        n,good=decode_header(x); self.assertTrue(good); self.assertNotEqual(n,12)
    def test_header_shape_nonfinite(self):
        for x in (np.ones(15),np.full(16,np.nan)):
            with self.assertRaises(ValueError): decode_header(x)
    def test_charge_resources(self):
        self.assertEqual(geometry('A0_legacy')['payload_complex_symbols'],256)
        self.assertEqual(geometry('A3_lora')['payload_complex_symbols'],264)
        self.assertEqual(geometry('A4_wide')['payload_complex_symbols'],1032)
        self.assertEqual(geometry('A5_common')['payload_complex_symbols'],520)
        self.assertEqual(geometry('A2_ramp'),geometry('A3_lora'))
    def test_schedule_not_abrupt(self):
        c=load(); self.assertEqual(schedule(199,1200,c,'A3_lora')['ce_weight'],0.)
        self.assertEqual(schedule(200,1200,c,'A3_lora')['ce_weight'],.005)
        self.assertEqual(schedule(399,1200,c,'A3_lora')['ce_weight'],1.)
        self.assertEqual(schedule(200,1200,c,'A1_length_mask')['ce_weight'],1.)
    def test_config_unknown_key_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json'; c=load(); c['oracle_length']=True; p.write_text(json.dumps(c))
            with self.assertRaises(ValueError): load(p)
    def test_config_nonfinite_and_bool_rejected(self):
        for k,value in [('bridge_lr',float('nan')),('rank',True),('warmup_steps',1200)]:
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'config.json'; c=load(); c[k]=value; p.write_text(json.dumps(c))
                with self.assertRaises(ValueError): load(p)
    def test_unknown_arm(self):
        with self.assertRaises(ValueError): spec('oracle')
    def test_whole_waveform_header_accounting(self):
        c=channel_config(); condition=c['conditions'][0]; path=paths(c,condition['weather'],1,'text-v3-contract')[0]
        for arm in ('A0_legacy','A3_lora','A4_wide','A5_common'):
            x=np.random.default_rng(4).standard_normal(geometry(arm)['total_reals']).astype(np.float32)
            if arm!='A0_legacy': x[:16]=encode_header(12)
            y,stats,t=send(c,condition,path,7,arm,x)
            res=resource(c,arm)
            self.assertEqual(y.shape,x.shape); self.assertTrue(np.isfinite(y).all())
            self.assertEqual(stats['used_samples'],res['scheduled_samples'])
            self.assertEqual(stats['payload_complex_symbols'],geometry(arm)['payload_complex_symbols'])
            self.assertEqual(len(t.events),2); self.assertIsNone(stats['pre_fec_ber'])
            self.assertEqual(stats['analog_repeats'],1)
    def example(self):
        item=dict(image_id=1,annotation_id=2,split='train',text='a cat sits',covered_text='a cat sits',covered_tokens=5,
            source_sha256='x',source_utf8_bytes=10,truncated=False)
        return row(item,'A3_lora','matched',dict(text='a cat sits',stopped_by_capacity=False,receiver_failure=False))
    def test_empty_groups_fail(self):
        with self.assertRaises(ValueError): summarize([])
    def test_quality_and_dependence(self):
        e=self.example(); good=summarize([e]); bad=dict(good,covered_wer=1.)
        self.assertTrue(gate(dict(matched=good,zero=bad,shuffled=bad),load())['passed'])
        self.assertFalse(gate(dict(matched=good,zero=good,shuffled=bad),load())['passed'])
    def test_quality_requires_natural_end(self):
        e=self.example(); e['stopped_by_capacity']=True; good=summarize([e]); bad=dict(good,covered_wer=1.)
        self.assertFalse(gate(dict(matched=good,zero=bad,shuffled=bad),load())['passed'])
    def test_report_recomputes_text(self):
        e=self.example(); check_text([e],[1],['matched'],'train'); e['covered_wer']=.2
        with self.assertRaises(ValueError): check_text([e],[1],['matched'],'train')
    def test_report_duplicate_missing_and_split(self):
        e=self.example()
        for rows,ids,split in [([e,e],[1],'train'),([e],[1,2],'train'),([e],[1],'validation')]:
            with self.assertRaises(ValueError): check_text(rows,ids,['matched'],split)


if __name__=='__main__': unittest.main(verbosity=2)
