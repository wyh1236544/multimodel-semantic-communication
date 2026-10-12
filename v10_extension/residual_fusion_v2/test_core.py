"""NumPy/stdlib contracts; no claim of real Torch execution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from residual_fusion_v2 import ARMS
from residual_fusion_v2.core import load,augmentation,donor_index,diagnostic_decision,parameter_counts,check_receipt_group,paired_bootstrap


def receipts():
    return [dict(arm=a,frame_key='1__clean__d0',received_sha256='rx',payload_sha256='tx',source_sha256='source',
        receiver_available=True,payload_complex_symbols=2304,analog_repeats=1,raw_utf8_wire_bytes=0,
        received_caption='same',latent_sha256='base' if a in ('B4_frozen','R1_off') else a) for a in ARMS]


class Contracts(unittest.TestCase):
    def test_default_load(self):
        c=load(); self.assertEqual(c['train_samples']*(c['clean_epochs']+c['noise_epochs']),4000)
    def test_bad_config_refused(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'config.json'
            for key,value in [('residual_hidden',18),('drop_fraction',.6),('diagnostic_wer_gate',.8),('seed',True),('diagnostic_lr',float('nan'))]:
                c=load(); c[key]=value; p.write_text(json.dumps(c))
                with self.assertRaises(ValueError): load(p)
    def test_real_gate_fails_and_smoke_does_not_fake_pass(self):
        m={'D0_precompression':.05,'D1_T32_clean':.7}
        r=diagnostic_decision(m,.1); self.assertFalse(r['proceed']); self.assertTrue(r['required'])
        s=diagnostic_decision(m,.1,True); self.assertTrue(s['proceed']); self.assertFalse(s['observed_pass']); self.assertFalse(s['required'])
    def test_gate_valid_and_missing_metrics(self):
        self.assertTrue(diagnostic_decision({'D0_precompression':.05,'D1_T32_clean':.1},.1)['proceed'])
        with self.assertRaises(ValueError): diagnostic_decision({'D0_precompression':.01},.1)
    def test_real_distinct_donors(self):
        for n in (2,8,32): self.assertTrue(all(donor_index(i,n)!=i for i in range(n)))
        with self.assertRaises(ValueError): donor_index(0,1)
    def test_augmentation_is_paired(self):
        c=load(); x=[augmentation(c,'fso',1,i) for i in range(1000)]
        self.assertEqual(x,[augmentation(c,'fso',1,i) for i in range(1000)])
        self.assertEqual(set(x),{'correct','zero','shuffle'})
    def test_parameter_scale(self): self.assertEqual(sum(parameter_counts(256,64).values()),49505)
    def test_shared_receipts_and_exact_off(self): check_receipt_group(receipts())
    def test_receipt_mutations_refused(self):
        for key,value in [('received_sha256','wrong'),('raw_utf8_wire_bytes',5),('analog_repeats',2),('received_caption','wrong')]:
            r=receipts(); r[2][key]=value
            with self.assertRaises(ValueError): check_receipt_group(r)
        r=receipts(); r[3]['latent_sha256']='changed'
        with self.assertRaises(ValueError): check_receipt_group(r)
    def test_missing_arm_refused(self):
        with self.assertRaises(ValueError): check_receipt_group(receipts()[:-1])
    def test_cluster_bootstrap_not_trial_count(self):
        rows=[]
        for i in range(4):
            for cond in ('x','y'):
                for draw in range(2):
                    for arm,offset in [('R0_image',0),('R1_joint',.2)]:
                        rows.append(dict(arm=arm,image_id=i,condition=cond,draw=draw,psnr=20+i+offset,receiver_available=True))
        r=paired_bootstrap(rows,'R0_image','R1_joint',['x','y'],'psnr',500)
        self.assertEqual(r['images'],4); self.assertEqual(r['trials'],16); self.assertAlmostEqual(r['mean'],.2)
        self.assertAlmostEqual(r['ci95'][0],.2)


if __name__=='__main__': unittest.main(verbosity=2)
