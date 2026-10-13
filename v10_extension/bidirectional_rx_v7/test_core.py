import json
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bidirectional_rx_v7.core import load,plan,geometry,mask_policy,array_hash,paired_ci,variants


class Contracts(unittest.TestCase):
    def test_budget_and_modes(self):
        c=load(); self.assertEqual(geometry()['payload_complex_symbols'],3072)
        self.assertEqual(plan(c,'diagnostic')['train_samples'],32); self.assertEqual(len(variants(['clean','fso'])),19)
        self.assertEqual(plan(c,'screen')['phases'],['fso'])

    def test_masks_are_paired_and_separate(self):
        c=load()
        for j in range(4):
            a=mask_policy(c,j,101); self.assertEqual(a,mask_policy(c,j,101))
            self.assertEqual(bool(a['image_slots']),j in (1,3)); self.assertEqual(bool(a['text_slots']),j in (2,3))

    def test_float32_hash_roundtrip(self):
        x=np.arange(6144,dtype=np.float32); self.assertEqual(array_hash(x),array_hash(x[None]))
        y=x.copy(); y[1]+=1; self.assertNotEqual(array_hash(x),array_hash(y))

    def test_paired_effect_keeps_outages_and_sign(self):
        rows=[]
        for iid in (1,2):
            for arm,psnr,wer in [('R0_self',20,1),('R3_bidirectional',21,.9)]:
                rows.append(dict(image_id=iid,arm=arm,phase='fso',control='normal',condition='moderate',psnr=psnr,normalized_wer=wer))
        result=paired_ci(rows,'R3_bidirectional','fso',['moderate'])
        self.assertEqual(result['delta_psnr_db'],1); self.assertAlmostEqual(result['delta_wer'],-.1)


if __name__=='__main__': unittest.main(verbosity=2)
