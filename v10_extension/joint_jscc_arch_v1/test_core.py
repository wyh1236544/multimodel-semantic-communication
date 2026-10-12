"""Local stdlib/NumPy contracts; not a Torch/pretrained/GPU performance test."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from joint_jscc_arch_v1.core import (load,chunks,IMAGE_INDICES,WIRE_GROUPS,
    DECODE_GROUPS,arm_options,check_pair_histories,expected_keys,bootstrap)


class Contracts(unittest.TestCase):
    def test_public_resource_layout(self):
        self.assertEqual(len(set(IMAGE_INDICES)),224)
        self.assertEqual(len(WIRE_GROUPS)*16//2,2304)
        self.assertEqual((len(WIRE_GROUPS),len(DECODE_GROUPS)),(288,320))
        self.assertEqual(WIRE_GROUPS.count(0),240)

    def test_chunking_covers_long_input_without_overlength_clip(self):
        r=chunks(list(range(200)),128,1000,1001,1001)
        self.assertEqual(r['target_length'],128)
        self.assertEqual(r['covered_content_tokens'],126)
        self.assertTrue(r['truncated'])
        self.assertEqual([x[2] for x in r['chunks']],[75,51])
        self.assertTrue(all(len(i)==len(m)==77 for i,m,_ in r['chunks']))
        r=chunks([],128,1000,1001,1001)
        self.assertEqual(r['target_length'],2)
        self.assertEqual(sum(r['chunks'][0][1]),2)

    def test_arms_and_configuration(self):
        c,channel=load()
        self.assertEqual(channel['fs_hz']/channel['sps'],16e9)
        self.assertFalse(arm_options('B1_text')['modern'])
        self.assertTrue(arm_options('B4_separate')['auxiliary'])
        with self.assertRaises(ValueError): arm_options('B0_legacy')
        for key,value in [('max_text_tokens',76),('codec_hidden',63),('new_lr',float('nan')),
                          ('reserve_free_gib',20.),('channel_config','../escape.json')]:
            bad=copy.deepcopy(c); bad[key]=value
            with tempfile.TemporaryDirectory() as root:
                file=Path(root)/'bad.json'; file.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): load(file)

    def test_pairing_rejects_changed_mask_or_order(self):
        r=dict(phase='fso',epoch=1,image_id=1,bank_index=0,anchor=False,
               task_loss_applied=True,noise_seed=123)
        check_pair_histories([[r],[copy.deepcopy(r)]])
        for key,value in [('image_id',2),('task_loss_applied',False),('noise_seed',124)]:
            wrong=dict(r,**{key:value})
            with self.assertRaises(ValueError): check_pair_histories([[r],[wrong]])

    def test_matrix_and_image_bootstrap(self):
        self.assertEqual(len(expected_keys(['a','b'],[1,2],[dict(id='fso')],2)),12)
        s=bootstrap([.2]*32,'constant',200)
        self.assertAlmostEqual(s['mean'],.2)
        self.assertTrue(all(abs(x-.2)<1e-10 for x in s['ci95']))


if __name__=='__main__': unittest.main(verbosity=2)

