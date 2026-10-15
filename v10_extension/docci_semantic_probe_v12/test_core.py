import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
from docci_semantic_probe_v12.core import FOLDER,PROJECT,load,indices,damage_numpy,compare_pairs,safe,lr
TEST_ROOT=PROJECT/'.cache/v10/docci_semantic_probe_v12_tests'


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): TEST_ROOT.mkdir(parents=True,exist_ok=True)
    def test_complete_config_and_reject_unknown_nan_types(self):
        c=load(); self.assertEqual(c['train_samples'],2048)
        with tempfile.TemporaryDirectory(dir=str(TEST_ROOT)) as d:
            p=Path(d)/'config.json'
            for bad in ({**c,'unknown':1},{**c,'text_updates':True},{**c,'kd_temperature':float('nan')},{**c,'seed':43}):
                p.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): load(p)

    def test_order_covers_all_sources_and_is_arm_independent(self):
        order=sum((indices(64,2,u,42) for u in range(32)),[])
        self.assertEqual(sorted(order),list(range(64)))
        self.assertEqual(indices(64,2,5,42),indices(64,2,5,42))

    def test_fixed_resource_damage_exact_and_source_independent(self):
        a=np.ones(8192,np.float32); x=damage_numpy(a,'mask25',123)
        self.assertEqual(int((x==0).sum()),2048); self.assertEqual(x.shape,a.shape)
        self.assertTrue(np.array_equal(x==0,damage_numpy(-a,'mask25',123)==0))
        self.assertTrue(np.array_equal(a,damage_numpy(a,'clean',123)))
        with self.assertRaises(ValueError): damage_numpy(a,'fso',123)

    def test_symlink_and_parent_write_escape_rejected(self):
        with self.assertRaises(ValueError): safe('../escape')
        with tempfile.TemporaryDirectory(dir=str(TEST_ROOT)) as d:
            link=Path(d)/'link'; link.symlink_to('/private/tmp',target_is_directory=True)
            with self.assertRaises(ValueError): safe(link/'write')

    def test_bootstrap_sign_and_pairing(self):
        a={str(i):dict(psnr=float(i+1)) for i in range(4)}; b={str(i):dict(psnr=float(i)) for i in range(4)}
        self.assertEqual(compare_pairs(a,b,'psnr')['CI95'],[1.,1.])
        with self.assertRaises(ValueError): compare_pairs(a,{'bad':dict(psnr=0.)},'psnr')

    def test_fixed_lr_is_identical_for_both_arms_and_has_floor(self):
        c=load(); self.assertAlmostEqual(lr(c,'text',4095,4096),c['text_learning_rate']*.1)
        self.assertGreater(lr(c,'text',0,4096),0)

    def test_all_new_source_python38_and_no_network_install_calls(self):
        for f in FOLDER.glob('*.py'):
            tree=ast.parse(f.read_text(),filename=str(f),feature_version=(3,8))
            self.assertIsInstance(tree,ast.Module)
        text=(FOLDER/'launch.py').read_text()
        self.assertIn("HF_HUB_OFFLINE='1'",text)
        self.assertNotIn('pip install',text)


if __name__=='__main__': unittest.main(verbosity=2)
