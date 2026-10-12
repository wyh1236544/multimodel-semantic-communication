"""Non-Torch caption-view and exact compact full-waveform contracts."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from bart_coco_modal_v2.core import load,caption_view,geometry,pack_numpy,expand_numpy,digest
from bart_coco_modal_v2.channel import transport,resource
from fso_fluctuating_v2.core import load as channel_config,paths,public_payload


class Contracts(unittest.TestCase):
    def fixture(self,root):
        ids=dict(train=list(range(1,9)),validation=[101,102],calibration=[201],test=[301],excluded_history=[401])
        split=root/'split.json'; split.write_text(json.dumps(ids)); (root/'annotations').mkdir()
        for part,chosen in [('train',ids['train']),('val',ids['validation'])]:
            data=dict(images=[dict(id=i,file_name=str(i).zfill(12)+'.jpg') for i in chosen],annotations=[])
            for i in chosen: data['annotations']+= [dict(id=i*10+5,image_id=i,caption='short'),dict(id=i*10,image_id=i,caption='  Original   longer caption.  ')]
            (root/'annotations'/('captions_'+part+'2017.json')).write_text(json.dumps(data))
        return split
    def test_min_annotation_not_shortest_and_no_rewrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); split=self.fixture(root); before=digest(split)
            rows,receipt=caption_view(root,split,'train',8,42)
            self.assertEqual(before,digest(split)); self.assertTrue(all(r['text']=='Original longer caption.' for r in rows))
            self.assertTrue(all(r['annotation_id']==r['image_id']*10 for r in rows)); self.assertTrue(receipt['original_split_not_rewritten'])
            self.assertEqual(rows,caption_view(root,split,'train',8,42)[0])
    def test_no_test_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); split=self.fixture(root)
            with self.assertRaises(ValueError): caption_view(root,split,'test',1,42)
    def test_missing_official_not_ln_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); split=self.fixture(root)
            with self.assertRaises(FileNotFoundError): caption_view(root/'missing',split,'train',8,42)
    def test_missing_label_not_resampled(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); split=self.fixture(root); path=root/'annotations/captions_val2017.json'
            data=json.loads(path.read_text()); data['annotations']=data['annotations'][:2]; path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): caption_view(root,split,'validation',2,42)
    def test_duplicate_annotation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); split=self.fixture(root); path=root/'annotations/captions_val2017.json'
            data=json.loads(path.read_text()); data['annotations'].append(data['annotations'][0]); path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): caption_view(root,split,'validation',2,42)
    def test_json_order_does_not_select_best(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); split=self.fixture(root); path=root/'annotations/captions_val2017.json'
            before=caption_view(root,split,'validation',2,42)[0]; data=json.loads(path.read_text()); data['annotations'].reverse(); path.write_text(json.dumps(data))
            after=caption_view(root,split,'validation',2,42)[0]
            self.assertEqual([r['annotation_id'] for r in before],[r['annotation_id'] for r in after])
    def test_exact_compact_slots(self):
        wire=public_payload(4)
        for mode,n in [('image_only',2048),('text_only',512),('joint',2304)]:
            g=geometry(mode); payload=pack_numpy(wire,mode); full=expand_numpy(payload,mode)
            self.assertEqual(g['complex_symbols'],n); self.assertEqual(payload.size,n*2)
            self.assertTrue(np.array_equal(payload,pack_numpy(full,mode)))
            if mode=='text_only': self.assertFalse(full[32:256].any())
            if mode=='image_only': self.assertFalse(full[256:].any())
    def test_invalid_pack(self):
        with self.assertRaises(ValueError): pack_numpy(np.zeros((64,16)),'text_only')
        with self.assertRaises(ValueError): expand_numpy(np.zeros((288,16)),'text_only')
    def test_actual_compact_physics_accounting(self):
        c=channel_config(); condition=c['conditions'][0]; path=paths(c,condition['weather'],1,'new-contract')[0]; sizes=[]
        for mode in ('text_only','image_only','joint'):
            t=transport(c,condition,path,123,mode); t.probe(); x=pack_numpy(public_payload(5),mode)
            yy,rel,s=t.continuous(x); res=resource(c,mode)
            self.assertEqual(yy.shape,x.shape); self.assertEqual(s['used_samples'],res['scheduled_samples'])
            self.assertAlmostEqual(s['scheduled_seconds'],res['scheduled_seconds'],places=15)
            self.assertEqual(s['analog_repeats'],1); self.assertIsNone(s['pre_fec_ber']); self.assertTrue(np.isfinite(yy).all())
            self.assertGreaterEqual(rel,0); self.assertLessEqual(rel,1); self.assertLessEqual(s['used_samples'],s['slot_samples']); sizes.append(s['used_samples'])
        self.assertLess(sizes[0],sizes[1]); self.assertLess(sizes[1],sizes[2])
    def test_not_multiple_transmissions(self):
        c=channel_config(); condition=c['conditions'][0]; t=transport(c,condition,paths(c,condition['weather'],1,'a')[0],2,'text_only')
        with self.assertRaises(ValueError): t.continuous(np.zeros((64,16)),repeats=24)
    def test_config_and_python38(self):
        load()
        for path in Path(__file__).parent.glob('*.py'): ast.parse(path.read_text(),feature_version=(3,8))
        with tempfile.TemporaryDirectory() as d:
            for key,val in [('width',8),('train_samples',True),('learning_rate',float('nan')),('unknown',1),('default_modes',['oops'])]:
                c=load(); c[key]=val; path=Path(d)/'config.json'; path.write_text(json.dumps(c))
                with self.assertRaises(ValueError): load(path)


if __name__=='__main__': unittest.main(verbosity=2)
