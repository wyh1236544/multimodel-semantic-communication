import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from text_jscc_v1.core import load,metrics,summarize,overfit_gate,bootstrap,file_under,text_records


class Tok:
    bos_token_id=1; eos_token_id=2; pad_token_id=2
    def __call__(self,text,**kwargs): return {'input_ids':[3]*len(text.split())}
    def decode(self,ids,**kwargs): return ' '.join('word' for i in ids if i not in (1,2))


class Core(unittest.TestCase):
    def test_strict_config_and_fixed_budget(self):
        c=load(); self.assertEqual(c['slots']*c['width']//2,384)
    def test_no_fictitious_accuracy_or_clipped_wer(self):
        r=metrics('a cat','a cat','a cat cat cat cat'); self.assertGreater(r['normalized_wer'],1.)
    def test_empty_delivery_scored_failure(self):
        r=metrics('a cat','a cat',''); self.assertEqual(r['covered_wer'],1.); self.assertEqual(r['bleu2'],0.)
    def test_exact_and_covered_separate(self):
        r=metrics('a cat on a table','a cat','a cat'); self.assertEqual(r['covered_wer'],0.); self.assertGreater(r['normalized_wer'],0.)
    def test_bleu2_identical(self): self.assertEqual(metrics('a cat','a cat','a cat')['bleu2'],1.)
    def test_memory_dependence_required(self):
        groups={k:{'covered_wer':0.} for k in ('matched','zero','shuffled')}
        self.assertFalse(overfit_gate(groups,load())['passed'])
    def test_quality_gate_and_margin(self):
        groups={k:{'covered_wer':v} for k,v in [('matched',.01),('zero',.6),('shuffled',.5)]}
        self.assertTrue(overfit_gate(groups,load())['passed'])
    def test_paired_bootstrap_unit(self):
        b=bootstrap([-.1]*4,42,100); self.assertEqual(b['images'],4); self.assertAlmostEqual(b['mean'],-.1)
    def test_paths_escape_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('../outside','/absolute','x/../outside'):
                with self.assertRaises(ValueError): file_under(root,name)
            (root/'link').symlink_to(root)
            with self.assertRaises(ValueError): file_under(root,'link/file')
    def test_text_records_no_images_or_test_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ann=root/'annotations/localized_narratives'; ann.mkdir(parents=True)
            train=ann/'coco_train_captions.jsonl'; val=ann/'coco_val_captions.jsonl'
            train.write_text(json.dumps({'image_id':1,'caption':'a cat'})+'\n'+json.dumps({'image_id':1,'caption':'a different cat'})+'\n')
            val.write_text(json.dumps({'image_id':2,'caption':'a dog'})+'\n')
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            manifest=root/'splits.json'; manifest.write_text(json.dumps(dict(train=[1],validation=[2],calibration=[3],test=[4],excluded_history=[],annotation_sha256={'train':sha(train),'val':sha(val)})))
            r=text_records(root,manifest,'train',1,42,Tok(),128)
            self.assertEqual(r[0]['text'],'a cat'); self.assertFalse((root/'train2017').exists())
            with self.assertRaises(ValueError): text_records(root,manifest,'test',1,42,Tok(),128)
            with self.assertRaises(ValueError): text_records(root,manifest,'train',2,42,Tok(),128)
    def test_annotation_drift_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ann=root/'annotations/localized_narratives'; ann.mkdir(parents=True)
            (ann/'coco_train_captions.jsonl').write_text('{}\n')
            m=root/'m.json'; m.write_text(json.dumps(dict(train=[1],validation=[2],calibration=[3],test=[4],excluded_history=[],annotation_sha256={'train':'wrong','val':'wrong'})))
            with self.assertRaises(ValueError): text_records(root,m,'train',1,42,Tok(),128)


if __name__=='__main__': unittest.main(verbosity=2)
