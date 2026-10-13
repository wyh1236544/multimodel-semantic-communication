"""V6 CPU contracts: masks, resource equality and strict published budgets."""
from pathlib import Path
import json
import sys
import unittest
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from clip_bart_joint_v6.core import load,geometry,private_mask,source_mask,attach_text_source_hashes
from clip_bart_joint_v5.report import controls


class Contracts(unittest.TestCase):
    def original_joint_digest(self, pixels, text):
        """Execute the original data.cache hash expression, not a new formula.

        NumPy tensor-interface shim only; no Torch/model loading is claimed.
        """
        import ast
        import hashlib
        path=EXT/'bart_coco_modal_v2/data.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        cache=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cache')
        assignment=next(n for n in ast.walk(cache) if isinstance(n,ast.Assign) and
            any(isinstance(t,ast.Name) and t.id=='digest_input' for t in n.targets))
        class Pixels:
            def numpy(self): return pixels
        namespace={'source':{'text':text},'item':{'image':Pixels()},'mode':'joint'}
        exec(compile(ast.Module(body=[assignment],type_ignores=[]),str(path),'exec'),namespace)
        return hashlib.sha256(namespace['digest_input']).hexdigest()

    def receipt_fixture(self):
        import hashlib
        import numpy as np
        items=[]; texts={}
        for iid,text in ((1,'A cat sits.'),(2,'Café — 两只猫。')):
            pixels=np.full((1,3,128,128),iid/3,dtype=np.float32)
            joint=self.original_joint_digest(pixels,text)
            texts[str(iid)]=hashlib.sha256(text.encode('utf-8')).hexdigest()
            items.append(dict(image_id=iid,annotation_id=1000+iid,split='train',text=text,
                covered_text='TRUNCATED DECODE DIAGNOSTIC',source_utf8_bytes=len(text.encode('utf-8')),
                source_sha256=joint))
        ids=[r['image_id'] for r in items]; annotations={str(r['image_id']):r['annotation_id'] for r in items}
        receipt=dict(mode='joint',split='train',image_ids=ids,annotation_ids=annotations,
            annotation_sha256='a'*64,source_sha256={str(r['image_id']):r['source_sha256'] for r in items})
        text_receipt=dict(split='train',image_ids=ids+[3],annotation_ids=dict(annotations),
            annotation_sha256='a'*64,source_sha256=texts)
        return items,receipt,text_receipt

    def test_original_joint_digest_preserved_and_text_added(self):
        import copy
        items,receipt,text=self.receipt_fixture(); before=copy.deepcopy(receipt); originals=copy.deepcopy(items)
        result=attach_text_source_hashes(items,receipt,text)
        self.assertEqual(result['source_sha256'],before['source_sha256'])
        self.assertEqual(result['text_source_sha256'],text['source_sha256'])
        self.assertNotEqual(result['source_sha256']['1'],result['text_source_sha256']['1'])
        self.assertEqual(receipt,before); self.assertEqual(items,originals)

    def test_teacher_uses_original_utf8_not_covered_text(self):
        import hashlib
        items,receipt,text=self.receipt_fixture(); result=attach_text_source_hashes(items,receipt,text)
        expected=hashlib.sha256(items[1]['text'].encode('utf-8')).hexdigest()
        self.assertEqual(result['text_source_sha256']['2'],expected)
        self.assertNotEqual(expected,hashlib.sha256(items[1]['covered_text'].encode('utf-8')).hexdigest())

    def test_changed_original_text_rejected(self):
        items,receipt,text=self.receipt_fixture(); items[0]['text']='Wrong text!'
        items[0]['source_utf8_bytes']=len(items[0]['text'].encode('utf-8'))
        with self.assertRaisesRegex(ValueError,'independent BART view'): attach_text_source_hashes(items,receipt,text)

    def test_source_order_or_duplicate_rejected(self):
        for duplicate in (False,True):
            with self.subTest(duplicate=duplicate):
                items,receipt,text=self.receipt_fixture(); items=[items[0],items[0]] if duplicate else items[::-1]
                with self.assertRaises(ValueError): attach_text_source_hashes(items,receipt,text)

    def test_caption_split_mismatch_rejected(self):
        for field in ('item','receipt'):
            with self.subTest(field=field):
                items,receipt,text=self.receipt_fixture()
                if field=='item': items[0]['split']='validation'
                else: text['split']='validation'
                with self.assertRaises(ValueError): attach_text_source_hashes(items,receipt,text)

    def test_original_annotation_file_or_selected_id_rejected(self):
        for field in ('annotation_sha256','annotation_ids'):
            with self.subTest(field=field):
                items,receipt,text=self.receipt_fixture()
                if field=='annotation_sha256': text[field]='b'*64
                else: text[field]['1']=9999
                with self.assertRaises(ValueError): attach_text_source_hashes(items,receipt,text)

    def test_joint_identity_mismatch_rejected(self):
        items,receipt,text=self.receipt_fixture(); items[0]['source_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Joint image/text identity'): attach_text_source_hashes(items,receipt,text)

    def test_missing_bart_text_identity_rejected(self):
        items,receipt,text=self.receipt_fixture(); del text['source_sha256']['1']
        with self.assertRaises(ValueError): attach_text_source_hashes(items,receipt,text)

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
