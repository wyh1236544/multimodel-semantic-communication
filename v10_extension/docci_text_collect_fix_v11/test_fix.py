"""CPU arithmetic and negative contracts; no neural re-inference or GPU claim."""
import copy
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
from docci_text_collect_fix_v11 import engine
from docci_text_collect_fix_v11.collect import inputs,copy_records,pack
from docci_text_continue_v11.core import atomic_json,digest


class Numeric(unittest.TestCase):
    def test_five_user_printed_cases_not_all31_raw_records(self):
        cases=[(3.2667744159698486,3.2667740844595046,138),(1.9448553323745728,1.944855100399739,111),
            (1.9448553323745728,1.944855100399739,111),(2.15571928024292,2.155719515710774,101),
            (.5508021712303162,.5508021129006371,187)]
        for actual,expected,n in cases:
            self.assertFalse(engine.legacy.close(actual,expected)); self.assertLessEqual(engine.compare(actual,expected,n,62,4)['absolute_difference'],engine.compare(actual,expected,n,62,4)['absolute_bound'])
    def test_positive_FP32_reduction_regression(self):
        rng=np.random.default_rng(42); old_rejected=0
        for _ in range(500):
            ns=rng.integers(1,57,9); xs=[rng.lognormal(1.,.9,int(n)).astype(np.float32) for n in ns]
            actual=float(np.concatenate(xs).mean(dtype=np.float32)); expected=math.fsum(float(x.mean(dtype=np.float32))*len(x) for x in xs)/int(ns.sum())
            out=engine.compare(actual,expected,int(ns.sum()),int(ns.max()),9); old_rejected+=not out['old_strict_check_passed']
        self.assertGreater(old_rejected,0)
    def test_material_discrepancy_still_rejected(self):
        with self.assertRaises(ValueError): engine.compare(float(np.float32(1.01)),1.,138,62,3)
    def test_nonfinite_negative_and_non_FP32_rejected(self):
        for v in (float('nan'),float('inf'),-1.,.1):
            with self.assertRaises(ValueError): engine.compare(v,1.,138,62,3)
    def test_bad_public_counts_rejected(self):
        for n,m,k in ((511,62,9),(0,1,1),(True,1,1),(10,11,1),(10,2,10)):
            with self.assertRaises(ValueError): engine.compare(1.,1.,n,m,k)
    def test_empty_content_must_have_None(self):
        row=dict(slots=[dict(n=0,value=None)]*9,value=None); self.assertIsNone(engine.aggregation(row,'value','n'))
        row['value']=0.
        with self.assertRaises(ValueError): engine.aggregation(row,'value','n')
    def test_zero_is_well_defined(self):
        self.assertEqual(engine.compare(0.,0.,62,62,1)['absolute_difference'],0.)


class Structural(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=str(engine.PROJECT),prefix='v11_collect_fix_test_'); self.root=Path(self.temp.name)
        s=dict(slot=0,gold_content_BPE=2,TF_content_CE=1.,native_TF_content_CE=float(np.float32(.1)),TF_content_accuracy=.5,
            TF_pEOS_before_gold_end=.1,TF_pEOS_at_gold_end=.5,TF_argmax_early_EOS=0,common_content_positions=2,TF_CE_common=1.,free_prefix_CE_common=2.)
        slots=[s]+[dict(slot=j,gold_content_BPE=0,TF_content_CE=None,native_TF_content_CE=None,TF_content_accuracy=None,
            TF_pEOS_before_gold_end=None,TF_pEOS_at_gold_end=.9,TF_argmax_early_EOS=0,common_content_positions=0,TF_CE_common=None,free_prefix_CE_common=None) for j in range(1,9)]
        self.row=dict(example_id='qual_dev_a',source_text_sha256='a'*64,source_tokens=4,actual_wire_sha256='b'*64,
            diagnostic_only=True,gold_prefix_is_privileged_NOT_actual_RX=True,native_source_memory_NOT_transmitted=True,
            slot_count=9,TF_total_CE=.5,TF_content_CE=1.,native_TF_content_CE=float(np.float32(.1)),TF_content_accuracy=.5,
            common_content_positions=2,TF_CE_common=1.,free_prefix_CE_common=2.,slots=slots)
        self.declared=dict(example_ids=['qual_dev_a']); self.receipts={'qual_dev_a':dict(text_sha256='a'*64,source_tokens=4)}
        self.matched={'qual_dev_a':dict(tx_wire_sha256='b'*64,chunk_generated_token_ids=[[2,0,3,4,2]]+[[2,0,2]]*8)}
    def tearDown(self): self.temp.cleanup()
    def seal(self):
        with (self.root/'rows.jsonl').open('w') as f: f.write(json.dumps(self.row)+'\n')
        atomic_json(self.root/'DIAGNOSTIC_COMPLETE.json',dict(rows=1,rows_sha256=digest(self.root/'rows.jsonl')))
    def call(self):
        self.seal(); audit=[]; out=engine.diagnostic(self.root,self.declared,self.receipts,self.matched,audit); return out,audit
    def test_full_prefix_checks_and_audit(self):
        original=copy.deepcopy(self.row); rows,audit=self.call(); self.assertEqual(rows,[original]); self.assertEqual(len(audit),5)
    def test_all_nonaggregation_validators_are_identical_AST(self):
        old=next(n for n in ast.parse(Path(engine.legacy.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='diagnostic')
        new=next(n for n in ast.parse(Path(engine.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='diagnostic')
        class RemoveAggregation(ast.NodeTransformer):
            def visit_For(self,node):
                if isinstance(node.target,ast.Tuple) and [getattr(n,'id',None) for n in node.target.elts]==['key','weight']: return ast.Pass()
                return self.generic_visit(node)
        new.args=old.args
        self.assertEqual(ast.dump(RemoveAggregation().visit(old),include_attributes=False),ast.dump(RemoveAggregation().visit(new),include_attributes=False))
    def test_privileged_scope_not_relaxed(self):
        self.row['gold_prefix_is_privileged_NOT_actual_RX']=False
        with self.assertRaises(ValueError): self.call()
    def test_wrong_wire_not_relaxed(self):
        self.row['actual_wire_sha256']='c'*64
        with self.assertRaises(ValueError): self.call()
    def test_wrong_count_not_relaxed(self):
        self.row['slots'][0]['common_content_positions']=1
        with self.assertRaises(ValueError): self.call()
    def test_changed_rows_hash_not_relaxed(self):
        self.seal()
        with (self.root/'rows.jsonl').open('a') as f: f.write(' \n')
        with self.assertRaises(ValueError): engine.diagnostic(self.root,self.declared,self.receipts,self.matched)
    def test_only_diagnostic_hook_restored_even_on_failure(self):
        original=engine.legacy.diagnostic; strict=engine.legacy.close
        def failure(root):
            self.assertIs(engine.legacy.close,strict); self.assertIsNot(engine.legacy.diagnostic,original); raise ValueError('other check failed')
        with patch.object(engine,'sources',return_value={}),patch.object(engine.legacy,'verify',side_effect=failure):
            with self.assertRaises(ValueError): engine.verify(self.root)
        self.assertIs(engine.legacy.diagnostic,original); self.assertIs(engine.legacy.close,strict)
    def test_light_copy_preserves_original_and_excludes_weights(self):
        source=self.root/'source'; source.mkdir(); (source/'data.json').write_text('{"raw":1}\n'); (source/'last.pth').write_bytes(b'not loaded')
        output=self.root/'output'; output.mkdir(); records=inputs(source); self.assertEqual([r['path'] for r in records],['data.json'])
        before=(source/'data.json').read_bytes(); copy_records(source,output,records); self.assertEqual((source/'data.json').read_bytes(),before)
        self.assertEqual((output/'data.json').read_bytes(),before); self.assertFalse((output/'last.pth').exists())
        archive=output/'records.tar.gz'; self.assertEqual(pack(output,archive),digest(archive))
        with self.assertRaises(FileExistsError): pack(output,archive)
    def test_symlink_input_rejected(self):
        source=self.root/'source'; source.mkdir(); outside=self.root/'outside.json'; outside.write_text('{}'); (source/'link.json').symlink_to(outside)
        with self.assertRaises(ValueError): inputs(source)


if __name__=='__main__': unittest.main(verbosity=2)
