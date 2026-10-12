"""Raw-generated-text gate verification, including false completion markers."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from residual_fusion_v2.report import verify_diagnostics
from residual_fusion_v2.core import diagnostic_decision
from final_compare_v1.metrics import caption_metrics


class DiagnosticProofs(unittest.TestCase):
    def fixture(self,root):
        design=dict(config=dict(diagnostic_samples=2,diagnostic_updates=2,diagnostic_wer_gate=.1),
            train_ids=[1,2],smoke=False,tokenizer_ids=dict(bos=1,eos=2,pad=2),diagnostic_conditions=['bad'])
        d=root/'text_diagnostics'; d.mkdir(); rows=[]; scores={}
        for label in ('D0_precompression','D1_T32_clean'):
            sub=d/label; sub.mkdir(); hist=[]
            for i in (1,2):
                row=dict(label=label,image_id=i,condition='clean',source_split='train',source_text_at_receiver=False,
                    covered_reference_evaluation_only='a cat',received_caption='a dog',generated_token_ids=[1,3,2],
                    generated_tokens=2,stopped_by_eos=True,stopped_by_capacity=False,termination='predicted_EOS',**caption_metrics('a cat','a dog'))
                rows.append(row); hist.append(dict(step=i,image_id=i,schedule_seed=i,text_ce=1.))
            scores[label]=.5
            (sub/'history.jsonl').write_text('\n'.join(json.dumps(r) for r in hist)+'\n')
            (sub/'COMPLETE.json').write_text(json.dumps(dict(updates=2,train_ids=[1,2],checkpoint_roundtrip_verified=True,
                changed_names=['embedding.weight'],observed_gradient_names=['embedding.weight'],actual_covered_wer=.5,
                initial_parameter_sha256={'embedding.weight':'same'})))
        for i in (1,2): rows.append(dict(rows[0],label='D2_T32_FSO',image_id=i,condition='bad'))
        (d/'samples.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        gate=diagnostic_decision(scores,.1); (d/'gate.json').write_text(json.dumps(gate))
        return design
    def test_diagnostic_stop_is_verified_not_failure_masked(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); design=self.fixture(root); r=verify_diagnostics(root,design); self.assertFalse(r['proceed'])
    def test_false_score_refused(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); design=self.fixture(root); p=root/'text_diagnostics/samples.jsonl'
            rows=[json.loads(s) for s in p.read_text().splitlines()]; rows[0]['normalized_wer']=0.; p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            with self.assertRaises(ValueError): verify_diagnostics(root,design)
    def test_false_eos_refused(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); design=self.fixture(root); p=root/'text_diagnostics/samples.jsonl'
            rows=[json.loads(s) for s in p.read_text().splitlines()]; rows[0]['stopped_by_eos']=False; p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            with self.assertRaises(ValueError): verify_diagnostics(root,design)
    def test_quality_gate_bypass_refused(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); design=self.fixture(root); p=root/'text_diagnostics/gate.json'; gate=json.loads(p.read_text()); gate['proceed']=True; p.write_text(json.dumps(gate))
            with self.assertRaises(ValueError): verify_diagnostics(root,design)
    def test_missing_noisy_diagnostic_refused(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); design=self.fixture(root); p=root/'text_diagnostics/samples.jsonl'; rows=p.read_text().splitlines(); p.write_text('\n'.join(rows[:-1])+'\n')
            with self.assertRaises(ValueError): verify_diagnostics(root,design)
    def test_wrong_initialization_refused(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); design=self.fixture(root); p=root/'text_diagnostics/D1_T32_clean/COMPLETE.json'; rec=json.loads(p.read_text()); rec['initial_parameter_sha256']={}; p.write_text(json.dumps(rec))
            with self.assertRaises(ValueError): verify_diagnostics(root,design)


if __name__=='__main__': unittest.main(verbosity=2)
