"""Synthetic record audits only; never presented as neural-model results."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
EXT=Path(__file__).resolve().parents[1]; PROJECT=EXT.parent; sys.path.insert(0,str(EXT))
from text_recovery_v3 import VERSION,ARMS
from text_recovery_v3.core import load,row,summarize,geometry,gate
from text_recovery_v3.report import verify,seal,render
from text_recovery_v3.collect import collect


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False))
def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text('\n'.join(json.dumps(v,ensure_ascii=False,allow_nan=False) for v in rows)+'\n')


class RecordContracts(unittest.TestCase):
    def fixture(self,root,passed):
        c=load(); ids=list(range(1,9)); rows=[]
        items=[dict(image_id=i,annotation_id=i,split='train',text='a cat '+str(i),covered_text='a cat '+str(i),
            covered_tokens=5,truncated=False,source_sha256='x'+str(i),source_utf8_bytes=7) for i in ids]
        for control in ('native_masked','native_all128','native_pad_zero'):
            for item in items:
                rows.append(row(item,'native',control,dict(text=item['text'] if passed else '',stopped_by_capacity=False,receiver_failure=False)))
        write_rows(root/'native_controls.jsonl',rows)
        groups={k:summarize([v for v in rows if v['control']==k]) for k in ('native_masked','native_all128','native_pad_zero')}
        save(root/'native_gate.json',dict(passed=passed,groups=groups,quality_limit=.2,smoke_quality_bypass=False))
        save(root/'bart_interface_audit.json',dict(passed=True,incremental_max_logit_error=0.,cached_source_max_logit_error=0.,tolerance=.0002))
        save(root/'frozen_verification.json',dict(passed=True))
        save(root/'design.json',dict(config=c,diagnostic_caption_view=dict(image_ids=ids),arms=list(ARMS)))
        save(root/'run_identity.json',dict(version=VERSION,status='screen_complete',smoke=False,expand=False,arms=list(ARMS)))
        statuses={}
        for arm in ARMS:
            if not passed: statuses[arm]=dict(status='native_stop',expanded=False,diagnostic_passed=False); continue
            output=[]
            for control in ('matched','zero','shuffled'):
                for item in items:
                    output.append(row(item,arm,control,dict(text=item['text'] if control=='matched' else '',stopped_by_capacity=False,receiver_failure=False),geometry=geometry(arm)))
            folder=root/arm; write_rows(folder/'diagnostic_controls.jsonl',output)
            ag={k:summarize([v for v in output if v['control']==k]) for k in ('matched','zero','shuffled')}
            decision=gate(ag,c); save(folder/'diagnostic_gate.json',dict(decision,groups=ag))
            statuses[arm]=dict(status='diagnostic_passed',diagnostic_passed=True,expanded=False)
            train=folder/'diagnostic'
            write_rows(train/'history.jsonl',[dict(step=i,total_steps=1200,optimizer_updates=i//2) for i in range(1,1201)])
            save(train/'TRAIN_COMPLETE.json',dict(steps=1200,optimizer_updates=600))
            save(train/'learning_audit.json',dict(passed=True,frozen_unchanged=True,groups=dict(source=dict(tensors=1,nonzero_gradient=True,changed=True))))
            save(train/'reload_audit.json',dict(passed=True))
            save(train/'forward_consistency.json',dict(passed=True,errors=dict(source=0.,receiver=0.)))
        save(root/'arm_status.json',statuses)
    def location(self):
        folder=PROJECT/'releases'; folder.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix='text_v3_record_test_',dir=str(folder))
    def test_native_negative_is_valid_not_quality_success(self):
        with self.location() as d:
            root=Path(d); self.fixture(root,False); render(root); seal(root)
            self.assertTrue(verify(root)['passed']); self.assertFalse(any(verify(root)['quality_success'].values()))
            self.assertTrue((root/'SCREEN_COMPLETE.json').exists()); self.assertFalse((root/'COMPLETE.json').exists())
    def test_full_diagnostic_record_matrix(self):
        with self.location() as d:
            root=Path(d); self.fixture(root,True); self.assertEqual(verify(root,False)['diagnostic_arms_checked'],5)
            render(root); seal(root); self.assertTrue(verify(root)['passed'])
    def test_metric_drift_rejected(self):
        with self.location() as d:
            root=Path(d); self.fixture(root,True); path=root/'A2_ramp/diagnostic_gate.json'; x=json.loads(path.read_text()); x['matched_wer']=.3; save(path,x)
            with self.assertRaises(ValueError): verify(root,False)
    def test_unpassed_training_missing_marker_rejected(self):
        with self.location() as d:
            root=Path(d); self.fixture(root,True); path=root/'A3_lora/diagnostic/TRAIN_COMPLETE.json'; x=json.loads(path.read_text()); x['steps']=8; save(path,x)
            with self.assertRaises(ValueError): verify(root,False)
    def test_postseal_hash_drift_rejected(self):
        with self.location() as d:
            root=Path(d); self.fixture(root,False); render(root); seal(root); save(root/'native_gate.json',dict(passed=True))
            with self.assertRaises(ValueError): verify(root)
    def test_failure_cannot_be_collected(self):
        with self.location() as d:
            root=Path(d); self.fixture(root,False); render(root); seal(root); save(root/'FAILED.json',dict(message='failure'))
            with self.assertRaises(ValueError): verify(root)
    def test_archive_reverify_without_weights(self):
        import tarfile
        with self.location() as d:
            root=Path(d)/'records'; root.mkdir(); self.fixture(root,True); render(root); seal(root)
            target=Path(d)/'records.tar.gz'; collect(root,target)
            restored=Path(d)/'restored'; restored.mkdir()
            with tarfile.open(target,'r:gz') as tar: tar.extractall(str(restored))
            self.assertTrue(verify(restored/'v10_text_recovery_v3_records')['passed'])
            with self.assertRaises(FileExistsError): collect(root,target)


if __name__=='__main__': unittest.main(verbosity=2)
