"""Negative artifact/generation/gate tests; fixtures are explicitly synthetic."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from text_jscc_v1 import VERSION,ARMS
from text_jscc_v1.core import load,metrics,digest,overfit_gate
from text_jscc_v1.report import verify,groups_for,report
from text_jscc_v1.collect import collect
from fso_fluctuating_v2.core import load as channel_config
PREFIXES=('encoder.','compressor.','tx.','rx.','receiver.','language.decoder.')


def fixture(root):
    c=load(); c.update(train_samples=8,val_samples=2,diagnostic_samples=2,diagnostic_updates=8)
    d=dict(version=VERSION,config=c,smoke=False,split='validation',train_ids=list(range(1,9)),validation_ids=[101,102],
        payload_complex_symbols=384,raw_utf8_wire_bytes=0,analog_repeats=1,arms=list(ARMS),
        tokenizer_ids=dict(bos=1,eos=2,pad=2),channel_config=dict(primary_conditions=['x']),limits='SYNTHETIC TEST, not actual performance')
    def record(arm,i,condition):
        return dict(arm=arm,image_id=i,condition=condition,source_split='train',
            source_caption_evaluation_only='a cat',covered_reference_evaluation_only='a cat',
            source_sha256=hashlib.sha256(b'a cat').hexdigest(),source_text_at_receiver=False,source_length_at_receiver=False,
            raw_utf8_wire_bytes=0,input_truncated=False,received_caption='a dog',generated_token_ids=[1,3,2],generated_tokens=2,
            stopped_by_eos=True,stopped_by_capacity=False,termination='predicted_EOS',**metrics('a cat','a cat','a dog'))
    def lines(path,rows): path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    (root/'design.json').write_text(json.dumps(d)); (root/'run_identity.json').write_text(json.dumps(dict(version=VERSION,status='overfit_stop')))
    old=root/'legacy_diagnostics'; old.mkdir(); rows=[]
    for label in ('D0_precompression','D1_compressed_clean'):
        sub=old/label; sub.mkdir(); rows.extend(record(label,i,cond) for i in (1,2) for cond in ('matched','zero','shuffled'))
        lines(sub/'history.jsonl',[dict(step=i+1) for i in range(8)])
        (sub/'proof.json').write_text(json.dumps(dict(gradient_names=['w'],changed_names=['w'],checkpoint_reload_verified=True,initial_sha256={'w':'same'})))
    lines(old/'samples.jsonl',rows)
    sub=root/'native_overfit'; sub.mkdir(); rows=[record('native_overfit',i,cond) for i in (1,2) for cond in ('matched','zero','shuffled')]
    lines(sub/'samples.jsonl',rows); gate=overfit_gate(groups_for(rows),c); gate['smoke_bypasses_quality_only']=False
    (sub/'gate.json').write_text(json.dumps(gate)); lines(sub/'history.jsonl',[dict(items=[{'image_id':1}]) for _ in range(8)])
    (sub/'proof.json').write_text(json.dumps(dict(optimizer_steps=8,checkpoint_reload_verified=True,gradient_energy={'source':1.,'compression':1.,'tx':1.,'rx':1.,'decoder':1.},gradient_names=[p+'w' for p in PREFIXES],parameter_changes={p+'w':.1 for p in PREFIXES},checkpoint_sha256='synthetic-no-weight')))
    (root/'frozen_verification.json').write_text(json.dumps(dict(old_model_unchanged=True,B4_unchanged=True,old_gradients_none=True,no_test_source_used=True)))
    report(root); end=dict(version=VERSION,status='overfit_stop',checkpoint_sha256={},bank_sha256={},result_sha256={str(f.relative_to(root)):digest(f) for f in root.rglob('*') if f.is_file()})
    (root/'DIAGNOSTIC_COMPLETE.json').write_text(json.dumps(end)); return d


def complete_fixture(root):
    d=fixture(root); d['config'].update(clean_epochs=1,continue_epochs=1)
    c=channel_config(); c['conditions']=c['conditions'][:1]; c['primary_conditions']=[c['conditions'][0]['id']]
    d.update(channel_config=c,validation_source_sha256={str(i):hashlib.sha256(b'a cat').hexdigest() for i in d['validation_ids']},resource=dict(slot_samples=3000,fs_hz=64e9))
    def write(path,value): path.write_text(json.dumps(value))
    def lines(path,rows): path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    def change(r,correct):
        r=dict(r); r['received_caption']='a cat' if correct else 'a dog'
        r.update(metrics('a cat','a cat',r['received_caption'])); return r
    native=[json.loads(s) for s in (root/'native_overfit/samples.jsonl').read_text().splitlines()]
    native=[change(r,r['condition']=='matched') for r in native]; lines(root/'native_overfit/samples.jsonl',native)
    gate=overfit_gate(groups_for(native),d['config']); gate['smoke_bypasses_quality_only']=False; write(root/'native_overfit/gate.json',gate)
    clean=[]
    for i in d['validation_ids']:
        for cond in ('matched','zero','shuffled'):
            r=change(native[0],cond=='matched'); r.update(image_id=i,source_split='validation',condition=cond); clean.append(r)
    lines(root/'clean_validation_samples.jsonl',clean)
    write(root/'clean_validation_gate.json',dict(passed=True,smoke_bypasses_quality_only=False,groups=groups_for(clean)))
    for label in ('T0_clean','T1_clean_continue','T2_fso'):
        sub=root/('train_'+label); sub.mkdir()
        lines(sub/'history.jsonl',[dict(items=[{'image_id':i} for i in range(start,start+4)]) for start in (1,5)])
        write(sub/'proof.json',dict(optimizer_steps=2,checkpoint_reload_verified=True,gradient_energy={'source':1.,'compression':1.,'tx':1.,'rx':1.,'decoder':1.},gradient_names=[p+'w' for p in PREFIXES],parameter_changes={p+'w':.1 for p in PREFIXES},checkpoint_sha256='synthetic-no-weight',initial_sha256={'w':'same'},trainable_parameters=1))
    bank=root/'short_calibration'; bank.mkdir(); records=[]; cid=c['conditions'][0]['id']
    for partition in ('train','holdout'):
        for _ in range(4): records.append(dict(index=len(records),partition=partition))
    fidelity=[dict(condition=cid,calibration_index=i,available_fit=True,available_fresh=True,actual_evm2=.1,predicted_evm2=.1,source_image_id=1,source_split='train') for i in range(4,8)]
    write(bank/'bank.json',dict(payload_complex_symbols=384,public_waveforms_only=True,records=records,fidelity={'rows':fidelity}))
    with (bank/'bank.npz').open('wb') as f: np.savez(f,coefficients=np.zeros((8,11),complex),residuals=np.zeros((8,384),complex))
    for label in ('T0_clean','T1_clean_continue','T2_fso'): write(root/('fidelity_'+label+'.json'),dict(rows=fidelity))
    rows=[]; events=[]
    for arm in ARMS:
        for i in d['validation_ids']:
            for cond in ('clean',cid):
                r=change(native[0],True); key='{}__{}__{}'.format(arm,i,cond)
                r.update(arm=arm,image_id=i,source_split='validation',condition=cond,frame_key=key,
                    receiver_available=True,payload_complex_symbols=384,analog_repeats=1,pre_fec_ber=None,post_fec_ber=None,
                    scheduled_seconds=2120/64e9,launch_signal_energy_j=2e-12,semantic_chunked_clip_cosine=1.)
                rows.append(r)
                if cond!='clean':
                    for purpose,count in (('probe',100),('features',2000)):
                        events.append(dict(frame_key=key,purpose=purpose,data_symbols=0 if purpose=='probe' else 384,awg_samples=count,guard_samples=10,launch_signal_energy_j=1e-12))
    lines(root/'samples.jsonl',rows); lines(root/'events.jsonl',events)
    write(root/'design.json',d); write(root/'run_identity.json',dict(version=VERSION,status='complete')); report(root)
    write(root/'COMPLETE.json',dict(version=VERSION,status='complete',checkpoint_sha256={},bank_sha256={},result_sha256={str(f.relative_to(root)):digest(f) for f in root.rglob('*') if f.is_file() and f.name!='COMPLETE.json'}))
    return d


class Records(unittest.TestCase):
    def temporary(self):
        folder=EXT.parent/'releases'; folder.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix='text_records_test_',dir=str(folder))
    def test_diagnostic_stop_and_light_collection(self):
        with self.temporary() as t:
            root=Path(t)/'run'; root.mkdir(); fixture(root); self.assertEqual(verify(root),[]); collect(root,Path(t)/'records.tar.gz')
    def mutation(self,field,value):
        with self.temporary() as t:
            root=Path(t)/'run'; root.mkdir(); fixture(root)
            p=root/'native_overfit/samples.jsonl'; rows=[json.loads(s) for s in p.read_text().splitlines()]; rows[0][field]=value
            p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            with self.assertRaises(ValueError): verify(root)
    def test_false_metric_refused(self): self.mutation('covered_wer',0.)
    def test_false_eos_refused(self): self.mutation('stopped_by_eos',False)
    def test_source_length_oracle_refused(self): self.mutation('source_length_at_receiver',True)
    def test_raw_utf8_bypass_refused(self): self.mutation('raw_utf8_wire_bytes',3)
    def test_empty_matrix_refused(self):
        with self.temporary() as t:
            root=Path(t)/'run'; root.mkdir(); fixture(root); (root/'native_overfit/samples.jsonl').write_text('')
            with self.assertRaises(ValueError): verify(root)
    def test_false_gate_refused(self):
        with self.temporary() as t:
            root=Path(t)/'run'; root.mkdir(); fixture(root); p=root/'native_overfit/gate.json'; gate=json.loads(p.read_text()); gate['passed']=True; p.write_text(json.dumps(gate))
            with self.assertRaises(ValueError): verify(root)
    def test_stop_cannot_be_marked_complete(self):
        with self.temporary() as t:
            root=Path(t)/'run'; root.mkdir(); fixture(root); (root/'COMPLETE.json').write_text('{}')
            with self.assertRaises(ValueError): verify(root)
    def test_result_tamper_refused(self):
        with self.temporary() as t:
            root=Path(t)/'run'; root.mkdir(); fixture(root); (root/'REPORT.md').write_text('changed')
            with self.assertRaises(ValueError): verify(root)
    def test_complete_matrix_and_light_collection(self):
        with self.temporary() as t:
            root=Path(t)/'run'; root.mkdir(); complete_fixture(root)
            self.assertEqual(len(verify(root)),16); collect(root,Path(t)/'complete.tar.gz')
    def complete_mutation(self,key,value):
        with self.temporary() as t:
            root=Path(t)/'run'; root.mkdir(); complete_fixture(root)
            p=root/'samples.jsonl'; rows=[json.loads(s) for s in p.read_text().splitlines()]
            next(r for r in rows if r['condition']!='clean')[key]=value; p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            with self.assertRaises(ValueError): verify(root)
    def test_free_time_refused(self): self.complete_mutation('scheduled_seconds',0.)
    def test_free_energy_refused(self): self.complete_mutation('launch_signal_energy_j',0.)
    def test_legacy_budget_refused(self): self.complete_mutation('payload_complex_symbols',2304)
    def test_unpaired_training_refused(self):
        with self.temporary() as t:
            root=Path(t)/'run'; root.mkdir(); complete_fixture(root)
            p=root/'train_T2_fso/proof.json'; proof=json.loads(p.read_text()); proof['initial_sha256']={'w':'different'}; p.write_text(json.dumps(proof))
            with self.assertRaises(ValueError): verify(root)


if __name__=='__main__': unittest.main(verbosity=2)
