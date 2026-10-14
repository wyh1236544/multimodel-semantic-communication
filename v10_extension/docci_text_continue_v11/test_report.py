"""Synthetic complete/negative matrices, never neural performance evidence."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
from docci_text_continue_v11.core import PROJECT,atomic_json,digest,load,choose_indices,phase_lr,schedule
from docci_text_continue_v11.metrics import metrics
from docci_text_continue_v11.report import evaluation,diagnostic,history,finish
from docci_reconstruction_v10.core import load as old_load
from docci_reconstruction_v10.channel import resource


class Reports(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=str(PROJECT),prefix='v11_report_test_'); self.root=Path(self.temp.name)
        (self.root/'frames').mkdir(); self.c=old_load(); self.ids=['qual_dev_a','qual_dev_b']; self.rows=[]; self.receipts={}
        self.declared=dict(example_ids=self.ids,stage='g20480',cohort='validation')
        matrix=dict(arm='T1_chunks',branch='text',stage='g20480',cohort='validation',example_ids=self.ids,
            controls=['matched','zero','shuffled'],conditions=['clean'],resources=resource(self.c,'T1_chunks'),source_count_or_mask_at_RX=False,Common_used=False)
        atomic_json(self.root/'matrix.json',matrix)
        for i,eid in enumerate(self.ids):
            text=('cat sits','dog runs')[i]; sha=hashlib.sha256(text.encode()).hexdigest(); self.receipts[eid]=dict(text_sha256=sha,source_tokens=4)
            y=np.full(18432,1 if i==0 else -1,np.float32); file='frames/'+eid+'.npz'
            with (self.root/file).open('xb') as f: np.savez_compressed(f,observed=y,available=np.asarray(True),pilot_snr_db=np.asarray(40.,np.float32),reliability=np.asarray(1.,np.float32))
            h=hashlib.sha256(y.tobytes()).hexdigest()
            for control in ('matched','zero','shuffled'):
                caption=text if control=='matched' else 'wrong text'; stats=dict(resources=matrix['resources'],payload_complex_symbols=9216,analog_repeats=1,raw_utf8_wire_bytes=0,
                    receiver_available=True,pilot_snr_db=40.,analog_reliability=1.,clean_sentinel_not_measured_snr=True,pre_fec_ber=None,post_fec_ber=None)
                self.rows.append(dict(example_id=eid,control=control,stage='g20480',cohort='validation',condition='clean',branch='text',arm='T1_chunks',noise_seed=None,
                    official_split='qual_dev',source_text=text,source_text_sha256=sha,source_tokens=4,input_truncated=False,received_caption=caption,
                    **metrics(text,caption),chunk_generated_token_ids=[[2,0,3,4,2]]+[[2,0,2]]*8,generated_token_ids=[3,4],stopped_by_capacity=False,receiver_failure=False,
                    frame_file=file,frame_sha256=digest(self.root/file),received_sha256=h,tx_wire_sha256=h,stats=stats,
                    donor_example_id=self.ids[1-i] if control=='shuffled' else None,
                    donor_received_sha256=hashlib.sha256(np.full(18432,-1 if i==0 else 1,np.float32).tobytes()).hexdigest() if control=='shuffled' else None))
    def tearDown(self): self.temp.cleanup()
    def write(self):
        with (self.root/'rows.jsonl').open('w') as f:
            for r in self.rows: f.write(json.dumps(r)+'\n')
        atomic_json(self.root/'EVAL_COMPLETE.json',dict(rows=len(self.rows),rows_sha256=digest(self.root/'rows.jsonl'),matrix_sha256=digest(self.root/'matrix.json')))
    def verify(self): self.write(); return evaluation(self.root,self.c,self.declared,self.receipts)
    def test_full_positive_matrix(self): self.assertEqual(len(self.verify()),6)
    def test_recomputed_metric_failure(self):
        self.rows[0]['normalized_wer']=.1
        with self.assertRaises(ValueError): self.verify()
    def test_duplicate_or_missing_rejected(self):
        self.rows[-1]=copy.deepcopy(self.rows[0])
        with self.assertRaises(ValueError): self.verify()
    def test_no_source_count_at_RX(self):
        m=json.loads((self.root/'matrix.json').read_text()); m['source_count_or_mask_at_RX']=True; atomic_json(self.root/'matrix.json',m)
        with self.assertRaises(ValueError): self.verify()
    def test_wrong_shuffled_donor(self):
        self.rows[2]['donor_example_id']=self.rows[2]['example_id']
        with self.assertRaises(ValueError): self.verify()
    def test_test_split_rejected(self):
        self.rows[0]['official_split']='test'
        with self.assertRaises(ValueError): self.verify()
    def test_changed_channel_stats_rejected(self):
        self.rows[0]['stats']['pilot_snr_db']=20
        with self.assertRaises(ValueError): self.verify()
    def test_EOS_declaration_rejected(self):
        self.rows[0]['stopped_by_capacity']=True
        with self.assertRaises(ValueError): self.verify()
    def test_diagnostic_counts_and_privileged_scope(self):
        slot=dict(slot=0,gold_content_BPE=2,TF_content_CE=1.,native_TF_content_CE=.1,TF_content_accuracy=.5,
            TF_pEOS_before_gold_end=.1,TF_pEOS_at_gold_end=.5,TF_argmax_early_EOS=0,
            common_content_positions=2,TF_CE_common=1.,free_prefix_CE_common=2.)
        slots=[slot]+[dict(slot=j,gold_content_BPE=0,TF_content_CE=None,native_TF_content_CE=None,TF_content_accuracy=None,
            TF_pEOS_before_gold_end=None,TF_pEOS_at_gold_end=.9,TF_argmax_early_EOS=0,common_content_positions=0,TF_CE_common=None,free_prefix_CE_common=None) for j in range(1,9)]
        row=dict(example_id=self.ids[0],source_text_sha256=self.receipts[self.ids[0]]['text_sha256'],source_tokens=4,
            actual_wire_sha256=self.rows[0]['tx_wire_sha256'],diagnostic_only=True,gold_prefix_is_privileged_NOT_actual_RX=True,
            native_source_memory_NOT_transmitted=True,slot_count=9,TF_total_CE=.5,TF_content_CE=1.,native_TF_content_CE=.1,
            TF_content_accuracy=.5,common_content_positions=2,TF_CE_common=1.,free_prefix_CE_common=2.,slots=slots)
        folder=self.root/'prefix'; folder.mkdir()
        def seal():
            with (folder/'rows.jsonl').open('w') as f: f.write(json.dumps(row)+'\n')
            atomic_json(folder/'DIAGNOSTIC_COMPLETE.json',dict(rows=1,rows_sha256=digest(folder/'rows.jsonl')))
        seal(); diagnostic(folder,dict(example_ids=[self.ids[0]]),self.receipts,{self.ids[0]:self.rows[0]})
        row['gold_prefix_is_privileged_NOT_actual_RX']=False; seal()
        with self.assertRaises(ValueError): diagnostic(folder,dict(example_ids=[self.ids[0]]),self.receipts,{self.ids[0]:self.rows[0]})
    def test_global_history_and_reset_cursor_rejected(self):
        folder=self.root/'history'; folder.mkdir(); phase=load(); planned=schedule(phase)
        ids=['train_'+str(i) for i in range(2048)]; hist=[]; seen=set()
        for i in range(planned['additional_updates']):
            g=10001+i; batch=[ids[j] for j in choose_indices(2048,2,g-1,0,self.c['seed'],1)]; seen.update(batch)
            hist.append(dict(local_update=i+1,global_update=g,batch_example_ids=batch,unique_examples_seen=len(seen),
                cumulative_presentations=g*2,phase_presentations=(i+1)*2,loss=1.,lr=phase_lr(phase,i,planned['additional_updates']),smoke=False))
        atomic_json(folder/'trainability.json',dict(parameters=25618432,frozen_BART=True,loss='pure cross entropy',optimizer='NEW AdamW',parent_optimizer_restored=False,
            updates=len(hist),batch=2,parent_global_update=10000,smoke=False))
        atomic_json(folder/'learning_audit.json',dict(passed=True,updates=len(hist),unique_examples_seen=len(seen),
            groups={k:dict(nonzero_gradient=True,changed=True,parameters=1) for k in ('compressor','tx_normalization','receiver','memory')}))
        atomic_json(folder/'TRAIN_COMPLETE.json',dict(updates=len(hist),global_update=20480,loss='pure CE',weights_only_warm_start=True))
        def write():
            with (folder/'history.jsonl').open('w') as f:
                for r in hist: f.write(json.dumps(r)+'\n')
        write(); self.assertEqual(len(history(folder,self.c,phase,planned,ids)),10480)
        hist[0]['batch_example_ids']=[ids[j] for j in choose_indices(2048,2,0,0,self.c['seed'],1)]; write()
        with self.assertRaises(ValueError): history(folder,self.c,phase,planned,ids)
    def test_finish_is_deterministic_and_partial_retry_preserves_source(self):
        self.verify(); ref=self.root/'parent_reference/clean_validation'; ref.mkdir(parents=True)
        with (ref/'rows.jsonl').open('w') as f:
            for row in self.rows:
                if row['control']=='matched': f.write(json.dumps(row)+'\n')
        ident=dict(schedule=dict(target_updates=20480),config=self.c,probes=[10000,20480],diagnostics=[dict(example_ids=self.ids)])
        atomic_json(self.root/'run_identity.json',ident)
        result=dict(identity=ident,evaluations={k:self.rows for k in ('g10000_validation','g20480_validation','g20480_train')},diagnostics={})
        out=finish(self.root,result); original=(self.root/'actual_text_recovery.html').read_bytes()
        self.assertEqual(finish(self.root,result),out); self.assertEqual((self.root/'actual_text_recovery.html').read_bytes(),original)
        changed=read_json(self.root/'summary.json'); changed['legacy_rule_unchanged']=False; atomic_json(self.root/'summary.json',changed)
        with self.assertRaises(FileExistsError): finish(self.root,result)
        self.assertEqual((self.root/'actual_text_recovery.html').read_bytes(),original)


def read_json(path): return json.loads(path.read_text())


if __name__=='__main__': unittest.main(verbosity=2)
