"""Production-shaped positive AND mutation-negative record tests, no GPU claim."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
import numpy as np
from docci_reconstruction_v10.core import load,digest,metrics,geometry,gate,plan,choose_indices
from docci_reconstruction_v10.channel import resource
from docci_reconstruction_v10.report import evaluation,text_row,verify,finish


def write(root,name,value): (root/name).write_text(json.dumps(value,allow_nan=False))


def rewrite(root,rows):
    (root/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(root,'EVAL_COMPLETE.json',dict(rows=len(rows),rows_sha256=digest(root/'rows.jsonl'),matrix_sha256=digest(root/'matrix.json')))


def fixture(root,c=None,ids=None,stage='initial',cohort='validation'):
    c=c or load(); ids=ids or ['qual_a','qual_b']; res=resource(c,'T1_chunks'); (root/'frames').mkdir(); rows=[]
    wire=np.ones(18432,np.float32); rx=hashlib.sha256(wire.tobytes()).hexdigest()
    for eid in ids:
        with (root/'frames'/(eid+'.npz')).open('wb') as f: np.savez(f,observed=wire,available=np.asarray(True),pilot_snr_db=np.asarray(40,np.float32),reliability=np.asarray(1,np.float32))
    for i,eid in enumerate(ids):
        for ctrl in ('matched','zero','shuffled'):
            rows.append(dict(arm='T1_chunks',branch='text',stage=stage,cohort=cohort,example_id=eid,official_split='train' if cohort=='diagnostic_train' else 'qual_dev',condition='clean',control=ctrl,
                frame_file='frames/'+eid+'.npz',frame_sha256=digest(root/'frames'/(eid+'.npz')),received_sha256=rx,tx_wire_sha256=rx,noise_seed=i,
                donor_example_id=ids[(i+1)%len(ids)] if ctrl=='shuffled' else None,donor_received_sha256=rx if ctrl=='shuffled' else None,
                source_text='the cat',source_text_sha256=hashlib.sha256(b'the cat').hexdigest(),source_tokens=4,input_truncated=False,
                received_caption='the cat',generated_token_ids=[5,6],chunk_generated_token_ids=[[2,0,5,6,2]]+[[2,0,2]]*8,stopped_by_capacity=False,receiver_failure=False,
                stats=dict(receiver_available=True,pilot_snr_db=40.,analog_reliability=1.,resources=res,payload_complex_symbols=9216,analog_repeats=1,
                    raw_utf8_wire_bytes=0,clean_sentinel_not_measured_snr=True,pre_fec_ber=None,post_fec_ber=None),**metrics('the cat','the cat','the cat')))
    matrix=dict(arm='T1_chunks',branch='text',stage=stage,cohort=cohort,example_ids=ids,controls=['matched','zero','shuffled'],conditions=['clean'],resources=res,
        source_count_or_mask_at_RX=False,Common_used=False,failures_retained=True)
    write(root,'matrix.json',matrix); rewrite(root,rows); return c,ids,rows


class Reporting(unittest.TestCase):
    def test_positive_full_chunks_and_missing_or_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); c,ids,rows=fixture(root); self.assertEqual(len(evaluation(root,ids,c,'T1_chunks','initial','validation')),6)
            rewrite(root,rows[:-1])
            with self.assertRaises(ValueError): evaluation(root,ids,c,'T1_chunks','initial','validation')
            rewrite(root,rows+[rows[0]])
            with self.assertRaises(ValueError): evaluation(root,ids,c,'T1_chunks','initial','validation')
    def test_hidden_source_count_and_bad_EOS_fail(self):
        with tempfile.TemporaryDirectory() as d:
            c,ids,rows=fixture(Path(d)); bad=copy.deepcopy(rows[0]); bad['chunk_generated_token_ids']=bad['chunk_generated_token_ids'][:1]
            with self.assertRaises(ValueError): text_row(bad,c,'T1_chunks')
            bad=copy.deepcopy(rows[0]); bad['stopped_by_capacity']=True
            with self.assertRaises(ValueError): text_row(bad,c,'T1_chunks')
    def test_omitted_resource_full_source_metrics_and_test_leakage_fail(self):
        for key,value in (('official_split','test'),('normalized_wer',.1),('source_tokens',513)):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d); c,ids,rows=fixture(root); rows[0][key]=value; rewrite(root,rows)
                with self.assertRaises(ValueError): evaluation(root,ids,c,'T1_chunks','initial','validation')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); c,ids,rows=fixture(root); rows[0]['stats']['payload_complex_symbols']=1024; rewrite(root,rows)
            with self.assertRaises(ValueError): evaluation(root,ids,c,'T1_chunks','initial','validation')
    def test_public_RX_file_and_wrong_donor_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); c,ids,rows=fixture(root); rows[2]['donor_example_id']='qual_a'; rewrite(root,rows)
            with self.assertRaises(ValueError): evaluation(root,ids,c,'T1_chunks','initial','validation')
            file=root/'frames/qual_a.npz'
            with file.open('wb') as f: np.savez(f,observed=np.ones(18432,np.float32),available=True,pilot_snr_db=40.,reliability=1.,source_count=1)
            for row in rows:
                if row['example_id']=='qual_a': row['frame_sha256']=digest(file)
            rewrite(root,rows)
            with self.assertRaises(ValueError): evaluation(root,ids,c,'T1_chunks','initial','validation')
    def test_complete_negative_screen_generates_HTML_without_fake_pass(self):
        with tempfile.TemporaryDirectory(dir=str(EXT.parent),prefix='v10_report_test_') as d:
            root=Path(d); c=load(); c.update(train_samples=32,val_samples=2,diagnostic_samples=2,save_images=2,text_diagnostic_updates=8)
            write(root,'config.json',c); ti=['train_'+str(i) for i in range(32)]; vi=['qual_a','qual_b']; di=ti[:2]
            files=dict(config_json='c4022c931216afcb778991477332e5312aca8be478027ee3a8bf34b7004c44e2',pytorch_model_bin='6e4fdb5e04114696ec4f7cc0cb5fc2a22a826f1c9eea234ca550b5390341a2fd',
                merges_txt='1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',vocab_json='9e7f63c2d15d666b52e21d250d2e513b87c9b713cfa6987a82ed89e5e6e50655')
            files={k.rsplit('_',1)[0]+'.'+k.rsplit('_',1)[1]:v for k,v in files.items()}
            identity=dict(schema='docci-reconstruction-v10',arm='T1_chunks',branch='text',mode='diagnostic',config=c,schedule=plan(c,'diagnostic','text'),
                geometry=geometry(c,'T1_chunks'),gpu_physical=1,Common_used=False,diffusion_used=False,opposite_modality_loaded=False,raw_UTF8_transmitted=False,
                source_mask_or_length_at_RX=False,backbone_signature_before='same',backbone_signature_after='same',backbone=dict(files=files),
                initialization=dict(checkpoint=None),source_snapshot_sha256={},evaluations=[],stages=['train_clean'],status='QUALITY_STOP')
            folder=Path(__file__).parent; snap=root/'source_snapshot'; snap.mkdir(); manifest=folder/'MANIFEST.json'
            source_manifest=json.loads(manifest.read_text()) if manifest.exists() else dict(schema='docci-reconstruction-v10-source',files={f.name:digest(f) for f in folder.iterdir() if f.is_file() and f.suffix in ('.py','.json','.md')})
            for name in source_manifest['files']: shutil.copyfile(folder/name,snap/name)
            if manifest.exists(): shutil.copyfile(manifest,snap/'MANIFEST.json')
            else: write(snap,'MANIFEST.json',source_manifest)
            identity['source_snapshot_sha256']=source_manifest['files']
            sha=hashlib.sha256(b'the cat').hexdigest()
            records=lambda ids,split:[dict(example_id=eid,official_split=split,text_sha256=sha,source_tokens=4,fixed_chunks=9,no_truncation=True) for eid in ids]
            write(root,'data_identity.json',dict(train=dict(records=records(ti,'train')),validation=dict(records=records(vi,'qual_dev')),diagnostic_ids=di,source=dict(official_counts=dict(train=9647,test=5000,qual_dev=100,qual_test=100))))
            native_rows=None
            for name,ids,stage,cohort in (('initial_validation',vi,'initial','validation'),('clean_train',di,'clean','diagnostic_train'),('clean_validation',vi,'clean','validation')):
                f=root/name; f.mkdir(); _,_,rows=fixture(f,c,ids,stage,cohort); identity['evaluations'].append(name); native_rows=rows
                if name=='clean_train': identity['train_gate']=gate(rows,'text',c,True)
                if name=='clean_validation': identity['validation_gate']=gate(rows,'text',c,False)
            for name,ids,split in (('native_train',di,'train'),('native_validation',vi,'qual_dev')):
                f=root/name; f.mkdir(); rows=[]
                for eid in ids:
                    row=copy.deepcopy(native_rows[0]); row.update(example_id=eid,official_split=split,scope='ORACLE no transmission',generated_token_ids=[2,0,5,6,2]); rows.append(row)
                (f/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows)); write(f,'NATIVE_COMPLETE.json',dict(rows=2,rows_sha256=digest(f/'rows.jsonl'),no_transmission=True))
            f=root/'native_chunk_validation'; f.mkdir(); rows=[]
            for eid in vi:
                row=copy.deepcopy(native_rows[0]); row.update(example_id=eid,official_split='qual_dev',scope='ORACLE chunk source memories'); rows.append(row)
            (f/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows)); write(f,'NATIVE_COMPLETE.json',dict(rows=2,rows_sha256=digest(f/'rows.jsonl'),no_transmission=True))
            f=root/'train_clean'; f.mkdir(); seen=set(); history=[]
            for u in range(8):
                ids=[ti[i] for i in choose_indices(32,1,u,0,c['seed'])]; seen.update(ids)
                history.append(dict(update=u+1,batch_example_ids=ids,unique_examples_seen=len(seen),cumulative_example_presentations=u+1,loss=1.,ce_or_latent_mse=1.,pixel_mse=0.))
            (f/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
            write(f,'learning_audit.json',dict(passed=True,actual_optimizer_updates=8,unique_examples_seen=len(seen),groups=dict(codec=dict(parameters=1,nonzero_gradient=True,changed=True))))
            write(f,'TRAIN_COMPLETE.json',dict(arm='T1_chunks',stage='clean',optimizer_updates=8,exact_disk_roundtrip=True))
            write(f,'trainability.json',dict(training_examples=ti,microbatch_size=1,gradient_accumulation=1,example_presentations=8))
            write(f,'forward_consistency.json',dict(passed=True)); write(root,'run_identity.json',identity)
            result=finish(root); self.assertTrue(result['verified']); self.assertTrue((root/'QUALITY_STOP.json').exists()); self.assertFalse((root/'CLEAN_READY.json').exists())
            self.assertIn('全文WER',(root/'actual_recovery.html').read_text())
            identity['status']='CLEAN_READY'; write(root,'run_identity.json',identity)
            with self.assertRaises(ValueError): verify(root)


if __name__=='__main__': unittest.main(verbosity=2)
