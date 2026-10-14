"""Mutation-negative checks for complete same-frame text/image reporting."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import shutil
import unittest
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
import numpy as np
from docci_modal_v9.core import load,geometry,digest,metrics
from docci_modal_v9.channel import resource
from docci_modal_v9.report import evaluation,text_row,image_row,verify,finish


def write(root,name,value): (root/name).write_text(json.dumps(value,allow_nan=False))


def fixture(root):
    c=load(); ids=['qual_a','qual_b']; res=resource(c,'text'); (root/'frames').mkdir(); rows=[]
    for eid in ids:
        file=root/'frames'/(eid+'.npz')
        with file.open('wb') as f: np.savez(f,observed=np.ones(2048,np.float32),available=np.asarray(True),pilot_snr_db=np.asarray(40,np.float32),reliability=np.asarray(1,np.float32))
    rx=hashlib.sha256(np.ones(2048,np.float32).tobytes()).hexdigest()
    for i,eid in enumerate(ids):
        for ctrl in ('matched','zero','shuffled'):
            row=dict(branch='text',stage='initial',cohort='validation',example_id=eid,official_split='qual_dev',condition='clean',control=ctrl,
                frame_file='frames/'+eid+'.npz',frame_sha256=digest(root/'frames'/(eid+'.npz')),received_sha256=rx,tx_wire_sha256=rx,noise_seed=i,
                donor_example_id=ids[1-i] if ctrl=='shuffled' else None,donor_received_sha256=rx if ctrl=='shuffled' else None,
                source_text='the cat',source_text_sha256=hashlib.sha256(b'the cat').hexdigest(),source_tokens=4,input_truncated=False,
                received_caption='the cat',generated_token_ids=[2,0,5,6,2],stopped_by_capacity=False,receiver_failure=False,
                stats=dict(receiver_available=True,pilot_snr_db=40.,analog_reliability=1.,resources=res,payload_complex_symbols=1024,analog_repeats=1,
                    raw_utf8_wire_bytes=0,clean_sentinel_not_measured_snr=True,pre_fec_ber=None,post_fec_ber=None),**metrics('the cat','the cat','the cat'))
            rows.append(row)
    matrix=dict(branch='text',stage='initial',cohort='validation',example_ids=ids,controls=['matched','zero','shuffled'],conditions=['clean'],resources=res,
        Common_or_other_modality_used=False,raw_text_never_RX=True,failures_retained=True)
    write(root,'matrix.json',matrix); rewrite(root,rows)
    return c,ids,rows


def rewrite(root,rows):
    (root/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(root,'EVAL_COMPLETE.json',dict(rows=len(rows),rows_sha256=digest(root/'rows.jsonl'),matrix_sha256=digest(root/'matrix.json')))


class Reporting(unittest.TestCase):
    def test_full_negative_diagnostic_and_html_do_not_fake_complete(self):
        # No Torch/GPU fixture claims: this exercises the entire record workflow.
        from docci_modal_v9.core import schedule,gate
        with tempfile.TemporaryDirectory(dir=str(EXT.parent),prefix='v9_report_test_') as d:
            root=Path(d); c=load(); c['train_samples']=8; c['diagnostic_samples']=2; c['val_samples']=2; c['save_images']=2; c['text_diagnostic_updates']=8
            write(root,'config.json',c); base=root/'template'; base.mkdir(); _,vi,base_rows=fixture(base)
            ti=['train_a','train_b']+['train_'+str(i) for i in range(6)]; diag=ti[:2]
            identity=dict(schema='docci-modal-v9',branch='text',mode='diagnostic',config=c,schedule=schedule(c,'diagnostic','text'),
                Common_used=False,opposite_modality_model_loaded=False,raw_UTF8_transmitted=False,diffusion_used=False,
                backbone_signature_before='same',backbone_signature_after='same',initialization=dict(checkpoint=None),
                source_snapshot_sha256={},evaluations=[],stages=['train_overfit'],status='QUALITY_STOP')
            snapshot=root/'source_snapshot'; snapshot.mkdir(); source=Path(__file__).parent
            if (source/'MANIFEST.json').exists():
                source_manifest=json.loads((source/'MANIFEST.json').read_text())
            else:
                source_manifest=dict(schema='docci-modal-v9-source',files={f.name:digest(f) for f in source.iterdir() if f.is_file() and f.suffix in ('.py','.json','.md')})
            for name in source_manifest['files']: shutil.copyfile(source/name,snapshot/name)
            write(snapshot,'MANIFEST.json',source_manifest)
            if (source/'MANIFEST.json').exists(): shutil.copyfile(source/'MANIFEST.json',snapshot/'MANIFEST.json')
            identity['source_snapshot_sha256']=source_manifest['files']
            text_sha=hashlib.sha256(b'the cat').hexdigest()
            records=lambda ids,split:[dict(example_id=eid,official_split=split,text_sha256=text_sha,source_tokens=4) for eid in ids]
            write(root,'data_identity.json',dict(train=dict(records=records(ti,'train')),validation=dict(records=records(vi,'qual_dev')),diagnostic_ids=diag,
                source=dict(official_counts=dict(train=9647,test=5000,qual_dev=100,qual_test=100))))
            for name,ids,cohort,stage in (('initial_validation',vi,'validation','initial'),('overfit_train',diag,'diagnostic_train','overfit'),('overfit_validation',vi,'validation','overfit_DIAGNOSTIC_WEIGHTS')):
                folder=root/name; folder.mkdir(); _,_,rows=fixture(folder)
                mapping=dict(zip(vi,ids))
                for row in rows:
                    row.update(example_id=mapping[row['example_id']],cohort=cohort,stage=stage,official_split='train' if cohort=='diagnostic_train' else 'qual_dev')
                    if row['donor_example_id']: row['donor_example_id']=mapping[row['donor_example_id']]
                m=json.loads((folder/'matrix.json').read_text()); m.update(example_ids=ids,cohort=cohort,stage=stage); write(folder,'matrix.json',m); rewrite(folder,rows)
                identity['evaluations'].append(name)
                if name=='overfit_train': identity['overfit_gate']=gate(rows,'text',c,True)
            for name,ids,split in (('native_train',diag,'train'),('native_validation',vi,'qual_dev')):
                folder=root/name; folder.mkdir(); rows=[]
                for eid in ids:
                    row=copy.deepcopy(base_rows[0]); row.update(example_id=eid,official_split=split,scope='ORACLE no transmission'); rows.append(row)
                (folder/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows)); write(folder,'NATIVE_COMPLETE.json',dict(rows=2,rows_sha256=digest(folder/'rows.jsonl'),no_transmission=True))
            folder=root/'train_overfit'; folder.mkdir()
            history=[dict(update=i,batch_example_ids=diag*2,cumulative_example_presentations=i*4,loss=1.) for i in range(1,9)]
            (folder/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
            write(folder,'learning_audit.json',dict(actual_optimizer_updates=8,passed=True,groups=dict(codec=dict(parameters=1,changed=True,nonzero_gradient=True))))
            write(folder,'TRAIN_COMPLETE.json',dict(optimizer_updates=8,exact_disk_roundtrip=True))
            write(folder,'trainability.json',dict(microbatch_size=4,gradient_accumulation=1,example_presentations=32))
            write(folder,'forward_consistency.json',dict(passed=True)); write(root,'run_identity.json',identity)
            # Diagnostic only2 TRAIN samples -> actual batch2, not configured4.
            history=[dict(update=i,batch_example_ids=diag,cumulative_example_presentations=i*2,loss=1.) for i in range(1,9)]
            (folder/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history)); write(folder,'trainability.json',dict(microbatch_size=2,gradient_accumulation=1,example_presentations=16))
            result=finish(root); self.assertTrue(result['verified']); self.assertTrue((root/'QUALITY_STOP.json').exists()); self.assertFalse((root/'COMPLETE.json').exists())
            self.assertIn('原始输入', (root/'actual_recovery.html').read_text())
            identity['status']='DIAGNOSTIC_COMPLETE'; write(root,'run_identity.json',identity)
            with self.assertRaises(ValueError): verify(root)
    def test_complete_and_missing_matrix(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); c,ids,rows=fixture(root); self.assertEqual(len(evaluation(root,ids,'text',c)),6)
            rewrite(root,rows[:-1])
            with self.assertRaises(ValueError): evaluation(root,ids,'text',c)
    def test_duplicate_rows_and_wrong_donor(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); c,ids,rows=fixture(root); changed=copy.deepcopy(rows); changed[2]['donor_example_id']='qual_a'; rewrite(root,changed)
            with self.assertRaises(ValueError): evaluation(root,ids,'text',c)
            rewrite(root,rows+[rows[0]])
            with self.assertRaises(ValueError): evaluation(root,ids,'text',c)
    def test_scores_full_source_and_natural_EOS(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); c,ids,rows=fixture(root); changed=copy.deepcopy(rows); changed[0]['normalized_wer']=.1; rewrite(root,changed)
            with self.assertRaises(ValueError): evaluation(root,ids,'text',c)
            bad=copy.deepcopy(rows[0]); bad['stopped_by_capacity']=True
            with self.assertRaises(ValueError): text_row(bad,c)
    def test_receiver_only_public_fields_and_resource(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); c,ids,rows=fixture(root); changed=copy.deepcopy(rows); changed[0]['stats']['raw_utf8_wire_bytes']=50; rewrite(root,changed)
            with self.assertRaises(ValueError): evaluation(root,ids,'text',c)
            file=root/'frames/qual_a.npz'
            with file.open('wb') as f: np.savez(f,observed=np.ones(2048,np.float32),available=True,pilot_snr_db=40.,reliability=1.,source_mask=[True])
            for row in rows:
                if row['example_id']=='qual_a': row['frame_sha256']=digest(file)
            rewrite(root,rows)
            with self.assertRaises(ValueError): evaluation(root,ids,'text',c)
    def test_test_split_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); c,ids,rows=fixture(root); rows[0]['official_split']='test'; rewrite(root,rows)
            with self.assertRaises(ValueError): evaluation(root,ids,'text',c)
    def test_image_scores_and_valid_rectangle(self):
        c=load(); row=dict(image_size=512,mse=.01,psnr=20.,ssim=.8,evaluation_rectangle=[0,0,512,512],restored_image_file=None)
        image_row(Path('.'),row,c); row['psnr']=24.
        with self.assertRaises(ValueError): image_row(Path('.'),row,c)


if __name__=='__main__': unittest.main(verbosity=2)
