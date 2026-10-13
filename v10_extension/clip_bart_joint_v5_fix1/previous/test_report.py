"""Synthetic positives/negative stops, never proof of learned image/text gain."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
EXT=Path(__file__).resolve().parents[1]; PROJECT=EXT.parent; sys.path.insert(0,str(EXT))
from clip_bart_joint_v5 import VERSION,ARMS
from clip_bart_joint_v5.core import load,text_gate,geometry,metrics,digest
from clip_bart_joint_v5.report import controls,verify,finish
from bart_latent_codec_v4.core import batch_indices
from clip_bart_joint_v5.collect import collect


def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f: json.dump(v,f)


def lines(p,rs):
    with p.open('w',encoding='utf-8') as f:
        for r in rs: f.write(json.dumps(r)+'\n')


class Reports(unittest.TestCase):
    def location(self): return tempfile.TemporaryDirectory(prefix='v5_report_',dir=str(PROJECT/'releases'))
    def fixture(self,root):
        c=load(); c.update(text_train_samples=8,text_updates=8,train_samples=8,val_samples=2,clean_epochs=1,joint_warm_steps=2,save_images=2)
        tc=dict(batch_size=2); tr=list(range(1,9)); vi=[11,12]
        (root/'source_snapshot').mkdir()
        with (root/'source_snapshot/model.py').open('w') as f: f.write('# synthetic fixture\n')
        save(root/'run_identity.json',dict(version=VERSION,status='screen_complete',config=c,text_config=tc,
            TRAIN8_weights_loaded=False,smoke=False,noise=False,source_sha256={'model.py':digest(root/'source_snapshot/model.py')}))
        save(root/'design.json',dict(config=c,arms=list(ARMS),geometry=geometry(),TRAIN8_weights_loaded=False,
            text_train_view=dict(image_ids=tr),text_validation_view=dict(image_ids=vi)))
        f=root/'text_prefit/clean'; f.mkdir(parents=True)
        save(f/'TRAIN_COMPLETE.json',dict(updates=8)); save(f/'learning_audit.json',dict(passed=True,frozen_unchanged=True))
        save(f/'reload_audit.json',dict(passed=True))
        lines(f/'history.jsonl',[dict(update=j+1,batch_image_ids=[tr[i] for i in batch_indices(8,2,j,c['seed'])]) for j in range(8)])
        rs=[]
        for control in ('matched','zero','shuffled'):
            for iid in vi:
                source='a cat sits'; text='wrong' if control=='matched' else ''
                rs.append(dict(image_id=iid,control=control,source_split='validation',source_text=source,covered_text=source,
                    received_caption=text,input_truncated=False,stopped_by_capacity=False,receiver_failure=False,
                    donor_image_id=23-iid if control=='shuffled' else None,**metrics(source,source,text)))
        lines(root/'text_prefit/validation_controls.jsonl',rs)
        g=text_gate(controls(root/'text_prefit/validation_controls.jsonl',vi),c); save(root/'text_prefit/validation_gate.json',g)
        save(root/'arm_status.json',{arm:dict(trained=False,status='text_prefit_stop',clean_passed=False,noise_trained=False) for arm in ARMS})
        save(root/'frozen_verification.json',dict(passed=True,backbones_unchanged=True,original_data_unchanged=True))
    def test_negative_text_prefit_is_valid_completed_screen(self):
        with self.location() as d:
            root=Path(d); self.fixture(root); finish(root); self.assertTrue(verify(root)['passed'])
            self.assertFalse(verify(root)['text_prefit_passed'])
    def test_no_train8_checkpoint_shortcut(self):
        with self.location() as d:
            root=Path(d); self.fixture(root); identity=json.loads((root/'run_identity.json').read_text()); identity['TRAIN8_weights_loaded']=True; save(root/'run_identity.json',identity)
            with self.assertRaises(ValueError): verify(root,False)
    def test_metric_tamper_rejected(self):
        with self.location() as d:
            root=Path(d); self.fixture(root); p=root/'text_prefit/validation_controls.jsonl'
            rs=[json.loads(v) for v in p.read_text().splitlines()]; rs[0]['covered_wer']=0.; lines(p,rs)
            with self.assertRaises(ValueError): verify(root,False)
    def test_duplicate_control_rejected(self):
        with self.location() as d:
            root=Path(d); self.fixture(root); p=root/'text_prefit/validation_controls.jsonl'; rs=[json.loads(v) for v in p.read_text().splitlines()]; lines(p,rs+[rs[0]])
            with self.assertRaises(ValueError): verify(root,False)
    def test_missing_prefit_update_rejected(self):
        with self.location() as d:
            root=Path(d); self.fixture(root); p=root/'text_prefit/clean/history.jsonl'; rs=[json.loads(v) for v in p.read_text().splitlines()]; lines(p,rs[:-1])
            with self.assertRaises(ValueError): verify(root,False)
    def test_block_joint_after_failed_prefit(self):
        with self.location() as d:
            root=Path(d); self.fixture(root); (root/'J1_joint').mkdir()
            with self.assertRaises(ValueError): verify(root,False)
    def test_failure_not_collected(self):
        with self.location() as d:
            root=Path(d); self.fixture(root); finish(root); save(root/'FAILED.json',{'failed':True})
            with self.assertRaises(ValueError): verify(root)
    def test_archive_roundtrip_no_weights_and_no_overwrite(self):
        import tarfile
        with self.location() as d:
            root=Path(d)/'run'; root.mkdir(); self.fixture(root); finish(root); archive=Path(d)/'records.tar.gz'; collect(root,archive)
            dest=Path(d)/'restore'; dest.mkdir()
            with tarfile.open(archive) as tar: tar.extractall(str(dest))
            self.assertTrue(verify(dest/'v10_clip_bart_joint_v5_records')['passed'])
            with self.assertRaises(FileExistsError): collect(root,archive)

    def full_fixture(self,root):
        self.fixture(root)
        from clip_bart_joint_v5.report import read
        from fso_fluctuating_v2.core import load as physical_config
        from clip_bart_joint_v5.channel import resource
        from clip_bart_joint_v5.core import make_row
        import math
        import numpy as np
        identity=read(root/'run_identity.json'); c=identity['config']; vi=[11,12]; tr=list(range(1,9))
        p=root/'text_prefit/validation_controls.jsonl'; rs=[json.loads(v) for v in p.read_text().splitlines()]
        for r in rs:
            if r['control']=='matched': r['received_caption']=r['source_text']; r.update(metrics(r['source_text'],r['covered_text'],r['received_caption']))
        lines(p,rs); save(root/'text_prefit/validation_gate.json',text_gate(controls(p,vi),c))
        cc=physical_config(); cc=dict(cc,conditions=cc['conditions'][:1]); cid=cc['conditions'][0]['id']; res=resource(cc)
        design=read(root/'design.json'); design.update(image_train_view=dict(image_ids=tr),image_validation_view=dict(image_ids=vi),channel_config=cc); save(root/'design.json',design)
        statuses={}
        for arm in ARMS:
            folder=root/arm; folder.mkdir(); tf=folder/'clean'; tf.mkdir()
            save(folder/'initial_state_hashes.json',{'one':'same synthetic value'})
            save(tf/'TRAIN_COMPLETE.json',dict(steps=8,optimizer_updates=4,checkpoint_sha256='weights excluded'))
            order=np.random.default_rng(c['seed']).permutation(8)
            lines(tf/'history.jsonl',[dict(step=j+1,total_steps=8,image_id=tr[int(order[j])],noisy=False,proxy=[]) for j in range(8)])
            gs={k:dict(tensors=1,nonzero_gradient=True,changed=True) for k in ('text_codec','common_source','common_receiver','common_tasks','image_tx','image_rx','text_fusion')}
            save(tf/'learning_audit.json',dict(passed=True,frozen_unchanged=True,groups=gs))
            save(tf/'reload_audit.json',dict(passed=True)); save(tf/'forward_consistency.json',dict(errors={'wire':0.},tolerance=c['forward_tolerance']))
            save(tf/'trainability.json',dict(registered_bridge_parameters=100))
            lines(folder/'clean_controls.jsonl',rs); save(folder/'clean_gate.json',text_gate(controls(folder/'clean_controls.jsonl',vi),c))
            ef=folder/'eval_clean'; ef.mkdir()
            ctrls=['normal','zero_common','no_common_fusion','shuffled_common'] if arm=='J1_joint' else ['normal']
            save(ef/'eval_design.json',dict(arm=arm,phase='clean',conditions=['clean',cid],controls=ctrls,
                validation_ids=vi,resources=res,channel_config=cc))
            ev=[dict(condition=cid,image_id=i,event=j) for i in vi for j in (0,1)]; lines(ef/'events.jsonl',ev)
            er=[]
            for condition in ['clean',cid]:
                for control in ctrls:
                    for iid in vi:
                        item=dict(image_id=iid,annotation_id=iid,split='validation',text='a cat sits',covered_text='a cat sits',
                            source_sha256='synthetic',source_utf8_bytes=10,truncated=False)
                        stats=None if condition=='clean' else dict(used_samples=res['scheduled_samples'],payload_complex_symbols=3072,
                            analog_repeats=1,raw_utf8_wire_bytes=0,receiver_available=True)
                        er.append(make_row(item,arm,'clean',condition,dict(text=item['text'],stopped_by_capacity=False,receiver_failure=False),
                            dict(mse=.04,psnr=-10*math.log10(.04),ssim=.8),control=control,stats=stats))
            if arm=='J1_joint':
                for iid in vi:
                    item=dict(image_id=iid,annotation_id=iid,split='validation',text='a cat sits',covered_text='a cat sits',source_sha256='synthetic',source_utf8_bytes=10,truncated=False)
                    er.append(make_row(item,arm,'clean','clean',dict(text='a dog runs',stopped_by_capacity=False,receiver_failure=False),
                        dict(mse=.04,psnr=-10*math.log10(.04),ssim=.8),control='wrong_text_source',stats=None))
            lines(ef/'samples.jsonl',er)
            statuses[arm]=dict(trained=True,status='clean_passed',clean_passed=True,noise_trained=False)
        save(root/'arm_status.json',statuses)

    def test_full_clean_positive_matrix(self):
        with self.location() as d:
            root=Path(d); self.full_fixture(root); finish(root); self.assertTrue(verify(root)['passed'])
    def test_registered_count_mismatch_rejected(self):
        with self.location() as d:
            root=Path(d); self.full_fixture(root); save(root/'J2_no_clip_text/clean/trainability.json',{'registered_bridge_parameters':101})
            with self.assertRaises(ValueError): verify(root,False)
    def test_physical_event_missing_rejected(self):
        with self.location() as d:
            root=Path(d); self.full_fixture(root); p=root/'J1_joint/eval_clean/events.jsonl'; rs=[json.loads(v) for v in p.read_text().splitlines()]; lines(p,rs[:-1])
            with self.assertRaises(ValueError): verify(root,False)
    def test_noise_not_started_without_request(self):
        with self.location() as d:
            root=Path(d); self.full_fixture(root); p=root/'arm_status.json'; s=json.loads(p.read_text()); s['J1_joint'].update(noise_trained=True,status='noise_evaluated'); save(p,s)
            with self.assertRaises(ValueError): verify(root,False)


if __name__=='__main__': unittest.main(verbosity=2)
