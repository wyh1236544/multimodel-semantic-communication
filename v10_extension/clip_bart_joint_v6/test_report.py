"""Synthetic positives/negative stops, never proof of learned image/text gain."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
EXT=Path(__file__).resolve().parents[1]; PROJECT=EXT.parent; sys.path.insert(0,str(EXT))
from clip_bart_joint_v6 import VERSION,ARMS
from clip_bart_joint_v6.core import load,text_gate,geometry,metrics,digest
from clip_bart_joint_v6.report import controls,verify,finish
from bart_latent_codec_v4.core import batch_indices
from clip_bart_joint_v6.collect import collect


def production_clean_snr():
    """Execute the unchanged clean_args BODY with an explicitly NumPy-only shim.

    This tests the production convention without importing Torch. It is not a
    neural or GPU test; the separate real-Torch test below covers that interface.
    """
    import ast
    import numpy as np
    from types import SimpleNamespace
    path = EXT/'clip_bart_joint_v5/data.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(v for v in tree.body if isinstance(v, ast.FunctionDef) and v.name == 'clean_args')
    class Wire:
        shape = (1, 6144)
        device = 'cpu'
        def new_full(self, shape, value): return np.full(shape, value, dtype=np.float32)
        def new_ones(self, count): return np.ones(count, dtype=np.float32)
    shim = SimpleNamespace(ones=lambda count, device, dtype: np.ones(count, dtype=bool), bool=bool)
    namespace = {'torch': shim}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return float(namespace['clean_args'](Wire())[2][0])


def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f: json.dump(v,f)


def lines(p,rs):
    with p.open('w',encoding='utf-8') as f:
        for r in rs: f.write(json.dumps(r)+'\n')


class Reports(unittest.TestCase):
    def location(self): return tempfile.TemporaryDirectory(prefix='v5_report_',dir=str(PROJECT/'releases'))
    def fixture(self,root):
        c=load(); c.update(text_train_samples=8,text_updates=3016,text_block_updates=8,train_samples=8,val_samples=2,common_epochs=1,clean_epochs=1,joint_warm_steps=2,save_images=2,generation_train_samples=4)
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
        # Actual V4 private_controls always writes the secondary beam rows too.
        for control in ('matched','zero','shuffled','matched_beam4'):
            for iid in vi:
                source='a cat sits'; text=source if control=='matched_beam4' else ('wrong' if control=='matched' else '')
                rs.append(dict(image_id=iid,control=control,source_split='validation',source_text=source,covered_text=source,
                    received_caption=text,input_truncated=False,stopped_by_capacity=False,receiver_failure=False,
                    donor_image_id=23-iid if control=='shuffled' else None,**metrics(source,source,text)))
        lines(root/'text_prefit/validation_controls.jsonl',rs)
        g=text_gate(controls(root/'text_prefit/validation_controls.jsonl',vi,secondary_beam=True),c); save(root/'text_prefit/validation_gate.json',g)
        save(root/'arm_status.json',{arm:dict(trained=False,status='text_prefit_stop',clean_passed=False,noise_trained=False) for arm in ARMS})
        save(root/'frozen_verification.json',dict(passed=True,backbones_unchanged=True,original_data_unchanged=True))
        self.text_extras(root,c)
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
    def test_actual_v4_four_controls_primary_gate_unchanged(self):
        with self.location() as d:
            root=Path(d); self.fixture(root); path=root/'text_prefit/validation_controls.jsonl'
            groups=controls(path,[11,12],secondary_beam=True)
            self.assertEqual(set(groups),{'matched','zero','shuffled'})
            self.assertEqual(groups['matched']['covered_wer'],1.)
            self.assertFalse(text_gate(groups,load())['passed'])
            self.assertTrue(verify(root,False)['passed'])
    def test_secondary_beam_missing_rejected(self):
        with self.location() as d:
            root=Path(d); self.fixture(root); p=root/'text_prefit/validation_controls.jsonl'
            rs=[json.loads(v) for v in p.read_text().splitlines()]
            lines(p,[r for r in rs if not (r['control']=='matched_beam4' and r['image_id']==11)])
            with self.assertRaises(ValueError): verify(root,False)
    def test_secondary_beam_duplicate_rejected(self):
        with self.location() as d:
            root=Path(d); self.fixture(root); p=root/'text_prefit/validation_controls.jsonl'
            rs=[json.loads(v) for v in p.read_text().splitlines()]
            lines(p,rs+[next(r for r in rs if r['control']=='matched_beam4')])
            with self.assertRaises(ValueError): verify(root,False)
    def test_secondary_beam_metric_tamper_rejected(self):
        with self.location() as d:
            root=Path(d); self.fixture(root); p=root/'text_prefit/validation_controls.jsonl'
            rs=[json.loads(v) for v in p.read_text().splitlines()]
            next(r for r in rs if r['control']=='matched_beam4')['covered_wer']=.5
            lines(p,rs)
            with self.assertRaises(ValueError): verify(root,False)
    def test_secondary_beam_wrong_source_split_rejected(self):
        with self.location() as d:
            root=Path(d); self.fixture(root); p=root/'text_prefit/validation_controls.jsonl'
            rs=[json.loads(v) for v in p.read_text().splitlines()]
            next(r for r in rs if r['control']=='matched_beam4')['source_split']='train'
            lines(p,rs)
            with self.assertRaises(ValueError): verify(root,False)
    def test_unknown_control_still_rejected(self):
        with self.location() as d:
            root=Path(d); self.fixture(root); p=root/'text_prefit/validation_controls.jsonl'
            rs=[json.loads(v) for v in p.read_text().splitlines()]; rs[-1]['control']='anything'
            lines(p,rs)
            with self.assertRaises(ValueError): verify(root,False)
    def test_joint_controls_still_only_three(self):
        with self.location() as d:
            root=Path(d); self.fixture(root)
            with self.assertRaises(ValueError): controls(root/'text_prefit/validation_controls.jsonl',[11,12])
    def test_beam_cannot_replace_missing_greedy(self):
        with self.location() as d:
            root=Path(d); self.fixture(root); p=root/'text_prefit/validation_controls.jsonl'
            rs=[json.loads(v) for v in p.read_text().splitlines()]
            lines(p,[r for r in rs if r['control']!='matched'])
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
            self.assertTrue(verify(dest/'v10_clip_bart_joint_v6_records')['passed'])
            with self.assertRaises(FileExistsError): collect(root,archive)

    def full_fixture(self,root):
        self.fixture(root)
        from clip_bart_joint_v6.report import read
        from fso_fluctuating_v2.core import load as physical_config
        from clip_bart_joint_v5.channel import resource
        from clip_bart_joint_v6.core import make_row
        import math
        import numpy as np
        identity=read(root/'run_identity.json'); c=identity['config']; vi=[11,12]; tr=list(range(1,9))
        p=root/'text_prefit/validation_controls.jsonl'; rs=[json.loads(v) for v in p.read_text().splitlines()]
        for r in rs:
            if r['control']=='matched': r['received_caption']=r['source_text']; r.update(metrics(r['source_text'],r['covered_text'],r['received_caption']))
        lines(p,rs); save(root/'text_prefit/validation_gate.json',text_gate(controls(p,vi,secondary_beam=True),c))
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
            lines(folder/'clean_controls.jsonl',[r for r in rs if r['control']!='matched_beam4']); save(folder/'clean_gate.json',text_gate(controls(folder/'clean_controls.jsonl',vi),c))
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
        self.joint_extras(root,c)

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

    def text_extras(self,root,c):
        import shutil
        from clip_bart_joint_v6.report import read
        identity=read(root/'run_identity.json'); identity['text_milestones']=[3000,3008,3016]; save(root/'run_identity.json',identity)
        save(root/'parent_text_identity.json',dict(source_proof=dict(passed=True,smoke=False,text_prefit_passed=True),source_updates=3000,optimizer_restored=False))
        save(root/'text_prefit/milestones.json',dict(global_updates=[3000,3008,3016],fixed_last_checkpoint=True,no_validation_checkpoint_selection=True))
        original=root/'text_prefit/validation_controls.jsonl'
        for u in (3000,3008):
            shutil.copyfile(original,root/'text_prefit'/('validation_u'+str(u)+'.jsonl'))
            shutil.copyfile(root/'text_prefit/validation_gate.json',root/'text_prefit'/('gate_u'+str(u)+'.json'))
        shutil.copytree(root/'text_prefit/clean',root/'text_prefit/u3008')
        rs=[r for r in [json.loads(v) for v in original.read_text().splitlines()] if r['control']=='matched']
        for r in rs: r.update(control='oracle_not_transmitted',original_source_memory_provided=True)
        lines(root/'text_prefit/native_bart_oracle.jsonl',rs)

    def joint_extras(self,root,c):
        import hashlib
        import shutil
        import numpy as np
        from clip_bart_joint_v6.report import read
        from clip_bart_joint_v6.core import private_mask
        design=read(root/'design.json')
        # Production joint cache hashes IMAGE FLOAT32 BYTES + UTF8 TEXT; never
        # substitute the text-only hash for that unchanged joint identity.
        text_hashes={str(i):hashlib.sha256(b'a cat sits').hexdigest() for i in range(1,9)}
        design['image_train_view']['source_sha256']={
            str(i):hashlib.sha256(np.full((1,3,128,128),i/8,dtype=np.float32).tobytes()+b'a cat sits').hexdigest()
            for i in range(1,9)}
        design['image_train_view']['text_source_sha256']=dict(text_hashes)
        design['text_train_view']['source_sha256']=dict(text_hashes)
        save(root/'design.json',design)
        for arm in ARMS:
            folder=root/arm; clean=folder/'clean'
            audit=read(clean/'learning_audit.json')
            for key in ('balanced_pool','common_caption'): audit['groups'][key]=dict(tensors=1,nonzero_gradient=True,changed=True)
            save(clean/'learning_audit.json',audit)
            hs=[json.loads(v) for v in (clean/'history.jsonl').read_text().splitlines()]
            for j,r in enumerate(hs): r['private_mask']=private_mask(c,j,r['image_id'],'clean')
            lines(clean/'history.jsonl',hs)
            shutil.copytree(clean,folder/'common')
            common=folder/'common'; ca=read(common/'learning_audit.json')
            ca['groups']={k:ca['groups'][k] for k in ('common_source','common_receiver','common_tasks','balanced_pool','common_caption')}
            save(common/'learning_audit.json',ca)
            for j,r in enumerate(hs): r['private_mask']=private_mask(c,j,r['image_id'],'common')
            lines(common/'history.jsonl',hs)
            for name in ('common_controls.jsonl','joint_common_controls.jsonl'):
                rs=[json.loads(v) for v in (folder/'clean_controls.jsonl').read_text().splitlines()]
                for r in rs: r.update(common_only_receiver_diagnostic=True,charged_payload_complex_symbols=3072)
                lines(folder/name,rs)
            er=[]
            for control in ('normal','zero_common','no_common_fusion','shuffled_common'):
                for r in [z for z in rs if z['control']=='matched']:
                    z=dict(r,control=control,geometry=geometry(),private_mask=private_mask(c,3,r['image_id'],'clean'),
                        not_a_physical_channel=True,donor_image_id=23-r['image_id'] if control=='shuffled_common' else None)
                    er.append(z)
            lines(folder/'masked_validation.jsonl',er)
            ef=folder/'eval_clean'
            entries=[json.loads(v) for v in (ef/'samples.jsonl').read_text().splitlines()]
            (ef/'images').mkdir()
            for r in entries:
                if r['control']=='normal':
                    fname='images/{}__{}__receiver.npz'.format(r['condition'],r['image_id'])
                    snr=production_clean_snr() if r['condition']=='clean' else 0.
                    if r['stats'] is not None: r['stats']['pilot_snr_db']=snr
                    r['receiver_condition_file']=fname
                    np.savez_compressed(ef/fname,common=np.ones((32,16)),received_caption=np.asarray(r['received_caption']),
                        receiver_available=np.asarray(True),image_id=np.asarray(r['image_id']),
                        source_split=np.asarray('validation'),pilot_snr_db=np.asarray(snr))
            lines(ef/'samples.jsonl',entries)
        np.savez_compressed(root/'J1_joint/generation_train.npz',common=np.ones((4,32,16)),
            received_caption=np.asarray(['a cat sits']*4),source_text_train_teacher=np.asarray(['a cat sits']*4),
            image_id=np.asarray([1,2,3,4]),source_split=np.asarray(['train']*4),
            proxy_noisy=np.asarray([False]*4),pilot_snr_db=np.asarray([20.]*4))

    def test_private_mask_tamper_rejected(self):
        with self.location() as d:
            root=Path(d); self.full_fixture(root); p=root/'J1_joint/clean/history.jsonl'
            rs=[json.loads(v) for v in p.read_text().splitlines()]; rs[1]['private_mask']['image_slots']=[]
            lines(p,rs)
            with self.assertRaises(ValueError): verify(root,False)
    def test_parent_smoke_not_accepted_as_generalized_source(self):
        with self.location() as d:
            root=Path(d); self.fixture(root); p=root/'parent_text_identity.json'; v=json.loads(p.read_text())
            v['source_proof']['smoke']=True; save(p,v)
            with self.assertRaises(ValueError): verify(root,False)
    def test_missing_intermediate_text_update_rejected(self):
        with self.location() as d:
            root=Path(d); self.fixture(root); p=root/'text_prefit/u3008/history.jsonl'
            rs=[json.loads(v) for v in p.read_text().splitlines()]; lines(p,rs[:-1])
            with self.assertRaises(ValueError): verify(root,False)
    def test_generation_teacher_validation_leak_rejected(self):
        import numpy as np
        with self.location() as d:
            root=Path(d); self.full_fixture(root)
            file=root/'J1_joint/generation_train.npz'
            with np.load(file,allow_pickle=False) as cache: data={k:cache[k].copy() for k in cache.files}
            data['source_split']=np.asarray(['validation']*4)
            np.savez_compressed(file,**data)
            with self.assertRaises(ValueError): verify(root,False)
    def test_full_paired_noise_positive(self):
        import shutil
        import numpy as np
        from clip_bart_joint_v6.report import read
        from clip_bart_joint_v6.core import private_mask
        with self.location() as d:
            root=Path(d); self.full_fixture(root); identity=read(root/'run_identity.json')
            identity['noise']=True; save(root/'run_identity.json',identity); c=identity['config']
            statuses=read(root/'arm_status.json')
            for arm in ARMS:
                folder=root/arm
                for phase in ('clean_continue','fso'):
                    shutil.copytree(folder/'clean',folder/phase)
                    total=c['train_samples']*c['continue_epochs']; history=[]
                    for j in range(total):
                        ep,pos=divmod(j,8); iid=int(np.random.default_rng(c['seed']+ep).permutation(8)[pos])+1
                        noisy=phase=='fso' and j%c['clean_anchor_period']!=0
                        history.append(dict(step=j+1,total_steps=total,image_id=iid,noisy=noisy,
                            proxy=[dict(payload_complex_symbols=3072,analog_repeats=1)] if noisy else [],
                            private_mask=private_mask(c,j,iid,phase)))
                    lines(folder/phase/'history.jsonl',history)
                    save(folder/phase/'TRAIN_COMPLETE.json',dict(steps=total,optimizer_updates=total//2,checkpoint_sha256='excluded'))
                    shutil.copyfile(folder/'clean_controls.jsonl',folder/(phase+'_controls.jsonl'))
                    dest=folder/('eval_'+phase); shutil.copytree(folder/'eval_clean',dest)
                    ed=read(dest/'eval_design.json'); ed['phase']=phase; save(dest/'eval_design.json',ed)
                    rs=[json.loads(v) for v in (dest/'samples.jsonl').read_text().splitlines()]
                    for r in rs: r['phase']=phase
                    lines(dest/'samples.jsonl',rs)
                statuses[arm].update(status='noise_evaluated',noise_trained=True)
            save(root/'arm_status.json',statuses); finish(root); self.assertTrue(verify(root)['passed'])
    def test_common_scope_not_free_bandwidth(self):
        with self.location() as d:
            root=Path(d); self.full_fixture(root); p=root/'J1_joint/common_controls.jsonl'
            rs=[json.loads(v) for v in p.read_text().splitlines()]; rs[0]['charged_payload_complex_symbols']=256; lines(p,rs)
            with self.assertRaises(ValueError): verify(root,False)
    def test_missing_masked_validation_rejected(self):
        with self.location() as d:
            root=Path(d); self.full_fixture(root); p=root/'J1_joint/masked_validation.jsonl'
            rs=[json.loads(v) for v in p.read_text().splitlines()]; lines(p,rs[:-1])
            with self.assertRaises(ValueError): verify(root,False)
    def test_oracle_not_mislabeled_actual_receiver(self):
        with self.location() as d:
            root=Path(d); self.fixture(root); p=root/'text_prefit/native_bart_oracle.jsonl'
            rs=[json.loads(v) for v in p.read_text().splitlines()]; rs[0]['control']='normal'; lines(p,rs)
            with self.assertRaises(ValueError): verify(root,False)
    def test_intermediate_frozen_failure_rejected(self):
        with self.location() as d:
            root=Path(d); self.fixture(root); p=root/'text_prefit/u3008/learning_audit.json'
            v=json.loads(p.read_text()); v['frozen_unchanged']=False; save(p,v)
            with self.assertRaises(ValueError): verify(root,False)
    def test_generation_teacher_content_tamper_rejected(self):
        import numpy as np
        with self.location() as d:
            root=Path(d); self.full_fixture(root); p=root/'J1_joint/generation_train.npz'
            with np.load(p,allow_pickle=False) as z: data={k:z[k].copy() for k in z.files}
            data['source_text_train_teacher']=np.asarray(['wrong']*4); np.savez_compressed(p,**data)
            with self.assertRaises(ValueError): verify(root,False)
    def test_receiver_condition_source_text_leak_rejected(self):
        import numpy as np
        with self.location() as d:
            root=Path(d); self.full_fixture(root); p=root/'J1_joint/eval_clean/images/clean__11__receiver.npz'
            with np.load(p,allow_pickle=False) as z: data={k:z[k].copy() for k in z.files}
            data['source_text']=np.asarray('a cat sits'); np.savez_compressed(p,**data)
            with self.assertRaises(ValueError): verify(root,False)
    def test_receiver_condition_caption_mismatch_rejected(self):
        import numpy as np
        with self.location() as d:
            root=Path(d); self.full_fixture(root); p=root/'J1_joint/eval_clean/images/clean__11__receiver.npz'
            with np.load(p,allow_pickle=False) as z: data={k:z[k].copy() for k in z.files}
            data['received_caption']=np.asarray('different'); np.savez_compressed(p,**data)
            with self.assertRaises(ValueError): verify(root,False)

    def replace_receiver_snr(self, root, condition, value, stats_value=None):
        import numpy as np
        folder=root/'J1_joint/eval_clean'
        p=folder/'images'/('{}__11__receiver.npz'.format(condition))
        with np.load(p,allow_pickle=False) as z: data={k:z[k].copy() for k in z.files}
        data['pilot_snr_db']=np.asarray(value)
        np.savez_compressed(p,**data)
        if stats_value is not None:
            rows=[json.loads(v) for v in (folder/'samples.jsonl').read_text().splitlines()]
            next(r for r in rows if r['condition']==condition and r['control']=='normal' and r['image_id']==11)['stats']['pilot_snr_db']=stats_value
            lines(folder/'samples.jsonl',rows)

    def test_production_clean_args_40_roundtrip(self):
        self.assertEqual(production_clean_snr(),40.)
        with self.location() as d:
            root=Path(d); self.full_fixture(root); finish(root)
            self.assertTrue(verify(root)['passed'])

    def test_actual_torch_clean_args_report_roundtrip(self):
        import importlib.util
        if importlib.util.find_spec('torch') is None:
            self.skipTest('Actual Torch unavailable locally; mandatory in server launcher')
        import torch
        from clip_bart_joint_v5.data import clean_args
        wire=torch.zeros(2,6144)
        args=clean_args(wire)
        self.assertIs(args[0],wire)
        self.assertTrue(args[1].all())
        self.assertTrue(torch.equal(args[2],torch.full((2,),40.)))
        self.assertTrue(torch.equal(args[3],torch.ones(2)))
        with self.location() as d:
            root=Path(d); self.full_fixture(root)
            self.replace_receiver_snr(root,'clean',float(args[2][0]))
            self.assertTrue(verify(root,False)['passed'])

    def test_obsolete_clean_20_rejected(self):
        with self.location() as d:
            root=Path(d); self.full_fixture(root); self.replace_receiver_snr(root,'clean',20.)
            with self.assertRaisesRegex(ValueError,'Received pilot SNR differs'): verify(root,False)

    def test_clean_zero_placeholder_rejected(self):
        with self.location() as d:
            root=Path(d); self.full_fixture(root); self.replace_receiver_snr(root,'clean',0.)
            with self.assertRaisesRegex(ValueError,'Received pilot SNR differs'): verify(root,False)

    def test_physical_snr_float32_rounding_accepted(self):
        import numpy as np
        with self.location() as d:
            root=Path(d); self.full_fixture(root); value=-13.764312893427
            self.replace_receiver_snr(root,'weak_p06',np.float32(value),stats_value=value)
            self.assertTrue(verify(root,False)['passed'])

    def test_physical_snr_mismatch_rejected(self):
        with self.location() as d:
            root=Path(d); self.full_fixture(root)
            self.replace_receiver_snr(root,'weak_p06',40.,stats_value=0.)
            with self.assertRaisesRegex(ValueError,'Received pilot SNR differs'): verify(root,False)

    def test_nonfinite_received_snr_rejected(self):
        for condition in ('clean','weak_p06'):
            for value in (float('nan'),float('inf'),float('-inf')):
                with self.subTest(condition=condition,value=value), self.location() as d:
                    root=Path(d); self.full_fixture(root); self.replace_receiver_snr(root,condition,value)
                    with self.assertRaisesRegex(ValueError,'Received pilot SNR differs'): verify(root,False)

    def test_real_joint_and_independent_text_hashes_roundtrip(self):
        with self.location() as d:
            root=Path(d); self.full_fixture(root)
            design=json.loads((root/'design.json').read_text())
            image_view=design['image_train_view']
            self.assertNotEqual(image_view['source_sha256']['1'],image_view['text_source_sha256']['1'])
            finish(root); self.assertTrue(verify(root)['passed'])

    def test_missing_independent_teacher_hash_rejected(self):
        with self.location() as d:
            root=Path(d); self.full_fixture(root); design=json.loads((root/'design.json').read_text())
            del design['image_train_view']['text_source_sha256']; save(root/'design.json',design)
            with self.assertRaises(ValueError): verify(root,False)

    def test_joint_hash_cannot_be_reused_as_text_hash(self):
        with self.location() as d:
            root=Path(d); self.full_fixture(root); design=json.loads((root/'design.json').read_text())
            design['image_train_view']['text_source_sha256']=dict(design['image_train_view']['source_sha256'])
            save(root/'design.json',design)
            with self.assertRaises(ValueError): verify(root,False)

    def test_teacher_hash_must_match_independent_bart_view(self):
        with self.location() as d:
            root=Path(d); self.full_fixture(root); design=json.loads((root/'design.json').read_text())
            design['text_train_view']['source_sha256']['1']='0'*64; save(root/'design.json',design)
            with self.assertRaises(ValueError): verify(root,False)

    def test_short_teacher_vector_not_silently_zipped(self):
        import numpy as np
        with self.location() as d:
            root=Path(d); self.full_fixture(root); p=root/'J1_joint/generation_train.npz'
            with np.load(p,allow_pickle=False) as z: data={k:z[k].copy() for k in z.files}
            data['source_text_train_teacher']=data['source_text_train_teacher'][:1]
            np.savez_compressed(p,**data)
            with self.assertRaises(ValueError): verify(root,False)


if __name__=='__main__': unittest.main(verbosity=2)
