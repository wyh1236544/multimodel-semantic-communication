"""Positive complete synthetic bookkeeping fixture + hostile mutations.

Fixtures do not execute Torch and do not establish model quality.
"""
import json
from pathlib import Path
import sys
import tempfile
import tarfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bidirectional_rx_v7 import VERSION,ARMS
from bidirectional_rx_v7.core import load,plan,geometry,mask_policy,array_hash,metrics,variants,seal
from bidirectional_rx_v7.report import verify,read
from bidirectional_rx_v7.report import finish
from bidirectional_rx_v7.collect import collect
from fso_fluctuating_v2.core import load as channel_config
from clip_bart_joint_v5.channel import resource


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value))


def rows(path,values):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text('\n'.join(json.dumps(v) for v in values)+'\n')


def fixture(root):
    c=load(); c['save_images']=0; s=plan(c,'smoke'); ti=list(range(1,9)); vi=[101,102]; cc=channel_config(); cc['conditions']=cc['conditions'][:1]; res=resource(cc)
    wire=np.ones(6144,np.float32); initial={'synthetic':'hash'}; hashes={str(i):array_hash(wire) for i in ti+vi}
    d=dict(config=c,schedule=s,geometry=geometry(),arms=list(ARMS),source_frozen=True,common_not_retrained=True,
        image_train_view=dict(image_ids=ti),image_validation_view=dict(image_ids=vi,source_sha256={str(i):'joint' for i in vi}),channel_config=cc,source_wire_hashes=hashes)
    put(root/'run_identity.json',dict(version=VERSION,status='complete',mode='smoke',config=c,schedule=s,TRAIN8_weights_loaded=False,text_global_milestone=9000,source_sha256={}))
    put(root/'design.json',d); put(root/'initial_state_hashes.json',initial); put(root/'initial_preservation.json',dict(passed=True))
    put(root/'frozen_verification.json',dict(passed=True,backbones_unchanged=True,old_weights_unchanged=True,original_data_unchanged=True))
    groups={k:dict(tensors=1,nonzero_gradient=True,changed=True) for k in ('image_input','text_input','image_attention','text_attention','image_output','text_output','image_gate','text_gate')}
    for phase in s['phases']:
        for arm in ARMS:
            f=root/arm/phase; hist=[]; order=np.random.default_rng(c['seed']).permutation(8)
            for j,k in enumerate(order):
                iid=ti[int(k)]; noisy=phase=='fso' and j%4!=0
                hist.append(dict(step=j+1,image_id=iid,phase=phase,noisy=noisy,proxy=[dict(payload_complex_symbols=3072,analog_repeats=1)] if noisy else [],
                    private_mask=mask_policy(c,j,iid),wire_sha256=hashes[str(iid)],observed_sha256='paired-'+str(j),pilot_snr_db=2 if noisy else 40,reliability=.5,loss=1))
            rows(f/'history.jsonl',hist); put(f/'TRAIN_COMPLETE.json',dict(steps=8,optimizer_updates=4,initial_hashes=initial,checkpoint_sha256='fixture-no-weights'))
            put(f/'learning_audit.json',dict(passed=True,frozen_unchanged=True,groups=groups,registered_parameters=1234))
            put(f/'forward_consistency.json',dict(errors=dict(image=0,text=0))); put(f/'reload_audit.json',dict(passed=True,actual_disk_reload=True,output_max_error=0))
    conditions=['clean','clean_drop_image','clean_drop_text']+[v['id'] for v in cc['conditions']]; frames=[]; events=[]; sample=[]; f=root/'evaluation'
    put(f/'eval_design.json',dict(conditions=conditions,variants=variants(s['phases']),resources=res,validation_ids=vi,channel_config=cc,same_physical_waveform_all_decoders=True))
    for cid in conditions:
        for iid in vi:
            physical=not cid.startswith('clean'); failure=physical and iid==102; pilot=2. if physical else 40.; w=wire.copy(); step={'clean_drop_image':1,'clean_drop_text':2}.get(cid)
            policy=None if step is None else mask_policy(c,step,iid)
            if policy:
                for j in policy['image_slots']: w[512+16*j:512+16*(j+1)]=0
                for j in policy['text_slots']: w[4096+64*j:4096+64*(j+1)]=0
            name='frames/{}__{}.npz'.format(cid,iid); (f/'frames').mkdir(exist_ok=True)
            np.savez_compressed(f/name,observed=w,available=np.asarray(not failure),pilot_snr_db=np.asarray(pilot),reliability=np.asarray(.5))
            stats=dict(resources=res,used_samples=res['scheduled_samples'],payload_complex_symbols=3072,analog_repeats=1,raw_utf8_wire_bytes=0,
                pre_fec_ber=None,receiver_available=not failure,pilot_snr_db=pilot) if physical else None
            frames.append(dict(condition=cid,image_id=iid,frame_file=name,observed_sha256=array_hash(w),receiver_available=not failure,pilot_snr_db=pilot,reliability=.5,stats=stats,synthetic_private_mask=policy))
            if physical: events.extend([dict(condition=cid,image_id=iid,kind=k) for k in ('probe','features')])
            for arm,phase,ctrl in variants(s['phases']):
                text='' if failure else 'a cat'; r=dict(arm=arm,phase=phase,control=ctrl,condition=cid,image_id=iid,source_split='validation',geometry=geometry(),
                    frame_file=name,observed_sha256=array_hash(w),stats=stats,pilot_snr_db=pilot,receiver_failure=failure,source_sha256='joint',
                    source_text='a cat',covered_text='a cat',received_caption=text,input_truncated=False,stopped_by_capacity=False,mse=.1,psnr=10.,ssim=.5,
                    donor_image_id=vi[(vi.index(iid)+1)%2] if ctrl.startswith('shuffle_') else None,receiver_intervention_not_new_transmission=ctrl!='normal',**metrics('a cat','a cat',text))
                sample.append(r)
    rows(f/'frames.jsonl',frames); rows(f/'events.jsonl',events); rows(f/'samples.jsonl',sample)
    put(root/'COMPLETE.json',dict(execution_only_not_quality_success=True,files=seal(root)))


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name); fixture(self.root)

    def tearDown(self): self.temp.cleanup()

    def test_positive_complete_and_sealed(self): self.assertEqual(verify(self.root)['rows'],152)

    def test_reject_missing_or_extra_decode(self):
        p=self.root/'evaluation/samples.jsonl'; p.write_text(p.read_text()+p.read_text().splitlines()[0]+'\n')
        with self.assertRaisesRegex(ValueError,'matrix'): verify(self.root,False)

    def test_reject_wrong_received_damage(self):
        p=self.root/'evaluation/samples.jsonl'; data=[json.loads(x) for x in p.read_text().splitlines()]; data[0]['observed_sha256']='other'; rows(p,data)
        with self.assertRaisesRegex(ValueError,'different waveform'): verify(self.root,False)

    def test_reject_source_feature_in_cache(self):
        p=next((self.root/'evaluation/frames').glob('*.npz'))
        with np.load(p) as z: data={k:z[k] for k in z.files}
        data['ground_truth_text']=np.asarray('a cat'); np.savez_compressed(p,**data)
        with self.assertRaisesRegex(ValueError,'source leak'): verify(self.root,False)

    def test_reject_unavailable_oracle_text(self):
        p=self.root/'evaluation/samples.jsonl'; data=[json.loads(x) for x in p.read_text().splitlines()]; r=next(r for r in data if r['receiver_failure']); r['received_caption']='a cat'; rows(p,data)
        with self.assertRaisesRegex(ValueError,'leaked text'): verify(self.root,False)

    def test_reject_different_arm_training_noise(self):
        p=self.root/'R1_text_to_image/fso/history.jsonl'; data=[json.loads(x) for x in p.read_text().splitlines()]; data[1]['observed_sha256']='not paired'; rows(p,data)
        with self.assertRaisesRegex(ValueError,'different received damage'): verify(self.root,False)

    def test_reject_seal_drift(self):
        (self.root/'untracked.txt').write_text('new')
        with self.assertRaisesRegex(ValueError,'Sealed'): verify(self.root)

    def test_finish_actual_html_and_lightweight_archive(self):
        project=Path(__file__).resolve().parents[2]; (project/'releases').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='v7_report_fixture_',dir=str(project/'releases')) as folder:
            root=Path(folder)/'records'; root.mkdir(); fixture(root)
            (root/'fixture_weights.pth').write_bytes(b'fixture-not-model')
            finish(root); self.assertTrue((root/'actual_image_text.html').is_file())
            archive=Path(folder)/'fixture.tar.gz'; collect(root,archive)
            with tarfile.open(archive) as tar:
                self.assertFalse(any(n.endswith('.pth') for n in tar.getnames()))
                self.assertTrue(any(n.endswith('COMPLETE.json') for n in tar.getnames()))


if __name__=='__main__': unittest.main(verbosity=2)
