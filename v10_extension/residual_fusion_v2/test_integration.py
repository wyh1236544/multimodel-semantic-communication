"""Synthetic complete/diagnostic archives; requires the built source MANIFEST."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from residual_fusion_v2 import VERSION,ARMS
from residual_fusion_v2.core import load,digest,parameter_counts,augmentation,donor_index,diagnostic_decision
from residual_fusion_v2.report import report,verify
from residual_fusion_v2.test_report import DiagnosticProofs
from residual_fusion_v2.collect import collect
from final_compare_v1.metrics import caption_metrics


def fixture(root,complete):
    design=DiagnosticProofs().fixture(root)
    c=load(); c.update(train_samples=8,clean_epochs=1,noise_epochs=1,val_samples=2,save_images=0,diagnostic_samples=2,diagnostic_updates=2,residual_hidden=16)
    cc=json.loads((EXT/'fso_fluctuating_v2/config.json').read_text())
    design.update(version=VERSION,config=c,channel_config=cc,architecture_config=dict(fidelity_frames_per_condition=6),
        arms=list(ARMS),split='validation',train_ids=list(range(1,9)),validation_ids=[101,102],
        validation_source_sha256={'101':'src101','102':'src102'},conditions=[cc['conditions'][1]],primary_conditions=[cc['conditions'][1]['id']],
        train_forward_steps=16,optimizer_steps=8,environment=[None,None,dict(slot_samples=100000)],
        limitations='SYNTHETIC TEST, no model performance')
    status='complete' if complete else 'diagnostic_stop'; folder=Path(__file__).parent
    identity=dict(version=VERSION,status=status,new_source_sha256={f.name:digest(f) for f in folder.iterdir() if f.suffix in ('.py','.json','.md')},source_sha256={})
    if complete:
        p=root/'text_diagnostics/samples.jsonl'; rows=[json.loads(s) for s in p.read_text().splitlines()]
        for r in rows: r.update(received_caption='a cat',**caption_metrics('a cat','a cat'))
        p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        for label in ('D0_precompression','D1_T32_clean'):
            p=root/'text_diagnostics'/label/'COMPLETE.json'; rec=json.loads(p.read_text()); rec['actual_covered_wer']=0.; p.write_text(json.dumps(rec))
        (root/'text_diagnostics/gate.json').write_text(json.dumps(diagnostic_decision({'D0_precompression':0.,'D1_T32_clean':0.},.1)))
    for label in ('D0_precompression','D1_T32_clean'):
        p=root/'text_diagnostics'/label/'COMPLETE.json'; rec=json.loads(p.read_text()); rec['checkpoint_sha256']='diagnostic-'+label; p.write_text(json.dumps(rec))
    bank=dict(records=[dict(index=0,partition='train',available=True)])
    fidelity=[]
    for cond in cc['conditions']:
        for j in range(6):
            ix=len(bank['records']); bank['records'].append(dict(index=ix,partition='holdout',condition=cond['id'],available=True))
            fidelity.append(dict(condition=cond['id'],calibration_index=ix,source_image_id=1,available_fit=True,available_fresh=True,
                actual_evm2=.1,predicted_evm2=.1,actual_launch_signal_energy_j=1e-12))
    (root/'calibration_summary.json').write_text(json.dumps(bank)); identity['calibration_sha256']=digest(root/'calibration_summary.json')
    (root/'design.json').write_text(json.dumps(design)); (root/'run_identity.json').write_text(json.dumps(identity))
    (root/'checkpoint_load.json').write_text('{}')
    (root/'frozen_verification.json').write_text(json.dumps(dict(all_B4_parameters_unchanged=True,all_legacy_parameters_unchanged=True,
        all_B4_gradients_none=True,all_legacy_gradients_none=True,no_test_split_used=True)))
    if complete:
        (root/'fidelity_fixed_B4.json').write_text(json.dumps(dict(stage='fixed_B4',passed=True,rows=fidelity)))
        counts=parameter_counts(256,16); initial={n:'same' for n in counts}
        (root/'residual_initial.json').write_text(json.dumps(dict(parameter_counts=counts,initial_parameter_sha256=initial)))
        for arm in ('R0_image','R1_joint'):
            sub=root/('train_'+arm); sub.mkdir(); history=[]
            for phase in ('clean','fso'):
                for pos in range(8):
                    step=len(history)+1; aug=augmentation(c,phase,0,pos)
                    history.append(dict(phase=phase,epoch=1,image_id=pos+1,forward_step=step,optimizer_step=step//2,
                        augmentation=aug,donor_id=donor_index(pos,8)+1 if aug=='shuffle' else None,noise_seed=pos,
                        bank_index=None if phase=='clean' else 0,task_loss_applied=True,pixel_mse=.01,base_pixel_mse=.01,
                        latent_mse=.01,loss=.01,gate=.1))
            (sub/'history.jsonl').write_text('\n'.join(json.dumps(r) for r in history)+'\n')
            (sub/'COMPLETE.json').write_text(json.dumps(dict(arm=arm,steps=16,optimizer_steps=8,checkpoint_roundtrip_verified=True,
                trainable_parameters=sum(counts.values()),initialization_sha256=initial,parameter_changes={n:.01 for n in counts},
                observed_gradient_names=list(counts),checkpoint_sha256='checkpoint-'+arm)))
        rows=[]; events=[]; empty=np.zeros((1,48,256),np.float32)
        zsha=hashlib.sha256(str(empty.dtype).encode()+json.dumps(list(empty.shape)).encode()+empty.tobytes()).hexdigest()
        for cond in ('clean',cc['conditions'][1]['id']):
            for i in (101,102):
                frame='{}__{}__d0'.format(i,cond)
                for arm in ARMS:
                    donor=102 if i==101 else 101
                    memory=None if arm in ('B4_frozen','R1_off') else zsha if arm=='R1_zero_text' else 'memory'+str(donor if arm=='R1_shuffled_text' else i)
                    r=dict(arm=arm,frame_key=frame,image_id=i,condition=cond,draw=0,source_sha256='src'+str(i),
                        received_sha256='rx'+str(i),payload_sha256='tx'+str(i),receiver_available=True,
                        payload_complex_symbols=2304,analog_repeats=1,raw_utf8_wire_bytes=0,received_caption='a cat',
                        source_caption_evaluation_only='a cat',covered_normalized_wer=0.,latent_sha256='base'+str(i),
                        fusion_memory_sha256=memory,fusion_donor_id=donor if arm=='R1_shuffled_text' else None,
                        fusion_gate=0. if arm in ('B4_frozen','R1_off') else .1,pre_fec_ber=None,post_fec_ber=None,psnr=20.,ssim=.8,
                        **caption_metrics('a cat','a cat'))
                    if cond!='clean': r.update(awg_samples=10400,guard_samples=64,scheduled_seconds=10464/cc['fs_hz'],launch_signal_energy_j=2e-12)
                    rows.append(r)
                if cond!='clean':
                    for purpose,count in (('probe',400),('features',10000)):
                        events.append(dict(frame_key=frame,purpose=purpose,awg_samples=count,guard_samples=32,
                            data_symbols=0 if purpose=='probe' else 2304,receiver_locked=True,launch_signal_energy_j=1e-12,
                            waveform_seconds=count/cc['fs_hz'],launch_peak_signal_bound_w=.1))
        (root/'samples.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        (root/'events.jsonl').write_text('\n'.join(json.dumps(r) for r in events)+'\n')
    report(root,status)
    names=[f for f in root.rglob('*') if f.is_file()]
    checks={'text_diagnostics/'+label+'/decoder_delta.pth':'diagnostic-'+label for label in ('D0_precompression','D1_T32_clean')}
    if complete: checks.update({'train_'+arm+'/last_delta.pth':'checkpoint-'+arm for arm in ('R0_image','R1_joint')})
    marker='COMPLETE.json' if complete else 'DIAGNOSTIC_COMPLETE.json'
    (root/marker).write_text(json.dumps(dict(version=VERSION,status=status,result_sha256={str(f.relative_to(root)):digest(f) for f in names},checkpoint_sha256=checks)))
    return root


class Integration(unittest.TestCase):
    def temp(self):
        folder=EXT.parent/'releases'; folder.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix='residual_synthetic_',dir=str(folder))
    def test_complete_verify_and_light_collection(self):
        with self.temp() as t:
            root=Path(t)/'run'; root.mkdir(); fixture(root,True); self.assertEqual(len(verify(root)),24)
            collect(root,Path(t)/'synthetic.tar.gz')
    def test_diagnostic_stop_light_collection(self):
        with self.temp() as t:
            root=Path(t)/'run'; root.mkdir(); fixture(root,False); self.assertEqual(verify(root),[])
            collect(root,Path(t)/'synthetic.tar.gz')
    def mutate(self,field,value):
        with self.temp() as t:
            root=Path(t)/'run'; root.mkdir(); fixture(root,True); p=root/'samples.jsonl'; rows=[json.loads(s) for s in p.read_text().splitlines()]
            r=next(r for r in rows if r['condition']!='clean' and r['arm']=='R1_joint'); r[field]=value
            p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            with self.assertRaises(ValueError): verify(root)
    def test_mismatched_receipt_refused(self): self.mutate('received_sha256','wrong')
    def test_free_time_refused(self): self.mutate('scheduled_seconds',0.)
    def test_wrong_energy_refused(self): self.mutate('launch_signal_energy_j',0.)
    def test_changed_raw_artifact_refused(self):
        with self.temp() as t:
            root=Path(t)/'run'; root.mkdir(); fixture(root,False); (root/'REPORT.md').write_text('changed')
            with self.assertRaises(ValueError): verify(root)


if __name__=='__main__': unittest.main(verbosity=2)
