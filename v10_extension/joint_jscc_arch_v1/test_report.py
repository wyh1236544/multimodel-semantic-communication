"""Negative evidence tests without Torch or real model outputs."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from joint_jscc_arch_v1.report import validate_row,contrast,verify,report
from joint_jscc_arch_v1.core import load,digest,arm_options
from joint_jscc_arch_v1 import VERSION,ARMS,TRAINED_ARMS


class ReportContracts(unittest.TestCase):
    def sample(self,arm='a',iid=1,condition='fso',draw=0,psnr=20.,available=True):
        return dict(arm=arm,image_id=iid,condition=condition,draw=draw,psnr=psnr,ssim=.7,
            normalized_wer=.4,covered_normalized_wer=.2,cer=.3,waveform_papr_db=10.,
            analog_repeats=1,raw_utf8_wire_bytes=0,payload_complex_symbols=2304,
            pre_fec_ber=None,post_fec_ber=None,receiver_available=available,
            neural_input_capacity_tokens=128,covered_content_tokens=126)

    def test_reject_hidden_repeat_oracle_fake_ber_and_nonfinite(self):
        row=self.sample(); validate_row(row)
        for key,value in [('analog_repeats',2),('raw_utf8_wire_bytes',1),('payload_complex_symbols',2305),
                          ('pre_fec_ber',.1),('psnr',float('nan')),('covered_content_tokens',128)]:
            bad=copy.deepcopy(row); bad[key]=value
            with self.assertRaises(ValueError): validate_row(bad)

    def test_failures_retained_and_images_not_repeats_are_resampled(self):
        rows=[]
        for iid in (1,2):
            for draw in range(3):
                rows.append(self.sample('a',iid,draw=draw,psnr=20.,available=draw!=2))
                rows.append(self.sample('b',iid,draw=draw,psnr=21. if draw!=2 else 17.,available=draw!=2))
        all_frames=contrast(rows,('a','b'),['fso'],'psnr',200)
        available=contrast(rows,('a','b'),['fso'],'psnr',200,True)
        self.assertEqual((all_frames['images'],all_frames['trials']),(2,6))
        self.assertAlmostEqual(all_frames['mean'],-1/3)
        self.assertEqual(available['trials'],4); self.assertAlmostEqual(available['mean'],1.)

    def test_incomplete_pair_not_silently_filtered(self):
        with self.assertRaises(KeyError): contrast([self.sample()],('a','b'),['fso'],'psnr',200)

    def fixture(self,d):
        """SYNTHETIC audit data only, never evidence of neural/physical performance."""
        cfg,cc=load(); cfg.update(train_samples=8,clean_epochs=1,noise_epochs=1,val_samples=2,draws=1,save_images=0)
        def save(name,value):
            file=d/name; file.parent.mkdir(parents=True,exist_ok=True); file.write_text(json.dumps(value,allow_nan=False))
        def lines(name,rows):
            file=d/name; file.parent.mkdir(parents=True,exist_ok=True); file.write_text('\n'.join(json.dumps(r,allow_nan=False) for r in rows)+'\n')
        design=dict(version=VERSION,config=cfg,channel_config=cc,arms=list(ARMS),split='validation',train_ids=list(range(1,9)),
                    validation_ids=[101,102],validation_source_sha256={'101':'source101','102':'source102'},conditions=cc['conditions'],
                    train_forward_steps=16,optimizer_steps=8,primary_conditions=cc['primary_conditions'],
                    environment=[None,None,dict(slot_samples=1000)],limitations='SYNTHETIC FIXTURE ONLY')
        save('design.json',design)
        bank=[]
        for c in cc['conditions']:
            for _ in range(6): bank.append(dict(index=len(bank),partition='holdout',condition=c['id'],available=True))
        for c in cc['conditions']: bank.append(dict(index=len(bank),partition='train',condition=c['id'],available=True))
        save('calibration_summary.json',dict(records=bank))
        counts={n:10 for n in ('source_transformer.weight','text_compressor.queries','language.output.bias',
            'old_tx.weight','old_rx.weight','modern_tx.down.weight','modern_rx.down.weight','spatial_tx.net.weight',
            'spatial_rx.net.weight','rx_condition.0.weight','rx_transformer.weight','common_image_head.weight')}
        save('checkpoint_load.json',dict(parameter_counts=counts))
        for arm in TRAINED_ARMS:
            o=arm_options(arm); folder='train_'+arm+'/'
            frozen_prefixes=('modern_tx.','modern_rx.','spatial_tx.','spatial_rx.','rx_condition.','missing_token')
            eligible={n for n in counts if o['modern'] or not n.startswith(frozen_prefixes)}
            required=['source','text_compression','ar','tx','rx','receiver']+(['conditioning','tx_spatial','rx_spatial'] if o['modern'] else [])+(['common_coarse'] if o['auxiliary'] else [])
            save(folder+'COMPLETE.json',dict(arm=arm,steps=16,optimizer_steps=8,checkpoint_roundtrip_verified=True,frozen_backbones=True,
                initial_core_verified=True,required_groups=required,parameter_group_changes={k:1. for k in required},
                trainable_parameters=sum(counts[n] for n in eligible),registered_parameters=sum(counts.values()),
                observed_nonzero_gradient_names=sorted(eligible),observed_nonzero_gradient_parameters=sum(counts[n] for n in eligible),
                parameter_changes={n:float(n in eligible) for n in counts},checkpoint_sha256='synthetic_fso',clean_checkpoint_sha256='synthetic_clean'))
            history=[]
            for phase in ('clean','fso'):
                for iid in range(1,9):
                    history.append(dict(phase=phase,epoch=1,image_id=iid,forward_step=len(history)+1,bank_index=None if phase=='clean' else 36,
                                        anchor=phase=='clean',task_loss_applied=True,noise_seed=iid,auxiliary_applied=o['auxiliary']))
            lines(folder+'history.jsonl',history)
            for stage in ('initial','clean','fso'):
                raw=[dict(condition=r['condition'],calibration_index=r['index'],source_image_id=1,available_fit=True,
                          available_fresh=True,actual_evm2=.1,predicted_evm2=.105,actual_launch_signal_energy_j=1e-12) for r in bank if r['partition']=='holdout']
                save(folder+'fidelity_'+stage+'.json',dict(stage=stage,passed=True,source_split='train',public_bank_partition='holdout',rows=raw,
                    groups={c['id']:dict(passed=True,both_available=6) for c in cc['conditions']}))
        rows=[]; events=[]
        for arm in ARMS:
            for condition in ['clean']+[c['id'] for c in cc['conditions']]:
                for index,iid in enumerate((101,102)):
                    r=self.sample(arm,iid,condition,psnr=20.+ARMS.index(arm)*.1)
                    r.update(frame_index=index,source_sha256='source'+str(iid),noise_seed=iid,weather_path=None if condition=='clean' else {'index':index},
                             received_caption='synthetic',source_caption_evaluation_only='synthetic fixture',input_truncated=False)
                    if condition!='clean':
                        r.update(event_count=2,awg_samples=200,guard_samples=0)
                        for purpose in ('probe','features'):
                            events.append(dict(arm=arm,image_id=iid,condition=condition,draw=0,purpose=purpose,
                                awg_samples=100,guard_samples=0,data_symbols=2304,receiver_locked=True,
                                launch_signal_energy_j=1e-12,launch_peak_signal_bound_w=1.,waveform_seconds=1e-9))
                    rows.append(r)
        lines('samples.jsonl',rows); lines('events.jsonl',events)
        lines('clean_stage.jsonl',[dict(arm=a,image_id=i,psnr=20.,covered_normalized_wer=.2) for a in TRAINED_ARMS for i in (101,102)])
        save('frozen_verification.json',dict(all_legacy_parameters_unchanged=True,VAE_CLIP_unchanged=True,no_test_split_used=True))
        save('run_identity.json',dict(version=VERSION,status='running',calibration_sha256=digest(d/'calibration_summary.json'),
             new_source_sha256={'report.py':digest(Path(__file__).with_name('report.py'))},source_sha256={},calibration_source_sha256={}))
        return save,lines

    def test_full_synthetic_verifier_report_and_completion_ledger(self):
        with tempfile.TemporaryDirectory() as root:
            d=Path(root); save,_=self.fixture(d)
            self.assertEqual(len(verify(d)),70); report(d)
            identity=json.loads((d/'run_identity.json').read_text()); identity['status']='complete'; save('run_identity.json',identity)
            ledger={str(f.relative_to(d)):digest(f) for f in d.rglob('*') if f.is_file()}
            checkpoints={'train_'+a+'/'+p+'_delta.pth':'synthetic_'+p for a in TRAINED_ARMS for p in ('clean','fso')}
            save('COMPLETE.json',dict(version=VERSION,rows=70,result_sha256=ledger,checkpoint_sha256=checkpoints))
            self.assertEqual(len(verify(d,complete=True)),70)
            (d/'REPORT.md').write_text('tampered')
            with self.assertRaises(ValueError): verify(d,complete=True)

    def test_full_verifier_rejects_scale_holdout_and_false_fidelity(self):
        for kind in ('scale','holdout','fidelity','missing'):
            with tempfile.TemporaryDirectory() as root:
                d=Path(root); save,lines=self.fixture(d)
                if kind=='scale':
                    r=json.loads((d/'train_B4_separate/COMPLETE.json').read_text()); r['trainable_parameters']+=1; save('train_B4_separate/COMPLETE.json',r)
                elif kind=='holdout':
                    rr=[json.loads(s) for s in (d/'train_B1_text/history.jsonl').read_text().splitlines()]; rr[-1]['bank_index']=0; lines('train_B1_text/history.jsonl',rr)
                elif kind=='fidelity':
                    r=json.loads((d/'train_B3_common/fidelity_fso.json').read_text())
                    for row in r['rows']: row['predicted_evm2']=3.
                    save('train_B3_common/fidelity_fso.json',r)
                else:
                    rr=[json.loads(s) for s in (d/'samples.jsonl').read_text().splitlines()]; lines('samples.jsonl',rr[:-1])
                with self.assertRaises(ValueError): verify(d)


if __name__=='__main__': unittest.main(verbosity=2)
