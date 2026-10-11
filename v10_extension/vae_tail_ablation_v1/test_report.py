"""Synthetic neural metrics over real NumPy link records; audit/HTML regression."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from vae_tail_ablation_v1 import VERSION, ARMS
from vae_tail_ablation_v1.core import load, paths, transport, environment, digest,tail_prefixes
from fso_fluctuating_v2.core import public_payload
from vae_tail_ablation_v1.report import verify, report


class ReportContract(unittest.TestCase):
    def test_synthetic_full_report_and_negative_matrix(self):
        with tempfile.TemporaryDirectory(dir=str(Path(__file__).parent)) as temp:
            d=Path(temp); c=load(); c.update(val_samples=2,draws=1,train_samples=8,epochs=1,primary_validation_offset=0,save_images=0)
            def save(n,value):
                (d/n).parent.mkdir(parents=True,exist_ok=True)
                (d/n).write_text(json.dumps(value))
            save('calibration_summary.json',dict(synthetic=True))
            prefixes=tail_prefixes(4)
            candidates={n:1 for n in ('base.multimodal_encoder.a','base.channel.tx_map.a','base.channel.rx_map.a','base.joint_decoder.a')}
            candidates.update({n+'a':1 for n in prefixes})
            save('checkpoint_load.json',dict(candidate_parameters=candidates,vae_tail_prefixes=prefixes,
                semantic_parameters=4,vae_tail_parameters=3,trainable_per_arm={ARMS[0]:0,ARMS[1]:4,ARMS[2]:7},CLIP_and_VAE_encoder_frozen=True))
            save('design.json',dict(version=VERSION,config=c,arms=ARMS,split='validation',
                train_ids=list(range(11,19)),validation_ids=[1,2],validation_source_sha256={'1':'one','2':'two'},
                primary_validation_ids=[1,2],vae_tail_prefixes=prefixes,
                train_forward_steps=8,optimizer_steps=4,environment=environment(c)))
            save('run_identity.json',dict(version=VERSION,status='running',new_source_sha256={},source_sha256={},calibration_source_sha256={},
                calibration_sha256=digest(d/'calibration_summary.json')))
            for arm in ARMS:
                trained=arm!=ARMS[0]
                changes={n:(1. if trained and (not n.startswith(prefixes) or arm==ARMS[2]) else 0.) for n in candidates}
                save('train_'+arm+'/COMPLETE.json',dict(arm=arm,steps=8 if trained else 0,optimizer_steps=4 if trained else 0,
                    gradient_steps=dict(source_encoder=int(trained),tx_mapping=int(trained),rx_mapping=int(trained),decoder=int(trained),vae_tail=int(arm==ARMS[2])),
                    parameter_changes=changes,frozen_parameters_unchanged=True,vae_tail_gradient_steps=int(arm==ARMS[2]),
                    checkpoint_roundtrip_verified=trained,
                    vae_tail_parameter_change=sum(v for n,v in changes.items() if n.startswith(prefixes)),
                    training_peak_allocated_gib=None,training_peak_reserved_gib=None,gpu_name='SYNTHETIC',wall_seconds=1.))
                if trained:
                    history=[dict(epoch=1,image_id=i+11,bank_index=i,anchor=False,task_loss_applied=True,forward_step=i+1,
                        network_seed=i,noise_seed=i,vae_tail_trainable=arm==ARMS[2]) for i in range(8)]
                    (d/('train_'+arm)/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
            diagnostics=[dict(arm=a,image_id=i,source_sha256='one' if i==1 else 'two',
                cached_vae_latent_sha256=str(i),fixed_reference_latent_sha256=str(i),
                direct_vae_psnr=24.,direct_vae_ssim=.8,fixed_U0_semantic_latent_psnr=20.+int(a==ARMS[2]),fixed_U0_semantic_latent_ssim=.7,
                direct_reference_not_mathematical_upper_bound=True) for a in ARMS for i in (1,2)]
            (d/'diagnostics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in diagnostics))
            rows=[]; events=[]
            for condition in [dict(id='clean',weather='weak')]+c['conditions']:
                trace=paths(c,condition['weather'],2,'synthetic-test')
                for ix,image_id in enumerate((1,2)):
                    if condition['id']=='clean':
                        stats=dict(analog_repeats=1,raw_utf8_wire_bytes=0,pre_fec_ber=None,post_fec_ber=None,
                            payload_complex_symbols=2304,scheduled_seconds=2560/16e9); ev=[]; valid=True; path=None
                    else:
                        t=transport(c,condition,trace[ix],ix+15); t.probe(); _,_,stats=t.continuous(public_payload(ix),1,'lmmse')
                        ev=t.events; valid=stats['analog_receiver_available']; path=trace[ix]
                    for arm in ARMS:
                        rows.append(dict(stats,arm=arm,image_id=image_id,condition=condition['id'],draw=0,frame_index=ix,
                            source_sha256='one' if image_id==1 else 'two',noise_seed=ix+15,weather_path=path,receiver_available=valid,
                            psnr=20.+ARMS.index(arm),ssim=.6,normalized_wer=.7,cer=.7,token_acc=.2,common_to_text_cosine=.3,
                            event_count=len(ev),received_caption='synthetic',source_caption_evaluation_only='synthetic reference'))
                        events.extend(dict(e,arm=arm,image_id=image_id,condition=condition['id'],draw=0) for e in ev)
            (d/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (d/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
            self.assertEqual(len(verify(d)),42)
            report(d)
            self.assertTrue((d/'quality_timeline.svg').read_text().startswith('<svg'))
            self.assertIn('Partial VAE decoder', (d/'report.html').read_text())
            # A changed cached latent would invalidate the paired ablation.
            damaged=copy.deepcopy(diagnostics); damaged[-1]['cached_vae_latent_sha256']='different'
            (d/'diagnostics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in damaged))
            with self.assertRaisesRegex(ValueError,'Unpaired reference latent'): verify(d)
            (d/'diagnostics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in diagnostics))
            bad=json.loads((d/('train_'+ARMS[1])/'COMPLETE.json').read_text())
            bad['vae_tail_gradient_steps']=1
            save('train_'+ARMS[1]+'/COMPLETE.json',bad)
            with self.assertRaisesRegex(ValueError,'Tail audit differs'): verify(d)
            bad['vae_tail_gradient_steps']=0; save('train_'+ARMS[1]+'/COMPLETE.json',bad)
            with (d/'samples.jsonl').open('a') as f: f.write(json.dumps(rows[0])+'\n')
            with self.assertRaisesRegex(ValueError,'Duplicate'): verify(d)


if __name__=='__main__': unittest.main(verbosity=2)
