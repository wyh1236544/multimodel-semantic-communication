"""Synthetic neural metrics over real NumPy link records; audit/HTML regression."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from fso_fluctuating_v2 import VERSION, ARMS
from fso_fluctuating_v2.core import load, paths, transport, public_payload, environment, digest
from fso_fluctuating_v2.report import verify, report


class ReportContract(unittest.TestCase):
    def test_synthetic_full_report_and_negative_matrix(self):
        with tempfile.TemporaryDirectory(dir=str(Path(__file__).parent)) as temp:
            d=Path(temp); c=load(); c.update(val_samples=2,draws=1,train_samples=8,epochs=1)
            def save(n,value):
                (d/n).parent.mkdir(parents=True,exist_ok=True)
                (d/n).write_text(json.dumps(value))
            save('calibration_summary.json',dict(synthetic=True))
            save('design.json',dict(version=VERSION,config=c,arms=ARMS,split='validation',
                train_ids=list(range(11,19)),validation_ids=[1,2],validation_source_sha256={'1':'one','2':'two'},
                train_forward_steps=8,optimizer_steps=4,environment=environment(c)))
            save('run_identity.json',dict(version=VERSION,status='running',new_source_sha256={},source_sha256={},calibration_source_sha256={},
                calibration_sha256=digest(d/'calibration_summary.json')))
            for arm in ARMS:
                trained=arm!=ARMS[0]
                save('train_'+arm+'/COMPLETE.json',dict(arm=arm,steps=8 if trained else 0,optimizer_steps=4 if trained else 0,
                    gradient_steps=dict(source_encoder=1,tx_mapping=1,rx_mapping=1,decoder=1),
                    parameter_changes={n:1. for n in ('base.multimodal_encoder.a','base.channel.tx_map.a','base.channel.rx_map.a','base.joint_decoder.a')}))
                if trained:
                    history=[dict(epoch=1,image_id=i+11,bank_index=i,anchor=False,task_loss_applied=True,forward_step=i+1) for i in range(8)]
                    (d/('train_'+arm)/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
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
            self.assertIn('Low-quality fluctuating', (d/'report.html').read_text())
            with (d/'samples.jsonl').open('a') as f: f.write(json.dumps(rows[0])+'\n')
            with self.assertRaisesRegex(ValueError,'Duplicate'): verify(d)


if __name__=='__main__': unittest.main(verbosity=2)
