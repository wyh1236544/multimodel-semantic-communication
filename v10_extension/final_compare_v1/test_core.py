"""Real NumPy optics + budget/codec/report negative cases; synthetic tests only."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from final_compare_v1 import VERSION, METHODS, REFERENCES
from final_compare_v1.spec import load_plan, resources, canonical_hash, sampled_count, digital_samples
from final_compare_v1.codec import encode_text, decode_text, encode_image, decode_image, choose_source, seal, TEXT
from final_compare_v1.transport import BudgetTransport
from final_compare_v1.metrics import caption_metrics, task_metrics
from final_compare_v1.report import verify, paired_ci, write_report
from final_compare_v1.collect import collect
from optical_link_v1.config import load
from optical_link_v1.wire import send_tokens


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=load_plan()[0]
        cls.spec,cls.cases,_=load()
        cls.optical=cls.cases['good']
        cls.resource=resources(cls.optical,cls.plan)
        cls.x=np.random.default_rng(17).normal(size=(288,16)).astype(np.float32)
        cls.image=np.random.default_rng(12).random((128,128,3)).astype(np.float32)
        cls.transports={}
        for method in METHODS:
            t=BudgetTransport(cls.optical,1,2,-9.,cls.resource['slot_samples'],cls.plan['laser_max_dbm'])
            t.probe()
            if method=='semantic_analog':
                y,r,stats=t.continuous(cls.x,cls.resource['analog_repeats'],'lmmse')
            elif method=='semantic_digital': y,stats=send_tokens(cls.x,t)
            else:
                t.transfer(encode_text('A test caption.'),'text',cls.plan['digital_policy'])
                t.transfer(b'image-unit-test','features',cls.plan['digital_policy'])
                stats=t.totals()
                stats.update(compressed_text_wire_bytes=len(encode_text('A test caption.')),image_wire_bytes=len(b'image-unit-test'),resource_refused_before_payload=False)
            cls.transports[method]=(t,stats)

    def test_text_exact_unicode(self):
        text='A cyclist 骑自行车。\n UTF-8 ✓'
        self.assertEqual(decode_text(encode_text(text)),(text,True))

    def test_text_crc_erases(self):
        data=bytearray(encode_text('caption')); data[-1]^=1
        self.assertEqual(decode_text(data),('',False))

    def test_text_length_rejects(self):
        data=seal(TEXT.pack(b'TXF1',65537)+b'bad')
        self.assertEqual(decode_text(data),('',False))
        with self.assertRaises(ValueError): encode_text('x'*65537)

    def test_image_codec_and_independent_text(self):
        data=encode_image(self.image,50,2); result,ok=decode_image(data)
        self.assertTrue(ok); self.assertEqual(result.shape,(128,128,3))
        result,ok=decode_image(data[:-1]); self.assertFalse(ok); self.assertTrue(np.all(result==.5))
        self.assertEqual(decode_text(encode_text('untouched')),('untouched',True))

    def test_exact_resource_formula(self):
        self.assertEqual(self.resource['slot_samples'],256000)
        self.assertEqual(self.resource['analog_repeats'],27)
        t,_=self.transports['semantic_digital']
        self.assertEqual(sum(e['awg_samples']+e['guard_samples'] for e in t.events),self.resource['semantic_digital_samples'])

    def test_codec_budget_selects_highest_fitting(self):
        c=choose_source(self.image,'a text',self.optical,self.plan,self.resource['slot_samples'])
        self.assertTrue(c['budget_fits']); self.assertLessEqual(c['predicted_samples'],self.resource['slot_samples'])
        if c['jpeg_quality']<95:
            higher=encode_image(self.image,c['jpeg_quality']+1,2)
            self.assertGreater(digital_samples([len(c['text']),len(higher)],self.optical['waveform'],self.plan['digital_policy']),self.resource['slot_samples'])

    def test_codec_no_budget_is_preserved(self):
        c=choose_source(self.image,'a text',self.optical,self.plan,1)
        self.assertFalse(c['budget_fits']); self.assertIsNone(c['image'])

    def test_same_probe_and_fading(self):
        values=[t.path_draw for t,s in self.transports.values()]
        self.assertTrue(all(v==values[0] for v in values))
        probes=[t.events[0]['estimated_snr_db'] for t,s in self.transports.values()]
        self.assertEqual(len(set(probes)),1)

    def test_same_launch_power_and_energy_cap(self):
        for t,s in self.transports.values():
            self.assertLessEqual(s['used_samples'],s['slot_samples'])
            self.assertLessEqual(s['launch_signal_energy_j'],s['signal_energy_budget_j']*(1+1e-8))
            for e in t.events: self.assertLess(e['launch_power_relative_error'],1e-8)

    def test_budget_rejects_before_transmit(self):
        t=BudgetTransport(self.optical,1,2,-9.,sampled_count(0,self.optical['waveform']),20.)
        t.probe()
        with self.assertRaises(ValueError): t.burst(np.ones(2,complex),'features')
        self.assertEqual(len(t.events),1)

    def test_infeasible_power_is_not_clamped(self):
        t=BudgetTransport(self.optical,1,2,10.,self.resource['slot_samples'],-40.)
        with self.assertRaises(ValueError): t.probe()
        self.assertFalse(t.events)

    def test_analog_not_binary_delivery(self):
        t,s=self.transports['semantic_analog']
        self.assertIsNone(s['pre_fec_ber']); self.assertIsNone(s['feature_object_crc_accepted'])
        self.assertEqual(t.events[1]['data_symbols'],27*2304)

    def test_incompatible_dsp_rejected(self):
        c=copy.deepcopy(self.optical); c['receiver']['carrier_recovery']='pilot_cfo_dd'
        with self.assertRaises(ValueError): resources(c,self.plan)

    def test_all_public_launch_points_are_feasible(self):
        for c in self.plan['conditions']:
            optical=self.cases[c['case']]
            t=BudgetTransport(optical,1,2,c['launch_signal_dbm'],self.resource['slot_samples'],self.plan['laser_max_dbm'])
            self.assertTrue(np.isfinite(t.probe()))

    def test_invalid_plan(self):
        p=copy.deepcopy(self.plan); p['conditions'].append(copy.deepcopy(p['conditions'][0]))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plan.json'; path.write_text(json.dumps(p))
            with self.assertRaises(ValueError): load_plan(path)

    def test_text_metrics_do_not_hide_punctuation(self):
        m=caption_metrics('A cat.','a CAT')
        self.assertEqual(m['normalized_wer'],0.); self.assertFalse(m['text_exact']); self.assertGreater(m['wer'],0.)

    def test_image_cluster_pairing(self):
        rows=[dict(condition='c',image_id=i,method=m,psnr=20+int(m=='a'),normalized_wer=.2,joint_task_success=False)
              for i in range(3) for repeat in range(3) for m in ('a','b')]
        r=paired_ci(rows,'a','b','c',100)
        self.assertEqual(r['independent_images'],3); self.assertEqual(r['psnr_ci95'],[1.,1.])

    def fixture(self,d):
        plan=copy.deepcopy(self.plan); plan['conditions']=[dict(id='good_p09',case='good',launch_signal_dbm=-9.)]
        design=dict(version=VERSION,split='validation',trained=False,val_samples=1,draws=1,expected_rows=6,
            plan=plan,optical_config=self.spec,effective_cases=self.cases,conditions=plan['conditions'])
        identity=dict(plan_sha256=canonical_hash(plan),optical_config_sha256=canonical_hash(self.spec),status='running')
        for name,value in [('design.json',design),('run_identity.json',identity),('validation_ids.json',[1])]:
            (d/name).write_text(json.dumps(value))
        rows=[]; events=[]
        for method in REFERENCES+METHODS:
            reference=method in REFERENCES
            row=dict(image_id=1,condition='clean' if reference else 'good_p09',draw=0,method=method,reference=reference,
                psnr=20.,ssim=.8,source_caption_evaluation_only='A test caption.',received_caption='A test caption.',
                source_caption_sha256=hashlib.sha256(b'A test caption.').hexdigest(),source_rgb_sha256='synthetic',normalized_payload_sha256=hashlib.sha256(self.x.astype('<f4').tobytes()).hexdigest(),
                nominal_snr_db=None,**caption_metrics('A test caption.','A test caption.'))
            row.update(task_metrics(row,plan))
            if not reference:
                t,s=self.transports[method]; row.update(s,event_count=len(t.events),routing_probe_snr_db=t.events[0]['estimated_snr_db'])
                for e in t.events: events.append(dict(e,image_id=1,condition='good_p09',draw=0,method=method))
            rows.append(row)
        for name,values in [('samples.jsonl',rows),('packets.jsonl',events)]:
            (d/name).write_text(''.join(json.dumps(r)+'\n' for r in values))
        return rows

    def test_independent_matrix_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); self.fixture(d); verify(d)
            result=write_report(d)
            self.assertEqual(len(result['groups']),6)
            self.assertIn('原文', (d/'report.html').read_text(encoding='utf-8'))

    def test_incomplete_matrix_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); rows=self.fixture(d)
            (d/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:-1]))
            with self.assertRaises(ValueError): verify(d)

    def test_duplicate_matrix_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); rows=self.fixture(d); rows[-1]=rows[0]
            (d/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError): verify(d)

    def test_wrong_analog_ber_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); rows=self.fixture(d); rows[-1]['pre_fec_ber']=0.
            (d/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError): verify(d)

    def complete_fixture(self,d):
        self.fixture(d); write_report(d)
        (d/'checkpoint_load.json').write_text(json.dumps(dict(synthetic_test_only=True)))
        identity=json.loads((d/'run_identity.json').read_text()); identity['status']='complete'
        (d/'run_identity.json').write_text(json.dumps(identity))
        names=['design.json','validation_ids.json','checkpoint_load.json','samples.jsonl','packets.jsonl','summary.json','REPORT.md','report.html']
        names += [p.name for p in d.glob('*.svg')]
        (d/'COMPLETE.json').write_text(json.dumps(dict(version=VERSION,rows=6,result_sha256={n:hashlib.sha256((d/n).read_bytes()).hexdigest() for n in names})))

    def test_complete_hash_change_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); self.complete_fixture(d); verify(d,require_complete=True)
            with (d/'samples.jsonl').open('a') as stream: stream.write('\n')
            with self.assertRaises(ValueError): verify(d,require_complete=True)

    def test_completed_collection(self):
        parent=Path(__file__).resolve().parents[2]/'experiment_results/final_compare_v1_checks'
        parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(parent)) as tmp:
            root=Path(tmp); d=root/'synthetic'; d.mkdir(); self.complete_fixture(d)
            archive=collect(d,root/'synthetic_records.tar.gz')
            self.assertTrue(archive.is_file()); self.assertTrue(archive.with_name(archive.name+'.sha256').is_file())
            with self.assertRaises(FileExistsError): collect(d,archive)

    def test_collection_refuses_unexpected_file(self):
        parent=Path(__file__).resolve().parents[2]/'experiment_results/final_compare_v1_checks'
        parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(parent)) as tmp:
            root=Path(tmp); d=root/'synthetic'; d.mkdir(); self.complete_fixture(d)
            (d/'unexpected.json').write_text('{}')
            with self.assertRaises(ValueError): collect(d,root/'must_not_exist.tar.gz')
            self.assertFalse((root/'must_not_exist.tar.gz').exists())


if __name__=='__main__': unittest.main(verbosity=2)
