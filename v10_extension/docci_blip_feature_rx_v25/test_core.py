"""Synthetic CPU protocol/I/O regressions. Never real neural performance."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT),str(EXT.parent)]
from docci_blip_feature_rx_v25 import core as c
from docci_blip_feature_rx_v25 import report as r

def text_hash(s): return hashlib.sha256(s.encode()).hexdigest()

def fixture(base):
    """EXPLICIT TEST_ONLY fake model/weight receipts, no Torch execution."""
    from docci_caption_calibration_v24.core import token_audit,text_audit
    from docci_text_continue_v11.metrics import metrics
    root = base/'pipeline'; root.mkdir(); cache = root/'cache'; cache.mkdir()
    (cache/'arrays').mkdir();(cache/'images').mkdir()
    cfg = c.config('smoke');c.atomic(cache/'config.json',cfg)
    entries = [];caps = []
    for i in range(10):
        eid = 'TEST_ONLY_'+str(i); cohort = 'train' if i<8 else 'dev'
        source = 'object'; ids = np.ones((9,64),np.int64);ids[:,0]=0;ids[:,1]=2
        ids[0,:3]=[0,4,2]
        arrays = dict(ids=ids,memory=np.full((3,9,32,768),.1+i/100,np.float32),
            gate=np.ones((3,9),np.float32),visual=np.full((3,32,768),i+1,np.float32),
            received_rgb=np.full((3,512,512,3),i,np.uint8))
        name = 'arrays/'+eid+'.npz';c.save_npz(cache/name,**arrays)
        trace = dict(shape=[1,32,768],input_has_text=False,source_image_at_RX=False,finite=True,
            processed_pixels_shape=[1,3,224,224],processed_pixels_float32_sha256='a'*64)
        entry = dict(example_id=eid,cohort=cohort,official_split='train',text=source,text_sha256=text_hash(source+str(i)),
            source_tokens=3,source_image_sha256=text_hash(str(i)),file=name,sha256=c.sha(cache/name),
            array_hashes={k:c.array_sha(v) for k,v in arrays.items()},
            visual_slice_sha256=[c.array_sha(arrays['visual'][j:j+1]) for j in range(3)],
            memory_slice_sha256=[c.array_sha(arrays['memory'][j:j+1]) for j in range(3)],
            image_stats=[dict(nominal_snr_db=s) for s in c.CHANNELS],text_stats=[dict(nominal_snr_db=s) for s in c.CHANNELS],
            visual_traces=[trace]*3,image_files={},input_truncated=False,source_mask_at_RX=False)
        # Unique source text avoids a synthetic TRAIN/dev text-duplicate leak.
        source = 'object'+str(i);entry.update(text=source,text_sha256=text_hash(source))
        if cohort=='dev':
            for snr in c.CHANNELS:
                name = 'images/'+eid+'_'+str(snr)+'.png';Image.fromarray(arrays['received_rgb'][0]).save(cache/name)
                entry['image_files'][str(snr)] = dict(file=name,sha256=c.sha(cache/name))
                audit = token_audit([2,4,2],2,[2],64)
                cap = dict(example_id=eid,image_snr_db=snr,image_sha256=c.sha(cache/name),output_text=source,
                    text_audit=text_audit(source),telemetry=dict(audit,eos_token_ids=[2],
                        processed_pixels=dict(float32_sha256='a'*64)))
                caps.append(cap)
        entries.append(entry)
    c.atomic(cache/'index.json',dict(records=entries))
    gray = np.full((32,768),.5,np.float32);c.save_npz(cache/'gray.npz',visual=gray)
    c.atomic(cache/'gray.json',dict(visual_slice_sha256=c.array_sha(gray[None])))
    c.atomic(cache/'frozen_audit.json',dict(BLIP_before='TEST_ONLY',BLIP_after='TEST_ONLY',pretrained_parameters_trained=0))
    with (cache/'caption_only.jsonl').open('x') as f:
        for row in caps:c.line(f,row)
    c.seal(cache,'cache')
    train = entries[:8]
    for arm in c.ARMS:
        path = root/('train_'+arm);path.mkdir()
        c.atomic(path/'identity.json',dict(TEST_ONLY=True,arm=arm,profile='smoke',config=cfg,
            cache_index_sha256=c.sha(cache/'index.json'),train_ids=[e['example_id'] for e in train],
            trainable_parameters=r.parameter_count(),initial_parameters={'TEST_ONLY':'same'},
            BART_frozen=True,BLIP_frozen=True,TX_RX_frozen=True,source_masks_at_RX=False,matching_gate=False))
        seen = set()
        with (path/'history.jsonl').open('x') as f:
            for step,epoch,ix,rs in c.stream(cfg):
                ids = [train[j]['example_id'] for j in ix];seen.update(ids)
                c.line(f,dict(update=step+1,epoch=epoch+1,ids=ids,regimes=rs,
                    valid_target_tokens=sum(train[j]['source_tokens']+16 for j in ix),CE=1.,lr=.0001,gradient_norm=.1))
        c.atomic(path/'learning_audit.json',dict(TEST_ONLY=True,passed=True,frozen_unchanged=True,
            BART_before='TEST_ONLY',BART_after='TEST_ONLY',groups={'unit':dict(parameters=1,changed=True,nonzero_gradient=True)}))
        with (path/'last.pth').open('xb') as f:f.write(b'EXPLICIT_TEST_ONLY_NOT_A_NEURAL_CHECKPOINT')
        c.atomic(path/'TRAIN_RESULT.json',dict(TEST_ONLY=True,updates=3,unique_training_ids=len(seen),
            actual_disk_roundtrip=True,fixed_final=True,checkpoint_sha256=c.sha(path/'last.pth')))
        c.seal(path,'train')
    evaluation = root/'evaluation';evaluation.mkdir();dev = entries[8:];donors = c.donor_map([e['example_id'] for e in dev])
    by = {e['example_id']:e for e in dev}
    with (evaluation/'rows.jsonl').open('x') as f:
        for entry in dev:
            for regime in c.regimes(cfg):
                it,tt = [c.CHANNELS.index(s) for s in c.REGIMES[regime]]
                for arm,controls in c.CONTROLS.items():
                    for ctrl in controls:
                        output = entry['text'] if ctrl!='shuffled' else 'wrong'
                        fields = dict(received_caption=output)
                        if arm!='caption':fields.update(chunk_generated_token_ids=[[2,0,4,2]]+[[2,0,2]]*8,
                            generated_token_ids=[4],stopped_by_capacity=False)
                        aux = None if arm in ('base','caption') else 'TEST_ONLY'
                        if arm=='visual':aux = (by[donors[entry['example_id']]]['visual_slice_sha256'][it] if ctrl=='shuffled' else
                            c.array_sha(gray[None]) if ctrl=='gray' else c.array_sha(np.zeros((1,32,768),np.float32)) if ctrl=='zero' else entry['visual_slice_sha256'][it])
                        row = dict(TEST_ONLY=True,example_id=entry['example_id'],source_text=entry['text'],
                            source_text_sha256=entry['text_sha256'],source_tokens=3,regime=regime,arm=arm,control=ctrl,
                            donor_id=donors[entry['example_id']] if ctrl=='shuffled' else None,
                            image_snr_db=c.REGIMES[regime][0],text_snr_db=c.REGIMES[regime][1],
                            image_stats=entry['image_stats'][it],text_stats=entry['text_stats'][tt],
                            memory_sha256=entry['memory_slice_sha256'][tt],auxiliary_sha256=aux,
                            source_text_at_RX=False,source_mask_at_RX=False,source_image_at_RX=False,
                            same_text_observation_all_controls=True,training_updates=0,
                            shared_trial_allocated_symbols=13376,
                            minimum_input_symbols=(4128 if arm=='caption' else 9248 if arm in ('base','self') or
                                ctrl in ('gray','zero','disabled') else 13376),
                            **fields,**metrics(entry['text'],output),output_audit=text_audit(output))
                        c.line(f,row)
    c.atomic(evaluation/'EVAL_RESULT.json',dict(TEST_ONLY=True,rows=48,BART_before='TEST_ONLY',BART_after='TEST_ONLY',frozen_unchanged=True))
    c.seal(evaluation,'evaluation')
    c.atomic(root/'PIPELINE.json',dict(schema=c.SCHEMA,status='COMPLETE',profile='smoke',
        source_manifest_sha256=c.sha(c.FOLDER/'MANIFEST.json'),TEST_ONLY=True))
    return root

def reseal(root,stage):
    # Only a newly created TEST_ONLY TemporaryDirectory is allowed by callers.
    (root/'COMPLETE.json').unlink();c.seal(root,stage)

class Contracts(unittest.TestCase):
    def test_fixed_profiles_budget_and_no_automatic_expansion(self):
        smoke,pilot = c.config('smoke'),c.config('pilot')
        self.assertEqual((smoke['train'],smoke['dev'],smoke['updates']),(8,2,3))
        self.assertEqual((pilot['train'],pilot['dev'],pilot['updates']),(512,32,512))
        with self.assertRaises(ValueError):c.config('full')
        self.assertNotIn('both_m5',c.TRAIN_REGIMES)
        self.assertEqual(c.PRIMARY,'both0')
    def test_training_stream_same_fixed_schedule_and_full_updates(self):
        cfg = c.config('pilot');a = list(c.stream(cfg));b = list(c.stream(cfg))
        self.assertEqual(a,b);self.assertEqual(len(a),512)
        self.assertEqual({i for _,_,ix,_ in a for i in ix},set(range(512)))
        self.assertTrue(all(s in c.TRAIN_REGIMES for _,_,_,rs in a for s in rs))
    def test_actual_donors_different_and_within_cohort(self):
        d = c.donor_map(['a','b','c']);self.assertEqual(set(d),set(d.values()))
        self.assertTrue(all(k!=v for k,v in d.items()))
        for ids in ([],['a'],['a','a']):
            with self.assertRaises(ValueError):c.donor_map(ids)
    def test_strict_json_no_partial_write_or_overwrite(self):
        with tempfile.TemporaryDirectory(dir=c.PROJECT/'experiment_results') as t:
            p = Path(t)/'test.json'
            with self.assertRaises(ValueError):c.atomic(p,{'x':float('nan')})
            self.assertFalse(p.exists());self.assertFalse(p.with_name('test.json.writing').exists())
            c.atomic(p,{'good':1})
            with self.assertRaises(FileExistsError):c.atomic(p,{})
    def test_path_symlinks_and_traversal_rejected(self):
        for n in ('../outside','/outside'):
            with self.assertRaises(ValueError):c.local(c.PROJECT,n)
        with self.assertRaises(ValueError):c.safe(c.PROJECT)
        with tempfile.TemporaryDirectory(dir=c.PROJECT/'experiment_results') as t:
            p=Path(t)/'linked';p.symlink_to(Path(t),target_is_directory=True)
            with self.assertRaises(ValueError):c.safe(p/'new')
    def test_exact_matrix_controls_and_complex_resource_contract(self):
        keys=c.expected_keys(c.config('smoke'),[{'example_id':'a'},{'example_id':'b'}])
        self.assertEqual(len(keys),48)
        self.assertEqual(c.REGIMES['text0'],(None,0));self.assertEqual(c.REGIMES['both0'],(0,0))
        self.assertEqual(4128+9248,13376)
    def test_actual_parameter_formula(self):
        self.assertEqual(r.parameter_count(),1516290)
        self.assertEqual(r.parameter_count(32,16,2),5042)
    def test_full_CPU_verifier_and_PRIVATE_collector_with_TEST_ONLY_fixtures(self):
        from docci_blip_feature_rx_v25.collect import collect
        with tempfile.TemporaryDirectory(prefix='V25_TEST_ONLY_',dir=c.PROJECT/'experiment_results') as t:
            root=fixture(Path(t));before=c.sha(root/'cache/index.json')
            value=r.verify_pipeline(root,True);self.assertEqual(value['evaluation']['rows'],48)
            self.assertFalse(value['results']['numeric_screen_passed']);self.assertFalse(value['results']['semantic_success'])
            self.assertIsNone(value['results']['comparisons']['vs_self']['paired95CI'])
            dest=Path(t)/'PRIVATE_TEST_ONLY.tar.gz';collect(root,dest)
            self.assertEqual(before,c.sha(root/'cache/index.json'))
            self.assertFalse(list(dest.with_name(dest.name+'.contents').rglob('*.npz')))
            self.assertFalse(list(dest.with_name(dest.name+'.contents').rglob('*.pth')))
            self.assertEqual(r.verify_pipeline(dest.with_name(dest.name+'.contents')/'data',False)['evaluation']['rows'],48)
    def test_missing_output_and_changed_auxiliary_independently_rejected(self):
        with tempfile.TemporaryDirectory(prefix='V25_TEST_ONLY_',dir=c.PROJECT/'experiment_results') as t:
            root=fixture(Path(t));p=root/'evaluation/rows.jsonl';data=c.rows(p)
            bad=next(x for x in data if x['control']=='shuffled');bad['auxiliary_sha256']='wrong'
            p.write_text('\n'.join(json.dumps(x) for x in data)+'\n');reseal(root/'evaluation','evaluation')
            with self.assertRaisesRegex(ValueError,'Auxiliary'):r.verify_pipeline(root,False)
            p.write_text('\n'.join(json.dumps(x) for x in data[:-1])+'\n');reseal(root/'evaluation','evaluation')
            with self.assertRaisesRegex(ValueError,'controls'):r.verify_pipeline(root,False)
    def test_changed_own_text_observation_cannot_fake_visual_gain(self):
        with tempfile.TemporaryDirectory(prefix='V25_TEST_ONLY_',dir=c.PROJECT/'experiment_results') as t:
            root=fixture(Path(t));p=root/'evaluation/rows.jsonl';data=c.rows(p)
            data[-1]['memory_sha256']='changed'
            p.write_text('\n'.join(json.dumps(x) for x in data)+'\n');reseal(root/'evaluation','evaluation')
            with self.assertRaisesRegex(ValueError,'observation'):r.verify_pipeline(root,False)
    def test_frozen_or_incomplete_training_is_not_success(self):
        with tempfile.TemporaryDirectory(prefix='V25_TEST_ONLY_',dir=c.PROJECT/'experiment_results') as t:
            root=fixture(Path(t));p=root/'train_visual/learning_audit.json';v=c.read(p);v['groups']['unit']['changed']=False
            p.write_text(json.dumps(v));reseal(root/'train_visual','train')
            with self.assertRaisesRegex(ValueError,'did not learn'):r.verify_pipeline(root,False)
    def test_no_skip_or_source_information_in_actual_visual_path(self):
        import ast
        tree=ast.parse((c.FOLDER/'visual.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
        self.assertEqual([a.arg for a in fn.args.args],['model','processor','image','device','dtype','budget'])
        code=(c.FOLDER/'test_hf.py').read_text();self.assertNotIn('skipTest',code);self.assertNotIn('skipIf',code)
        c.source_check()
    def test_actual_visual_trace_must_omit_text_inputs(self):
        with tempfile.TemporaryDirectory(prefix='V25_TEST_ONLY_',dir=c.PROJECT/'experiment_results') as t:
            root=fixture(Path(t));p=root/'cache/index.json';v=c.read(p)
            v['records'][0]['visual_traces'][0]['input_has_text']=True
            p.write_text(json.dumps(v));reseal(root/'cache','cache')
            with self.assertRaisesRegex(ValueError,'Q-Former'):r.cache(root/'cache',False)
    def test_receiver_gallery_escapes_output_instead_of_executing_markup(self):
        from docci_text_continue_v11.metrics import metrics
        from docci_caption_calibration_v24.core import text_audit
        with tempfile.TemporaryDirectory(prefix='V25_TEST_ONLY_',dir=c.PROJECT/'experiment_results') as t:
            root=fixture(Path(t));p=root/'evaluation/rows.jsonl';data=c.rows(p)
            row=next(x for x in data if x['arm']=='visual' and x['control']=='matched')
            row['received_caption']='<script>alert(1)</script>'
            row.update(metrics(row['source_text'],row['received_caption']),output_audit=text_audit(row['received_caption']))
            p.write_text('\n'.join(json.dumps(x) for x in data)+'\n');reseal(root/'evaluation','evaluation')
            text=r.render(root,r.verify_pipeline(root,False))
            self.assertNotIn('<script>',text);self.assertIn('&lt;script&gt;',text)
    def test_smoke_never_promotes_numeric_success_or_small_sample_CI(self):
        self.assertIsNone(r.bootstrap([.1,.2]))
        cfg=c.config('smoke')
        with tempfile.TemporaryDirectory(prefix='V25_TEST_ONLY_',dir=c.PROJECT/'experiment_results') as t:
            root=fixture(Path(t));summary=r.summarize(c.rows(root/'evaluation/rows.jsonl'),cfg)
            self.assertFalse(summary['numeric_screen_passed']);self.assertFalse(summary['semantic_success'])
            self.assertEqual(summary['facts'],'PENDING')
if __name__=='__main__':unittest.main(verbosity=2)
