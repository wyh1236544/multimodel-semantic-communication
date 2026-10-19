"""CPU production-path fixtures and negative tests; no downloaded neural model."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_caption_calibration_v24 import core as c
from docci_caption_calibration_v24 import report

def fixture(root):
    from docci_evidence_v23.test_core import fixture as original_fixture
    from docci_caption_calibration_v24.run import import_inputs, complete
    from types import SimpleNamespace
    from PIL import Image
    original = original_fixture(root/'old')
    out = root/'new'; out.mkdir()
    manifest = c.read(c.FOLDER/'MANIFEST.json')
    plan = import_inputs(SimpleNamespace(prepared=str(original), action='caption'), out)
    lock = dict(kind='caption', revision='1ef7f63a8f0a144c13fdca8103eb7b4691c74cec',
        origin='Salesforce/blip2-opt-2.7b', tree={'path':'UNIT_ONLY','files':{}})
    c.atomic(out/'ASSET_LOCK.json', lock)
    c.atomic(out/'RUN.json', dict(schema=c.SCHEMA, stage='caption', protocol=c.PROTOCOL,
        dtype='float16', training_updates=0, automatic_next_stage=False,
        source_manifest_sha256=c.sha(c.FOLDER/'MANIFEST.json'),
        prepared_complete_sha256=c.sha(out/'prepared/COMPLETE.json'),
        asset_lock_sha256=c.sha(out/'ASSET_LOCK.json')))
    c.atomic(out/'MODEL.json', dict(config={'vision_config':{'image_size':224}},
        trainable_parameters=0, state_before_sha256='UNIT', loading_info={},
        explicit_generation_kwargs=c.generation_kwargs('caption')))
    calls = []; cache = {}
    with (out/'rows.jsonl').open('x') as f:
        for r in plan:
            call = c.digest([r['image_sha256'], r['prompt'], 'caption', 'float16'])
            reused = call in cache
            if not reused:
                t = c.token_audit([2, 10, 2], 2, [2], 64)
                t.update(implicit_BOS=True, input_token_ids=None, eos_token_ids=[2],
                    processed_pixels=dict(shape=[1,3,224,224], finite=True),
                    forward_probes={k:dict(finite=True) for k in ('vision_model','qformer','language_projection','language_model')},
                    score_trace=[dict(step=j, selected_token_id=v, top5_ids=[2,10,11,12,13],
                        top5_probabilities=[.2]*5, eos_probability=.2, masked_negative_infinity_count=0) for j,v in enumerate([10,2])])
                text = 'UNIT_ONLY object'
                cache[call] = dict(output_text=text, raw_decoded_text='RAW_UNIT', telemetry=t, text_audit=c.text_audit(text))
                calls.append(call)
            value = cache[call]
            c.line(f, dict(r, inference_id=call, reused_inference=reused,
                output_sha256=hashlib.sha256(value['output_text'].encode()).hexdigest(), **value))
    c.atomic(out/'INFERENCE.json', dict(unique_inferences=len(calls), unique_inference_ids=calls,
        rows=len(plan), state_before_sha256='UNIT', state_after_sha256='UNIT', training_updates=0,
        semantic_success=False, facts='PENDING', GPU='physical1', source_text_at_model=False,
        received_text_at_model=False, source_images_privileged_diagnosis_only=True,
        peak_allocated_gib=1., peak_reserved_gib=2., startup_free_gib=46.))
    c.atomic(out/'ASSET_AUDIT.json', dict(before={}, after={}, full_hash_before_load=True, full_hash_after_load=False))
    snapshot = out/'source_snapshot'; snapshot.mkdir()
    import shutil
    for name in list(manifest['files'])+['MANIFEST.json']: shutil.copyfile(c.FOLDER/name, snapshot/name)
    complete(out)
    return out

def reseal(root):
    p = root/'COMPLETE.json'; value = c.read(p)
    value['files'] = {f.relative_to(root).as_posix():c.sha(f) for f in root.rglob('*') if f.is_file() and f != p}
    # Mechanical fixture mutation, not an allowed operation on historical records.
    p.write_text(json.dumps(value))

class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='v24_UNIT_', dir=c.PROJECT/'experiment_results')
        self.root = Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def test_fixed_scope_no_training_expansion_or_transmissions(self):
        self.assertEqual(c.PROTOCOL['training_updates'], 0)
        self.assertEqual(c.PROTOCOL['new_transmissions'], 0)
        self.assertEqual(c.PROTOCOL['image_frame_symbols']+c.PROTOCOL['text_frame_symbols'], 13376)
        from docci_caption_calibration_v24.run import parser
        with self.assertRaises(SystemExit): parser().parse_args(['train','x'])
        with self.assertRaises(SystemExit): parser().parse_args(['caption','x','--profile','pilot'])
    def test_native_processor_omits_text_and_groundtruth(self):
        im = object(); v = c.processor_kwargs(im, None)
        self.assertEqual(set(v), {'images','return_tensors'})
        self.assertIs(v['images'], im)
        q = c.processor_kwargs(im, c.QUESTIONS['object'])
        self.assertFalse(q['truncation'])
        self.assertNotIn('cat', q['text'])
    def test_generation_limits_and_no_repeat_suppression(self):
        self.assertEqual(c.generation_kwargs('caption')['max_new_tokens'], 64)
        self.assertEqual(c.generation_kwargs('qa')['max_new_tokens'], 32)
        self.assertEqual(c.generation_kwargs('caption')['no_repeat_ngram_size'], 0)
        self.assertIsNone(c.generation_kwargs('caption')['forced_eos_token_id'])
    def test_BOS_equal_EOS_is_not_natural_end(self):
        t = c.token_audit([2]+[10]*64, 64, [2], 64)
        self.assertFalse(t['natural_EOS']); self.assertTrue(t['capacity_stop'])
    def test_EOS_and_empty_output_valid_engineering(self):
        self.assertTrue(c.token_audit([2,2], 1, [2], 64)['natural_EOS'])
        self.assertTrue(c.text_audit('')['empty'])
    def test_token_alignment_rejects_false_end_and_shortstop(self):
        for ids, steps in (([2,10],2),([2,10],1),([2,2,10],2)):
            with self.assertRaises(ValueError): c.token_audit(ids,steps,[2],64)
    def test_loop_diagnostics_cover_multitoken_cycles(self):
        r = c.text_audit('a bowl on the table '*20)
        self.assertEqual(r['adjacent_repeat_fraction'],0)
        self.assertGreater(r['duplicate_8gram_fraction'],.8)
    def test_safe_paths_and_no_overwrite(self):
        with self.assertRaises(ValueError): c.local(self.root,'../x')
        with self.assertRaises(ValueError): c.safe('/tmp/outside_PROJECT')
        (self.root/'link').symlink_to(self.root)
        with self.assertRaises(ValueError): c.local(self.root,'link/x')
        c.atomic(self.root/'x.json',{})
        with self.assertRaises(FileExistsError): c.atomic(self.root/'x.json',{})
    def test_budget_and_finite_json(self):
        for n in (0,-1,121,float('nan')):
            with self.assertRaises(ValueError): c.Budget(n)
        b=c.Budget(.01); b.deadline=0
        with self.assertRaises(TimeoutError): b()
        with self.assertRaises(ValueError): report.finite({'x':[float('inf')]})
    def test_actual_schema_fixture_and_input_preservation(self):
        p = fixture(self.root); before=c.sha(self.root/'old/rows.jsonl')
        result=report.verify(p)
        self.assertEqual(result['rows'],20)
        self.assertFalse(result['semantic_success'])
        self.assertEqual(before,c.sha(self.root/'old/rows.jsonl'))
        # All three received conditions in this synthetic fixture have equal pixels:
        # input-content dedup, not the production expectation of nine unique images.
        self.assertEqual(result['unique_inferences'],3)
    def test_QA_plan_has_80_rows_no_text_information(self):
        p=fixture(self.root); index=c.read(p/'prepared/index.json'); data=c.rows(p/'prepared/rows.jsonl')
        plan=c.make_plan(index,data,'qa')
        self.assertEqual(len(plan),80)
        self.assertTrue(all('source_text' not in r and 'received_caption' not in r for r in plan))
        index['profile']='pilot'
        with self.assertRaises(ValueError): c.make_plan(index,data,'caption')
    def test_donor_truth_and_caption_prompt_tamper_rejected(self):
        p=fixture(self.root); plan=c.read(p/'PLAN.json'); plan[2]['prompt']='gold answer'
        (p/'PLAN.json').write_text(json.dumps(plan)); reseal(p)
        with self.assertRaisesRegex(ValueError,'requests'): report.verify(p)
    def test_output_matrix_duplicate_rejected(self):
        p=fixture(self.root); rr=c.rows(p/'rows.jsonl'); rr[-1]=rr[0]
        (p/'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rr)); reseal(p)
        with self.assertRaisesRegex(ValueError,'matrix'): report.verify(p)
    def test_false_reused_inference_and_changed_tokens_rejected(self):
        p=fixture(self.root); rr=c.rows(p/'rows.jsonl'); rr[0]['reused_inference']=True
        (p/'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rr)); reseal(p)
        with self.assertRaisesRegex(ValueError,'reuse'): report.verify(p)
    def test_physical_GPU_and_frozen_hash_boundary_rejected(self):
        p=fixture(self.root); info=c.read(p/'INFERENCE.json'); info['state_after_sha256']='CHANGED'
        (p/'INFERENCE.json').write_text(json.dumps(info)); reseal(p)
        with self.assertRaisesRegex(ValueError,'Frozen'): report.verify(p)
    def test_raw_hash_tamper_rejected(self):
        p=fixture(self.root); (p/'rows.jsonl').write_text('')
        with self.assertRaisesRegex(ValueError,'hash'): report.verify(p)
    def test_manual_review_is_exact_run_scoped_and_fail_closed(self):
        p=fixture(self.root); r=report.review_template(p); dest=self.root/'review.json'
        c.atomic(dest,r)
        with self.assertRaises(ValueError): c.require_review(p,dest,report.verify(p))
        r.update(decision='APPROVE_QA_ONLY',reviewer='UNIT')
        for x in r['items']: x.update(on_topic=True,main_object_supported=True,no_long_loop=True)
        dest.write_text(json.dumps(r))
        self.assertEqual(c.require_review(p,dest,report.verify(p))['reviewer'],'UNIT')
        r['items'][0]['main_object_supported']=False; dest.write_text(json.dumps(r))
        with self.assertRaises(ValueError): c.require_review(p,dest,report.verify(p))
    def test_gallery_render_full_outputs_not_prompts_as_HTML(self):
        p=fixture(self.root); result=report.verify(p)
        value=report.render(p,result)
        self.assertIn('原图：特权诊断',value); self.assertIn('UNIT_ONLY object',value)
        self.assertIn('data/prepared/',value)
        self.assertNotIn('semantic_success": true',value)
    def test_real_collector_archive_and_immutability(self):
        from docci_caption_calibration_v24.collect import collect
        p=fixture(self.root); before=c.sha(p/'COMPLETE.json')
        dest=self.root/'records.tar.gz'; collect(p,dest)
        self.assertEqual(before,c.sha(p/'COMPLETE.json'))
        self.assertTrue(dest.is_file()); self.assertTrue(dest.with_name(dest.name+'.sha256').is_file())
        with self.assertRaises(FileExistsError): collect(p,dest)
    def test_collector_refuses_nested_and_missing_records(self):
        from docci_caption_calibration_v24.collect import collect
        p=fixture(self.root)
        with self.assertRaises(ValueError): collect(p,p/'records.tar.gz')
        (p/'INFERENCE.json').unlink()
        with self.assertRaises(ValueError): report.verify(p)
    def test_symlink_and_nonfinite_telemetry_fail(self):
        p=fixture(self.root); rr=c.rows(p/'rows.jsonl'); rr[0]['telemetry']['elapsed_seconds']=float('nan')
        (p/'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rr)); reseal(p)
        with self.assertRaisesRegex(ValueError,'Nonfinite'): report.verify(p)
    def test_tiny_HF_gate_is_not_silently_skipped(self):
        source=(c.FOLDER/'test_hf.py').read_text()
        self.assertNotIn('skipTest',source); self.assertNotIn('skipIf',source)
        self.assertNotIn('from_pretrained',source)
    def test_existing_actual_V23_prepared_plan_reuses_exact_PNGs(self):
        from types import SimpleNamespace
        from docci_caption_calibration_v24.run import import_inputs
        source=c.PROJECT/'experiment_results/v23_caption_review.K3KyAHsL/records/experiment_results/v23_inputs_smoke_01'
        if not source.is_dir():
            # On server/isolated restores test synthetic schema; private uploaded
            # actual inputs are NOT shipped in a public source package.
            return
        before={p.relative_to(source).as_posix():c.sha(p) for p in source.rglob('*') if p.is_file()}
        out=self.root/'actual_import';out.mkdir()
        plan=import_inputs(SimpleNamespace(prepared=str(source),action='caption'),out)
        self.assertEqual(len(plan),20)
        self.assertEqual(len({r['image_sha256'] for r in plan}),9)
        self.assertEqual(before,{p.relative_to(source).as_posix():c.sha(p) for p in source.rglob('*') if p.is_file()})
    def test_probability_trace_and_processor_geometry_refuse_tamper(self):
        p=fixture(self.root);rr=c.rows(p/'rows.jsonl');rr[0]['telemetry']['processed_pixels']['shape']=[1,3,512,512]
        (p/'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rr));reseal(p)
        with self.assertRaisesRegex(ValueError,'geometry'):report.verify(p)
    def test_official_asset_shape_rejects_unknown_files_before_load(self):
        folder=self.root/'model';folder.mkdir();(folder/'evil.py').write_text('UNIT_ONLY')
        lock=dict(kind='caption',origin='Salesforce/blip2-opt-2.7b',revision='1ef7f63a8f0a144c13fdca8103eb7b4691c74cec',
            tree=dict(path=str(folder),files={'evil.py':'bad'}))
        with self.assertRaisesRegex(ValueError,'files'):c.verify_assets(lock,lambda:None)
    def test_owned_child_failure_and_timeout_preserve_logs(self):
        import os,time
        from docci_caption_calibration_v24.launch import invoke
        log=self.root/'child.log'
        with self.assertRaisesRegex(RuntimeError,'exit 7'):
            invoke([sys.executable,'-c','raise SystemExit(7)'],log,dict(os.environ),time.monotonic()+5)
        self.assertTrue(log.is_file())
        with self.assertRaises(TimeoutError):
            invoke([sys.executable,'-c','import time; time.sleep(10)'],self.root/'timeout.log',dict(os.environ),time.monotonic()+.1)
    def test_real_run_orchestrator_with_explicit_TEST_ONLY_neural_substitute(self):
        import os,shutil
        from types import SimpleNamespace
        from docci_caption_calibration_v24 import run,runtime
        from docci_evidence_v23.test_core import fixture as old_fixture
        prepared=old_fixture(self.root/'prepared_input')
        modeldir=self.root/'MODEL_UNIT';modeldir.mkdir()
        c.atomic(modeldir/'config.json',dict(model_type='blip-2'))
        lock=dict(kind='caption',origin='Salesforce/blip2-opt-2.7b',revision='1ef7f63a8f0a144c13fdca8103eb7b4691c74cec',
            tree=dict(path=str(modeldir),files={'config.json':c.sha(modeldir/'config.json')}))
        c.atomic(self.root/'lock.json',lock)
        # Build a separate full producer fixture, then use only its declared TEST_ONLY
        # telemetry as the neural substitute. No uploaded code or neural inference.
        seedbase=self.root/'seed';seedbase.mkdir(); seedrun=fixture(seedbase)
        sample=c.rows(seedrun/'rows.jsonl')[0]
        value={k:sample[k] for k in ('output_text','raw_decoded_text','text_audit','telemetry')}
        fake_torch=SimpleNamespace(__version__='1.12.0',manual_seed=lambda x:None,
            use_deterministic_algorithms=lambda x:None,backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=False)),
            device=lambda x:x,float16='TEST_FP16',float32='TEST_FP32',
            cuda=SimpleNamespace(is_available=lambda:True,manual_seed_all=lambda x:None,
                mem_get_info=lambda:(46*2**30,48*2**30),reset_peak_memory_stats=lambda:None,
                max_memory_allocated=lambda:2**30,max_memory_reserved=lambda:2*2**30))
        fake_model=SimpleNamespace(parameters=lambda:[],config=SimpleNamespace(to_dict=lambda:{'vision_config':{'image_size':224}}),
            generation_config=SimpleNamespace(to_dict=lambda:{}))
        processor=SimpleNamespace(image_processor=SimpleNamespace(to_dict=lambda:{}))
        output=self.root/'production_entry'
        argv=['run.py','caption',str(output),'--prepared',str(prepared),'--asset-lock',str(self.root/'lock.json')]
        with patch.dict(sys.modules,{'torch':fake_torch,'transformers':SimpleNamespace(__version__='4.30.2')}), \
             patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1'}),patch.object(sys,'argv',argv), \
             patch.object(run,'verify_assets',return_value=modeldir),patch.object(run.importlib.metadata,'version',return_value='TEST_ONLY'), \
             patch.object(runtime,'load_model',return_value=(fake_model,processor,{})), \
             patch.object(runtime,'state_digest',return_value='UNIT'),patch.object(runtime,'generate_one',return_value=value) as generate:
            run.main()
        self.assertEqual(generate.call_count,3)
        self.assertEqual(report.verify(output)['rows'],20)
        self.assertTrue(all(call.args[3] is None for call in generate.call_args_list))
    def test_gallery_escapes_actual_candidate_markup(self):
        p=fixture(self.root);rr=c.rows(p/'rows.jsonl')
        rr[0]['output_text']='<script>alert("UNIT")</script>'
        native=report.rows
        with patch.object(report,'rows',side_effect=lambda path:rr if Path(path)==p/'rows.jsonl' else native(path)):
            rendered=report.render(p,{'semantic_success':False})
        self.assertNotIn('<script>',rendered)
        self.assertIn('&lt;script&gt;',rendered)

if __name__ == '__main__': unittest.main(verbosity=2)
