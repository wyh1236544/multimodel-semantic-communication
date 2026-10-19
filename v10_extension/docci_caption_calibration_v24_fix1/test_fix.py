"""CPU regression for exact V24 config-read expressions and repair scope."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_caption_calibration_v24 import core as c
from docci_caption_calibration_v24_fix1 import fix

class Config:
    def __init__(self):self.eos_token_id=2;self.pad_token_id=1
    def to_dict(self):return dict(eos_token_id=self.eos_token_id,pad_token_id=self.pad_token_id)

def model():
    m=SimpleNamespace(generation_config=None,language_model=SimpleNamespace(generation_config=Config()))
    def generate(*args,**kwargs):return {'original':True,'kwargs':kwargs}
    m.generate=generate
    return m

def expressions(name):
    tree=ast.parse((c.FOLDER/name).read_text())
    return [n for n in ast.walk(tree) if isinstance(n,ast.Attribute) and
        isinstance(n.value,ast.Attribute) and isinstance(n.value.value,ast.Name) and
        n.value.value.id=='model' and n.value.attr=='generation_config']

class Contracts(unittest.TestCase):
    def test_actual_production_eos_and_pad_reads_reproduce_failure_then_pass(self):
        original=model();view=fix.model_view(original)
        nodes=expressions('runtime.py')
        self.assertEqual({n.attr for n in nodes},{'eos_token_id','pad_token_id'})
        for n in nodes:
            code=compile(ast.Expression(n),'ORIGINAL_V24_READ','eval')
            with self.assertRaises(AttributeError):eval(code,{'model':original})
            self.assertEqual(eval(code,{'model':view}),getattr(original.language_model.generation_config,n.attr))
    def test_actual_production_MODEL_serialization_reproduces_failure_then_pass(self):
        original=model();nodes=expressions('run.py');self.assertEqual(len(nodes),1)
        call=ast.Call(func=nodes[0],args=[],keywords=[]);ast.fix_missing_locations(call)
        code=compile(ast.Expression(call),'ORIGINAL_MODEL_SERIALIZATION','eval')
        with self.assertRaises(AttributeError):eval(code,{'model':original})
        self.assertEqual(eval(code,{'model':fix.model_view(original)}),original.language_model.generation_config.to_dict())
    def test_view_keeps_original_model_and_method_identity(self):
        original=model();before=copy.deepcopy(original.language_model.generation_config.to_dict())
        view=fix.model_view(original)
        self.assertIs(view.generate,original.generate)
        self.assertIs(view.language_model,original.language_model)
        self.assertEqual(view.generate(max_new_tokens=64),original.generate(max_new_tokens=64))
        self.assertIsNone(original.generation_config)
        self.assertEqual(before,original.language_model.generation_config.to_dict())
    def test_inner_always_owns_metadata_even_if_outer_present_but_wrong(self):
        original=model();original.generation_config=SimpleNamespace(eos_token_id=19)
        self.assertEqual(fix.model_view(original).generation_config.eos_token_id,2)
        self.assertEqual(original.generation_config.eos_token_id,19)
    def test_missing_actual_inner_config_is_not_fabricated(self):
        for original in (SimpleNamespace(generation_config=Config()),
                SimpleNamespace(language_model=SimpleNamespace(generation_config=None)),
                SimpleNamespace(language_model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=None)))):
            with self.assertRaises(ValueError):fix.model_view(original)
    def test_view_idempotent_and_tracks_actual_inner_owner(self):
        original=model();view=fix.model_view(original)
        self.assertIs(fix.model_view(view),view)
        original.language_model.generation_config=Config()
        self.assertIs(view.generation_config,original.language_model.generation_config)
    def test_context_restores_runtime_after_exception(self):
        from docci_caption_calibration_v24 import runtime,run
        before=runtime.load_model,runtime.generate_one,run.complete
        with self.assertRaisesRegex(RuntimeError,'UNIT'):
            with fix.repaired():
                self.assertIsNot(runtime.generate_one,before[1]);raise RuntimeError('UNIT')
        self.assertEqual(before,(runtime.load_model,runtime.generate_one,run.complete))
    def test_original_runtime_generation_receives_view_not_altered_kwargs(self):
        from docci_caption_calibration_v24 import runtime
        def actual(m,*args,**kw):
            return m.generation_config.to_dict(),m.generate,args,kw
        with patch.object(runtime,'generate_one',side_effect=actual):
            with fix.repaired():
                original=model();r=runtime.generate_one(original,'im',prompt=None,stage='caption')
                self.assertEqual(r[0],dict(eos_token_id=2,pad_token_id=1))
                self.assertIs(r[1],original.generate)
                self.assertEqual(r[2],('im',));self.assertEqual(r[3],dict(prompt=None,stage='caption'))
    def test_load_and_complete_keep_explicit_snapshot_and_provenance(self):
        from docci_caption_calibration_v24 import runtime,run
        with tempfile.TemporaryDirectory(prefix='v24_fix1_UNIT_',dir=c.PROJECT/'experiment_results') as tmp:
            root=Path(tmp);original=model()
            with patch.object(runtime,'load_model',return_value=(original,'processor',{})),patch.object(run,'complete') as finish:
                with fix.repaired():
                    view,processor,info=runtime.load_model('UNIT',dtype='UNIT',device='UNIT')
                    self.assertIsNone(original.generation_config)
                    self.assertEqual(view.generation_config.eos_token_id,2)
                    run.complete(root)
                finish.assert_called_once_with(root)
            receipt=c.read(root/'FIX1.json')
            self.assertTrue(receipt['observations'][0]['outer_generation_config_is_none'])
            self.assertEqual(receipt['observations'][0]['generation_config_owner'],'language_model')
            self.assertFalse(receipt['generation_kwargs_changed'])
            self.assertEqual(c.sha(root/'fix1_source_snapshot/fix.py'),c.sha(fix.FOLDER/'fix.py'))
    def test_no_generation_rule_change(self):
        self.assertEqual(c.generation_kwargs('caption')['max_new_tokens'],64)
        self.assertEqual(c.generation_kwargs('qa')['max_new_tokens'],32)
        self.assertEqual(c.generation_kwargs('caption')['no_repeat_ngram_size'],0)
        self.assertFalse(c.PROTOCOL['semantic_success'])
    def test_original_and_repair_source_pins(self):
        m=fix.source_check()
        self.assertEqual(m['dependencies']['v10_extension/docci_caption_calibration_v24/MANIFEST.json'],c.sha(c.FOLDER/'MANIFEST.json'))
    def test_no_skip_of_real_HF_tests(self):
        code=(fix.FOLDER/'test_hf.py').read_text()
        self.assertNotIn('skipTest',code);self.assertNotIn('skipIf',code)
        self.assertIn('with repaired()',code)
    def test_original_run_MODEL_and_fixed_collection_with_TEST_ONLY_neural_substitute(self):
        import os
        from docci_caption_calibration_v24 import run,runtime
        from docci_evidence_v23.test_core import fixture as old_prepared
        from docci_caption_calibration_v24.test_core import fixture as old_result
        with tempfile.TemporaryDirectory(prefix='v24_fix1_entry_UNIT_',dir=c.PROJECT/'experiment_results') as tmp:
            base=Path(tmp);prepared=old_prepared(base/'inputs')
            seed=base/'seed';seed.mkdir();previous=old_result(seed)
            sample=c.rows(previous/'rows.jsonl')[0]
            value={k:sample[k] for k in ('output_text','raw_decoded_text','text_audit','telemetry')}
            assets=base/'MODEL_UNIT';assets.mkdir();c.atomic(assets/'config.json',dict(model_type='blip-2'))
            lock=dict(kind='caption',origin='Salesforce/blip2-opt-2.7b',revision='1ef7f63a8f0a144c13fdca8103eb7b4691c74cec',
                tree=dict(path=str(assets),files={'config.json':c.sha(assets/'config.json')}))
            c.atomic(base/'lock.json',lock)
            original=model();original.parameters=lambda:[]
            original.config=SimpleNamespace(to_dict=lambda:{'vision_config':{'image_size':224}})
            processor=SimpleNamespace(image_processor=SimpleNamespace(to_dict=lambda:{}))
            fake_torch=SimpleNamespace(__version__='1.12.0',manual_seed=lambda n:None,use_deterministic_algorithms=lambda x:None,
                backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=False)),device=lambda x:x,float16='UNIT_HALF',float32='UNIT_FLOAT',
                cuda=SimpleNamespace(is_available=lambda:True,manual_seed_all=lambda x:None,mem_get_info=lambda:(46*2**30,48*2**30),
                    reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:2**30,max_memory_reserved=lambda:2*2**30))
            output=base/'production';argv=['run.py','caption',str(output),'--prepared',str(prepared),'--asset-lock',str(base/'lock.json')]
            with patch.dict(sys.modules,{'torch':fake_torch,'transformers':SimpleNamespace(__version__='4.30.2')}), \
                 patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1'}),patch.object(sys,'argv',argv), \
                 patch.object(run,'verify_assets',return_value=assets),patch.object(run.importlib.metadata,'version',return_value='TEST_ONLY'), \
                 patch.object(runtime,'load_model',return_value=(original,processor,{})), \
                 patch.object(runtime,'state_digest',return_value='UNIT'),patch.object(runtime,'generate_one',return_value=value):
                with fix.repaired():run.main()
            self.assertIsNone(original.generation_config)
            self.assertEqual(c.read(output/'MODEL.json')['generation_config'],original.language_model.generation_config.to_dict())
            self.assertEqual(fix.verify(output)['rows'],20)
            fixed_before=c.sha(output/'COMPLETE.json')
            dest=base/'private_records.tar.gz';fix.collect(output,dest)
            self.assertEqual(fixed_before,c.sha(output/'COMPLETE.json'))
            self.assertTrue(dest.is_file())
            self.assertTrue((base/'private_records.tar.gz.contents/data/FIX1.json').is_file())
            # Re-seal ONLY this owned synthetic fixture to test independent owner audit.
            info=c.read(output/'FIX1.json');info['observations'][0]['generation_config_owner']='outer'
            (output/'FIX1.json').write_text(json.dumps(info))
            from docci_caption_calibration_v24.test_core import reseal
            reseal(output)
            with self.assertRaisesRegex(ValueError,'owner'):fix.verify(output)
    def test_complete_refuses_missing_actual_load_provenance(self):
        from docci_caption_calibration_v24 import run
        with tempfile.TemporaryDirectory(prefix='v24_fix1_no_load_UNIT_',dir=c.PROJECT/'experiment_results') as tmp:
            with fix.repaired():
                with self.assertRaisesRegex(ValueError,'load provenance'):run.complete(Path(tmp))

if __name__=='__main__':unittest.main(verbosity=2)
