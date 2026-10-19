"""CPU-only metadata regressions; TEST_ONLY substitutes are not HF evidence."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_caption_calibration_v24 import core as c
from docci_caption_calibration_v24_fix2 import fix
from docci_caption_calibration_v24_fix1.test_fix import model

class FakeDtype:
    """TEST ONLY. The production path never infers dtype from a class name."""
    def __init__(self, name='float16'): self.name = name
    def __str__(self): return 'torch.'+self.name

class Contracts(unittest.TestCase):
    def test_nested_dtype_copy_is_explicit_and_original_unchanged(self):
        dtype = FakeDtype()
        source = dict(config={'torch_dtype':dtype, 'nested':[dtype, (dtype,)]}, text='中文')
        with self.assertRaises(TypeError): json.dumps(source, allow_nan=False)
        result, events = fix.metadata_copy(source, FakeDtype)
        self.assertEqual(result['config']['nested'], ['float16', ['float16']])
        self.assertEqual(len(events), 3)
        self.assertIs(source['config']['torch_dtype'], dtype)
        self.assertIsInstance(source['config']['nested'][1], tuple)
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_standard_scalar_and_integer_config_keys(self):
        source = {'id2label':{0:'zero', 1:'one'}, 'ordinary':[None, True, 4, 1.5, 'float16']}
        result, events = fix.metadata_copy(source)
        self.assertEqual(result, json.loads(json.dumps(source)))
        self.assertEqual(events, [])
        with self.assertRaisesRegex(ValueError, 'Colliding'):
            fix.metadata_copy({1:'one', '1':'collision'})

    def test_unknown_objects_dtype_imitation_and_nonfinite_rejected(self):
        for value in (object(), FakeDtype(), {'bad':float('nan')}, [float('inf')],
                {float('-inf'):'bad'}, {object():'key'}):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises((TypeError, ValueError)): fix.metadata_copy(value)

    def test_cycles_refused_shared_noncircular_containers_allowed(self):
        value = []; value.append(value)
        with self.assertRaisesRegex(ValueError, 'Cyclic'): fix.metadata_copy(value)
        shared = [1, 2]
        self.assertEqual(fix.metadata_copy([shared, shared])[0], [[1, 2], [1, 2]])

    def test_scoped_atomic_preflights_before_write_and_preserves_strict_others(self):
        from docci_caption_calibration_v24 import run
        with tempfile.TemporaryDirectory(prefix='v24_fix2_atomic_UNIT_', dir=c.PROJECT/'experiment_results') as tmp:
            root = Path(tmp); value = {'torch_dtype':FakeDtype()}
            with patch.dict(sys.modules, {'torch':SimpleNamespace(dtype=FakeDtype)}):
                with fix.repaired(root):
                    with self.assertRaises(TypeError): run.atomic(root/'MODEL.json', {'unknown':object()})
                    self.assertFalse((root/'MODEL.json.writing').exists())
                    with self.assertRaises(TypeError): run.atomic(root/'OTHER.json', value)
                    run.atomic(root/'MODEL.json', value)
                    self.assertEqual(c.read(root/'MODEL.json'), {'torch_dtype':'float16'})
                    with self.assertRaises(FileExistsError): run.atomic(root/'MODEL.json', value)

    def test_atomic_context_restores_all_original_bindings_after_failure(self):
        from docci_caption_calibration_v24 import run, runtime
        before = (c.atomic, run.atomic, runtime.load_model, runtime.generate_one, run.complete)
        with tempfile.TemporaryDirectory(prefix='v24_fix2_scope_UNIT_', dir=c.PROJECT/'experiment_results') as tmp:
            with self.assertRaisesRegex(RuntimeError, 'UNIT'):
                with fix.repaired(Path(tmp)): raise RuntimeError('UNIT')
        self.assertEqual(before, (c.atomic, run.atomic, runtime.load_model, runtime.generate_one, run.complete))

    def test_real_server_gate_is_mandatory_and_source_pins_intact(self):
        m = fix.source_check()
        self.assertIn('v10_extension/docci_caption_calibration_v24_fix1/fix.py', m['dependencies'])
        code = (fix.FOLDER/'test_hf.py').read_text()
        self.assertIn('from_pretrained', code); self.assertIn('torch_dtype=torch.float16', code)
        self.assertNotIn('skipTest', code); self.assertNotIn('skipIf', code)

    def test_full_production_write_verify_collect_with_TEST_ONLY_dtype_model(self):
        from docci_caption_calibration_v24 import run, runtime
        from docci_evidence_v23.test_core import fixture as old_prepared
        from docci_caption_calibration_v24.test_core import fixture as old_result, reseal
        with tempfile.TemporaryDirectory(prefix='v24_fix2_entry_UNIT_', dir=c.PROJECT/'experiment_results') as tmp:
            base = Path(tmp); prepared = old_prepared(base/'inputs')
            seed = base/'seed'; seed.mkdir(); previous = old_result(seed)
            sample = c.rows(previous/'rows.jsonl')[0]
            neural = {k:sample[k] for k in ('output_text', 'raw_decoded_text', 'text_audit', 'telemetry')}
            assets = base/'MODEL_UNIT'; assets.mkdir(); c.atomic(assets/'config.json', {'model_type':'blip-2'})
            lock = dict(kind='caption', origin='Salesforce/blip2-opt-2.7b',
                revision='1ef7f63a8f0a144c13fdca8103eb7b4691c74cec',
                tree=dict(path=str(assets), files={'config.json':c.sha(assets/'config.json')}))
            c.atomic(base/'lock.json', lock)
            original = model(); original.parameters = lambda:[]
            dtype = FakeDtype(); configs = {'vision_config':{'image_size':224}, 'torch_dtype':dtype}
            original.config = SimpleNamespace(to_dict=lambda:configs)
            inner = dict(eos_token_id=2, pad_token_id=1, torch_dtype=dtype)
            original.language_model.generation_config.to_dict = lambda:inner
            processor = SimpleNamespace(image_processor=SimpleNamespace(to_dict=lambda:{'mode':[dtype]}))
            fake_torch = SimpleNamespace(__version__='1.12.0', dtype=FakeDtype,
                manual_seed=lambda n:None, use_deterministic_algorithms=lambda x:None,
                backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=False)), device=lambda x:x,
                float16=dtype, float32=FakeDtype('float32'),
                cuda=SimpleNamespace(is_available=lambda:True, manual_seed_all=lambda x:None,
                    mem_get_info=lambda:(46*2**30,48*2**30), reset_peak_memory_stats=lambda:None,
                    max_memory_allocated=lambda:2**30, max_memory_reserved=lambda:2*2**30))
            output = base/'production'
            argv = ['run.py', 'caption', str(output), '--prepared', str(prepared), '--asset-lock', str(base/'lock.json')]
            before = (c.sha(c.FOLDER/'run.py'), c.sha(fix.previous.FOLDER/'fix.py'))
            with patch.dict(sys.modules, {'torch':fake_torch, 'transformers':SimpleNamespace(__version__='4.30.2')}), \
                    patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'1'}), patch.object(sys, 'argv', argv), \
                    patch.object(run, 'verify_assets', return_value=assets), \
                    patch.object(run.importlib.metadata, 'version', return_value='TEST_ONLY'), \
                    patch.object(runtime, 'load_model', return_value=(original, processor, {})), \
                    patch.object(runtime, 'state_digest', return_value='UNIT'), \
                    patch.object(runtime, 'generate_one', return_value=neural):
                with fix.repaired(output): run.main()
            self.assertIsNone(original.generation_config)
            self.assertIs(configs['torch_dtype'], dtype); self.assertIs(inner['torch_dtype'], dtype)
            result = fix.verify(output)
            self.assertEqual(result['rows'], 20); self.assertEqual(result['dtype_fields_converted'], 4)
            self.assertEqual(result['semantic_success'], False)
            self.assertEqual(before, (c.sha(c.FOLDER/'run.py'), c.sha(fix.previous.FOLDER/'fix.py')))
            complete_before = c.sha(output/'COMPLETE.json')
            dest = base/'PRIVATE_records.tar.gz'; fix.collect(output, dest)
            self.assertTrue(dest.is_file()); self.assertEqual(c.sha(output/'COMPLETE.json'), complete_before)
            self.assertEqual(c.read(base/'PRIVATE_records.tar.gz.contents/data/FIX2.json')['repair'], fix.REPAIR)
            receipt = c.read(output/'FIX2.json')
            receipt['metadata_dtype_conversions']['MODEL.json'][0]['after'] = 'float32'
            (output/'FIX2.json').write_text(json.dumps(receipt))
            reseal(output)
            with self.assertRaisesRegex(ValueError, 'audit|match'): fix.verify(output)

if __name__ == '__main__': unittest.main(verbosity=2)
