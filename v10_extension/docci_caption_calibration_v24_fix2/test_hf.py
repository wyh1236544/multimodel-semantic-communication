"""Mandatory server-only real HF tests; tiny local FP16 reload before full BLIP2."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_caption_calibration_v24_fix1.test_hf import HFRepairContracts
from docci_caption_calibration_v24_fix1.fix import model_view
from docci_caption_calibration_v24_fix2.fix import metadata_copy, repaired
from docci_caption_calibration_v24 import core as c

class HFMetadataContracts(HFRepairContracts):
    def test_actual_torch_dtype_nested_copy_and_tensor_rejection(self):
        import torch
        source = {'config':{'torch_dtype':torch.float16, 'id2label':{0:'zero'}},
            'generation_config':{'nested':[torch.float32]}}
        with self.assertRaises(TypeError):
            json.dumps(source, allow_nan=False)
        value, events = metadata_copy(source)
        self.assertEqual(value['config']['torch_dtype'], 'float16')
        self.assertEqual(value['generation_config']['nested'], ['float32'])
        self.assertIs(source['config']['torch_dtype'], torch.float16)
        self.assertEqual(len(events), 2)
        with self.assertRaises(TypeError):
            metadata_copy({'tensor':torch.tensor([1.])})

    def test_actual_tiny_checkpoint_float16_reload_and_MODEL_write(self):
        import torch
        from transformers import Blip2ForConditionalGeneration
        from docci_caption_calibration_v24 import run
        with tempfile.TemporaryDirectory(prefix='v24_fix2_real_HF_', dir=c.PROJECT/'experiment_results') as tmp:
            root = Path(tmp); checkpoint = root/'tiny'
            self.model.save_pretrained(checkpoint)
            loaded, info = Blip2ForConditionalGeneration.from_pretrained(checkpoint,
                local_files_only=True, torch_dtype=torch.float16, output_loading_info=True)
            view = model_view(loaded)
            self.assertEqual(next(loaded.parameters()).dtype, torch.float16)
            before = loaded.config.to_dict()
            value = dict(loading_info=info, config=before,
                generation_config=view.generation_config.to_dict(), processor={})
            normalized, events = metadata_copy(value)
            json.dumps(normalized, allow_nan=False)
            # No FP16 CPU neural forward; inherited tests exercise real generation in FP32.
            with repaired(root):
                run.atomic(root/'MODEL.json', value)
            self.assertEqual(c.read(root/'MODEL.json'), normalized)
            self.assertEqual(loaded.config.to_dict(), before)
            self.assertIsNone(loaded.generation_config)
            self.assertFalse((root/'MODEL.json.writing').exists())
            print('REAL_HF_FP16_METADATA_GATE_PASSED dtype_fields={}'.format(len(events)), flush=True)

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(HFMetadataContracts)
    # The inherited original two generation tests + Fix1 two tests are not removed.
    with tempfile.TemporaryDirectory(prefix='v24_fix2_hf_scope_', dir=c.PROJECT/'experiment_results') as tmp:
        with repaired(Path(tmp)):
            result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
