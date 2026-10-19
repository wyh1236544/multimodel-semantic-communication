"""Mandatory real tiny-HF tests, including original None outer-config behavior."""
from pathlib import Path
import sys
import unittest
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_caption_calibration_v24.test_hf import ActualHFContracts
from docci_caption_calibration_v24_fix1.fix import model_view,repaired

class HFRepairContracts(ActualHFContracts):
    def test_outer_none_and_native_inner_config_without_mutation(self):
        self.assertIsNone(self.model.generation_config)
        before=self.model.language_model.generation_config.to_dict()
        v=model_view(self.model)
        self.assertIs(v.generation_config,self.model.language_model.generation_config)
        self.assertEqual(v.generation_config.to_dict(),before)
        self.assertIs(v.generate.__self__,self.model)
        self.assertIsNone(self.model.generation_config)
    def test_metadata_and_real_generation_use_inner_config(self):
        import torch
        from docci_caption_calibration_v24.runtime import generate_one
        v=model_view(self.model)
        self.assertEqual(v.generation_config.to_dict()['eos_token_id'],2)
        r=generate_one(self.model,self.processor,object(),None,'caption','cpu',torch.float32,lambda:None)
        self.assertEqual(r['telemetry']['eos_token_ids'],[2])
        self.assertEqual(r['telemetry']['pad_token_id'],1)
        self.assertIsNone(self.model.generation_config)

if __name__=='__main__':
    # Both inherited original tests run, plus two repair-specific real HF checks.
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(HFRepairContracts)
    with repaired():
        result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
