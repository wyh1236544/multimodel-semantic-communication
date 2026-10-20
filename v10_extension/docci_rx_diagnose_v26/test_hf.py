"""Mandatory actual tiny BART tests in fixed HF4.30.2; no dependency/model downloads."""
from pathlib import Path
import sys
import unittest
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
import torch
import transformers
from docci_rx_diagnose_v26.neural import teacher,Trace
from docci_blip_feature_rx_v25.model import ce,generate
from docci_blip_feature_rx_v25.engine import signature

class ActualBARTContracts(unittest.TestCase):
    def setUp(self):
        from docci_blip_feature_rx_v25.test_hf import ModelContracts
        if torch.__version__.split('+')[0]!='1.12.0' or transformers.__version__!='4.30.2':
            raise RuntimeError('Use actual original Torch1.12/HF4.30.2; no skips/upgrades')
        torch.set_num_threads(2)
        helper=ModelContracts();self.bart=helper.bart();self.adapter=helper.adapter(self.bart).requires_grad_(False)
        self.memory,self.aux,self.gate,self.ids=helper.inputs()
    def tearDown(self):self.adapter.close()
    def test_actual_TF_matches_original_V25_CE_and_gold_counts(self):
        value=teacher(self.bart,self.memory,self.adapter,self.aux,self.gate,self.ids)
        with torch.no_grad():old,n=ce(self.bart,self.adapter,self.memory,self.aux,self.gate,self.ids)
        self.assertEqual(n,20);self.assertEqual(value['counts']['all'],20);self.assertEqual(value['counts']['content'],2)
        self.assertAlmostEqual(value['nll']['all']/n,float(old)/n,places=5)
        self.assertEqual(value['counts']['eos'],9);self.assertEqual(value['counts']['bos'],9)
    def test_actual_trace_observes_zero_and_nonzero_effect_without_changing_CE(self):
        zero=teacher(self.bart,self.memory,self.adapter,self.aux,self.gate,self.ids)
        self.assertTrue(all(s['delta_l2_sq']==0 for x in zero['trace']['layers'] for s in x['scopes']['all']))
        with torch.no_grad():
            for x in self.adapter.cross:x.alpha.fill_(.3)
        value=teacher(self.bart,self.memory,self.adapter,self.aux,self.gate,self.ids)
        self.assertTrue(any(s['delta_l2_sq']>0 for x in value['trace']['layers'] for s in x['scopes']['content']))
        with torch.no_grad():old,n=ce(self.bart,self.adapter,self.memory,self.aux,self.gate,self.ids)
        self.assertAlmostEqual(value['nll']['all']/n,float(old)/n,places=5)
    def test_disabled_TF_is_base_and_zero_content_gate_is_identity(self):
        with torch.no_grad():
            for x in self.adapter.cross:x.alpha.fill_(.3)
        base=teacher(self.bart,self.memory,None,None,self.gate,self.ids)
        disabled=teacher(self.bart,self.memory,self.adapter,self.aux,self.gate,self.ids,False)
        gated=teacher(self.bart,self.memory,self.adapter,self.aux,torch.zeros_like(self.gate),self.ids)
        self.assertEqual(base['nll'],disabled['nll']);self.assertEqual(base['nll'],gated['nll'])
        self.assertEqual(disabled['trace']['layers'],[])
    def test_actual_instrumentation_preserves_greedy_and_all_frozen_tensors(self):
        from types import SimpleNamespace
        tok=SimpleNamespace(bos_token_id=0,eos_token_id=2,pad_token_id=1,decode=lambda seq,**kw:' '.join(map(str,seq)))
        b,a=signature(self.bart),signature(self.adapter)
        before=generate(self.bart,tok,self.memory,self.adapter,self.aux,self.gate)
        teacher(self.bart,self.memory,self.adapter,self.aux,self.gate,self.ids)
        after=generate(self.bart,tok,self.memory,self.adapter,self.aux,self.gate)
        self.assertEqual(before,after);self.assertEqual(signature(self.bart),b);self.assertEqual(signature(self.adapter),a)
        self.assertTrue(all(p.grad is None for p in list(self.bart.parameters())+list(self.adapter.parameters())))
    def test_real_exception_cleans_trace_hooks_and_adapter_context(self):
        with self.assertRaisesRegex(ValueError,'shape'):
            with Trace(self.adapter,torch.ones(9,1,dtype=torch.bool),torch.ones(9,1,dtype=torch.bool),True):
                with torch.no_grad():ce(self.bart,self.adapter,self.memory,self.aux,self.gate,self.ids)
        self.assertIsNone(self.adapter.context)
        self.assertTrue(all(not x._forward_hooks for x in self.adapter.cross))
    def test_empty_content_measurements_null_and_gold_PAD_has_no_loss(self):
        value=teacher(self.bart,self.memory,self.adapter,self.aux,self.gate,self.ids)
        for slot in value['slots'][1:]:
            self.assertEqual(slot['counts']['content'],0);self.assertEqual(slot['nll']['content'],0)
        for layer in value['trace']['layers']:
            self.assertTrue(all(s['effective_ratio'] is None for s in layer['scopes']['content'][1:]))
        self.assertEqual(value['counts']['all'],sum(s['counts']['all'] for s in value['slots']))

if __name__=='__main__':unittest.main(verbosity=2)
