"""Real numerical contracts, mandatory on server; explicit local dependency skip."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
AVAILABLE=all(importlib.util.find_spec(n) is not None for n in ('torch','transformers'))
if os.environ.get('V18_REQUIRE_TORCH')=='1' and not AVAILABLE:raise RuntimeError('Existing Torch/Transformers required; no dependency upgrade')
if AVAILABLE:
    import torch
    from transformers import BartConfig,BartForConditionalGeneration
    from docci_baseline_v14.model import TextModel
    from docci_text_channel_v18.channel import transmit,objective

@unittest.skipUnless(AVAILABLE,'Torch/Transformers unavailable locally: numerical contracts NOT_EXECUTED')
class Neural(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        c=BartConfig(vocab_size=97,d_model=64,encoder_layers=1,decoder_layers=1,
            encoder_attention_heads=4,decoder_attention_heads=4,encoder_ffn_dim=128,decoder_ffn_dim=128,
            max_position_embeddings=128,pad_token_id=1,bos_token_id=0,eos_token_id=2,decoder_start_token_id=2,
            dropout=0.,attention_dropout=0.,activation_dropout=0.,forced_eos_token_id=None)
        self.model=TextModel(BartForConditionalGeneration(c),'released64').eval()
        self.ids=torch.tensor([0,4,5,6,2]+[1]*59)[None,None].expand(1,9,-1).clone()
        self.ids[:,1:]=torch.tensor([0,2]+[1]*62)
        with torch.no_grad():self.hidden=self.model.bart.model.encoder(input_ids=self.ids[0],attention_mask=self.ids[0].ne(1).long()).last_hidden_state[None]
        self.frozen={n:p.clone() for n,p in self.model.bart.named_parameters()}
    def tearDown(self):
        self.assertTrue(all(not p.requires_grad and p.grad is None and torch.equal(p,self.frozen[n]) for n,p in self.model.bart.named_parameters()))
    def test_noisy_CE_reaches_transmitter_and_receiver_and_both_update(self):
        active={n:p for n,p in self.model.named_parameters() if p.requires_grad};before={n:p.clone() for n,p in active.items()}
        opt=torch.optim.AdamW(list(active.values()),lr=.001)
        wire=self.model.encode(self.ids,self.hidden);obs,_=transmit(wire,0,123)
        summed,n=objective(self.model,self.ids,self.hidden,obs);(summed/n).backward()
        for prefixes in self.model.groups().values():
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for name,p in active.items() if name.startswith(prefixes)))
        opt.step()
        for prefixes in self.model.groups().values():self.assertTrue(any(not torch.equal(before[n],p) for n,p in active.items() if n.startswith(prefixes)))
    def test_clean_identity_unit_power_and_common_epsilon(self):
        wire=self.model.encode(self.ids,self.hidden);clean,s=transmit(wire,None,42)
        self.assertTrue(torch.equal(clean,wire));self.assertAlmostEqual(s['signal_power'],1.,places=5)
        a,s=transmit(wire,10,42);b,t=transmit(wire,10,42)
        self.assertTrue(torch.equal(a,b));self.assertEqual(s,t);self.assertFalse(torch.equal(a,wire))
        altered=wire.detach().clone()*2
        with self.assertRaises(ValueError):transmit(altered,0,42)
    def test_no_source_PAD_leak_and_receiver_has_only_observed_argument(self):
        a=self.model.encode(self.ids,self.hidden);x=self.hidden.clone();x[self.ids.eq(1)]=12345.
        self.assertTrue(torch.allclose(a,self.model.encode(self.ids,x),atol=1e-6,rtol=1e-6))
        import inspect
        self.assertEqual(list(inspect.signature(self.model.receive).parameters),['observed'])
    def test_two_view_gradient_weighting_matches_sum(self):
        active=[p for p in self.model.parameters() if p.requires_grad]
        for p in active:p.grad=None
        w=self.model.encode(self.ids,self.hidden);o,_=transmit(w,None,42);v,n=objective(self.model,self.ids,self.hidden,o);(v/n).backward();expected=[p.grad.clone() if p.grad is not None else None for p in active]
        for p in active:p.grad=None
        for weight in [.25,.75]:
            w=self.model.encode(self.ids,self.hidden);o,_=transmit(w,None,42);v,n=objective(self.model,self.ids,self.hidden,o);(v/n*weight).backward()
        self.assertTrue(all(e is None and p.grad is None or e is not None and torch.allclose(e,p.grad,atol=3e-6,rtol=3e-4) for p,e in zip(active,expected)))
    def test_saved_codec_checkpoint_roundtrip(self):
        state=self.model.saved()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'codec.pth';torch.save(state,str(p));self.model.restore(torch.load(str(p),map_location='cpu'))
        actual=self.model.saved();self.assertTrue(all(torch.equal(v,actual[k]) for k,v in state.items()))
if __name__=='__main__':unittest.main(verbosity=2)
