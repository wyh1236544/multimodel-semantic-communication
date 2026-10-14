"""Mandatory server Torch/Transformers CPU contracts, never labelled local pass."""
import hashlib
import inspect
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
try:
    import torch
    from transformers import BartConfig,BartForConditionalGeneration
except ImportError:
    if os.environ.get('V9_REQUIRE_TORCH')=='1': raise
    print('NOT_EXECUTED: local Torch/Transformers unavailable; mandatory server test_model + authentic GPU smoke remain.'); sys.exit(0)
import numpy as np
from docci_modal_v9.core import load,digest
from docci_modal_v9.model import TextBranch,ImageBranch,warm,generate
from docci_modal_v9.channel import clean,noisy
from docci_modal_v9.workflow import choose_indices,fit
torch.set_num_threads(2)


def text_model():
    bart=BartForConditionalGeneration(BartConfig(vocab_size=64,d_model=128,encoder_layers=1,decoder_layers=1,
        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=256,decoder_ffn_dim=256,max_position_embeddings=1024,
        bos_token_id=0,eos_token_id=2,pad_token_id=1,decoder_start_token_id=2))
    return TextBranch(bart,load())


class Models(unittest.TestCase):
    def test_real_optimizer_updates_disk_reload_and_backbone_freeze(self):
        m=text_model(); c=load(); items=[]
        for i in range(2):
            ids=np.ones(512,np.int64); ids[:8]=[0,3,4,5,6,7,8,2]
            items.append(dict(example_id='train_'+str(i),ids=ids,bart_hidden=np.random.default_rng(i).standard_normal((512,128)).astype(np.float32)))
        before={n:p.detach().clone() for n,p in m.bart.named_parameters()}; calls=[]; original=torch.optim.AdamW.step
        def counted(opt,*args,**kwargs): calls.append(1); return original(opt,*args,**kwargs)
        with tempfile.TemporaryDirectory(dir=str(EXT.parent),prefix='v9_torch_contract_') as d:
            with mock.patch.object(torch.optim.AdamW,'step',counted): saved=fit(m,items,c,m.bart,torch.device('cpu'),Path(d)/'train_overfit','overfit',2,lambda:None)
            self.assertEqual(len(calls),2); self.assertTrue((Path(d)/'train_overfit/last.pth').is_file())
            self.assertTrue(all(torch.equal(v,m.saved()[n]) for n,v in saved.items()))
        self.assertTrue(all(torch.equal(before[n],p) and p.grad is None for n,p in m.bart.named_parameters()))
    def test_RX_boundary_cannot_accept_source(self):
        for cls in (TextBranch,ImageBranch): self.assertEqual(list(inspect.signature(cls.receive).parameters),['self','observed','available','pilot_snr','reliability'])
    def test_text_padding_power_failure_and_gradients(self):
        m=text_model(); ids=torch.ones(2,512,dtype=torch.long); ids[:,:8]=3; hidden=torch.randn(2,512,128)
        x=m.encode(dict(ids=ids,bart_hidden=hidden)); hidden[:,8:]=500; again=m.encode(dict(ids=ids,bart_hidden=hidden))
        self.assertTrue(torch.allclose(x,again)); self.assertAlmostEqual(float(x.square().mean()),1.,places=5)
        output=m.receive(x,torch.zeros(2,dtype=torch.bool),x.new_zeros(2),x.new_zeros(2)); self.assertFalse(bool(output['memory'].any()))
        y,ok,snr,rel,_=noisy(x,m.c,1,['a','b']); output=m.receive(y,ok,snr,rel)
        output['memory'][:,:,0].square().mean().backward()
        for prefixes in m.groups().values(): self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for n,p in m.named_parameters() if n.startswith(prefixes)))
        self.assertFalse(any(p.requires_grad or p.grad is not None for p in m.bart.parameters()))
    def test_historical_text_prefix_preserved_new_zero(self):
        m=text_model(); state=m.saved(); old={n:v.clone() for n,v in state.items()}; old['source_position']=old['source_position'][:,:128].clone()
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/'last.pth'; torch.save(dict(arm='P0_ce',width=64,state=old),str(file))
            warm(m,dict(checkpoint=str(file),sha256=digest(file),mode='historical_weights_only'))
        now=m.saved(); self.assertTrue(torch.equal(now['queries'],old['queries'])); self.assertTrue(torch.equal(now['source_position'][:,:128],old['source_position']))
        self.assertFalse(bool(now['source_position'][:,128:].any())); self.assertEqual(now['memory_position'].shape[1],32)
    def test_image_only_shape_group_gradients_and_failure(self):
        m=ImageBranch(load(image_size=256)); x=m.encode(dict(latent=torch.randn(1,4,32,32)))
        self.assertEqual(tuple(x.shape),(1,2048)); self.assertAlmostEqual(float(x.square().mean()),1.,places=5)
        output=m.receive(*clean(x)); output['latent'].square().mean().backward()
        for prefixes in m.groups().values(): self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for n,p in m.named_parameters() if n.startswith(prefixes)))
        self.assertFalse(hasattr(m,'bart')); self.assertFalse(hasattr(m,'vae'))
        failed=m.receive(x,torch.zeros(1,dtype=torch.bool),x.new_zeros(1),x.new_zeros(1)); self.assertFalse(bool(failed['latent'].any()))
    def test_image_history_whitelist_and_checkpoint_roundtrip(self):
        m=ImageBranch(load()); state=m.saved()
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/'last.pth'; torch.save(dict(schema='docci-longclip-joint-v8',stage='image',config=dict(image_size=512),state=dict(state,**{'text.failed.weight':torch.ones(2)})),str(file))
            record=warm(m,dict(checkpoint=str(file),sha256=digest(file),mode='historical_weights_only'))
            self.assertTrue(record['discarded_other_modal_and_Common_tensors']); self.assertTrue(all(torch.equal(v,m.saved()[n]) for n,v in state.items()))
    def test_image_accumulation_order_no_overlap_cursor(self):
        pairs=[choose_indices(8,1,u,j,42,2)[0] for u in range(4) for j in range(2)]
        self.assertEqual(len(set(pairs)),8)
    def test_greedy_uses_no_repetition_mask_or_forced_EOS(self):
        m=text_model(); captured={}
        def fake(**kwargs): captured.update(kwargs); return torch.tensor([[2,0,5,2]])
        m.bart.generate=fake
        class Tok:
            bos_token_id=0; eos_token_id=2; pad_token_id=1
            def decode(self,x,**kwargs): return 'a'
        x=m.receive(*clean(torch.ones(1,2048))); result=generate(m,x,Tok())
        self.assertFalse(result['stopped_by_capacity']); self.assertIsNone(captured['forced_eos_token_id']); self.assertEqual(captured['no_repeat_ngram_size'],0)
        self.assertEqual(captured['repetition_penalty'],1.)


if __name__=='__main__': unittest.main(verbosity=2)
