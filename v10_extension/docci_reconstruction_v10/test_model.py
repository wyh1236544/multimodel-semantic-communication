"""Mandatory SERVER Torch/Transformers tests; skip locally means NOT_EXECUTED."""
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
    if os.environ.get('V10_REQUIRE_TORCH')=='1': raise
    print('NOT_EXECUTED: Torch/Transformers unavailable locally. Mandatory server CPU tests + real256/512 GPU smoke pending.'); sys.exit(0)
import numpy as np
from docci_reconstruction_v10.core import load,chunks,chunk_plan,geometry,digest
from docci_reconstruction_v10.model import TextBranch,ImageBranch,Window,warm,generate
from docci_reconstruction_v10.workflow import fit
from docci_reconstruction_v10.channel import clean,noisy
torch.set_num_threads(2)


def bart():
    return BartForConditionalGeneration(BartConfig(vocab_size=128,d_model=128,encoder_layers=1,decoder_layers=1,
        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=256,decoder_ffn_dim=256,max_position_embeddings=1024,
        bos_token_id=0,eos_token_id=2,pad_token_id=1,decoder_start_token_id=2))


def source(c):
    ids=torch.ones(1,512,dtype=torch.long); ids[0,:6]=torch.tensor([0,3,4,5,6,2])
    chunk=torch.tensor(chunk_plan([3,4,5,6],c))[None]
    return dict(ids=ids,bart_hidden=torch.randn(1,512,128),chunk_ids=chunk,chunk_hidden=torch.randn(1,9,64,128))


class Models(unittest.TestCase):
    def test_fresh_shared_text_initialization_and_public_receive(self):
        c=load(); b=bart(); torch.manual_seed(42); a=TextBranch(b,c,'T0_whole'); torch.manual_seed(42); d=TextBranch(b,c,'T1_chunks')
        for k,v in a.saved().items(): self.assertTrue(torch.equal(v[:,:64] if k=='source_position' else v,d.saved()[k]),k)
        self.assertEqual(list(inspect.signature(d.receive).parameters),['observed','available','pilot_snr','reliability'])
        wire=d.encode(source(c)); out=d.receive(*clean(wire)); self.assertEqual(wire.shape,(1,18432)); self.assertEqual(out['memory'].shape,(1,9,32,128))
        self.assertAlmostEqual(float(wire.square().mean()),1.,places=5)
    def test_PAD_features_cannot_leak_and_failed_RX_is_zero(self):
        c=load(); m=TextBranch(bart(),c,'T1_chunks'); src=source(c); original=m.encode(src)
        src['chunk_hidden'][src['chunk_ids']==1]=torch.randn_like(src['chunk_hidden'][src['chunk_ids']==1])*100
        self.assertTrue(torch.allclose(original,m.encode(src),atol=1e-5))
        out=m.receive(original,torch.tensor([False]),torch.tensor([40.]),torch.tensor([1.]))
        self.assertFalse(bool(out['memory'].any())); self.assertFalse(bool(out['available'][0]))
    def test_shift_mask_wrap_and_cross_window_gradients(self):
        module=Window(64,shift=4); mask=module.mask(16,16,torch.device('cpu'))
        self.assertEqual(mask.shape,(4,64,64)); self.assertTrue(bool(mask.any())); self.assertFalse(bool(mask.diagonal(dim1=-2,dim2=-1).any()))
        x=torch.randn(1,64,16,16,requires_grad=True); module(x).square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all()); self.assertGreater(float(x.grad.abs().sum()),0)
        self.assertGreater(float(module.relative.grad.abs().sum()),0)
    def test_multiscale_encoder_decoder_budget_and_real_all_group_gradients(self):
        c=load(image_size=256); m=ImageBranch(c,'I2_shifted'); x=torch.randn(1,4,32,32); wire=m.encode(dict(latent=x)); output=m.receive(*clean(wire))
        self.assertEqual(wire.shape,(1,2048)); self.assertEqual(output['latent'].shape,x.shape)
        self.assertAlmostEqual(float(wire.square().mean()),1.,places=5)
        (output['latent']-x).square().mean().backward()
        for prefix in ('image_encoder.','image_decoder.'):
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for n,p in m.named_parameters() if n.startswith(prefix)))
    def test_same_CNN_loss_arms_and_no_clean_encoder_skip_at_RX(self):
        c=load(image_size=256); torch.manual_seed(42); a=ImageBranch(c,'I0_latent'); torch.manual_seed(42); b=ImageBranch(c,'I1_pixel')
        self.assertTrue(all(torch.equal(v,b.saved()[k]) for k,v in a.saved().items()))
        self.assertEqual(list(inspect.signature(a.receive).parameters),['observed','available','pilot_snr','reliability'])
    def test_explicit_historical64_prefix_transfer(self):
        c=load(); old=TextBranch(bart(),c,'T0_whole'); state=old.saved(); state['source_position']=state['source_position'][:,:128]
        with tempfile.TemporaryDirectory(dir=str(EXT.parent),prefix='v10_warm_test_') as d:
            file=Path(d)/'own.pth'; torch.save(dict(arm='P0_ce',width=64,state=state),file)
            new=TextBranch(old.bart,c,'T1_chunks'); warm(new,dict(checkpoint=str(file),sha256=digest(file)))
            self.assertTrue(torch.equal(new.saved()['source_position'],state['source_position'][:,:64]))
            self.assertTrue(all(torch.equal(v,new.saved()[k]) for k,v in state.items() if k!='source_position'))
    def test_all_text_groups_update_frozen_backbone_and_checkpoint_roundtrip(self):
        c=load(); m=TextBranch(bart(),c,'T1_chunks'); src=source(c); items=[]
        for i in range(2): items.append(dict(example_id='train_'+str(i),**{k:v[0].numpy() for k,v in src.items()}))
        frozen={n:p.detach().clone() for n,p in m.bart.named_parameters()}; calls=[]; original=torch.optim.AdamW.step
        def counted(opt,*args,**kwargs): calls.append(1); return original(opt,*args,**kwargs)
        with tempfile.TemporaryDirectory(dir=str(EXT.parent),prefix='v10_fit_test_') as d:
            with mock.patch.object(torch.optim.AdamW,'step',counted): state=fit(m,items,c,m.bart,torch.device('cpu'),Path(d)/'clean','clean',2,lambda:None)
            self.assertEqual(len(calls),2); self.assertTrue((Path(d)/'clean/TRAIN_COMPLETE.json').is_file())
            self.assertTrue(all(torch.equal(v,m.saved()[k]) for k,v in state.items()))
        self.assertTrue(all(p.grad is None and torch.equal(p,frozen[n]) for n,p in m.bart.named_parameters()))
    def test_pixel_gradient_loss_probe_and_image_optimizer_roundtrip(self):
        c=load(image_size=256); m=ImageBranch(c,'I2_shifted')
        class FakeVAE:
            config=type('Config',(),dict(scaling_factor=1.))()
            def decode(self,x): return type('Output',(),dict(sample=torch.nn.functional.interpolate(x[:,:3],scale_factor=8,mode='nearest')))()
        items=[dict(example_id='train_a',latent=np.random.default_rng(1).standard_normal((4,32,32)).astype(np.float32),rgb=np.full((256,256,3),127,np.uint8),rectangle=[0,0,256,256])]
        with tempfile.TemporaryDirectory(dir=str(EXT.parent),prefix='v10_image_fit_') as d:
            fit(m,items,c,FakeVAE(),torch.device('cpu'),Path(d)/'clean','clean',2,lambda:None)
            import json
            row=json.loads((Path(d)/'clean/history.jsonl').read_text().splitlines()[0]); probes=row['loss_gradient_probes']
            self.assertGreater(probes[0]['pixel_weighted_gradient_norm'],0); self.assertGreater(probes[0]['latent_weighted_gradient_norm'],0)
    def test_generation_public_chunk_count_and_no_forcedEOS(self):
        c=load(); m=TextBranch(bart(),c,'T1_chunks'); calls=[]
        def decode(**kw):
            calls.append(kw); return torch.tensor([[2,0,3,2]]+[[2,0,2,1]]*8)
        class Tok:
            bos_token_id=0; eos_token_id=2; pad_token_id=1
            def decode(self,ids,**kw): return str(ids)
        with mock.patch.object(m.bart,'generate',decode): result=generate(m,m.receive(*clean(m.encode(source(c)))),Tok())
        self.assertEqual(result['generated_token_ids'],[3]); self.assertEqual(len(result['chunk_generated_token_ids']),9)
        self.assertIsNone(calls[0]['forced_eos_token_id']); self.assertEqual(calls[0]['no_repeat_ngram_size'],0)
    def test_noisy_symbol_gradients_remain_connected(self):
        c=load(); wire=torch.randn(1,18432,requires_grad=True); y,ok,snr,rel,stats=noisy(wire,c,2,['train_a'])
        y.square().mean().backward(); self.assertTrue(torch.isfinite(wire.grad).all()); self.assertGreater(float(wire.grad.abs().sum()),0)


if __name__=='__main__': unittest.main(verbosity=2)
