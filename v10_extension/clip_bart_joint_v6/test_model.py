"""Mandatory Torch/HF tiny-model tests on server; no network/download/skip."""
from pathlib import Path
import sys
import unittest
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
import torch
from torch import nn
from transformers import BartConfig,BartForConditionalGeneration
from clip_bart_joint_v6.model import JointCodec,masked_wire
from clip_bart_joint_v6.core import load
from clip_bart_joint_v5.data import clean_args
from bart_latent_codec_v4.core import load as text_config
from bart_latent_codec_v4.model import prediction


class Reference(nn.Module):
    def __init__(self): super().__init__(); self.gain=nn.Parameter(torch.tensor(1.))
    def receive(self,wire,reliability,snr,available):
        latent=wire[:,32:256,:4].mean(1)[:,:,None,None].expand(-1,-1,16,16)*self.gain
        return dict(latent=latent*available[:,None,None,None])


class TorchContracts(unittest.TestCase):
    def fixture(self,arm='J1_joint'):
        torch.manual_seed(7); torch.set_num_threads(1)
        cfg=BartConfig(vocab_size=31,d_model=32,encoder_layers=1,decoder_layers=1,encoder_attention_heads=4,
            decoder_attention_heads=4,encoder_ffn_dim=64,decoder_ffn_dim=64,max_position_embeddings=128,
            dropout=0.,attention_dropout=0.,activation_dropout=0.,bos_token_id=0,pad_token_id=1,eos_token_id=2,decoder_start_token_id=2)
        bart=BartForConditionalGeneration(cfg).eval().requires_grad_(False)
        tc=text_config(); tc.update(head_dim=8,depth=1); c=load(); c.update(hidden=16,heads=4)
        model=JointCodec(bart,Reference(),tc,c,8).eval(); model.configure(arm)
        ids=torch.ones(1,128,dtype=torch.long); ids[:,:5]=torch.tensor([0,4,5,6,2]); mask=ids.ne(1)
        with torch.no_grad(): h=bart.model.encoder(input_ids=ids,attention_mask=mask.long()).last_hidden_state
        inputs=[h,mask,torch.randn(1,224,16),torch.randn(1,8),torch.randn(1,8)]
        return model,inputs,ids
    def test_initial_text_exactly_preserves_private_codec(self):
        m,x,_=self.fixture(); wire=m.source_encode(*x); out=m.receive(*clean_args(wire))
        private=m.text.receive_text(wire[:,4096:])
        self.assertTrue(torch.equal(out['text']['memory'],private['memory']))
    def test_source_separate_image_and_text_independent(self):
        m,x,_=self.fixture('J0_separate'); a=m.source_encode(*x)
        changed=[v.clone() for v in x]; changed[0].add_(1); changed[4].mul_(-2); b=m.source_encode(*changed)
        self.assertTrue(torch.equal(a[:,:256],b[:,:256])); self.assertTrue(torch.equal(a[:,512:4096],b[:,512:4096]))
        changed=[v.clone() for v in x]; changed[2].mul_(-1); changed[3].add_(3); d=m.source_encode(*changed)
        self.assertTrue(torch.equal(a[:,256:512],d[:,256:512])); self.assertTrue(torch.equal(a[:,4096:],d[:,4096:]))
    def test_source_no_clip_text_is_invariant(self):
        m,x,_=self.fixture('J2_no_clip_text'); a=m.source_encode(*x)
        x[-1]=x[-1]*10+2; b=m.source_encode(*x); self.assertTrue(torch.equal(a,b))
    def test_rx_separate_has_no_cross_modal_information(self):
        m,x,_=self.fixture('J0_separate')
        with torch.no_grad(): m.text_residual.weight.fill_(.01); m.image_residual.weight.fill_(.01)
        wire=m.source_encode(*x); a=m.receive(*clean_args(wire))
        changed=wire.clone(); changed[:,:256]*=-1; changed[:,512:4096]*=-1
        b=m.receive(*clean_args(changed)); self.assertTrue(torch.equal(a['text']['memory'],b['text']['memory']))
        changed=wire.clone(); changed[:,256:512]*=-1; changed[:,4096:]*=-1
        b=m.receive(*clean_args(changed)); self.assertTrue(torch.equal(a['latent'],b['latent']))
    def test_all_arms_have_same_registered_bridge_count(self):
        counts=[]
        for arm in ('J0_separate','J1_joint','J2_no_clip_text'):
            m,_,_=self.fixture(arm); counts.append(sum(v.numel() for v in m.bridge_state().values()))
        self.assertEqual(len(set(counts)),1)
    def test_unit_group_power_and_wire_shape(self):
        m,x,_=self.fixture(); wire=m.source_encode(*x); self.assertEqual(wire.shape,(1,6144))
        for a,b in [(0,256),(256,512),(512,4096),(4096,6144)]: self.assertAlmostEqual(float(wire[:,a:b].square().mean()),1.,places=5)
    def test_no_oracle_receiver_and_erasure(self):
        m,x,_=self.fixture(); wire=m.source_encode(*x)
        with self.assertRaises(TypeError): m.receive(*clean_args(wire),source_mask=x[1])
        args=list(clean_args(wire)); args[1]=torch.tensor([False]); out=m.receive(*args)
        self.assertEqual(float(out['text']['memory'].abs().sum()),0.); self.assertEqual(float(out['latent'].abs().sum()),0.)
    def test_required_groups_learn_and_backbones_frozen(self):
        m,x,ids=self.fixture(); initial=m.bridge_state(); seen=set()
        frozen={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001)
        for _ in range(4):
            out=m.receive(*clean_args(m.source_encode(*x)))
            pred,labels=prediction(m.text.bart,ids,x[1],out['text']['memory'],out['text']['receiver_mask'])
            loss=torch.nn.functional.cross_entropy(pred.reshape(-1,31),labels.reshape(-1),ignore_index=-100)
            loss+=out['latent'].square().mean()+out['alignment_image'].square().mean()+out['alignment_text'].square().mean()+out['coarse_latent'].square().mean()
            cp,cl=prediction(m.text.bart,ids,x[1],out['common_text']['memory'],out['common_text']['receiver_mask'])
            loss+=torch.nn.functional.cross_entropy(cp.reshape(-1,31),cl.reshape(-1),ignore_index=-100)
            loss.backward(); seen.update(n for n,p in m.named_parameters() if p.grad is not None and float(p.grad.abs().sum())>0)
            opt.step(); opt.zero_grad(set_to_none=True)
        after=m.bridge_state()
        for prefixes in m.groups().values():
            self.assertTrue(any(n.startswith(prefixes) for n in seen))
            self.assertTrue(any(n.startswith(prefixes) and not torch.equal(v,after[n]) for n,v in initial.items()))
        self.assertTrue(all(p.grad is None and torch.equal(frozen[n],p) for n,p in m.named_parameters() if n in frozen))
    def test_grad_no_grad_actual_interfaces(self):
        m,x,_=self.fixture(); a=m.source_encode(*x); out=m.receive(*clean_args(a))
        with torch.no_grad(): b=m.source_encode(*x); other=m.receive(*clean_args(b))
        self.assertTrue(torch.equal(a.detach(),b)); self.assertTrue(torch.equal(out['text']['memory'].detach(),other['text']['memory']))
        self.assertTrue(torch.equal(out['latent'].detach(),other['latent']))
    def test_warm_freeze_and_restore(self):
        m,_,_=self.fixture(); state=m.bridge_state(); m.configure('J1_joint',True)
        self.assertFalse(any(p.requires_grad for p in m.text.parameters()))
        m.configure('J1_joint',False)
        self.assertTrue(m.text.queries.requires_grad); self.assertFalse(any(p.requires_grad for p in m.text.bart.parameters()))
        with torch.no_grad(): m.common_queries.add_(.1)
        m.restore(state); self.assertTrue(torch.equal(state['common_queries'],m.common_queries))



class AddedContracts(TorchContracts):
    # Subclass inherits actual fixed-memory and source independence contracts.
    def test_common_caption_uses_only_received_c(self):
        m,x,_=self.fixture(); a=m.source_encode(*x); b=a.clone(); b[:,512:]=torch.randn_like(b[:,512:])
        oa=m.receive(*clean_args(a))['common_text']['memory']
        ob=m.receive(*clean_args(b))['common_text']['memory']
        self.assertTrue(torch.equal(oa,ob))
    def test_common_warm_freezes_private_and_normal_fusion(self):
        m,_,_=self.fixture(); m.phase='common'; m.configure(m.arm)
        self.assertFalse(any(p.requires_grad for n,p in m.named_parameters() if n.startswith(('text.','image_tx.','image_rx.','text_residual.'))))
        self.assertTrue(m.caption_projection.weight.requires_grad); self.assertTrue(m.pool_image.out.weight.requires_grad)
    def test_train_mask_preserves_c_and_fixed_shape(self):
        from clip_bart_joint_v6.core import private_mask
        m,x,_=self.fixture(); a=m.source_encode(*x); b=masked_wire(a,private_mask(load(),3,42,'clean'))
        self.assertTrue(torch.equal(a[:,:512],b[:,:512])); self.assertEqual(a.shape,b.shape)
    def test_prompt_adapter_zero_init_and_gradients(self):
        from clip_bart_joint_v6.prompt_adapter import PromptAdapter
        torch.manual_seed(9); a=PromptAdapter(32,16,4); common=torch.randn(2,32,16); base=torch.randn(2,77,32)
        self.assertTrue(torch.equal(a(common,base),base))
        target=torch.randn_like(base); seen=set(); before={n:p.detach().clone() for n,p in a.named_parameters()}
        opt=torch.optim.AdamW(a.parameters(),lr=.001)
        for _ in range(4):
            loss=(a(common,base)-target).square().mean(); loss.backward()
            seen.update(n for n,p in a.named_parameters() if p.grad is not None and bool(p.grad.abs().sum()>0))
            opt.step(); opt.zero_grad(set_to_none=True)
        for prefix in ('common.','attention.','output.','gate'):
            self.assertTrue(any(n.startswith(prefix) for n in seen))
            self.assertTrue(any(n.startswith(prefix) and not torch.equal(before[n],p) for n,p in a.named_parameters()))
    def test_receiver_batch_two_erasure_shape(self):
        m,x,_=self.fixture(); wire=m.source_encode(*x).repeat(2,1)
        args=list(clean_args(wire)); args[1]=torch.tensor([True,False])
        out=m.receive(*args)
        self.assertEqual(out['latent'].shape,(2,4,16,16))
        self.assertEqual(float(out['latent'][1].abs().sum()),0.)
        self.assertEqual(float(out['common_text']['memory'][1].abs().sum()),0.)
    def test_common_warm_required_groups_learn_without_private_updates(self):
        m,x,ids=self.fixture(); before=m.bridge_state(); m.phase='common'; m.configure(m.arm); seen=set()
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001)
        for _ in range(4):
            out=m.receive(*clean_args(m.source_encode(*x)))
            cp,cl=prediction(m.text.bart,ids,x[1],out['common_text']['memory'],out['common_text']['receiver_mask'])
            loss=torch.nn.functional.cross_entropy(cp.reshape(-1,31),cl.reshape(-1),ignore_index=-100)
            loss+=out['alignment_image'].square().mean()+out['alignment_text'].square().mean()+out['coarse_latent'].square().mean()
            loss.backward(); seen.update(n for n,p in m.named_parameters() if p.grad is not None and bool(p.grad.abs().sum()>0))
            opt.step(); opt.zero_grad(set_to_none=True)
        after=m.bridge_state()
        for prefixes in m.groups().values():
            self.assertTrue(any(n.startswith(prefixes) for n in seen))
            self.assertTrue(any(n.startswith(prefixes) and not torch.equal(v,after[n]) for n,v in before.items()))
        for n,v in before.items():
            if n.startswith(('text.','image_tx.','image_rx.','text_attention.','text_residual.','coarse_gate','caption_mix_gate')):
                self.assertTrue(torch.equal(v,after[n]),n)


if __name__=='__main__': unittest.main(verbosity=2)
