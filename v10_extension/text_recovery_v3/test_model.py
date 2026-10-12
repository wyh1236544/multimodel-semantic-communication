"""Real Torch/HF tiny models, no download. Server must run; no local skip."""
from pathlib import Path
import sys
import unittest
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
import torch
from transformers import BartConfig,BartForConditionalGeneration
from text_recovery_v3.core import load,geometry
from text_recovery_v3.model import TextBridge,install_lora,adapters,reset_lora,lora_state,LoRALinear,consistency,logits


class TorchContracts(unittest.TestCase):
    def fixture(self,arm='A3_lora'):
        torch.manual_seed(9); torch.set_num_threads(1)
        config=BartConfig(vocab_size=31,d_model=32,encoder_layers=2,decoder_layers=2,encoder_attention_heads=4,
            decoder_attention_heads=4,encoder_ffn_dim=64,decoder_ffn_dim=64,max_position_embeddings=128,
            dropout=0.,attention_dropout=0.,activation_dropout=0.,pad_token_id=1,bos_token_id=0,eos_token_id=2,decoder_start_token_id=2)
        bart=BartForConditionalGeneration(config).eval().requires_grad_(False); c=load(); c.update(hidden=16,layers=1)
        install_lora(bart,c); model=TextBridge(bart,c,arm,8).eval()
        ids=torch.ones(1,128,dtype=torch.long); ids[0,:5]=torch.tensor([0,4,5,6,2]); mask=ids.ne(1)
        with torch.no_grad(): hidden=bart.model.encoder(input_ids=ids,attention_mask=mask.long(),return_dict=True).last_hidden_state
        item=dict(input_ids=ids,mask=mask,bart_hidden=hidden,clip_text=torch.randn(1,8))
        return model,item
    def test_lora_zero_initial_identity(self):
        base=torch.nn.Linear(5,7).requires_grad_(False); l=LoRALinear(base,2); x=torch.randn(2,5)
        l.enabled=True; self.assertTrue(torch.equal(base(x),l(x))); l(x).sum().backward()
        self.assertIsNone(base.weight.grad); self.assertGreater(float(l.b.grad.abs().sum()),0)
    def test_header_from_observation_only(self):
        model,item=self.fixture(); x=model.transmit(item['bart_hidden'],item['mask'],item['clip_text'])
        out=model.receive_text(x); self.assertEqual(out['received_length'],[5]); self.assertEqual(int(out['receiver_mask'].sum()),5)
        corrupted=x.detach().clone(); corrupted[:,:2]*=-1
        self.assertNotEqual(model.receive_text(corrupted)['received_length'],[5])
        with self.assertRaises(TypeError): model.receive_text(x,source_length=5)
    def test_source_padding_cannot_leak(self):
        model,item=self.fixture(); h=item['bart_hidden'].clone(); h[:,5:]+=100
        a=model.transmit(item['bart_hidden'],item['mask'],item['clip_text']); b=model.transmit(h,item['mask'],item['clip_text'])
        self.assertTrue(torch.equal(a,b))
    def test_unit_power_and_resource_shapes(self):
        for arm in ('A0_legacy','A1_length_mask','A3_lora','A4_wide','A5_common'):
            model,item=self.fixture(arm); x=model.transmit(item['bart_hidden'],item['mask'],item['clip_text'])
            self.assertEqual(x.shape,(1,geometry(arm)['total_reals'])); self.assertAlmostEqual(float(x.square().mean()),1.,places=5)
    def test_teacher_and_incremental_logits(self):
        model,item=self.fixture(); check=consistency(model.bart,item,torch.device('cpu'))
        self.assertLess(check['incremental_max_logit_error'],1e-5); self.assertLess(check['cached_source_max_logit_error'],1e-5)
    def test_pad_labels_ignored(self):
        model,item=self.fixture(); out=model.receive_text(model.transmit(item['bart_hidden'],item['mask'],item['clip_text']))
        _,labels=logits(model.bart,item['input_ids'],item['mask'],out['memory'],out['receiver_mask'])
        self.assertTrue(torch.all(labels[:,5:]==-100)); self.assertEqual(int(labels[0,4]),2)
    def test_codec_and_lora_get_gradients(self):
        model,item=self.fixture(); opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
        before=model.state(); grads=set()
        for _ in range(3):
            out=model.receive_text(model.transmit(item['bart_hidden'],item['mask'],item['clip_text']))
            pred,labels=logits(model.bart,item['input_ids'],item['mask'],out['memory'],out['receiver_mask'])
            loss=torch.nn.functional.cross_entropy(pred.reshape(-1,31),labels.reshape(-1),ignore_index=-100)+(out['memory']-item['bart_hidden']).square().mean()
            loss.backward()
            grads.update(n for n,p in model.named_parameters() if p.grad is not None and float(p.grad.abs().sum())>0)
            opt.step(); opt.zero_grad(set_to_none=True)
        for prefixes in model.groups().values():
            self.assertTrue(any(n.startswith(prefixes) for n in grads))
            self.assertTrue(any(n.startswith(prefixes) and not torch.equal(v,model.state()[n]) for n,v in before.items()))
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if not p.requires_grad))
    def test_lora_disabled_and_reset(self):
        model,item=self.fixture(); initial=lora_state(model.bart)
        self.assertEqual(sum(v.numel() for v in initial.values()),2048)
        with torch.no_grad():
            for m in model.bart.modules():
                if isinstance(m,LoRALinear): m.b.add_(.1)
        reset_lora(model.bart,initial)
        self.assertTrue(all(torch.equal(x,lora_state(model.bart)[n]) for n,x in initial.items()))
        self.assertTrue(all(not m.enabled for m in model.bart.modules() if isinstance(m,LoRALinear)))
    def test_failure_zero_memory(self):
        model,item=self.fixture(); x=model.transmit(item['bart_hidden'],item['mask'],item['clip_text'])
        out=model.receive_text(x,available=torch.tensor([False])); self.assertFalse(bool(out['available'][0])); self.assertEqual(float(out['memory'].abs().sum()),0.)
    def test_saved_state_roundtrip(self):
        model,_=self.fixture(); state=model.state()
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad: p.add_(.01)
        model.restore(state); self.assertTrue(all(torch.equal(v,model.state()[n]) for n,v in state.items()))


if __name__=='__main__': unittest.main(verbosity=2)
