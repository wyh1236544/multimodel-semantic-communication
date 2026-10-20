"""Mandatory actual tiny BLIP-2/BART tests on original server environment."""
from pathlib import Path
import sys
import tempfile
import unittest
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT),str(EXT.parent)]
from docci_caption_calibration_v24.test_hf import ActualHFContracts
from docci_caption_calibration_v24_fix1.fix import repaired,model_view
from docci_blip_feature_rx_v25.core import PROJECT

class ModelContracts(ActualHFContracts):
    def bart(self):
        import torch
        from transformers import BartConfig,BartForConditionalGeneration
        torch.manual_seed(25)
        c = BartConfig(vocab_size=32,d_model=32,encoder_layers=1,decoder_layers=2,
            encoder_attention_heads=4,decoder_attention_heads=4,encoder_ffn_dim=64,decoder_ffn_dim=64,
            max_position_embeddings=128,bos_token_id=0,eos_token_id=2,pad_token_id=1,
            decoder_start_token_id=2,forced_eos_token_id=None,dropout=0.,attention_dropout=0.)
        return BartForConditionalGeneration(c).requires_grad_(False).eval()
    def inputs(self,b=1):
        import torch
        torch.manual_seed(250)
        memory=torch.randn(b,9,32,32);aux=torch.randn(b,32,32);gate=torch.ones(b,9)
        ids=torch.ones(b,9,64,dtype=torch.long);ids[:,:,0]=0;ids[:,:,1]=2;ids[:,0,:4]=torch.tensor([0,4,5,2])
        return memory,aux,gate,ids
    def adapter(self,bart):
        from docci_blip_feature_rx_v25.model import Adapter
        return Adapter(bart,dict(width=16,heads=4,decoder_layers=2)).eval()
    def logits(self,bart,adapter,memory,aux,gate):
        import torch
        from transformers.modeling_outputs import BaseModelOutput
        n=memory.shape[0]*9
        with adapter.use(memory,aux,gate):
            return bart(encoder_outputs=BaseModelOutput(last_hidden_state=memory.flatten(0,1)),
                attention_mask=torch.ones(n,32,dtype=torch.long),
                decoder_input_ids=torch.tensor([2,0,4]).expand(n,-1),use_cache=False,return_dict=True).logits
    def test_actual_QFormer_matches_native_generate_and_extract_has_no_LM_forward(self):
        import torch
        from docci_blip_feature_rx_v25.visual import extract
        from docci_caption_calibration_v24.runtime import generate_one
        lm=[];q=[]
        h=self.model.language_model.register_forward_hook(lambda *args:lm.append(True))
        try:
            feature,trace=extract(self.model,self.processor,object(),'cpu',torch.float32,lambda:None)
            self.assertEqual(lm,[]);self.assertNotIn('text',self.processor.last)
            self.assertEqual(list(feature.shape),[1,4,32]);self.assertFalse(trace['input_has_text'])
        finally:h.remove()
        h=self.model.qformer.register_forward_hook(lambda module,args,output:q.append(output.last_hidden_state.detach().clone()))
        try:generate_one(model_view(self.model),self.processor,object(),None,'caption','cpu',torch.float32,lambda:None)
        finally:h.remove()
        torch.testing.assert_close(feature,q[0],rtol=0,atol=0)
    def test_zero_adapter_and_disabled_gate_preserve_actual_BART_generation(self):
        import torch
        from types import SimpleNamespace
        from docci_blip_feature_rx_v25.model import generate
        b=self.bart();memory,aux,gate,_=self.inputs();adapter=self.adapter(b)
        tokenizer=SimpleNamespace(bos_token_id=0,eos_token_id=2,pad_token_id=1,
            decode=lambda ids,**kw:' '.join(str(i) for i in ids))
        try:
            original=generate(b,tokenizer,memory)
            self.assertEqual(generate(b,tokenizer,memory,adapter,aux,gate),original)
            with torch.no_grad():
                for cross in adapter.cross:cross.alpha.fill_(.4)
            self.assertEqual(generate(b,tokenizer,memory,adapter,aux,gate,False),original)
        finally:adapter.close()
    def test_all_adapter_groups_learn_frozen_BART_and_real_checkpoint_roundtrip(self):
        import torch
        from docci_blip_feature_rx_v25.model import ce
        from docci_blip_feature_rx_v25.report import parameter_count
        from docci_longclip_joint_v8.backbones import signature
        b=self.bart();before=signature(b);memory,aux,gate,ids=self.inputs();adapter=self.adapter(b)
        initial={n:p.detach().clone() for n,p in adapter.named_parameters()};seen=set()
        opt=torch.optim.AdamW(adapter.parameters(),lr=.01)
        try:
            self.assertEqual(sum(p.numel() for p in adapter.parameters()),parameter_count(32,16,2))
            for _ in range(3):
                opt.zero_grad(set_to_none=True);loss,n=ce(b,adapter,memory,aux,gate,ids)
                self.assertEqual(n,20);(loss/n).backward()
                seen.update(name for name,p in adapter.named_parameters() if p.grad is not None and bool(p.grad.abs().sum()))
                opt.step();self.assertIsNone(adapter.context)
            for group,prefix in adapter.groups().items():
                names=[n for n in initial if n.startswith(prefix)]
                self.assertTrue(any(n in seen for n in names),group)
                self.assertTrue(any(not torch.equal(initial[n],dict(adapter.named_parameters())[n]) for n in names),group)
            self.assertEqual(signature(b),before);self.assertTrue(all(p.grad is None and not p.requires_grad for p in b.parameters()))
            with tempfile.TemporaryDirectory(prefix='V25_real_tiny_',dir=PROJECT/'experiment_results') as t:
                path=Path(t)/'adapter.pth';torch.save(adapter.state_dict(),path)
                saved=torch.load(path,map_location='cpu');adapter.load_state_dict(saved,strict=True)
                self.assertTrue(all(torch.equal(saved[n],v) for n,v in adapter.state_dict().items()))
        finally:adapter.close()
    def test_actual_visual_content_changes_logits_and_zero_content_gate_is_identity(self):
        import torch
        b=self.bart();memory,aux,gate,_=self.inputs();adapter=self.adapter(b)
        try:
            with torch.no_grad():
                for cross in adapter.cross:cross.alpha.fill_(.4)
                x=self.logits(b,adapter,memory,aux,gate)
                y=self.logits(b,adapter,memory,aux+torch.randn_like(aux),gate)
                self.assertFalse(torch.equal(x,y))
                z=self.logits(b,adapter,memory,aux,torch.zeros_like(gate))
                for cross in adapter.cross:cross.alpha.zero_()
                identity=self.logits(b,adapter,memory,aux,gate)
                torch.testing.assert_close(z,identity,rtol=0,atol=0)
        finally:adapter.close()
    def test_batch_isolation_and_context_cleanup_after_exception(self):
        import torch
        b=self.bart();memory,aux,gate,_=self.inputs(2);adapter=self.adapter(b)
        try:
            with torch.no_grad():
                for cross in adapter.cross:cross.alpha.fill_(.4)
                x=self.logits(b,adapter,memory,aux,gate);modified=aux.clone();modified[1].add_(1)
                y=self.logits(b,adapter,memory,modified,gate)
                torch.testing.assert_close(x[:9],y[:9],rtol=0,atol=0)
            with self.assertRaisesRegex(RuntimeError,'UNIT'):
                with adapter.use(memory,aux,gate):raise RuntimeError('UNIT')
            self.assertIsNone(adapter.context)
        finally:adapter.close()
        self.assertTrue(all(not l.encoder_attn_layer_norm._forward_hooks for l in b.model.decoder.layers))
    def test_self_tokens_use_only_received_memory_and_soft_gate(self):
        import torch
        from docci_blip_feature_rx_v25.model import self_tokens
        memory,aux,gate,_=self.inputs()
        before=self_tokens(memory,gate);aux.add_(100)
        torch.testing.assert_close(before,self_tokens(memory,gate),rtol=0,atol=0)
        self.assertEqual(list(before.shape),[1,32,32])
        self.assertTrue(torch.equal(self_tokens(memory,torch.zeros_like(gate)),torch.zeros_like(before)))

if __name__=='__main__':
    with repaired():
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ModelContracts))
    if not result.wasSuccessful():raise SystemExit(1)
