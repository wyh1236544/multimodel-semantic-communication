"""Tiny random-fixture Torch contracts, NOT pretrained recovery results."""
import inspect
import os
from pathlib import Path
import sys
import unittest
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
try:
    import torch
    from transformers import BartConfig,BartForConditionalGeneration
except ImportError:
    if os.environ.get('V8_REQUIRE_TORCH')=='1': raise
    print('NOT EXECUTED: Torch/Transformers unavailable; real server CPU and authentic GPU smoke remain mandatory.')
    raise SystemExit(0)
from docci_longclip_joint_v8.core import load,geometry
from docci_longclip_joint_v8.model import JointModel,prediction,generate
from docci_longclip_joint_v8.channel import training_receive


class TorchContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(7); self.c=load(image_size=256,max_text_tokens=256)
        bart=BartForConditionalGeneration(BartConfig(vocab_size=96,d_model=64,encoder_layers=1,decoder_layers=1,encoder_attention_heads=1,
            decoder_attention_heads=1,encoder_ffn_dim=128,decoder_ffn_dim=128,max_position_embeddings=1024,bos_token_id=0,eos_token_id=2,
            pad_token_id=1,decoder_start_token_id=2,dropout=0.,attention_dropout=0.,activation_dropout=0.))
        self.model=JointModel(bart,self.c).eval()
        ids=torch.ones(1,256,dtype=torch.long); ids[0,:6]=torch.tensor([0,5,6,7,8,2])
        self.source=dict(ids=ids,bart_hidden=torch.randn(1,256,64),latent=torch.randn(1,4,32,32),
            clip_tokens=torch.randn(1,300,512),clip_image=torch.randn(1,512),clip_summary=torch.randn(1,512))

    def clean(self,wire,control='normal',donor=None,available=True):
        return self.model.receive(wire,torch.tensor([available]),torch.tensor([40.]),torch.tensor([1.]),control,donor)

    def test_full_codec_256_512_shapes_and_unit_power(self):
        for size in (256,512):
            c=dict(self.c,image_size=size); self.model=JointModel(self.model.text.bart,c)
            source=dict(self.source,latent=torch.randn(1,4,size//8,size//8)); wire=self.model.encode(source)
            self.assertEqual(wire.shape,(1,geometry(c)['total_reals'])); self.assertAlmostEqual(float(wire.square().mean()),1.,places=4)
            out=self.clean(wire); self.assertEqual(out['latent'].shape,source['latent'].shape)

    def test_separate_no_cross_modality_leak(self):
        self.model.configure('separate'); a=self.source; b=dict(a,clip_tokens=a['clip_tokens']+3,bart_hidden=a['bart_hidden']*3)
        wa=self.model.encode(a); wb=self.model.encode(b); g=geometry(self.c); mid=g['common_reals']+g['image_reals']
        self.assertTrue(torch.equal(wa[:,:256],wb[:,:256])); self.assertTrue(torch.equal(wa[:,512:mid],wb[:,512:mid]))
        outa=self.clean(wa); outb=self.clean(wb); self.assertTrue(torch.equal(outa['latent'],outb['latent']))
        wc=wa.clone(); wc[:,512:mid]+=4; self.assertTrue(torch.equal(self.clean(wc)['text']['memory'],outa['text']['memory']))

    def test_no_source_padding_or_length_at_receiver(self):
        modified=dict(self.source,bart_hidden=self.source['bart_hidden'].clone()); modified['bart_hidden'][:,6:]+=1000
        self.assertTrue(torch.equal(self.model.encode(self.source),self.model.encode(modified)))
        params=set(inspect.signature(self.model.receive).parameters)
        self.assertEqual(params,{'observed','available','pilot_snr','reliability','control','donor'})

    def test_proxy_reference_and_encoder_gradient(self):
        wire=self.model.encode(self.source); y,ok,snr,rel,_=training_receive(wire,self.c,4,'train_fixture',True)
        self.clean(wire); out=self.model.receive(y,ok,snr,rel)
        logits,labels=prediction(self.model.text.bart,self.source['ids'],self.source['ids'].ne(1),out['text']['memory'],out['text']['receiver_mask'])
        ce=torch.nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100)
        loss=ce+out['latent'].square().mean()+out['image_semantic'].square().mean()+out['text_semantic'].square().mean()+out['common_coarse'].square().mean()
        loss.backward()
        for prefix in ('text.compressor.','text.memory_projection.','image_encoder.','image_decoder.','common_attention.','common_tx.','common_rx.','image_fusion.','text_fusion.'):
            p=[p for n,p in self.model.named_parameters() if n.startswith(prefix)]
            self.assertTrue(any(v.grad is not None and bool(v.grad.abs().sum()>0) for v in p),prefix)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.model.text.bart.parameters()))

    def test_failure_zero_memory_and_empty_text(self):
        wire=self.model.encode(self.source); out=self.clean(wire,available=False)
        self.assertTrue(torch.equal(out['latent'],torch.zeros_like(out['latent']))); self.assertTrue(torch.equal(out['text']['memory'],torch.zeros_like(out['text']['memory'])))
        result=generate(self.model,out['text'],self.model.text.bart.config); self.assertEqual(result['text'],''); self.assertTrue(result['receiver_failure'])

    def test_controls_and_checkpoint_roundtrip(self):
        wire=self.model.encode(self.source); donor=wire+2
        base=self.clean(wire); zero=self.clean(wire,'zero_common'); off=self.clean(wire,'fusion_off')
        self.assertFalse(torch.equal(base['latent'],zero['latent'])); self.assertFalse(torch.equal(base['text']['memory'],off['text']['memory']))
        self.assertFalse(torch.equal(base['latent'],self.clean(wire,'shuffle_text',donor)['latent']))
        before=self.model.saved(); self.model.restore(before); self.assertTrue(all(torch.equal(v,self.model.saved()[n]) for n,v in before.items()))
        with self.assertRaises(ValueError): self.clean(wire,'shuffle_image')


if __name__=='__main__': unittest.main(verbosity=2)
