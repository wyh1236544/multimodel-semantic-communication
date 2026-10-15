"""Real Torch/HF contracts; local missing libraries are NOT reported as passes."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
try:
    import torch
    from transformers import BartConfig,BartForConditionalGeneration
    AVAILABLE=True
except ImportError:
    AVAILABLE=False
    if os.environ.get('V12_REQUIRE_TORCH')=='1':
        raise SystemExit('Real Torch/Transformers missing. Use existing server environment; do NOT install or claim CPU model gates passed.')


@unittest.skipUnless(AVAILABLE,'Torch/Transformers NOT AVAILABLE locally; mandatory real server check remains')
class Models(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(42)

    def text(self):
        from docci_reconstruction_v10.model import TextBranch
        from docci_reconstruction_v10.core import load,chunk_plan
        bart=BartForConditionalGeneration(BartConfig(vocab_size=64,d_model=32,encoder_layers=1,decoder_layers=1,
            encoder_attention_heads=4,decoder_attention_heads=4,encoder_ffn_dim=64,decoder_ffn_dim=64,
            pad_token_id=1,bos_token_id=0,eos_token_id=2,decoder_start_token_id=2,max_position_embeddings=1024,dropout=0.))
        c=load(); c.update(depth=1,head_dim=8,text_slots=4,text_width=8)
        model=TextBranch(bart,c,'T1_chunks')
        chunks=torch.tensor(chunk_plan([4,5,6,7],c))[None]
        with torch.no_grad(): hidden=bart.model.encoder(input_ids=chunks.flatten(0,1),attention_mask=chunks.flatten(0,1).ne(1).long(),return_dict=True).last_hidden_state
        src=dict(ids=torch.ones(1,512,dtype=torch.long),chunk_ids=chunks,chunk_hidden=hidden[None])
        return model,src

    def test_ce_and_full_distribution_kd_backward_frozen_backbone(self):
        from docci_semantic_probe_v12.train import text_loss
        model,src=self.text(); count=[]; original=model.bart.forward
        def forward(*args,**kw): count.append(1); return original(*args,**kw)
        model.bart.forward=forward
        loss,terms=text_loss(model,src,0.,2.); self.assertEqual(len(count),1); self.assertFalse(terms['teacher_executed'])
        model.zero_grad(set_to_none=True); count.clear()
        loss,terms=text_loss(model,src,.5,2.); self.assertEqual(len(count),2); self.assertTrue(terms['teacher_executed']); loss.backward()
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.bart.parameters()))
        for group,prefixes in model.groups().items():
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.startswith(prefixes)),group)

    def test_kd_teacher_detached_and_pad_does_not_contribute(self):
        from bart_latent_codec_v4.model import distillation
        student=torch.randn(2,3,8,requires_grad=True); teacher=torch.randn(2,3,8,requires_grad=True)
        labels=torch.tensor([[1,2,-100],[3,-100,-100]])
        loss=distillation(student,teacher,labels,2.); loss.backward()
        self.assertIsNone(teacher.grad); self.assertEqual(float(student.grad[labels==-100].abs().sum()),0.)

    def base(self):
        from docci_reconstruction_v10.core import load
        from docci_reconstruction_v10.model import ImageBranch
        c=load(image_size=256); return ImageBranch(c,'I2_shifted').requires_grad_(False).eval()

    def test_actual_I2_decoder_identity_initialization(self):
        from docci_semantic_probe_v12.model import ImageProbe
        base=self.base(); probe=ImageProbe(base,'A1_oracle_text'); x=torch.randn(1,2048); text=torch.randn(1,512)
        with torch.no_grad():
            a=probe.receive(x,text); b=base.receive(x,torch.tensor([True]),torch.tensor([40.]),torch.tensor([1.]))['latent']
        self.assertTrue(torch.allclose(a,b,atol=1e-6,rtol=1e-6))

    def test_image_only_ignores_text_and_same_parameter_initialization(self):
        from docci_semantic_probe_v12.model import ImageProbe
        base=self.base(); a=ImageProbe(base,'A0_image_only'); state=a.saved(); b=ImageProbe(base,'A1_oracle_text'); b.restore(state)
        self.assertEqual(sum(p.numel() for p in a.parameters() if p.requires_grad),sum(p.numel() for p in b.parameters() if p.requires_grad))
        for block in a.adapters: torch.nn.init.normal_(block.affine[-1].weight,std=.01)
        x=torch.randn(1,2048)
        self.assertTrue(torch.equal(a.receive(x,torch.randn(1,512)),a.receive(x,torch.randn(1,512))))

    def test_three_spatial_groups_learn_and_frozen_I2_does_not(self):
        from docci_semantic_probe_v12.model import ImageProbe
        from docci_reconstruction_v10.core import state_hash
        base=self.base(); before=state_hash(base.saved()); probe=ImageProbe(base,'A1_oracle_text')
        optim=torch.optim.AdamW(probe.adapters.parameters(),lr=.001)
        x=torch.randn(1,2048); text=torch.randn(1,512)
        for _ in range(2):
            optim.zero_grad(set_to_none=True); probe.receive(x,text).square().mean().backward(); optim.step()
        for group,prefixes in probe.groups().items():
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in probe.named_parameters() if n.startswith(prefixes)),group)
        self.assertEqual(before,state_hash(base.saved())); self.assertTrue(all(p.grad is None for p in base.parameters()))
        for block in probe.adapters: torch.nn.init.normal_(block.affine[-1].weight,std=.02)
        self.assertFalse(torch.allclose(probe.receive(x,text),probe.receive(x,-text),atol=1e-7,rtol=1e-7))

    def test_adapter_actual_disk_roundtrip(self):
        from docci_semantic_probe_v12.model import ImageProbe
        from docci_semantic_probe_v12.core import PROJECT
        probe=ImageProbe(self.base(),'A1_oracle_text'); state=probe.saved()
        temporary=PROJECT/'.cache/v10/docci_semantic_probe_v12_tests'; temporary.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(temporary)) as d:
            file=Path(d)/'own.pth'; torch.save(state,str(file)); probe.restore(torch.load(str(file),map_location='cpu'))
        self.assertTrue(all(torch.equal(v,probe.saved()[k]) for k,v in state.items()))


if __name__=='__main__': unittest.main(verbosity=2)
