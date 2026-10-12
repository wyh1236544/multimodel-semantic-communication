"""Required server Torch toy tests. Missing Torch is an error, never a pass/skip."""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from joint_jscc_arch_v1.model import ArchitectureModel,ARTextDecoder,normalize_wire
from joint_jscc_arch_v1.core import load
from models.joint_multimodal_encoder import JointMultimodalEncoder
from models.joint_multimodal_decoder import JointMultimodalDecoder
from models.joint_multimodal_heads import ImageReconstructionHead
from models.text_reconstruction_head import TextReconstructionHead
from models.shared_bottleneck import SharedBottleneck
from fso_jscc_v1.model import ResidualMap
from v10.channels import ReliabilityAdapter


def toy():
    h=32; b=nn.Module(); b.hidden_dim=h
    b.vae_tokenizer=nn.Module(); b.vae_tokenizer.latent_proj=nn.Conv2d(4,h,1)
    b.multimodal_encoder=JointMultimodalEncoder(h,256,77,32,256,77,4,1,True,True)
    b.bottleneck=SharedBottleneck(h,16)
    b.channel=nn.Module(); b.channel.tx_map=ResidualMap(); b.channel.rx_map=ResidualMap(True)
    b.channel.adapter=ReliabilityAdapter(16)
    b.joint_decoder=JointMultimodalDecoder(16,h,32,256,77,4,1,'joint',True)
    b.image_head=ImageReconstructionHead(h,4,16,4,1,symmetric_cross_attention=True,spatial_direct=True)
    b.text_head=TextReconstructionHead(h,64,77,4,1)
    b.text_encoder=SimpleNamespace(tokenizer=SimpleNamespace(bos_token_id=1,eos_token_id=2,pad_token_id=2))
    cfg,_=load(); cfg.update(codec_hidden=16,codec_layers=1)
    return ArchitectureModel(b,cfg,[0.,1.,0.])


class TorchContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_causal_teacher_and_eos_pad_loss(self):
        torch.manual_seed(1); d=ARTextDecoder(TextReconstructionHead(16,32,77,4,1),128,1,2,2).eval()
        memory=torch.randn(1,32,16); a=torch.tensor([[1,3,4,5]])
        b=a.clone(); b[:,2:]=torch.tensor([[9,10]])
        self.assertTrue(torch.allclose(d.logits(a,memory)[:,:2],d.logits(b,memory)[:,:2],atol=1e-6))
        target=torch.tensor([[1,3,2,2,2]])
        loss=d.teacher_loss(target,torch.tensor([3]),memory)
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        self.assertGreater(float(d.embedding.weight.grad.square().sum()),0.)
        with torch.no_grad(): d.output.bias.fill_(-100); d.output.bias[2]=100
        self.assertEqual(d.generate(memory).tolist(),[[1,2]])

    def test_budget_normalization_and_long_source_gradient(self):
        torch.manual_seed(2); m=toy(); m.configure('B3_common')
        latent=torch.randn(1,4,16,16); text=torch.randn(1,128,32,requires_grad=True)
        x=m.source_encode(latent,text,torch.ones(1,128,dtype=torch.bool))
        self.assertEqual(tuple(x.shape),(1,288,16))
        self.assertAlmostEqual(float(x.square().mean()),1.,places=5)
        x[:,:32].square().sum().backward()
        # A nonconstant projection loss is needed because RMS is constrained.
        m.zero_grad(); text.grad=None
        x=m.source_encode(latent,text,torch.ones(1,128,dtype=torch.bool))
        (x[:,:,0]*torch.arange(288)[None,:]).mean().backward()
        self.assertGreater(float(text.grad[:,90:].square().sum()),0.)

    def test_strict_neural_separation_and_equal_parameter_scale(self):
        torch.manual_seed(3); m=toy(); m.configure('B3_common')
        joint=sum(p.numel() for p in m.parameters() if p.requires_grad)
        m.configure('B4_separate')
        self.assertEqual(joint,sum(p.numel() for p in m.parameters() if p.requires_grad))
        z=torch.randn(1,4,16,16); t=torch.randn(1,128,32); mask=torch.ones(1,128,dtype=torch.bool)
        a=m.source_encode(z,t,mask); b=m.source_encode(z,t+torch.randn_like(t),mask)
        image=torch.tensor(list(range(16))+list(range(32,256)))
        self.assertTrue(torch.allclose(a[:,image],b[:,image],atol=1e-5))
        r=torch.randn(1,288,16); rel=torch.ones(1,288,1); snr=torch.tensor([0.]); ok=torch.tensor([True])
        a=m.receive(r,rel,snr,ok); changed=r.clone(); changed[:,256:]+=3.; changed[:,16:32]+=3.
        b=m.receive(changed,rel,snr,ok)
        self.assertTrue(torch.allclose(a['latent'],b['latent'],atol=1e-5))

    def test_common_reads_no_private_receiver_symbols(self):
        torch.manual_seed(4); m=toy(); m.configure('B3_common')
        r=torch.randn(1,288,16); rel=torch.ones(1,288,1); snr=torch.tensor([0.]); ok=torch.tensor([True])
        a=m.receive(r,rel,snr,ok); r[:,32:]=torch.randn_like(r[:,32:])*20
        b=m.receive(r,rel,snr,ok)
        self.assertTrue(torch.equal(a['common_text_memory'],b['common_text_memory']))
        self.assertTrue(torch.equal(a['common_coarse'],b['common_coarse']))

    def test_codec_condition_and_source_parameter_updates(self):
        torch.manual_seed(5); m=toy(); m.configure('B3_common')
        before={n:p.detach().clone() for n,p in m.named_parameters()}
        opt=torch.optim.Adam(m.parameters(),lr=.001)
        for _ in range(3):
            opt.zero_grad(); x=m.source_encode(torch.randn(1,4,16,16),torch.randn(1,128,32),torch.ones(1,128,dtype=torch.bool))
            out=m.receive(x+torch.randn_like(x)*.1,torch.ones(1,288,1)*.5,torch.tensor([-2.]),torch.tensor([True]))
            loss=out['latent'].square().mean()+m.language.teacher_loss(torch.tensor([[1,3,4,2]]),torch.tensor([4]),out['text_memory'])+out['common_coarse'].square().mean()
            loss.backward(); opt.step()
        for prefix in ('source_transformer.','text_compressor.','modern_tx.','modern_rx.','spatial_tx.','spatial_rx.','rx_condition.','language.'):
            self.assertGreater(sum(float((p.detach()-before[n]).square().sum()) for n,p in m.named_parameters() if n.startswith(prefix)),0.)

    def test_checkpoint_exact_roundtrip(self):
        torch.manual_seed(6); a=toy(); a.configure('B4_separate'); b=toy(); b.load_state_dict(a.state_dict(),strict=True)
        self.assertTrue(all(torch.equal(v,b.state_dict()[n]) for n,v in a.state_dict().items()))
        with self.assertRaises(RuntimeError): b.load_state_dict({},strict=True)


if __name__=='__main__': unittest.main(verbosity=2)
