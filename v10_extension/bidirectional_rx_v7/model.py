"""Residual RX only. Separate own/companion views prevent ablation leakage."""
import torch
from torch import nn
from clip_bart_joint_v5.model import Attention
from bidirectional_rx_v7 import ARMS, CONTROLS


class Block(nn.Module):
    def __init__(self,h,heads):
        super().__init__(); self.attention=Attention(h,h,h,heads); self.n1=nn.LayerNorm(h)
        self.ff=nn.Sequential(nn.Linear(h,2*h),nn.GELU(),nn.Linear(2*h,h)); self.n2=nn.LayerNorm(h)

    def forward(self,q,m,blocked):
        q=self.n1(q+self.attention(q,m,blocked)); return self.n2(q+self.ff(q))


class ReceiverAdapter(nn.Module):
    def __init__(self,dim,c,clip_dim=512):
        super().__init__(); self.c=dict(c); h=c['hidden']; self.arm=ARMS[0]
        self.image_input=nn.Linear(4,h); self.text_input=nn.Linear(dim,h)
        self.image_semantic=nn.Linear(clip_dim,h)
        self.image_position=nn.Parameter(torch.randn(1,256,h)*.02)
        self.text_position=nn.Parameter(torch.randn(1,32,h)*.02)
        self.image_blocks=nn.ModuleList([Block(h,c['heads']) for _ in range(c['depth'])])
        self.text_blocks=nn.ModuleList([Block(h,c['heads']) for _ in range(c['depth'])])
        self.image_output=nn.Linear(h,4); self.text_output=nn.Linear(h,dim)
        # Gates use own observation + public pilot only; R0 cannot see companion.
        self.image_gate=nn.Linear(h+2,1); self.text_gate=nn.Linear(h+2,1)
        for layer in (self.image_output,self.text_output): nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def configure(self,arm):
        if arm not in ARMS: raise ValueError('Unknown arm')
        self.arm=arm

    def forward(self,own,pilot,reliability,control='normal',donor=None):
        if control not in CONTROLS: raise ValueError('Unknown companion control')
        image=own['latent']; memory=own['text']['memory']; ok=own['text']['available'].bool()
        if image.shape[1:]!=(4,16,16) or memory.shape[1]!=32: raise ValueError('Fixed RX layout')
        b=memory.shape[0]; factor=ok.to(memory)[:,None,None]
        it=self.image_input(image.flatten(2).transpose(1,2))+self.image_position
        tt=self.text_input(memory)+self.text_position
        # Semantics extracted by frozen CLIP from the RECEIVED coarse image.
        # No clean image or transmitter CLIP feature enters this boundary.
        sem=self.image_semantic(own['received_clip'])[:,None]
        image_memory=torch.cat((it,sem),1); ci=image_memory; ct=tt
        if control=='zero_text_companion': ct=torch.zeros_like(tt)
        if control=='zero_image_companion': ci=torch.zeros_like(image_memory)
        if control.startswith('shuffle_'):
            if donor is None: raise ValueError('Already-received real donor required')
            dok=donor['text']['available'].to(memory)[:,None,None]
            if control=='shuffle_text_companion': ct=(self.text_input(donor['text']['memory'])+self.text_position)*dok
            else:
                di=self.image_input(donor['latent'].flatten(2).transpose(1,2))+self.image_position
                ds=self.image_semantic(donor['received_clip'])[:,None]
                ci=torch.cat((di,ds),1)*dok
        # No other direction reads an intervened companion: own views untouched.
        ti=self.arm in (ARMS[1],ARMS[3]) and control!='cross_off'
        it_enabled=self.arm in (ARMS[2],ARMS[3]) and control!='cross_off'
        ib=torch.zeros((256,289),dtype=torch.bool,device=memory.device); ib[:,257:]=not ti
        tb=torch.zeros((32,289),dtype=torch.bool,device=memory.device); tb[:,32:]=not it_enabled
        qi,qt=it,tt
        im=torch.cat((image_memory,ct),1); tm=torch.cat((tt,ci),1)
        for block in self.image_blocks: qi=block(qi,im,ib)
        for block in self.text_blocks: qt=block(qt,tm,tb)
        public=torch.stack((pilot.to(memory).reshape(b).clamp(-40,40)/40,reliability.to(memory).reshape(b)),1)
        gi=torch.sigmoid(self.image_gate(torch.cat((it.mean(1),public),1)))[:,None]
        gt=torch.sigmoid(self.text_gate(torch.cat((tt.mean(1),public),1)))[:,None]
        di=self.c['image_delta_limit']*torch.tanh(self.image_output(qi))*gi
        dt=self.c['text_delta_limit']*torch.tanh(self.text_output(qt))*gt
        latent=(image+di.transpose(1,2).reshape_as(image))*factor.reshape(b,1,1,1)
        text=dict(memory=(memory+dt)*factor,receiver_mask=own['text']['receiver_mask'],available=ok)
        return dict(latent=latent,text=text)

    def saved(self): return {n:p.detach().cpu().clone() for n,p in self.named_parameters()}

    def restore(self,state):
        named=dict(self.named_parameters())
        if set(named)!=set(state): raise ValueError('Adapter checkpoint keys differ')
        with torch.no_grad():
            for n,p in named.items():
                if p.shape!=state[n].shape or not torch.isfinite(state[n]).all(): raise ValueError('Bad saved '+n)
                p.copy_(state[n].to(p))

    @staticmethod
    def groups():
        return dict(image_input=('image_input.','image_position','image_semantic.'),text_input=('text_input.','text_position'),
            image_attention=('image_blocks.',),text_attention=('text_blocks.',),
            image_output=('image_output.',),text_output=('text_output.',),
            image_gate=('image_gate.',),text_gate=('text_gate.',))
