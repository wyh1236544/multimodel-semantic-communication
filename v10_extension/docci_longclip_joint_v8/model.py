"""Scalable spatial JSCC + decodable BART private memory + learned C/I/T fusion.

Frozen pretrained models are not replaced by random backbones. New communication
layers are deliberately learned. RX accepts only received symbols/public CSI.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from bart_latent_codec_v4.model import LatentCodec, prediction
from docci_longclip_joint_v8.core import geometry


def unit(x):
    return x/x.square().mean(tuple(range(1,x.ndim)),keepdim=True).clamp_min(1e-8).sqrt()


def positions(n,d,device):
    p=torch.arange(n,device=device).float()[:,None]
    frequency=torch.exp(torch.arange(0,d,2,device=device).float()*(-math.log(10000.)/d))
    out=torch.empty(n,d,device=device); out[:,0::2]=torch.sin(p*frequency); out[:,1::2]=torch.cos(p*frequency)
    return out[None]


def spatial_positions(h,w,d,device):
    # Public grid coordinates, never source-image rectangles or hidden metadata.
    row=positions(h,d//2,device)[0][:,None].expand(h,w,-1)
    col=positions(w,d//2,device)[0][None].expand(h,w,-1)
    return torch.cat((row,col),-1).reshape(1,h*w,d)


class Attention(nn.Module):
    def __init__(self,qdim,mdim,outdim,hidden=64,heads=4):
        super().__init__(); self.heads=heads; self.dim=hidden//heads
        self.q=nn.Linear(qdim,hidden); self.k=nn.Linear(mdim,hidden); self.v=nn.Linear(mdim,hidden); self.out=nn.Linear(hidden,outdim)

    def forward(self,q,m,blocked=None):
        b,n,_=q.shape
        qq=self.q(q).reshape(b,n,self.heads,self.dim).transpose(1,2)
        kk=self.k(m).reshape(b,-1,self.heads,self.dim).transpose(1,2)
        vv=self.v(m).reshape(b,-1,self.heads,self.dim).transpose(1,2)
        score=(qq@kk.transpose(-1,-2))/math.sqrt(self.dim)
        if blocked is not None: score=score.masked_fill(blocked[None,None],-torch.finfo(score.dtype).max)
        weights=score.softmax(-1)
        return self.out((weights@vv).transpose(1,2).reshape(b,n,-1))


class SpatialBlock(nn.Module):
    """Window attention plus overlapping3x3 convolution, not a Swin reproduction."""
    def __init__(self,h,window):
        super().__init__(); self.window=window
        self.norm=nn.LayerNorm(h); self.attention=Attention(h,h,h,h)
        self.conv=nn.Sequential(nn.Conv2d(h,h,3,padding=1),nn.GELU(),nn.Conv2d(h,h,3,padding=1))

    def forward(self,x):
        b,d,h,w=x.shape; k=self.window
        if h%k or w%k: raise ValueError('Public spatial grid not divisible by window')
        windows=x.permute(0,2,3,1).reshape(b,h//k,k,w//k,k,d).permute(0,1,3,2,4,5).reshape(-1,k*k,d)
        n=self.norm(windows); windows=windows+self.attention(n,n)
        out=windows.reshape(b,h//k,w//k,k,k,d).permute(0,1,3,2,4,5).reshape(b,h,w,d).permute(0,3,1,2)
        return out+self.conv(out)


class ImageEncoder(nn.Module):
    def __init__(self,c):
        super().__init__(); h=c['hidden']
        self.down=nn.Conv2d(4,h,3,stride=2,padding=1)
        self.blocks=nn.Sequential(SpatialBlock(h,c['window']),SpatialBlock(h,c['window']))
        self.wire=nn.Conv2d(h,c['image_width'],1)

    def forward(self,latent):
        hidden=self.blocks(self.down(latent)); return unit(self.wire(hidden)),hidden


class ImageDecoder(nn.Module):
    def __init__(self,c):
        super().__init__(); h=c['hidden']
        self.up=nn.Conv2d(c['image_width'],h,1)
        self.condition=nn.Sequential(nn.Linear(2,h),nn.GELU(),nn.Linear(h,h))
        self.blocks=nn.Sequential(SpatialBlock(h,c['window']),SpatialBlock(h,c['window']))
        self.output=nn.Sequential(nn.ConvTranspose2d(h,h,4,stride=2,padding=1),nn.GELU(),nn.Conv2d(h,4,3,padding=1))

    def features(self,wire,snr,reliability):
        c=torch.stack((snr.clamp(-40,40)/40,reliability.clamp(0,1)),1)
        return self.blocks(self.up(wire)+self.condition(c)[:,:,None,None])


class JointModel(nn.Module):
    def __init__(self,bart,c):
        super().__init__(); self.c=dict(c); self.g=geometry(c); h=c['hidden']; dim=bart.config.d_model
        self.text=LatentCodec(bart,dict(slots=c['text_slots'],depth=3,head_dim=64,max_tokens=c['max_text_tokens']),c['text_width'])
        self.image_encoder=ImageEncoder(c); self.image_decoder=ImageDecoder(c)
        self.clip_image=nn.Linear(512,h); self.clip_text=nn.Linear(512,h)
        self.pool_i=Attention(h,h,h,h); self.pool_t=Attention(h,h,h,h)
        self.pool_i_queries=nn.Parameter(torch.randn(1,16,h)*.02); self.pool_t_queries=nn.Parameter(torch.randn(1,16,h)*.02)
        self.common_queries=nn.Parameter(torch.randn(1,32,h)*.02); self.common_attention=Attention(h,h,h,h)
        self.common_norm=nn.LayerNorm(h); self.common_tx=nn.Linear(h,c['common_width']); self.common_rx=nn.Linear(c['common_width'],h)
        self.image_semantic=nn.Linear(h,512); self.text_semantic=nn.Linear(h,512); self.coarse=nn.Linear(h,4*8*8)
        self.text_to_h=nn.Linear(dim,h)
        self.image_fusion=Attention(h,h,h,h); self.text_fusion=Attention(dim,h,dim,h)
        self.image_gate=nn.Parameter(torch.tensor(-2.)); self.text_gate=nn.Parameter(torch.tensor(-2.))
        # Small nonzero outputs let gates AND attention learn from the first step.
        for module in (self.image_fusion,self.text_fusion):
            nn.init.normal_(module.out.weight,std=1e-4); nn.init.zeros_(module.out.bias)
        self.arm='joint'; self.configure('joint','joint')

    def configure(self,arm,stage='joint'):
        if arm not in ('separate','joint') or stage not in ('text','image','joint'): raise ValueError('Unknown arm/stage')
        self.arm=arm; self.eval()
        for n,p in self.named_parameters():
            eligible=not n.startswith('text.bart.')
            if stage=='text': eligible=eligible and n.startswith('text.')
            if stage=='image': eligible=eligible and n.startswith(('image_encoder.','image_decoder.'))
            p.requires_grad_(eligible)

    def encode(self,source):
        image,spatial=self.image_encoder(source['latent'].float())
        if image.shape[-2:]!=(self.g['image_grid'],self.g['image_grid']): raise ValueError('Image wire layout differs')
        text=self.text.transmit(source['bart_hidden'].float(),source['ids'].ne(self.text.bart.config.pad_token_id))
        im=spatial.flatten(2).transpose(1,2)+.1*spatial_positions(*spatial.shape[-2:],spatial.shape[1],spatial.device)
        im=torch.cat((im,self.clip_image(source['clip_image'].float())[:,None]),1)
        # Sequence offsets preserve ordering across Long-CLIP chunks.
        tm=self.clip_text(source['clip_tokens'].float())
        tm=tm+positions(tm.shape[1],tm.shape[2],tm.device)*.1
        iq=self.pool_i_queries.expand(image.shape[0],-1,-1); tq=self.pool_t_queries.expand(image.shape[0],-1,-1)
        memory=torch.cat((iq+self.pool_i(iq,im),tq+self.pool_t(tq,tm)),1)
        q=self.common_queries.expand(image.shape[0],-1,-1)
        blocked=None
        if self.arm=='separate':
            groups=torch.arange(32,device=image.device)>=16; blocked=groups[:,None]!=groups[None,:]
        c=self.common_tx(self.common_norm(q+self.common_attention(q,memory,blocked)))
        c=torch.cat((unit(c[:,:16]),unit(c[:,16:])),1)
        return torch.cat((c.flatten(1),image.flatten(1),text),1)

    def receive(self,observed,available,pilot_snr,reliability,control='normal',donor=None):
        # No source masks, text IDs, image pixels, aspect rectangle or oracle CSI here.
        if observed.ndim!=2 or observed.shape[1]!=self.g['total_reals'] or not torch.isfinite(observed).all(): raise ValueError('Finite full public wire required')
        b=observed.shape[0]; ok=available.bool().reshape(b); factor=ok.to(observed)[:,None,None]
        nc,ni=self.g['common_reals'],self.g['image_reals']
        cwire=observed[:,:nc].reshape(b,32,16)
        iw=observed[:,nc:nc+ni].reshape(b,self.c['image_width'],self.g['image_grid'],self.g['image_grid'])
        tw=observed[:,nc+ni:]
        if control=='zero_common': cwire=torch.zeros_like(cwire)
        elif control=='shuffle_common':
            if donor is None: raise ValueError('Explicit other received donor required')
            cwire=donor[:,:nc].reshape_as(cwire)
        elif control=='shuffle_image':
            if donor is None: raise ValueError('Explicit received donor required')
            iw=donor[:,nc:nc+ni].reshape_as(iw)
        elif control=='shuffle_text':
            if donor is None: raise ValueError('Explicit received donor required')
            tw=donor[:,nc+ni:]
        elif control not in ('normal','fusion_off'): raise ValueError('Unknown RX intervention')
        common=self.common_rx(cwire)*factor
        image=self.image_decoder.features(iw,pilot_snr.reshape(b),reliability.reshape(b))*factor[:,:,:,None]
        text=self.text.receive_text(tw,ok)
        cm_i=common[:,:16] if self.arm=='separate' else common
        cm_t=common[:,16:] if self.arm=='separate' else common
        im=image.flatten(2).transpose(1,2)+.1*spatial_positions(*image.shape[-2:],image.shape[1],image.device)
        image_memory=cm_i if self.arm=='separate' else torch.cat((cm_i,self.text_to_h(text['memory'])),1)
        text_memory=cm_t if self.arm=='separate' else torch.cat((cm_t,im),1)
        if control!='fusion_off':
            image=image+torch.sigmoid(self.image_gate)*self.image_fusion(im,image_memory).transpose(1,2).reshape_as(image)
            text['memory']=(text['memory']+torch.sigmoid(self.text_gate)*self.text_fusion(text['memory'],text_memory))*factor
        latent=self.image_decoder.output(image)*factor[:,:,:,None]
        return dict(latent=latent,text=text,
            image_semantic=self.image_semantic(cm_i.mean(1)),text_semantic=self.text_semantic(cm_t.mean(1)),
            common_coarse=self.coarse(cm_i.mean(1)).reshape(b,4,8,8),common_wire=cwire)

    def saved(self):
        return {n:v.detach().cpu().clone() for n,v in self.named_parameters() if not n.startswith('text.bart.')}

    def restore(self,state):
        named={n:v for n,v in self.named_parameters() if not n.startswith('text.bart.')}
        if set(state)!=set(named): raise ValueError('New codec saved keys differ')
        with torch.no_grad():
            for n,p in named.items():
                if p.shape!=state[n].shape or not torch.isfinite(state[n]).all(): raise ValueError('Invalid codec saved '+n)
                p.copy_(state[n].to(p))


def generate(model,text,tok):
    if not bool(text['available'][0]): return dict(text='',receiver_failure=True,stopped_by_capacity=False)
    ids=model.text.bart.generate(encoder_outputs=BaseModelOutput(last_hidden_state=text['memory']),attention_mask=text['receiver_mask'].long(),
        max_length=model.c['max_text_tokens']+1,num_beams=1,do_sample=False,use_cache=True,
        forced_bos_token_id=tok.bos_token_id,forced_eos_token_id=None,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,
        no_repeat_ngram_size=0,repetition_penalty=1.,decoder_start_token_id=model.text.bart.config.decoder_start_token_id)
    ended=bool(ids[0,1:].eq(tok.eos_token_id).any())
    return dict(text=tok.decode(ids[0].tolist(),skip_special_tokens=True),receiver_failure=False,stopped_by_capacity=not ended)


def warm_text(model,path):
    """Explicit V4/V6 codec weight transfer, not optimizer resume or a new backbone."""
    from docci_longclip_joint_v8.core import digest
    state=torch.load(str(path),map_location='cpu')
    if state.get('arm')!='P0_ce' or state.get('width')!=64 or 'state' not in state: raise ValueError('Expected completed private P0 width64 codec')
    old=state['state']; named={n:p for n,p in model.text.named_parameters() if not n.startswith('bart.')}
    if set(old)!=set(named): raise ValueError('Old private codec key set differs')
    copied={}
    with torch.no_grad():
        for n,p in named.items():
            t=old[n]
            if not torch.isfinite(t).all(): raise ValueError('Nonfinite old codec')
            if t.shape==p.shape: p.copy_(t.to(p)); copied[n]='full'
            elif n in ('queries','source_position','memory_position') and t.shape[0]==p.shape[0] and t.shape[2]==p.shape[2] and t.shape[1]<=p.shape[1]:
                p[:,:t.shape[1]].copy_(t.to(p)); copied[n]='old prefix copied; new positions remain explicit initialization'
            else: raise ValueError('Unsupported text warm-start shape: '+n)
    return dict(checkpoint_sha256=digest(path),copied=copied,optimizer_resumed=False,old_checkpoint_unchanged=True)
