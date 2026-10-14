"""Frozen BART/VAE, learnable private codecs; multiscale Swin-inspired NOT replica."""
import math
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
from bart_latent_codec_v4.model import LatentCodec,prediction
from docci_modal_v9.model import ImageBranch as OldImage,warm as old_warm,copy_state
from docci_longclip_joint_v8.model import unit
from docci_reconstruction_v10.core import chunks,geometry,digest,joined_ids


class TextBranch(nn.Module):
    def __init__(self,bart,c,arm):
        super().__init__(); self.c=dict(c); self.arm=arm; self.branch='text'
        self.count=chunks(c) if arm=='T1_chunks' else 1
        self.maximum=c['chunk_tokens'] if arm=='T1_chunks' else c['max_text_tokens']
        # Build SAME seeded tensors before slicing the public64-position prefix.
        self.codec=LatentCodec(bart,dict(slots=c['text_slots'],depth=c['depth'],head_dim=c['head_dim'],max_tokens=c['max_text_tokens']),c['text_width'])
        if self.maximum<c['max_text_tokens']:
            self.codec.source_position=nn.Parameter(self.codec.source_position[:,:self.maximum].detach().clone())
            self.codec.maximum=self.maximum
        self.configure()
    @property
    def bart(self): return self.codec.bart
    def configure(self):
        self.eval()
        for name,p in self.named_parameters(): p.requires_grad_(not name.startswith('codec.bart.'))
    def encode(self,source):
        ids,hidden=self.sources(source); b=source['ids'].shape[0]
        return self.codec.transmit(hidden.float(),ids.ne(self.bart.config.pad_token_id)).reshape(b,-1)
    def sources(self,source):
        if self.arm=='T1_chunks': return source['chunk_ids'].flatten(0,1),source['chunk_hidden'].flatten(0,1)
        return source['ids'],source['bart_hidden']
    def receive(self,observed,available,pilot_snr,reliability):
        if observed.ndim!=2 or observed.shape[1]!=geometry(self.c,self.arm)['total_reals']: raise ValueError('Wrong public text layout')
        b=observed.shape[0]; ok=available.bool().reshape(b)
        out=self.codec.receive_text(observed.reshape(b*self.count,-1),ok[:,None].expand(b,self.count).reshape(-1))
        return dict(memory=out['memory'].reshape(b,self.count,*out['memory'].shape[1:]),
            receiver_mask=out['receiver_mask'].reshape(b,self.count,-1),available=ok)
    def saved(self): return {n:p.detach().cpu().clone() for n,p in self.codec.named_parameters() if not n.startswith('bart.')}
    def restore(self,state): copy_state({n:p for n,p in self.codec.named_parameters() if not n.startswith('bart.')},state)
    def groups(self): return {n:tuple('codec.'+p for p in prefixes) for n,prefixes in self.codec.groups().items()}


class Window(nn.Module):
    """True alternating shifted windows, relative offsets and cyclic-wrap mask."""
    def __init__(self,dim,window=8,shift=0,heads=4):
        super().__init__(); self.window=window; self.shift=shift; self.heads=heads; self.dim=dim
        self.norm=nn.LayerNorm(dim); self.qkv=nn.Linear(dim,3*dim); self.out=nn.Linear(dim,dim)
        self.ff=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,4*dim),nn.GELU(),nn.Linear(4*dim,dim))
        self.relative=nn.Parameter(torch.zeros((2*window-1)**2,heads))
        y,x=torch.meshgrid(torch.arange(window),torch.arange(window),indexing='ij')
        coords=torch.stack((y.flatten(),x.flatten()),1); delta=coords[:,None]-coords[None,:]+window-1
        self.register_buffer('offset',delta[:,:,0]*(2*window-1)+delta[:,:,1],persistent=False)
    def split(self,x):
        b,h,w,d=x.shape; k=self.window
        return x.reshape(b,h//k,k,w//k,k,d).permute(0,1,3,2,4,5).reshape(-1,k*k,d)
    def mask(self,h,w,device):
        k,s=self.window,self.shift
        if not s: return None
        regions=torch.zeros(1,h,w,1,device=device); value=0
        for ys in (slice(0,-k),slice(-k,-s),slice(-s,None)):
            for xs in (slice(0,-k),slice(-k,-s),slice(-s,None)):
                regions[:,ys,xs]=value; value+=1
        ids=self.split(regions)[...,0]; return ids[:,:,None]!=ids[:,None,:]
    def forward(self,x):
        b,d,h,w=x.shape; k=self.window
        if h%k or w%k: raise ValueError('Window requires divisible public grid')
        view=x.permute(0,2,3,1)
        if self.shift: view=torch.roll(view,(-self.shift,-self.shift),(1,2))
        win=self.split(view); n=self.norm(win)
        q,key,v=[a.reshape(-1,k*k,self.heads,d//self.heads).transpose(1,2) for a in self.qkv(n).chunk(3,-1)]
        logits=(q@key.transpose(-1,-2))/math.sqrt(d//self.heads)
        logits=logits+self.relative[self.offset].permute(2,0,1)[None]
        mask=self.mask(h,w,x.device)
        if mask is not None:
            logits=logits.reshape(b,-1,self.heads,k*k,k*k).masked_fill(mask[None,:,None],-torch.finfo(logits.dtype).max).reshape(-1,self.heads,k*k,k*k)
        win=win+self.out((logits.softmax(-1)@v).transpose(1,2).reshape(-1,k*k,d)); win=win+self.ff(win)
        view=win.reshape(b,h//k,w//k,k,k,d).permute(0,1,3,2,4,5).reshape(b,h,w,d)
        if self.shift: view=torch.roll(view,(self.shift,self.shift),(1,2))
        return view.permute(0,3,1,2).contiguous()


def blocks(dim): return nn.Sequential(Window(dim),Window(dim,shift=4))


class MultiEncoder(nn.Module):
    def __init__(self,width):
        super().__init__(); self.input=nn.Conv2d(4,64,3,padding=1); self.a=blocks(64)
        self.down=nn.Conv2d(64,96,4,2,1); self.b=blocks(96)
        self.coarse=nn.Sequential(nn.Conv2d(96,128,4,2,1),blocks(128),nn.ConvTranspose2d(128,96,4,2,1))
        self.merge=blocks(96); self.wire=nn.Conv2d(96,width,1)
    def forward(self,x):
        middle=self.b(self.down(self.a(self.input(x)))); return unit(self.wire(self.merge(middle+self.coarse(middle))))


class MultiDecoder(nn.Module):
    def __init__(self,width):
        super().__init__(); self.input=nn.Conv2d(width,96,1); self.a=blocks(96)
        self.condition=nn.Sequential(nn.Linear(2,96),nn.GELU(),nn.Linear(96,96))
        self.coarse=nn.Sequential(nn.Conv2d(96,128,4,2,1),blocks(128),nn.ConvTranspose2d(128,96,4,2,1))
        self.b=blocks(96); self.up=nn.ConvTranspose2d(96,64,4,2,1); self.c=blocks(64); self.output=nn.Conv2d(64,4,3,padding=1)
    def forward(self,x,snr,reliability):
        cond=torch.stack((snr.clamp(-40,40)/40,reliability.clamp(0,1)),1)
        middle=self.a(self.input(x)+self.condition(cond)[:,:,None,None])
        return self.output(self.c(self.up(self.b(middle+self.coarse(middle)))))


class ImageBranch(nn.Module):
    def __init__(self,c,arm):
        super().__init__(); self.c=dict(c); self.arm=arm; self.branch='image'
        if arm=='I2_shifted': self.image_encoder=MultiEncoder(c['image_width']); self.image_decoder=MultiDecoder(c['image_width'])
        else:
            old=OldImage(c); self.image_encoder=old.image_encoder; self.image_decoder=old.image_decoder
        self.configure()
    def configure(self): self.requires_grad_(True).eval()
    def encode(self,source):
        value=self.image_encoder(source['latent'].float())
        return (value if self.arm=='I2_shifted' else value[0]).flatten(1)
    def receive(self,observed,available,pilot_snr,reliability):
        if observed.ndim!=2 or observed.shape[1]!=geometry(self.c,self.arm)['total_reals'] or not torch.isfinite(observed).all(): raise ValueError('Wrong finite image payload')
        b=observed.shape[0]; grid=self.c['image_size']//16; wire=observed.reshape(b,self.c['image_width'],grid,grid)
        if self.arm=='I2_shifted': latent=self.image_decoder(wire,pilot_snr.reshape(b),reliability.reshape(b))
        else: latent=self.image_decoder.output(self.image_decoder.features(wire,pilot_snr.reshape(b),reliability.reshape(b)))
        ok=available.bool().reshape(b); return dict(latent=latent*ok[:,None,None,None],available=ok)
    def saved(self): return {n:p.detach().cpu().clone() for n,p in self.named_parameters()}
    def restore(self,state): copy_state(dict(self.named_parameters()),state)
    def groups(self): return dict(encoder=('image_encoder.',),decoder=('image_decoder.',))


def warm(model,info):
    if info['checkpoint'] is None: return dict(info,loaded=False,reason='declared fresh init')
    if model.branch=='image':
        if model.arm=='I2_shifted' or model.c['image_width']!=8: raise ValueError('New spatial codec/rate requires explicit fresh init')
        return old_warm(model,info)
    if model.c['text_width']!=64: raise ValueError('Historical codec only width64; explicitly use fresh init for rate sweep')
    if digest(info['checkpoint'])!=info['sha256']: raise ValueError('Own checkpoint pin differs before pickle load')
    record=torch.load(info['checkpoint'],map_location='cpu')
    if record.get('arm')!='P0_ce' or record.get('width')!=64: raise ValueError('Expected pinned successful V6 text codec')
    state=record['state']; target=model.saved()
    if set(target)!=set(state): raise ValueError('Codec historical keys differ')
    old=state['source_position']; new=target['source_position']; n=min(old.shape[1],new.shape[1])
    if old.shape[0]!=new.shape[0] or old.shape[2]!=new.shape[2]: raise ValueError('Source-position shape differs')
    target['source_position'].zero_(); target['source_position'][:,:n].copy_(old[:,:n])
    for k in target:
        if k!='source_position': target[k]=state[k]
    model.restore(target)
    if digest(info['checkpoint'])!=info['sha256']: raise ValueError('Historical checkpoint changed')
    return dict(info,loaded=True,copied_source_prefix=n,source_position_capacity=model.maximum,
        removed_positions_used_at_RX=False,optimizer_resumed=False,chunk_codec_shared_across_all_chunks=True)


@torch.no_grad()
def generate(model,output,tok):
    if not bool(output['available'][0]): return dict(text='',receiver_failure=True,stopped_by_capacity=False,generated_token_ids=[],chunk_generated_token_ids=[])
    memory=output['memory'][0]; mask=output['receiver_mask'][0]
    # One generate batch includes ALL public slots, including empty padded segments.
    ids=model.bart.generate(encoder_outputs=BaseModelOutput(last_hidden_state=memory),attention_mask=mask.long(),max_length=model.maximum+1,
        num_beams=1,do_sample=False,use_cache=True,forced_bos_token_id=tok.bos_token_id,forced_eos_token_id=None,
        eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,no_repeat_ngram_size=0,repetition_penalty=1.,
        decoder_start_token_id=model.bart.config.decoder_start_token_id)
    generated=ids.tolist(); joined,capacity=joined_ids(generated,tok.bos_token_id,tok.eos_token_id,tok.pad_token_id)
    return dict(text=tok.decode(joined,skip_special_tokens=True,clean_up_tokenization_spaces=False),receiver_failure=False,
        stopped_by_capacity=capacity,generated_token_ids=joined,chunk_generated_token_ids=generated)
