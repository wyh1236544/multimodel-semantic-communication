"""Single-modality entry points. No Common, opposite modality or clean RX mask."""
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
from bart_latent_codec_v4.model import LatentCodec,prediction
from docci_longclip_joint_v8.model import ImageEncoder,ImageDecoder
from docci_modal_v9.core import geometry,digest,read


def copy_state(named,state):
    if set(named)!=set(state): raise ValueError('Saved codec key set differs')
    with torch.no_grad():
        for name,param in named.items():
            value=state[name]
            if value.shape!=param.shape or not torch.isfinite(value).all(): raise ValueError('Saved shape/finiteness differs: '+name)
            param.copy_(value.to(param))


class TextBranch(nn.Module):
    def __init__(self,bart,c):
        super().__init__(); self.c=dict(c); self.branch='text'
        self.codec=LatentCodec(bart,dict(slots=c['text_slots'],depth=c['depth'],head_dim=c['head_dim'],max_tokens=c['max_text_tokens']),c['text_width'])
        self.configure()
    @property
    def bart(self): return self.codec.bart
    def configure(self):
        self.eval()
        for name,param in self.named_parameters(): param.requires_grad_(not name.startswith('codec.bart.'))
    def encode(self,source):
        return self.codec.transmit(source['bart_hidden'].float(),source['ids'].ne(self.bart.config.pad_token_id))
    def receive(self,observed,available,pilot_snr,reliability):
        return self.codec.receive_text(observed,available)
    def saved(self): return {n:p.detach().cpu().clone() for n,p in self.codec.named_parameters() if not n.startswith('bart.')}
    def restore(self,state): copy_state({n:p for n,p in self.codec.named_parameters() if not n.startswith('bart.')},state)
    def groups(self):
        return {name:tuple('codec.'+prefix for prefix in prefixes) for name,prefixes in self.codec.groups().items()}


class ImageBranch(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=dict(c); self.branch='image'; self.g=geometry(c,'image')
        self.image_encoder=ImageEncoder(c); self.image_decoder=ImageDecoder(c); self.configure()
    def configure(self): self.requires_grad_(True).eval()
    def encode(self,source):
        wire,_=self.image_encoder(source['latent'].float()); return wire.flatten(1)
    def receive(self,observed,available,pilot_snr,reliability):
        if observed.ndim!=2 or observed.shape[1]!=self.g['total_reals'] or not torch.isfinite(observed).all(): raise ValueError('Wrong finite public image payload')
        b=observed.shape[0]; grid=self.c['image_size']//16; ok=available.bool().reshape(b)
        wire=observed.reshape(b,self.c['image_width'],grid,grid)
        h=self.image_decoder.features(wire,pilot_snr.reshape(b),reliability.reshape(b))
        return dict(latent=self.image_decoder.output(h)*ok[:,None,None,None].to(h),available=ok)
    def saved(self): return {n:p.detach().cpu().clone() for n,p in self.named_parameters()}
    def restore(self,state): copy_state(dict(self.named_parameters()),state)
    def groups(self): return dict(encoder=('image_encoder.',),decoder=('image_decoder.',))


def warm(model,info):
    if info['checkpoint'] is None: return dict(info,loaded=False,reason='explicit fresh initialization')
    # A pinned own completed checkpoint, checked BEFORE deserializing old Torch pickle.
    if digest(info['checkpoint'])!=info['sha256']: raise ValueError('Warm checkpoint changed before load')
    checkpoint=torch.load(info['checkpoint'],map_location='cpu'); state=checkpoint.get('state')
    if not isinstance(state,dict): raise ValueError('Expected own state tensor dictionary')
    if model.branch=='text':
        if checkpoint.get('arm')!='P0_ce' or checkpoint.get('width')!=64: raise ValueError('Expected successful historical P0 CE width64 private codec')
        named={n:p for n,p in model.codec.named_parameters() if not n.startswith('bart.')}
        if set(state)!=set(named): raise ValueError('Historical text key set differs')
        copied={}
        with torch.no_grad():
            for n,p in named.items():
                old=state[n]
                if not torch.isfinite(old).all(): raise ValueError('Historical text nonfinite')
                if old.shape==p.shape: p.copy_(old.to(p)); copied[n]='full'
                elif n=='source_position' and old.shape[0]==p.shape[0] and old.shape[2]==p.shape[2] and old.shape[1]<p.shape[1]:
                    # Preserve learned128 prefix; BART already encodes absolute positions.
                    # New offsets start at ZERO, not fresh noise corrupting longer sources.
                    p.zero_(); p[:,:old.shape[1]].copy_(old.to(p)); copied[n]='old prefix retained; new source offsets ZERO/trainable'
                else: raise ValueError('Historical text shape differs: '+n)
        result=dict(info,loaded=True,copied=copied,decoder_memory_slots_preserved=32,text_token_capacity=model.c['max_text_tokens'])
    else:
        if checkpoint.get('schema')!='docci-longclip-joint-v8' or checkpoint.get('stage')!='image' or checkpoint['config']['image_size']!=512:
            raise ValueError('Expected completed V8 image prefit, not failed text or joint arm')
        image={n:v for n,v in state.items() if n.startswith(('image_encoder.','image_decoder.'))}
        model.restore(image)
        result=dict(info,loaded=True,copied=list(image),discarded_other_modal_and_Common_tensors=True)
    if digest(info['checkpoint'])!=info['sha256']: raise ValueError('Historical checkpoint changed during copy')
    return result


@torch.no_grad()
def generate(model,output,tok):
    if not bool(output['available'][0]): return dict(text='',receiver_failure=True,stopped_by_capacity=False,generated_token_ids=[])
    ids=model.bart.generate(encoder_outputs=BaseModelOutput(last_hidden_state=output['memory']),attention_mask=output['receiver_mask'].long(),
        max_length=model.c['max_text_tokens']+1,num_beams=1,do_sample=False,use_cache=True,forced_bos_token_id=tok.bos_token_id,
        forced_eos_token_id=None,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,no_repeat_ngram_size=0,repetition_penalty=1.,
        decoder_start_token_id=model.bart.config.decoder_start_token_id)
    eos=bool(ids[0,1:].eq(tok.eos_token_id).any())
    return dict(text=tok.decode(ids[0].tolist(),skip_special_tokens=True),receiver_failure=False,stopped_by_capacity=not eos,generated_token_ids=ids[0].tolist())
