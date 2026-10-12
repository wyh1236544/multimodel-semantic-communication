"""Dedicated trainable text autoencoder; RX never consumes source IDs or length."""
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
from joint_jscc_arch_v1.model import ARTextDecoder


def normalize(x):
    return x/x.square().mean((1, 2), keepdim=True).clamp_min(1e-8).sqrt()


class TextJSCC(nn.Module):
    def __init__(self, legacy, config, rrc=None):
        super().__init__(); h=config['hidden']; self.slots=config['slots']
        self.maximum=config['max_tokens']; tok=legacy.text_encoder.tokenizer
        self.register_buffer('rrc',torch.as_tensor([1.] if rrc is None else rrc).float().reshape(1,1,-1))
        self.embedding=nn.Embedding(tok.vocab_size,h)
        with torch.no_grad():
            # Token identity initialization, not pooled CLIP sentence embeddings.
            weight=legacy.text_encoder.backbone.embeddings.token_embedding.weight
            self.embedding.weight.copy_(legacy.text_encoder.projection(weight))
        self.position=nn.Parameter(torch.randn(1,self.maximum,h)*.02)
        layer=nn.TransformerEncoderLayer(h,config['heads'],h*4,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,config['layers'])
        self.queries=nn.Parameter(torch.randn(1,self.slots,h)*.02)
        self.compressor=nn.MultiheadAttention(h,config['heads'],dropout=0.,batch_first=True)
        self.tx=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,h),nn.GELU(),nn.Linear(h,16))
        self.rx=nn.Sequential(nn.Linear(16,h),nn.GELU(),nn.LayerNorm(h))
        self.rx_position=nn.Parameter(torch.randn(1,self.slots,h)*.02)
        self.rx_condition=nn.Linear(2,h)
        nn.init.zeros_(self.rx_condition.weight); nn.init.zeros_(self.rx_condition.bias)
        self.receiver=nn.TransformerEncoder(copy.deepcopy(layer),config['layers'])
        self.language=ARTextDecoder(legacy.text_head,self.maximum,tok.bos_token_id,tok.eos_token_id,tok.pad_token_id)
        # Proper input/output token embeddings are trained jointly with source.
        self.language.embedding=self.embedding
        self.language.output.weight=self.embedding.weight

    def encode(self, ids, lengths):
        # Source-side padding only. Neither source length nor mask goes over wire.
        pad=torch.arange(ids.shape[1],device=ids.device)[None,:] >= lengths[:,None]
        h=self.encoder(self.embedding(ids)*math.sqrt(self.embedding.embedding_dim)+self.position[:,:ids.shape[1]],src_key_padding_mask=pad)
        q=self.queries.expand(ids.size(0),-1,-1)
        compressed,_=self.compressor(q,h,h,key_padding_mask=pad,need_weights=False)
        return normalize(self.tx(compressed))

    def receive(self, wire, reliability, pilot_snr, available):
        if wire.shape[1:] != (48,16): raise ValueError('Text-only fixed48x16 wire required')
        b=wire.size(0); ok=available.reshape(b,1,1).to(wire)
        cond=torch.stack((reliability.reshape(b),pilot_snr.reshape(b).clamp(-40,40)/40),-1).to(wire)
        h=self.rx(wire*ok)+self.rx_position+self.rx_condition(cond)[:,None]
        return self.receiver(h)*ok

    def waveform_penalty(self,wire,target_db):
        b=wire.size(0); axes=(wire.reshape(b,-1,2)/math.sqrt(2)).transpose(1,2)
        wave=F.conv_transpose1d(axes.reshape(b*2,1,-1),self.rrc,stride=4).reshape(b,2,-1)*2
        power=wave.square().sum(1); relative=power/power.mean(1,keepdim=True).clamp_min(1e-8)
        return F.relu(relative-10**(target_db/10)).square().mean()


def tensors(items, device):
    return (torch.tensor([r['target_ids'] for r in items],dtype=torch.long,device=device),
            torch.tensor([r['target_length'] for r in items],dtype=torch.long,device=device))


@torch.no_grad()
def generate(decoder, memory, tokenizer, available=True):
    if not available:
        return dict(received_caption='',generated_token_ids=[],generated_tokens=0,
                    stopped_by_eos=False,stopped_by_capacity=False,termination='receiver_rejected')
    if memory.size(0)!=1: raise ValueError('Record true generation one sample at a time')
    out=torch.full((1,1),decoder.bos,dtype=torch.long,device=memory.device); eos=False
    for _ in range(decoder.maximum-1):
        nxt=decoder.output(decoder.features(out,memory)[:,-1]).argmax(-1)
        out=torch.cat((out,nxt[:,None]),1)
        if int(nxt[0])==decoder.eos: eos=True; break
    return dict(received_caption=tokenizer.batch_decode(out,skip_special_tokens=True)[0],
                generated_token_ids=out[0].cpu().tolist(),generated_tokens=int(out.shape[1]-1),
                stopped_by_eos=eos,stopped_by_capacity=not eos,
                termination='predicted_EOS' if eos else 'public_capacity_limit')


def legacy_wire(core, item, device):
    # Frozen B4 text paths must be independent of this non-business dummy image.
    latent=torch.zeros(1,4,16,16,device=device)
    x=core.source_encode(latent,item['text_features'].to(device),item['text_mask'].to(device))
    return torch.cat((x[:,16:32],x[:,256:]),1)


def legacy_receive(core, wire, reliability, snr, available):
    full=wire.new_zeros(wire.size(0),288,16)
    full[:,16:32]=wire[:,:16]; full[:,256:]=wire[:,16:]
    rel=wire.new_full((wire.size(0),288,1),float(reliability))
    return core.receive(full,rel,snr,available)['text_memory']


def grad_norms(model):
    groups=dict(source=0.,compression=0.,tx=0.,rx=0.,decoder=0.)
    for name,p in model.named_parameters():
        if p.grad is None: continue
        if name.startswith(('embedding.','position','encoder.')): key='source'
        elif name.startswith(('queries','compressor.')): key='compression'
        elif name.startswith('tx.'): key='tx'
        elif name.startswith(('rx.','rx_position','rx_condition.','receiver.')): key='rx'
        else: key='decoder'
        groups[key]+=float(p.grad.detach().square().sum())
    return groups
