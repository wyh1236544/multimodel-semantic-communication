"""B4 is entirely frozen. Equal-size residuals differ only in side memory."""
import torch
from torch import nn
import torch.nn.functional as F
from joint_jscc_arch_v1.model import attention_mask
from joint_jscc_arch_v1.core import source_groups


class ResidualFusion(nn.Module):
    def __init__(self,input_hidden=256,hidden=64):
        super().__init__()
        self.query=nn.Linear(input_hidden,hidden,bias=False)
        self.memory=nn.Linear(input_hidden,hidden,bias=False)
        self.attention=nn.MultiheadAttention(hidden,4,dropout=0.,bias=False,batch_first=True)
        self.gate=nn.Sequential(nn.Linear(4,16),nn.GELU(),nn.Linear(16,1))
        self.output=nn.Linear(hidden,4,bias=False)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.gate[-1].bias,-2.)

    def forward(self,base_latent,image_memory,side_memory,reliability,pilot_snr,available,disabled=False):
        if disabled: return base_latent,base_latent.new_zeros(base_latent.size(0))
        if image_memory.shape[1]!=272 or side_memory.shape[1]!=48: raise ValueError('Public B4 memory layout required')
        q=self.query(F.layer_norm(image_memory[:,16:],(image_memory.shape[-1],)))
        kv=self.memory(F.layer_norm(side_memory,(side_memory.shape[-1],)))
        a,_=self.attention(q,kv,kv,need_weights=False)
        cosine=F.cosine_similarity(q.mean(1),kv.mean(1),dim=-1)
        features=torch.stack((reliability.reshape(reliability.size(0),-1).mean(1),
            pilot_snr.reshape(-1).clamp(-40,40)/40,cosine,kv.square().mean((1,2)).log1p()),-1)
        g=torch.sigmoid(self.gate(features)).reshape(-1)*available.reshape(-1).to(q)
        delta=self.output(a).transpose(1,2).reshape_as(base_latent)
        return base_latent+g[:,None,None,None]*delta,g


def image_side(memory):
    return F.adaptive_avg_pool1d(memory.transpose(1,2),48).transpose(1,2)


@torch.no_grad()
def frozen_receive(core,received,reliability,pilot_snr,available):
    capture=[]
    handle=core.rx_norm.register_forward_hook(lambda module,args,out:capture.append(out))
    try: out=core.receive(received,reliability,pilot_snr,available)
    finally: handle.remove()
    if len(capture)!=1: raise RuntimeError('B4 receiver hook did not capture exactly one memory')
    h=capture[0]
    out['image_memory']=torch.cat((h[:,:16],h[:,32:288]),1)
    if out['text_memory'].shape[1]!=48: raise RuntimeError('B4 separation must be active')
    return out


@torch.no_grad()
def precompression_memory(core,item,device):
    """TRAIN-only oracle feature diagnostic. Never used in main receiver."""
    latent=item['latent'].to(device); t=item['text_features'].to(device); mask=item['text_mask'].to(device)
    v=core.image_projection(latent).flatten(2).transpose(1,2)+core.image_position+core.image_type
    t=t+core.text_position[:,:t.shape[1]]+core.text_type
    joined=torch.cat((core.common_queries.expand(1,-1,-1),v,t),1)
    pad=torch.cat((torch.zeros(1,288,device=device,dtype=torch.bool),~mask.bool()),1)
    h=core.source_norm(core.source_transformer(joined,
        mask=attention_mask(source_groups(t.shape[1]),True,device),src_key_padding_mask=pad))
    return h[:,288:][:,mask[0].bool()]


@torch.no_grad()
def generate_record(decoder,memory,tokenizer):
    """Same greedy algorithm as V1, now recording actual EOS/cap termination."""
    if memory.size(0)!=1: raise ValueError('Actual generation diagnosis uses batch1')
    ids=torch.full((1,1),decoder.bos,dtype=torch.long,device=memory.device)
    eos=False
    for _ in range(decoder.maximum-1):
        nxt=decoder.output(decoder.features(ids,memory)[:,-1]).argmax(-1)
        ids=torch.cat((ids,nxt[:,None]),1)
        if int(nxt[0])==decoder.eos: eos=True; break
    return dict(received_caption=tokenizer.batch_decode(ids,skip_special_tokens=True)[0],
        generated_token_ids=ids[0].cpu().tolist(),generated_tokens=int(ids.shape[1]-1),
        stopped_by_eos=eos,stopped_by_capacity=not eos,
        termination='predicted_EOS' if eos else 'public_capacity_limit',source_text_at_receiver=False)
