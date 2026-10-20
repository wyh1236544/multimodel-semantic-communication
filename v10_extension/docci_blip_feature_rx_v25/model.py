"""Pretrained visual-query tokens -> external BART residual cross attention.

No source IDs, gold masks, text lengths or SNR truth are receiver arguments.
Original encoder memory and original HF KV/cache objects are never replaced.
"""
from contextlib import contextmanager
import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import shift_tokens_right

@torch.no_grad()
def content_gate(bart,memory):
    flat = memory.flatten(0,1)
    ids = torch.tensor([bart.config.decoder_start_token_id,bart.config.bos_token_id],
        device=memory.device).expand(len(flat),-1)
    logits = bart(encoder_outputs=BaseModelOutput(last_hidden_state=flat),
        attention_mask=torch.ones(flat.shape[:2],device=flat.device,dtype=torch.long),
        decoder_input_ids=ids,use_cache=False,return_dict=True).logits[:,-1]
    return (1-logits.softmax(-1)[:,bart.config.eos_token_id]).reshape(memory.shape[:2])

def self_tokens(memory,gate):
    weights = gate.detach()/gate.detach().sum(1,keepdim=True).clamp_min(1e-8)
    return (memory.detach()*weights[:,:,None,None]).sum(1)

class Cross(nn.Module):
    def __init__(self,dim,width,heads):
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,width))
        self.attn = nn.MultiheadAttention(width,heads,dropout=0.,batch_first=True)
        self.output = nn.Linear(width,dim)
        self.alpha = nn.Parameter(torch.zeros(()))
    def forward(self,x,visual,gate):
        q = self.query(x)
        delta = self.output(self.attn(q,visual,visual,need_weights=False)[0])
        return x+torch.tanh(self.alpha)*gate[:,None,None]*delta

class Adapter(nn.Module):
    def __init__(self,bart,c):
        super().__init__()
        d,w = bart.config.d_model,c['width']
        self.project = nn.Sequential(nn.LayerNorm(d),nn.Linear(d,w))
        self.cross = nn.ModuleList(Cross(d,w,c['heads']) for _ in range(c['decoder_layers']))
        self.context = None; self.handles = []
        if len(bart.model.decoder.layers)<len(self.cross): raise ValueError('Too few BART layers')
        for i,layer in enumerate(bart.model.decoder.layers[-len(self.cross):]):
            self.handles.append(layer.encoder_attn_layer_norm.register_forward_hook(self._hook(i)))
    def _hook(self,i):
        def hook(module,inputs,output):
            if self.context is None: return output
            visual,gate = self.context
            if output.shape[0]!=gate.shape[0]: raise ValueError('Only registered greedy/no-beam geometry')
            return self.cross[i](output,visual,gate)
        return hook
    @contextmanager
    def use(self,memory,aux,gate,enabled=True):
        if self.context is not None: raise RuntimeError('Nested use forbidden')
        b,ch,slots,dim = memory.shape
        if ch!=9 or slots!=32 or aux.shape!=(b,32,dim) or gate.shape!=(b,9):
            raise ValueError('Fixed public receiver geometry differs')
        if not all(torch.isfinite(x).all() for x in (memory,aux,gate)) or not ((gate>=0)&(gate<=1)).all():
            raise ValueError('Invalid received states')
        if enabled:
            visual = self.project(aux.float())[:,None].expand(-1,9,-1,-1).flatten(0,1)
            self.context = visual,gate.detach().flatten()
        try: yield
        finally: self.context = None
    def close(self):
        for h in self.handles: h.remove()
        self.handles = []
    def groups(self):
        return dict(project=('project.',),cross0=('cross.0.',),cross1=('cross.1.',))

def ce(bart,adapter,memory,aux,gate,ids):
    labels = ids.flatten(0,1).masked_fill(ids.flatten(0,1).eq(bart.config.pad_token_id),-100)
    labels = labels[:,:int(labels.ne(-100).sum(1).max())]
    shifted = shift_tokens_right(labels,bart.config.pad_token_id,bart.config.decoder_start_token_id)
    with adapter.use(memory,aux,gate):
        logits = bart(encoder_outputs=BaseModelOutput(last_hidden_state=memory.flatten(0,1)),
            attention_mask=torch.ones(memory.shape[:3],device=memory.device,dtype=torch.long).flatten(0,1),
            decoder_input_ids=shifted,use_cache=False,return_dict=True).logits
    value = F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100,reduction='sum')
    return value,int(labels.ne(-100).sum())

@torch.no_grad()
def generate(bart,tok,memory,adapter=None,aux=None,gate=None,enabled=True):
    from docci_baseline_v14.model import decode
    rx = dict(memory=memory,receiver_mask=torch.ones(memory.shape[:3],device=memory.device,dtype=torch.bool))
    if adapter is None: return decode(bart,rx,tok,'greedy')
    with adapter.use(memory,aux,gate,enabled): return decode(bart,rx,tok,'greedy')
