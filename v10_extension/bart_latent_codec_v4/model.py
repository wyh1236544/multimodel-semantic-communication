"""Learned semantic slots; no four-token folding or original-position MSE.

Independent implementation inspired by LD4LG's public Perceiver autoencoder.
BART is frozen, but decoder operations retain gradients to received memory.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import shift_tokens_right


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x*x.square().mean(-1, keepdim=True).clamp_min(1e-8).rsqrt()*self.scale


class QueryBlock(nn.Module):
    def __init__(self, source_dim, latent_dim, head_dim):
        super().__init__(); self.heads = source_dim//head_dim; self.head_dim = head_dim
        if source_dim % head_dim: raise ValueError('Source dimension/head mismatch')
        self.source_norm = nn.LayerNorm(source_dim); self.query_norm = nn.LayerNorm(latent_dim)
        self.q = nn.Linear(latent_dim, source_dim, bias=False)
        self.source_kv = nn.Linear(source_dim, 2*source_dim, bias=False)
        self.latent_kv = nn.Linear(latent_dim, 2*source_dim, bias=False)
        self.q_norm = RMSNorm(head_dim); self.k_norm = RMSNorm(head_dim)
        self.out = nn.Linear(source_dim, latent_dim)
        self.ff = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, 4*latent_dim),
            nn.GELU(), nn.Linear(4*latent_dim, latent_dim))

    def forward(self, source, latent, source_mask):
        b, slots, _ = latent.shape; qlatent = self.query_norm(latent)
        q = self.q(qlatent).reshape(b, slots, self.heads, self.head_dim).transpose(1, 2)
        kv = torch.cat((self.source_kv(self.source_norm(source)), self.latent_kv(qlatent)), 1)
        k, v = kv.chunk(2, -1)
        k = k.reshape(b, -1, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, -1, self.heads, self.head_dim).transpose(1, 2)
        visible = torch.cat((source_mask.bool(), torch.ones(b, slots, device=source.device, dtype=torch.bool)), 1)
        score = (self.q_norm(q) @ self.k_norm(k).transpose(-1, -2))/math.sqrt(self.head_dim)
        score = score.masked_fill(~visible[:, None, None], -torch.finfo(score.dtype).max)
        attention = score.softmax(-1)
        pooled = (attention @ v).transpose(1, 2).reshape(b, slots, -1)
        latent = latent+self.out(pooled)
        return latent+self.ff(latent)


class MemoryBlock(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__(); self.heads = dim//head_dim; self.head_dim = head_dim
        self.norm = nn.LayerNorm(dim); self.qkv = nn.Linear(dim, 3*dim, bias=False)
        self.q_norm = RMSNorm(head_dim); self.k_norm = RMSNorm(head_dim); self.out = nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = [t.reshape(b, n, self.heads, self.head_dim).transpose(1, 2) for t in self.qkv(self.norm(x)).chunk(3, -1)]
        attention = ((self.q_norm(q) @ self.k_norm(k).transpose(-1, -2))/math.sqrt(self.head_dim)).softmax(-1)
        x = x+self.out((attention @ v).transpose(1, 2).reshape(b, n, d))
        return x+self.ff(x)


class LatentCodec(nn.Module):
    def __init__(self, bart, c, width):
        super().__init__(); self.bart = bart.requires_grad_(False).eval()
        self.width = width; self.slots = c['slots']; self.maximum = c['max_tokens']; dim = bart.config.d_model
        self.source_position = nn.Parameter(torch.randn(1, self.maximum, dim)*.02)
        self.queries = nn.Parameter(torch.randn(1, self.slots, width)*.02)
        self.compressor = nn.ModuleList(QueryBlock(dim, width, c['head_dim']) for _ in range(c['depth']))
        self.latent_norm = nn.LayerNorm(width)
        self.memory_projection = nn.Linear(width, dim)
        self.memory_position = nn.Parameter(torch.randn(1, self.slots, dim)*.02)
        self.reconstructor = nn.ModuleList(MemoryBlock(dim, c['head_dim']) for _ in range(c['depth']))
        self.memory_norm = nn.LayerNorm(dim)
        self.eval()  # dropout-free, gradients still enabled

    def encode(self, hidden, source_mask):
        if hidden.shape[1:] != (self.maximum, self.bart.config.d_model) or source_mask.shape != hidden.shape[:2]:
            raise ValueError('Wrong source feature shape')
        if not torch.isfinite(hidden).all() or not source_mask.bool().any(1).all(): raise ValueError('Invalid source')
        # Mask PAD values before normalization too: padded values cannot affect TX.
        source = hidden*source_mask.unsqueeze(-1).to(hidden)+self.source_position
        z = self.queries.expand(hidden.shape[0], -1, -1)
        for block in self.compressor: z = block(source, z, source_mask)
        z = self.latent_norm(z)
        # Per-slot unit mean-square power; no source-dependent scale is sent free.
        return F.normalize(z, dim=-1, eps=1e-8)*math.sqrt(self.width)

    def transmit(self, hidden, source_mask):
        return self.encode(hidden, source_mask).reshape(hidden.shape[0], -1)

    def receive_text(self, observed, available=None):
        # This boundary deliberately cannot accept source mask, length, IDs or features.
        if observed.ndim != 2 or observed.shape[1] != self.slots*self.width: raise ValueError('Wrong public payload')
        if not torch.isfinite(observed).all(): raise FloatingPointError('Nonfinite received symbols')
        b = observed.shape[0]
        ok = torch.ones(b, device=observed.device, dtype=torch.bool) if available is None else available.bool().reshape(b)
        memory = self.memory_projection(observed.reshape(b, self.slots, self.width))+self.memory_position
        for block in self.reconstructor: memory = block(memory)
        memory = self.memory_norm(memory)*ok[:, None, None]
        return dict(memory=memory, receiver_mask=torch.ones(b, self.slots, device=observed.device, dtype=torch.bool), available=ok)

    def state(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def restore(self, state):
        named = {n: p for n, p in self.named_parameters() if p.requires_grad}
        if set(named) != set(state): raise ValueError('Codec checkpoint keys differ')
        with torch.no_grad():
            for n, p in named.items():
                if state[n].shape != p.shape or not torch.isfinite(state[n]).all(): raise ValueError('Invalid saved '+n)
                p.copy_(state[n].to(p))

    def groups(self):
        return dict(compressor=('queries', 'source_position', 'compressor.'),
            tx_normalization=('latent_norm.',), receiver=('memory_projection.', 'memory_position', 'reconstructor.'),
            memory=('memory_norm.',))


def prediction(bart, ids, mask, memory, memory_mask):
    labels = ids.masked_fill(~mask.bool(), -100)
    end = int(mask.sum(1).max())
    labels = labels[:, :end]
    shifted = shift_tokens_right(labels, bart.config.pad_token_id, bart.config.decoder_start_token_id)
    logits = bart(encoder_outputs=BaseModelOutput(last_hidden_state=memory), attention_mask=memory_mask.long(),
        decoder_input_ids=shifted, use_cache=False, return_dict=True).logits
    return logits, labels


def distillation(student, teacher, labels, temperature):
    # Word distribution KD, teacher forced by the SAME gold prefix; teacher TRAIN only.
    valid = labels.ne(-100)
    if not valid.any(): raise ValueError('No valid target labels')
    s = student[valid]/temperature; t = teacher.detach()[valid]/temperature
    return F.kl_div(F.log_softmax(s, -1), F.softmax(t, -1), reduction='batchmean')*temperature**2


@torch.no_grad()
def generate(bart, output, tokenizer, constrained=False):
    if not bool(output['available'][0]): return dict(text='', stopped_by_capacity=False, receiver_failure=True)
    tokens = bart.generate(encoder_outputs=BaseModelOutput(last_hidden_state=output['memory']),
        attention_mask=output['receiver_mask'].long(), max_length=129,
        num_beams=4 if constrained else 1, do_sample=False, use_cache=True,
        no_repeat_ngram_size=3 if constrained else 0, repetition_penalty=1.2 if constrained else 1.,
        forced_bos_token_id=bart.config.bos_token_id, forced_eos_token_id=None,
        eos_token_id=bart.config.eos_token_id, pad_token_id=bart.config.pad_token_id,
        decoder_start_token_id=bart.config.decoder_start_token_id)
    ended = bool(tokens[0, 1:].eq(bart.config.eos_token_id).any())
    return dict(text=tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True), stopped_by_capacity=not ended, receiver_failure=False)
