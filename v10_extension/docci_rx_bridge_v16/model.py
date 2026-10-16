"""Frozen BART + external gated adapters, after old cross-attention BEFORE FFN.

Hooks return the untouched norm output when disabled. Original BART modules,
parameters, original text K/V and HuggingFace cache format are never replaced.
Adapters are registered ONLY in Bridge, not on BART. No inference source masks.
"""
from contextlib import contextmanager
import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import shift_tokens_right


def canonical(aux, kind):
    """Equal196x768 fixed support; I0 padding has no learned extra parameters."""
    if kind == 'self':
        aux = F.adaptive_avg_pool1d(aux.flatten(1, 2).transpose(1, 2), 196).transpose(1, 2)
    elif kind == 'rx_i0':
        if aux.shape[1:] != (196, 96):
            raise ValueError('Expected fixed received I0 grid')
        aux = F.pad(aux, (0, 672))
    elif kind not in ('oracle_clip', 'rx_clip'):
        raise ValueError('Unknown auxiliary source')
    if aux.shape[1:] != (196, 768) or not torch.isfinite(aux).all():
        raise ValueError('Invalid public auxiliary features')
    return F.normalize(aux.float(), dim=-1, eps=1e-8)*math.sqrt(768)


class Resample(nn.Module):
    def __init__(self, c, dim=768):
        super().__init__()
        w = c['width']
        self.input = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, w))
        self.query = nn.Parameter(torch.randn(1, c['slots'], w)*.02)
        self.attn = nn.ModuleList(nn.MultiheadAttention(w, c['heads'], dropout=0., batch_first=True)
                                  for _ in range(2))
        self.norm = nn.ModuleList(nn.LayerNorm(w) for _ in range(2))
        self.ff = nn.ModuleList(nn.Sequential(nn.LayerNorm(w), nn.Linear(w, w*4),
                               nn.GELU(), nn.Linear(w*4, w)) for _ in range(2))

    def forward(self, x):
        x = self.input(x)
        q = self.query.expand(x.shape[0], -1, -1)
        for attn, norm, ff in zip(self.attn, self.norm, self.ff):
            q = q+attn(norm(q), x, x, need_weights=False)[0]
            q = q+ff(q)
        return q


class AddCross(nn.Module):
    def __init__(self, dim, c):
        super().__init__()
        self.q = nn.LayerNorm(dim)
        self.kv = nn.Linear(c['width'], dim)
        self.attn = nn.MultiheadAttention(dim, c['heads'], dropout=0., batch_first=True)
        # Only gate is zero: block gradients begin after its first update.
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, q, slots, content_gate):
        kv = self.kv(slots)
        delta = self.attn(self.q(q), kv, kv, need_weights=False)[0]
        return q+torch.tanh(self.gate)*content_gate[:, None, None]*delta


class Bridge(nn.Module):
    def __init__(self, bart, c):
        super().__init__()
        self.resampler = Resample(c, bart.config.d_model)
        self.cross = nn.ModuleList(AddCross(bart.config.d_model, c) for _ in range(2))
        gen = torch.Generator().manual_seed(160042)
        # Frozen target projection; no trainable target which can collapse jointly.
        projection = torch.randn(bart.config.d_model, c['width'], generator=gen)
        self.register_buffer('target_projection', projection/math.sqrt(bart.config.d_model))
        self.context = None
        self.hooks = []
        if len(bart.model.decoder.layers) < 2:
            raise ValueError('At least two BART decoder layers required')
        for j, layer in enumerate(bart.model.decoder.layers[-2:]):
            self.hooks.append(layer.encoder_attn_layer_norm.register_forward_hook(self._hook(j)))

    def _hook(self, j):
        def hook(module, inputs, output):
            if self.context is None:
                return output
            slots, gate = self.context
            if output.shape[0] % slots.shape[0]:
                raise ValueError('Unexpected generation beam expansion')
            beams = output.shape[0]//slots.shape[0]
            return self.cross[j](output, slots.repeat_interleave(beams, 0),
                                 gate.repeat_interleave(beams, 0))
        return hook

    @contextmanager
    def use(self, slots, gate, enabled=True):
        if self.context is not None:
            raise RuntimeError('Nested/reentrant BART use rejected')
        self.context = (slots, gate) if enabled else None
        try:
            yield
        finally:
            self.context = None

    def close(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def target(self, memory, gate):
        # Receiver-only soft content probabilities, not true source lengths.
        # Otherwise the six or more empty public frames dominate the mean.
        w = gate.detach()/gate.detach().sum(1, keepdim=True).clamp_min(1e-8)
        target = (memory.detach().mean(2)*w[:, :, None]).sum(1)
        return F.normalize(target@self.target_projection, dim=-1)

    def alignment(self, aux, memory, gate, temperature):
        z = F.normalize(self.resampler(aux).mean(1), dim=-1)
        target = self.target(memory, gate)
        scores = z@target.T/temperature
        labels = torch.arange(z.shape[0], device=z.device)
        return (F.cross_entropy(scores, labels)+F.cross_entropy(scores.T, labels))/2

    def groups(self):
        return dict(resampler=('resampler.',), cross0=('cross.0.',), cross1=('cross.1.',))


@torch.no_grad()
def content_gate(bart, memory):
    """First FREE prediction after [decoder_start=2,BOS=0], not forced BOS."""
    flat = memory.flatten(0, 1)
    ids = torch.tensor([bart.config.decoder_start_token_id, bart.config.bos_token_id],
                       device=memory.device).expand(flat.shape[0], -1)
    logits = bart(encoder_outputs=BaseModelOutput(last_hidden_state=flat),
                  attention_mask=torch.ones(flat.shape[:2], device=flat.device, dtype=torch.long),
                  decoder_input_ids=ids, use_cache=False, return_dict=True).logits[:, -1]
    return (1-logits.softmax(-1)[:, bart.config.eos_token_id]).reshape(memory.shape[:2])


def prepared(bridge, memory, aux, gate):
    slots = bridge.resampler(aux)
    slots = slots[:, None].expand(-1, 9, -1, -1).flatten(0, 1)
    return slots, gate.flatten()


def ce(bart, bridge, memory, aux, gate, ids):
    labels = ids.flatten(0, 1).masked_fill(ids.flatten(0, 1).eq(bart.config.pad_token_id), -100)
    labels = labels[:, :int(labels.ne(-100).sum(1).max())]
    shifted = shift_tokens_right(labels, bart.config.pad_token_id, bart.config.decoder_start_token_id)
    slots, g = prepared(bridge, memory, aux, gate)
    with bridge.use(slots, g):
        logits = bart(encoder_outputs=BaseModelOutput(last_hidden_state=memory.flatten(0, 1)),
            attention_mask=torch.ones(memory.shape[0]*9, 32, device=memory.device, dtype=torch.long),
            decoder_input_ids=shifted, use_cache=False, return_dict=True).logits
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction='sum'), int(labels.ne(-100).sum())


@torch.no_grad()
def generate(bart, bridge, memory, aux, gate, tokenizer, strategy, enabled=True):
    from docci_baseline_v14.model import decode
    rx = dict(memory=memory, receiver_mask=torch.ones(memory.shape[:3], device=memory.device, dtype=torch.bool))
    slots, g = prepared(bridge, memory, aux, gate)
    with bridge.use(slots, g, enabled):
        return decode(bart, rx, tokenizer, strategy)
