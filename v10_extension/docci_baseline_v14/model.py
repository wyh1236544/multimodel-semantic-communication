"""Legacy vs released-code-aligned text codec; no claim of full paper replication."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from bart_latent_codec_v4.model import LatentCodec, QueryBlock, MemoryBlock, prediction
from docci_baseline_v14.core import DECODE
from docci_reconstruction_v10.core import joined_ids


class AlignedCodec(LatentCodec):
    """Independent PyTorch implementation of public LD4LG AE conventions.

    released64 follows the RELEASED narrow-query code, not Table13's width768.
    wide768 is an explicitly different capacity experiment. Both include a final
    projection to64, fixed32 slots, q/k RMSNorm and unit per-slot mean power.
    Shared kv at equal dimensions matches public PerceiverAttention semantics.
    No upstream package, network access or pretrained diffusion is required.
    """
    def __init__(self, bart, arm, depth=3, head_dim=64, slots=32, maximum=64):
        nn.Module.__init__(self)
        self.bart = bart.requires_grad_(False).eval()
        self.width, self.slots, self.maximum = 64, slots, maximum
        d = bart.config.d_model
        qdim = 64 if arm == 'released64' else d
        self.source_position = nn.Parameter(torch.randn(1, maximum, d))
        self.queries = nn.Parameter(torch.randn(1, slots, qdim)*.02)
        self.compressor = nn.ModuleList(QueryBlock(d, qdim, head_dim) for _ in range(depth))
        if qdim == d:
            for block in self.compressor:
                block.latent_kv = block.source_kv
        self.query_final_norm = nn.LayerNorm(qdim)
        self.to_wire = nn.Linear(qdim, 64) if qdim != 64 else nn.Identity()
        self.memory_projection = nn.Linear(64, d)
        self.memory_position = nn.Parameter(torch.randn(1, slots, d))
        self.reconstructor = nn.ModuleList(MemoryBlock(d, head_dim) for _ in range(depth))
        self.memory_norm = nn.LayerNorm(d)
        self.eval()

    def encode(self, hidden, source_mask):
        if hidden.shape[1:] != (self.maximum, self.bart.config.d_model):
            raise ValueError('Wrong public source feature shape')
        if source_mask.shape != hidden.shape[:2] or not source_mask.bool().any(1).all():
            raise ValueError('Wrong source mask')
        source = hidden*source_mask.unsqueeze(-1).to(hidden)+self.source_position/math.sqrt(self.bart.config.d_model)
        z = self.queries.expand(hidden.shape[0], -1, -1)
        for block in self.compressor:
            z = block(source, z, source_mask)
        z = self.to_wire(self.query_final_norm(z))
        return F.normalize(z, dim=-1, eps=1e-8)*8.

    def receive_text(self, observed, available=None):
        if observed.ndim != 2 or observed.shape[1] != self.slots*64 or not torch.isfinite(observed).all():
            raise ValueError('Invalid public observed text frame')
        b = observed.shape[0]
        ok = torch.ones(b,device=observed.device,dtype=torch.bool) if available is None else available.bool().reshape(b)
        memory = self.memory_projection(observed.reshape(b,self.slots,64))
        memory = memory+self.memory_position/math.sqrt(self.bart.config.d_model)
        for block in self.reconstructor:
            memory = block(memory)
        memory = self.memory_norm(memory)*ok[:,None,None]
        return dict(memory=memory, receiver_mask=torch.ones(b,self.slots,device=observed.device,dtype=torch.bool), available=ok)

    def groups(self):
        return dict(compressor=('queries', 'source_position', 'compressor.',
                                'query_final_norm.', 'to_wire.'),
                    receiver=('memory_projection.', 'memory_position', 'reconstructor.', 'memory_norm.'))


class TextModel(nn.Module):
    def __init__(self, bart, arm):
        super().__init__()
        self.arm = arm
        if arm == 'legacy64':
            self.codec = LatentCodec(bart, dict(slots=32, depth=3, head_dim=64, max_tokens=64), 64)
        elif arm in ('released64', 'wide768'):
            self.codec = AlignedCodec(bart, arm)
        else:
            raise ValueError('Unknown text arm')
        self.eval()

    @property
    def bart(self):
        return self.codec.bart

    def encode(self, ids, hidden):
        b = ids.shape[0]
        wire = self.codec.transmit(hidden.flatten(0, 1).float(),
                                   ids.flatten(0, 1).ne(self.bart.config.pad_token_id))
        return wire.reshape(b, -1)

    def receive(self, observed):
        if observed.ndim != 2 or observed.shape[1] != 18432:
            raise ValueError('Wrong public9-chunk layout')
        b = observed.shape[0]
        value = self.codec.receive_text(observed.reshape(b*9, 2048))
        return {k: v.reshape(b, 9, *v.shape[1:]) for k, v in value.items() if k != 'available'}

    def saved(self):
        return {n: p.detach().cpu().clone() for n, p in self.codec.named_parameters()
                if not n.startswith('bart.')}

    def restore(self, state):
        current = {n: p for n, p in self.codec.named_parameters() if not n.startswith('bart.')}
        if set(current) != set(state):
            raise ValueError('Text checkpoint parameter names differ')
        with torch.no_grad():
            for name, p in current.items():
                if state[name].shape != p.shape or not torch.isfinite(state[name]).all():
                    raise ValueError('Bad checkpoint tensor: ' + name)
                p.copy_(state[name].to(p))

    def groups(self):
        return {name: tuple('codec.'+x for x in prefixes)
                for name, prefixes in self.codec.groups().items()}


def loss(model, ids, hidden):
    rx = model.receive(model.encode(ids, hidden))
    flat = ids.flatten(0, 1)
    logits, labels = prediction(model.bart, flat, flat.ne(model.bart.config.pad_token_id),
                                rx['memory'].flatten(0, 1), rx['receiver_mask'].flatten(0, 1))
    total = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction='sum')
    return total, int(labels.ne(-100).sum())


@torch.no_grad()
def decode(bart, rx, tokenizer, strategy):
    if strategy not in DECODE or rx['memory'].shape[0] != 1:
        raise ValueError('Single-description public decode required')
    ids = bart.generate(encoder_outputs=BaseModelOutput(last_hidden_state=rx['memory'][0]),
        attention_mask=rx['receiver_mask'][0].long(), max_length=65,
        do_sample=False, use_cache=True, forced_bos_token_id=tokenizer.bos_token_id,
        forced_eos_token_id=None, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, decoder_start_token_id=bart.config.decoder_start_token_id,
        **DECODE[strategy]).tolist()
    joined, capacity = joined_ids(ids, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
    return dict(received_caption=tokenizer.decode(joined, skip_special_tokens=True,
                clean_up_tokenization_spaces=False), chunk_generated_token_ids=ids,
                generated_token_ids=joined, stopped_by_capacity=capacity)
