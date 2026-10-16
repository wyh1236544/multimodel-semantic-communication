"""Segment-query residual memory repair. No BART hooks or source-side RX inputs."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import shift_tokens_right
from docci_rx_bridge_v16.model import Resample, content_gate


def canonical(memory, visual, source):
    raw = (F.adaptive_avg_pool1d(memory.flatten(1, 2).transpose(1, 2), 196).transpose(1, 2)
           if source == 'self' else visual)
    if raw.shape[1] != 196 or raw.shape[2] != memory.shape[-1]:
        raise ValueError('Equal public auxiliary support required')
    return F.normalize(raw.float(), dim=-1, eps=1e-8)*math.sqrt(raw.shape[-1])


class RepairBlock(nn.Module):
    def __init__(self, w, heads):
        super().__init__()
        self.norm = nn.LayerNorm(w)
        self.attn = nn.MultiheadAttention(w, heads, dropout=0., batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, 4*w), nn.GELU(), nn.Linear(4*w, w))

    def forward(self, q, slots):
        q = q+self.attn(self.norm(q), slots, slots, need_weights=False)[0]
        return q+self.ff(q)


class SegmentBridge(nn.Module):
    def __init__(self, dim, c):
        super().__init__()
        w = c['width']
        self.resampler = Resample(c, dim)
        self.query_norm = nn.LayerNorm(dim)
        self.query_projection = nn.Linear(dim, w)
        self.segment_position = nn.Parameter(torch.randn(1, 9, 1, w)*.02)
        self.blocks = nn.ModuleList(RepairBlock(w, c['heads']) for _ in range(2))
        self.to_memory = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, dim))
        self.matching = nn.Sequential(nn.Linear(4*w, w), nn.GELU(), nn.Linear(w, 1))
        self.image_semantic = nn.Linear(w, dim)
        self.segment_semantic = nn.Linear(w, dim)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.bound = c['delta_bound']

    def features(self, memory, aux, gate):
        if memory.shape[1:3] != (9, 32) or gate.shape != memory.shape[:2]:
            raise ValueError('Fixed public text memory/gate geometry')
        slots = self.resampler(aux)
        q = self.query_projection(self.query_norm(memory))+self.segment_position
        weights = gate.detach()/gate.detach().sum(1, keepdim=True).clamp_min(1e-8)
        text = (q.mean(2)*weights[:, :, None]).sum(1)
        return slots, q, text

    def match(self, slots, text):
        image = slots.mean(1)
        return self.matching(torch.cat((image, text, image*text, (image-text).abs()), -1)).squeeze(-1)

    def repair(self, memory, aux, gate, enabled=True):
        if not enabled:
            return memory, None
        slots, q, text = self.features(memory, aux, gate)
        score = self.match(slots, text)
        q = q.flatten(1, 2)
        for block in self.blocks:
            q = block(q, slots)
        q = q.reshape(memory.shape[0], 9, 32, -1)
        delta = torch.tanh(self.to_memory(q))*self.bound
        repaired = memory+torch.tanh(self.alpha)*gate[:, :, None, None]*score.sigmoid()[:, None, None, None]*delta
        return repaired, dict(slots=slots, segment=q.mean(2), score=score, text=text)

    def auxiliary(self, memory, aux, gate, teacher, counts, wrong_aux, contrastive=False, temperature=.1, features=None):
        if features is None:
            _, f = self.repair(memory, aux, gate)
        else:
            f = features
        wrong_slots = self.resampler(wrong_aux)
        wrong_score = self.match(wrong_slots, f['text'])
        matching = .5*(F.binary_cross_entropy_with_logits(f['score'], torch.ones_like(f['score']))+
                        F.binary_cross_entropy_with_logits(wrong_score, torch.zeros_like(wrong_score)))
        prediction = F.normalize(self.segment_semantic(f['segment']), dim=-1)
        target = F.normalize(teacher[:, :9].detach(), dim=-1)
        weight = counts.to(prediction).detach()
        segment = ((1-(prediction*target).sum(-1))*weight).sum()/weight.sum().clamp_min(1)
        contrast = prediction.sum()*0.
        if contrastive:
            if memory.shape[0] < 2:
                raise ValueError('Contrastive learning requires an actual multi-example batch')
            image = F.normalize(self.image_semantic(f['slots'].mean(1)), dim=-1)
            text = F.normalize(teacher[:, 9].detach(), dim=-1)
            scores = image@text.T/temperature
            label = torch.arange(len(image), device=image.device)
            contrast = .5*(F.cross_entropy(scores, label)+F.cross_entropy(scores.T, label))
        return contrast, segment, matching

    def groups(self):
        return dict(resampler=('resampler.',), repair=('query_norm.', 'query_projection.',
                    'segment_position', 'blocks.'), memory_output=('to_memory.',), residual_gate=('alpha',),
                    matching=('matching.',), semantic_image=('image_semantic.',), semantic_segment=('segment_semantic.',))


def ce(bart, bridge, memory, aux, gate, ids, return_features=False):
    memory, features = bridge.repair(memory, aux, gate)
    labels = ids.flatten(0, 1).masked_fill(ids.flatten(0, 1).eq(bart.config.pad_token_id), -100)
    labels = labels[:, :int(labels.ne(-100).sum(1).max())]
    shifted = shift_tokens_right(labels, bart.config.pad_token_id, bart.config.decoder_start_token_id)
    logits = bart(encoder_outputs=BaseModelOutput(last_hidden_state=memory.flatten(0, 1)),
                  attention_mask=torch.ones(memory.shape[0]*9, 32, device=memory.device, dtype=torch.long),
                  decoder_input_ids=shifted, use_cache=False, return_dict=True).logits
    value = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction='sum')
    count = int(labels.ne(-100).sum())
    return (value, count, features) if return_features else (value, count)


@torch.no_grad()
def generate(bart, bridge, tokenizer, memory, aux, gate, enabled=True):
    from docci_baseline_v14.model import decode
    if bridge is not None:
        memory, _ = bridge.repair(memory, aux, gate, enabled)
    return decode(bart, dict(memory=memory, receiver_mask=torch.ones(memory.shape[:3],
                  device=memory.device, dtype=torch.bool)), tokenizer, 'greedy')


@torch.no_grad()
def native_teacher(bart, ids):
    """TRAIN labels only. Never called by the deployed receiver/evaluator."""
    b, chunks, length = ids.shape
    content = ids.gt(2)
    count = content.sum(-1)
    flat = ids.flatten(0, 1)
    active = count.flatten().gt(0)
    teacher = torch.zeros(b*chunks, bart.config.d_model, device=ids.device)
    hidden = bart.model.encoder(input_ids=flat[active], attention_mask=flat[active].ne(1).long(),
                               return_dict=True).last_hidden_state.float()
    mask = content.flatten(0, 1)[active]
    teacher[active] = (hidden*mask[:, :, None]).sum(1)/mask.sum(1)[:, None].clamp_min(1)
    teacher = teacher.reshape(b, chunks, -1)
    paragraph = (teacher*count[:, :, None]).sum(1)/count.sum(1)[:, None].clamp_min(1)
    return torch.cat((teacher, paragraph[:, None]), 1), count
