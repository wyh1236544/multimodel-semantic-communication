"""Frozen pretrained BART decoder; trainable position-preserving C/I/T bridge.

No source ids, source mask, length, image or teacher vector enters receive().
Frozen modules remain differentiable with respect to received memories.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from bart_clip_joint_v1 import CANDIDATES


def unit(x):
    return x/x.square().mean((1, 2), keepdim=True).clamp_min(1e-8).sqrt()


def encoder(hidden, cfg):
    layer = nn.TransformerEncoderLayer(hidden, cfg['heads'], hidden*4, dropout=0.,
        activation='gelu', batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, cfg['layers'])


class JointBridge(nn.Module):
    def __init__(self, bart, image_reference, cfg, clip_width=512):
        super().__init__()
        self.bart = bart.requires_grad_(False).eval()
        self.image_reference = image_reference.requires_grad_(False).eval()
        self.cfg = dict(cfg); self.arm = 'J1_joint'
        d, h = bart.config.d_model, cfg['hidden']
        self.text_fold = nn.Sequential(nn.Linear(4*d, h), nn.GELU(), nn.LayerNorm(h))
        self.text_source = encoder(h, cfg)
        self.text_source_position = nn.Parameter(torch.randn(1, 32, h)*.02)
        self.text_tx = nn.Linear(h, 16)
        self.text_rx = nn.Linear(16, h)
        self.text_receiver = encoder(h, cfg)
        self.text_receiver_position = nn.Parameter(torch.randn(1, 32, h)*.02)
        self.text_expand = nn.Linear(h, 4*d)
        self.bart_memory_norm = nn.LayerNorm(d)
        self.bart_memory_adapter = nn.Linear(d, d)
        nn.init.eye_(self.bart_memory_adapter.weight); nn.init.zeros_(self.bart_memory_adapter.bias)
        self.bart_common = nn.Linear(d, h)
        self.image_common = nn.Linear(16, h)
        self.clip_image = nn.Linear(clip_width, h)
        self.clip_text = nn.Linear(clip_width, h)
        self.common_queries = nn.Parameter(torch.randn(1, 32, h)*.02)
        self.common_attention = nn.MultiheadAttention(h, cfg['heads'], dropout=0., batch_first=True)
        self.common_norm = nn.LayerNorm(h)
        self.common_tx = nn.Linear(h, 16)
        self.common_rx = nn.Linear(16, h)
        self.text_cross = nn.MultiheadAttention(d, 8 if d % 8 == 0 else cfg['heads'], dropout=0.,
                                               kdim=h, vdim=h, batch_first=True)
        self.text_gate = nn.Linear(2, 1)
        nn.init.zeros_(self.text_gate.weight); nn.init.constant_(self.text_gate.bias, -2.)
        self.image_tx = nn.Sequential(nn.Linear(16, h), nn.GELU(), nn.Linear(h, 16))
        self.image_rx = nn.Sequential(nn.Linear(16, h), nn.GELU(), nn.Linear(h, 16))
        self.image_query = nn.Linear(4, h)
        self.image_cross = nn.MultiheadAttention(h, cfg['heads'], dropout=0., batch_first=True)
        self.image_residual = nn.Linear(h, 4)
        self.common_alignment = nn.Linear(h, clip_width)
        for block in (self.image_tx, self.image_rx):
            nn.init.normal_(block[-1].weight, std=1e-4); nn.init.zeros_(block[-1].bias)
        nn.init.normal_(self.image_residual.weight, std=1e-4); nn.init.zeros_(self.image_residual.bias)
        self.eval()

    def configure(self, arm):
        if arm not in CANDIDATES: raise ValueError('Unknown matched candidate')
        self.arm = arm; self.eval()

    def train(self, mode=True):
        # Deterministic pretrained dropout policy; autograd is still enabled.
        super().train(False)
        return self

    def source_encode(self, bart_hidden, source_mask, image_wire, clip_image, clip_text):
        b, n, d = bart_hidden.shape
        if n != 128 or image_wire.shape != (b, 224, 16): raise ValueError('Source capacity/layout changed')
        # Mask is TX-only. It is not transmitted or used to mask RX memory.
        source = bart_hidden*source_mask.to(bart_hidden).unsqueeze(-1)
        t = self.text_fold(source.reshape(b, 32, 4*d))
        t = unit(self.text_tx(self.text_source(t+self.text_source_position)))
        i = unit(image_wire+self.image_tx(image_wire))
        use_clip = 0. if self.arm == 'J0_no_clip_input' else 1.
        memory = torch.cat((self.image_common(image_wire), self.bart_common(source),
            self.clip_image(clip_image[:, None])*use_clip, self.clip_text(clip_text[:, None])*use_clip), 1)
        # Only public group layout plus TX text padding; no source-dependent norms across groups.
        pad = torch.cat((torch.zeros(b, 224, device=t.device, dtype=torch.bool), ~source_mask.bool(),
                         torch.zeros(b, 2, device=t.device, dtype=torch.bool)), 1)
        mask = None
        if self.arm == 'J2_separate':
            qgroup = torch.arange(32, device=t.device) >= 16
            mgroup = torch.tensor([0]*224+[1]*128+[0, 1], device=t.device, dtype=torch.bool)
            mask = qgroup[:, None] != mgroup[None, :]
        queries = self.common_queries.expand(b, -1, -1)
        c, _ = self.common_attention(queries, memory, memory, attn_mask=mask,
                                   key_padding_mask=pad, need_weights=False)
        c = self.common_tx(self.common_norm(c+queries))
        c = torch.cat((unit(c[:, :16]), unit(c[:, 16:])), 1)
        return torch.cat((c, i, t), 1)

    def receive(self, wire, reliability, pilot_snr, available, text_only=False):
        if wire.shape[1:] != (288, 16): raise ValueError('Public C32/I224/T32 required')
        b = wire.shape[0]; ok = available.to(wire).reshape(b, 1, 1)
        wire = wire*ok
        # Group-local RX: private I never enters common text memory or T codec.
        raw_common = self.common_rx(wire[:, :32])
        text_common = raw_common[:, 16:] if self.arm == 'J2_separate' else raw_common
        image_common = raw_common[:, :16] if self.arm == 'J2_separate' else raw_common
        h = self.text_receiver(self.text_rx(wire[:, 256:])+self.text_receiver_position)
        memory = self.text_expand(h).reshape(b, 128, self.bart.config.d_model)
        memory = self.bart_memory_adapter(self.bart_memory_norm(memory))
        hint, _ = self.text_cross(memory, text_common, text_common, need_weights=False)
        public = torch.stack((reliability.to(wire).reshape(b), pilot_snr.to(wire).clamp(-40, 40).reshape(b)/40), -1)
        memory = (memory+torch.sigmoid(self.text_gate(public))[:, None]*hint)*ok
        out = dict(text_memory=memory, physical_common=wire[:, :32], common_features=raw_common,
                   common_alignment=self.common_alignment(raw_common.mean(1))*ok[:, 0])
        if not text_only:
            corrected = wire[:, 32:256]+self.image_rx(wire[:, 32:256])
            image_frame = torch.cat((wire.new_zeros(b, 32, 16), corrected, wire.new_zeros(b, 32, 16)), 1)*ok
            rel = reliability.to(wire).reshape(b, 1, 1).expand(-1, 288, 1)
            # Fixed B4 image-only reference; no original source or text decoding at RX.
            base = self.image_reference.receive(image_frame, rel, pilot_snr, available)['latent']
            query = self.image_query(base.flatten(2).transpose(1, 2))
            hint, _ = self.image_cross(query, image_common, image_common, need_weights=False)
            delta = self.image_residual(hint).transpose(1, 2).reshape_as(base)
            out['latent'] = (base+delta)*ok.reshape(b, 1, 1, 1)
        return out

    def teacher_loss(self, ids, mask, memory):
        from transformers.modeling_outputs import BaseModelOutput
        labels = ids.masked_fill(~mask.bool(), -100)
        # HF shifts BART labels with decoder_start_token_id=EOS, not CLIP BOS.
        result = self.bart(encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=torch.ones(memory.shape[:2], device=memory.device, dtype=torch.long),
            labels=labels, use_cache=False, return_dict=True)
        return result.loss

    @torch.no_grad()
    def generate(self, memory, tokenizer, available=True):
        from transformers.modeling_outputs import BaseModelOutput
        if not available: return dict(text='', stopped_by_capacity=False, receiver_failure=True)
        tokens = self.bart.generate(encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=torch.ones(memory.shape[:2], device=memory.device, dtype=torch.long),
            max_length=129, min_length=0, num_beams=1, do_sample=False, use_cache=True,
            no_repeat_ngram_size=0, repetition_penalty=1., length_penalty=1.,
            forced_bos_token_id=self.bart.config.bos_token_id, forced_eos_token_id=None,
            eos_token_id=self.bart.config.eos_token_id, pad_token_id=self.bart.config.pad_token_id,
            decoder_start_token_id=self.bart.config.decoder_start_token_id)
        # Leading EOS is BART decoder start and MUST NOT count as predicted EOS.
        ended = bool((tokens[0, 1:] == self.bart.config.eos_token_id).any())
        return dict(text=tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True),
                    stopped_by_capacity=not ended, receiver_failure=False)

    def trainable_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable(self, weights):
        named = dict(self.named_parameters()); expected = {n for n, p in named.items() if p.requires_grad}
        if set(weights) != expected: raise ValueError('Candidate trainable tensor set differs')
        with torch.no_grad():
            for n, value in weights.items():
                if value.shape != named[n].shape or not torch.isfinite(value).all(): raise ValueError('Bad saved bridge '+n)
                named[n].copy_(value.to(named[n]))
