"""Preserve V1 ordered text bridge; receive only observed header/features.

LoRA is explicitly toggled. Source BART uses disabled adapters. No image path
is executed; JointBridge's unused historical image layers stay frozen.
"""
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import shift_tokens_right
from bart_clip_joint_v1.model import JointBridge, unit
from text_recovery_v3.core import spec, geometry, encode_header, decode_header


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__(); self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.randn(rank, base.in_features)*.01)
        self.b = nn.Parameter(torch.zeros(base.out_features, rank)); self.enabled = False

    def forward(self, x):
        y = self.base(x)
        return y+(x@self.a.t())@self.b.t() if self.enabled else y


def install_lora(bart, c):
    blocks = bart.model.decoder.layers[-c['lora_layers']:]
    modules = []
    for block in blocks:
        for key in ('q_proj', 'v_proj'):
            old = getattr(block.encoder_attn, key)
            if isinstance(old, LoRALinear): raise ValueError('Already adapted')
            new = LoRALinear(old, c['rank']).to(old.weight)
            setattr(block.encoder_attn, key, new); modules.append(new)
    bart.requires_grad_(False).eval()
    return modules


def adapters(bart, enabled=False, trainable=False):
    for m in bart.modules():
        if isinstance(m, LoRALinear):
            m.enabled = enabled
            m.a.requires_grad_(trainable); m.b.requires_grad_(trainable)
            m.a.grad = m.b.grad = None


def reset_lora(bart, initial):
    named = dict(bart.named_parameters())
    with torch.no_grad():
        for n, x in initial.items(): named[n].copy_(x.to(named[n]))
    adapters(bart)


def lora_state(bart):
    return {n: p.detach().cpu().clone() for n, p in bart.named_parameters() if n.endswith(('.a', '.b'))}


def logits(bart, ids, mask, memory, receiver_mask):
    labels = ids.masked_fill(~mask.bool(), -100)
    shifted = shift_tokens_right(labels, bart.config.pad_token_id, bart.config.decoder_start_token_id)
    out = bart(encoder_outputs=BaseModelOutput(last_hidden_state=memory), attention_mask=receiver_mask.long(),
        decoder_input_ids=shifted, use_cache=False, return_dict=True)
    return out.logits, labels


@torch.no_grad()
def generate(bart, memory, mask, tokenizer, available=True):
    if not available: return dict(text='', stopped_by_capacity=False, receiver_failure=True)
    tokens = bart.generate(encoder_outputs=BaseModelOutput(last_hidden_state=memory), attention_mask=mask.long(),
        max_length=129, num_beams=1, do_sample=False, use_cache=True, no_repeat_ngram_size=0,
        repetition_penalty=1., forced_bos_token_id=bart.config.bos_token_id, forced_eos_token_id=None,
        eos_token_id=bart.config.eos_token_id, pad_token_id=bart.config.pad_token_id,
        decoder_start_token_id=bart.config.decoder_start_token_id)
    ended = bool((tokens[0, 1:] == bart.config.eos_token_id).any())
    return dict(text=tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True),
        stopped_by_capacity=not ended, receiver_failure=False)


@torch.no_grad()
def consistency(bart, item, device):
    ids = item['input_ids'].to(device); mask = item['mask'].to(device)
    memory = item['bart_hidden'].to(device)
    labels = ids.masked_fill(~mask, -100)
    shifted = shift_tokens_right(labels, bart.config.pad_token_id, bart.config.decoder_start_token_id)
    length = int(mask.sum()); shifted = shifted[:, :length]
    enc = BaseModelOutput(last_hidden_state=memory)
    full = bart(encoder_outputs=enc, attention_mask=mask.long(), decoder_input_ids=shifted,
        use_cache=False, return_dict=True).logits
    cache = None; errors = []
    for k in range(length):
        inc = bart(encoder_outputs=enc, attention_mask=mask.long(), decoder_input_ids=shifted[:, k:k+1],
            past_key_values=cache, use_cache=True, return_dict=True)
        errors.append(float((inc.logits[:, -1]-full[:, k]).abs().max())); cache = inc.past_key_values
    native = bart(input_ids=ids, attention_mask=mask.long(), decoder_input_ids=shifted,
        use_cache=False, return_dict=True).logits
    return dict(incremental_max_logit_error=max(errors), cached_source_max_logit_error=float((native-full).abs().max()),
        tested_tokens=length, label_shift='official shift_tokens_right, start=EOS', source_mask_oracle_diagnostic_only=True)


class TextBridge(JointBridge):
    def __init__(self, bart, c, arm, clip_width=512):
        super().__init__(bart, nn.Identity(), c, clip_width)
        self.experiment_arm = arm; self.s = spec(arm)
        if self.s['width'] != 16:
            self.text_tx = nn.Linear(c['hidden'], self.s['width'])
            self.text_rx = nn.Linear(self.s['width'], c['hidden'])
        text = ('text_fold.', 'text_source.', 'text_source_position', 'text_tx.', 'text_rx.',
            'text_receiver.', 'text_receiver_position', 'text_expand.', 'bart_memory_norm.', 'bart_memory_adapter.',
            'common_rx.', 'text_cross.', 'text_gate.')
        common = ('bart_common.', 'clip_text.', 'common_queries', 'common_attention.', 'common_norm.',
            'common_tx.', 'common_rx.', 'text_cross.', 'text_gate.')
        for n, p in self.named_parameters(): p.requires_grad_(n.startswith(text+(common if self.s['common'] else ())))
        adapters(bart, self.s['lora'], self.s['lora'])
        self.eval()

    def transmit(self, hidden, source_mask, clip_text):
        if hidden.shape[1:] != (128, self.bart.config.d_model) or source_mask.shape != hidden.shape[:2]:
            raise ValueError('Ordered 128-position source required')
        b, _, d = hidden.shape; source = hidden*source_mask.to(hidden).unsqueeze(-1)
        t = unit(self.text_tx(self.text_source(self.text_fold(source.reshape(b, 32, 4*d))+self.text_source_position)))
        parts = []
        if self.s['length_mask']:
            lengths = source_mask.sum(1).detach().cpu().tolist()
            head = torch.tensor([encode_header(int(n)).tolist() for n in lengths], device=hidden.device, dtype=hidden.dtype)
            parts.append(head)
        if self.s['common']:
            memory = torch.cat((self.bart_common(source), self.clip_text(clip_text[:, None])), 1)
            pad = torch.cat((~source_mask.bool(), torch.zeros(b, 1, device=hidden.device, dtype=torch.bool)), 1)
            q = self.common_queries.expand(b, -1, -1)
            c, _ = self.common_attention(q, memory, memory, key_padding_mask=pad, need_weights=False)
            c = self.common_tx(self.common_norm(c+q)); c = torch.cat((unit(c[:, :16]), unit(c[:, 16:])), 1)
            parts.append(c.reshape(b, -1))
        parts.append(t.reshape(b, -1))
        return torch.cat(parts, 1)

    def receive_text(self, observed, available=None, pilot_snr=None, reliability=None):
        # NO source length/mask/tokenizer/ids permitted at this boundary.
        if observed.ndim != 2 or observed.shape[1] != geometry(self.experiment_arm)['total_reals']:
            raise ValueError('Wrong public frame geometry')
        if not torch.isfinite(observed).all(): raise FloatingPointError('Nonfinite RX')
        b = observed.shape[0]; cursor = 0
        ok = torch.ones(b, device=observed.device, dtype=torch.bool) if available is None else available.bool().reshape(b)
        mask = torch.ones(b, 128, device=observed.device, dtype=torch.bool); lengths = [None]*b
        if self.s['length_mask']:
            lengths = []; valid = []
            for x in observed[:, :16].detach().cpu().numpy():
                n, good = decode_header(x); lengths.append(n); valid.append(good)
            ok = ok & torch.tensor(valid, device=observed.device)
            mask = torch.arange(128, device=observed.device)[None] < torch.tensor(lengths, device=observed.device)[:, None]
            cursor = 16
        # Keep V2 private diagnosis' zero-C receiver bias/attention path. This is
        # a constant neural branch, NOT untransmitted source information.
        common = self.common_rx(observed.new_zeros(b, 32, 16))
        if self.s['common']:
            common = self.common_rx(observed[:, cursor:cursor+512].reshape(b, 32, 16)); cursor += 512
        h = self.text_receiver(self.text_rx(observed[:, cursor:].reshape(b, 32, self.s['width']))+self.text_receiver_position)
        memory = self.text_expand(h).reshape(b, 128, self.bart.config.d_model)
        memory = self.bart_memory_adapter(self.bart_memory_norm(memory))
        if common is not None:
            hint, _ = self.text_cross(memory, common, common, need_weights=False)
            snr = observed.new_full((b,), 40.) if pilot_snr is None else pilot_snr.to(observed).reshape(b)
            rel = observed.new_ones(b) if reliability is None else reliability.to(observed).reshape(b)
            public = torch.stack((rel, snr.clamp(-40, 40)/40), -1)
            memory = memory+torch.sigmoid(self.text_gate(public))[:, None]*hint
        return dict(memory=memory*ok[:, None, None], receiver_mask=mask, available=ok,
            received_length=lengths, header_has_crc=False)

    def state(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def restore(self, saved):
        named = {n: p for n, p in self.named_parameters() if p.requires_grad}
        if set(named) != set(saved): raise ValueError('Checkpoint trainable keys differ')
        with torch.no_grad():
            for n, p in named.items():
                x = saved[n]
                if x.shape != p.shape or not torch.isfinite(x).all(): raise ValueError('Invalid checkpoint '+n)
                p.copy_(x.to(p))

    def groups(self):
        groups = dict(source=('text_fold.', 'text_source.'), tx=('text_tx.',),
            rx=('text_rx.', 'text_receiver.', 'text_expand.'), memory=('bart_memory_norm.', 'bart_memory_adapter.'))
        if self.s['lora']: groups['decoder_lora'] = ('bart.model.decoder.',)
        if self.s['common']: groups.update(common_tx=('common_attention.', 'common_tx.'), common_rx=('common_rx.', 'text_cross.'))
        else: groups['constant_common_receiver'] = ('common_rx.', 'text_cross.', 'text_gate.')
        return groups
