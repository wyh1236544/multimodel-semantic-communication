"""V4 text codec + frozen B4 image reference + learned CLIP common semantics.

Private sources remain modality-local. Only the C encoder and C receiver fusion
can connect modalities. Same registered layers for all three candidate arms.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from clip_bart_joint_v5.core import geometry, source_mask
from bart_latent_codec_v4.model import LatentCodec


class Attention(nn.Module):
    """Explicit dropout-free attention avoids mode-dependent fused MHA kernels."""
    def __init__(self, qdim, mdim, hidden, heads):
        super().__init__(); self.heads = heads; self.dim = hidden//heads
        self.q = nn.Linear(qdim, hidden); self.k = nn.Linear(mdim, hidden)
        self.v = nn.Linear(mdim, hidden); self.out = nn.Linear(hidden, hidden)

    def forward(self, queries, memory, blocked=None):
        b, n, _ = queries.shape
        q = self.q(queries).reshape(b, n, self.heads, self.dim).transpose(1, 2)
        k = self.k(memory).reshape(b, -1, self.heads, self.dim).transpose(1, 2)
        v = self.v(memory).reshape(b, -1, self.heads, self.dim).transpose(1, 2)
        score = (q @ k.transpose(-1, -2))/math.sqrt(self.dim)
        if blocked is not None: score = score.masked_fill(blocked[None, None], -torch.finfo(score.dtype).max)
        weight = score.softmax(-1)
        return self.out((weight @ v).transpose(1, 2).reshape(b, n, -1))


def unit(x):
    return x/x.square().mean((1, 2), keepdim=True).clamp_min(1e-8).sqrt()


class JointCodec(nn.Module):
    def __init__(self, bart, reference, text_config, c, clip_dim=512):
        super().__init__(); self.text = LatentCodec(bart, text_config, 64)
        self.reference = reference.requires_grad_(False).eval(); self.c = dict(c); h = c['hidden']
        self.image_source = nn.Linear(16, h); self.clip_image = nn.Linear(clip_dim, h); self.clip_text = nn.Linear(clip_dim, h)
        self.common_queries = nn.Parameter(torch.randn(1, 32, h)*.02)
        self.common_attention = Attention(h, h, h, c['heads']); self.common_norm = nn.LayerNorm(h)
        self.common_tx = nn.Linear(h, 16); self.common_rx = nn.Linear(16, h)
        self.common_image_teacher = nn.Linear(h, clip_dim); self.common_text_teacher = nn.Linear(h, clip_dim)
        self.common_coarse = nn.Linear(h, 64)
        self.image_tx = nn.Sequential(nn.Linear(16, h), nn.GELU(), nn.Linear(h, 16))
        self.image_rx = nn.Sequential(nn.Linear(16, h), nn.GELU(), nn.Linear(h, 16))
        self.image_attention = Attention(4, h, h, c['heads']); self.image_residual = nn.Linear(h, 4)
        self.text_attention = Attention(bart.config.d_model, h, h, c['heads'])
        self.text_residual = nn.Linear(h, bart.config.d_model)
        self.text_gate = nn.Parameter(torch.tensor(-2.))
        for network in (self.image_tx, self.image_rx):
            nn.init.normal_(network[-1].weight, std=1e-4); nn.init.zeros_(network[-1].bias)
        # Initially preserve the proven private text/image decoder interfaces.
        nn.init.zeros_(self.text_residual.weight); nn.init.zeros_(self.text_residual.bias)
        nn.init.zeros_(self.image_residual.weight); nn.init.zeros_(self.image_residual.bias)
        self.arm = 'J1_joint'; self.configure(self.arm)

    def configure(self, arm, warm=False):
        source_mask(arm); self.arm = arm; self.eval()
        for name, p in self.named_parameters():
            eligible = not name.startswith(('text.bart.', 'reference.'))
            p.requires_grad_(eligible and not (warm and name.startswith('text.')))

    def source_encode(self, bart_hidden, mask, image_wire, clip_image, clip_text):
        if image_wire.shape[1:] != (224, 16): raise ValueError('Fixed B4 private image wire required')
        b = image_wire.shape[0]
        t = self.text.transmit(bart_hidden, mask)
        i = unit(image_wire+self.image_tx(image_wire))
        text_input = clip_text*0 if self.arm == 'J2_no_clip_text' else clip_text
        memory = torch.cat((self.image_source(image_wire), self.clip_image(clip_image[:, None]), self.clip_text(text_input[:, None])), 1)
        queries = self.common_queries.expand(b, -1, -1)
        blocked = torch.tensor(source_mask(self.arm), device=memory.device)
        c = self.common_tx(self.common_norm(queries+self.common_attention(queries, memory, blocked)))
        c = torch.cat((unit(c[:, :16]), unit(c[:, 16:])), 1)
        return torch.cat((c.reshape(b, -1), i.reshape(b, -1), t), 1)

    def receive(self, observed, available, pilot_snr, reliability, intervention='normal'):
        # Only observed wire and public/pilot observations. No source text/features/mask/length.
        if observed.ndim != 2 or observed.shape[1] != geometry()['total_reals'] or not torch.isfinite(observed).all():
            raise ValueError('Finite public C/I/T payload required')
        b = observed.shape[0]; ok = available.bool().reshape(b); factor = ok.to(observed)[:, None, None]
        cwire = observed[:, :512].reshape(b, 32, 16)
        if intervention == 'zero_common': cwire = torch.zeros_like(cwire)
        elif intervention not in ('normal', 'no_common_fusion'): raise ValueError('Unknown receive intervention')
        common = self.common_rx(cwire)*factor
        cm_image = common[:, :16] if self.arm == 'J0_separate' else common
        cm_text = common[:, 16:] if self.arm == 'J0_separate' else common
        text = self.text.receive_text(observed[:, 4096:], ok)
        hint = self.text_residual(self.text_attention(text['memory'], cm_text))
        enabled = 0. if intervention == 'no_common_fusion' else 1.
        text['memory'] = (text['memory']+enabled*torch.sigmoid(self.text_gate)*hint)*factor
        i = observed[:, 512:4096].reshape(b, 224, 16)
        corrected = (i+self.image_rx(i))*factor
        legacy = torch.cat((i.new_zeros(b, 32, 16), corrected, i.new_zeros(b, 32, 16)), 1)
        rel = reliability.to(i).reshape(b, 1, 1).expand(-1, 288, -1)
        # Call frozen image reference with gradients through the observed I route.
        latent = self.reference.receive(legacy, rel, pilot_snr, ok)['latent']
        queries = latent.flatten(2).transpose(1, 2)
        delta = self.image_residual(self.image_attention(queries, cm_image)).transpose(1, 2).reshape_as(latent)
        latent = (latent+enabled*delta)*factor.reshape(b, 1, 1, 1)
        return dict(text=text, latent=latent, common_wire=cwire,
            alignment_image=self.common_image_teacher(cm_image.mean(1)),
            alignment_text=self.common_text_teacher(cm_text.mean(1)),
            coarse_latent=self.common_coarse(cm_image.mean(1)).reshape(b, 4, 4, 4))

    def bridge_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if not n.startswith(('text.bart.', 'reference.'))}

    def restore(self, state):
        named = {n: p for n, p in self.named_parameters() if not n.startswith(('text.bart.', 'reference.'))}
        if set(named) != set(state): raise ValueError('Bridge checkpoint keys differ')
        with torch.no_grad():
            for n, p in named.items():
                if p.shape != state[n].shape or not torch.isfinite(state[n]).all(): raise ValueError('Invalid saved '+n)
                p.copy_(state[n].to(p))

    def groups(self):
        return dict(text_codec=('text.compressor.', 'text.reconstructor.', 'text.memory_projection.'),
            common_source=('common_attention.', 'common_tx.', 'common_queries'),
            common_receiver=('common_rx.',), common_tasks=('common_image_teacher.', 'common_text_teacher.', 'common_coarse.'),
            image_tx=('image_tx.',), image_rx=('image_rx.', 'image_attention.', 'image_residual.'),
            text_fusion=('text_attention.', 'text_residual.', 'text_gate'))
