"""Balanced modality pooling plus explicit common-only caption receiver."""
import torch
from torch import nn
import torch.nn.functional as F
from clip_bart_joint_v5.model import JointCodec as Parent, Attention, unit
from clip_bart_joint_v6.core import source_mask


class JointCodec(Parent):
    def __init__(self, bart, reference, text_config, c, clip_dim=512):
        super().__init__(bart, reference, text_config, c, clip_dim)
        h = c['hidden']
        self.pool_image_queries = nn.Parameter(torch.randn(1, 16, h)*.02)
        self.pool_text_queries = nn.Parameter(torch.randn(1, 16, h)*.02)
        self.pool_image = Attention(h, h, h, c['heads'])
        self.pool_text = Attention(h, h, h, c['heads'])
        self.pool_norm_image = nn.LayerNorm(h)
        self.pool_norm_text = nn.LayerNorm(h)
        self.caption_queries = nn.Parameter(torch.randn(1, 32, h)*.02)
        self.caption_attention = Attention(h, h, h, c['heads'])
        self.caption_projection = nn.Linear(h, bart.config.d_model)
        self.caption_norm = nn.LayerNorm(bart.config.d_model)
        self.caption_mix_gate = nn.Parameter(torch.tensor(0.))
        self.coarse_gate = nn.Parameter(torch.tensor(0.))
        self.phase = 'clean'
        self.configure(self.arm)

    def configure(self, arm, warm=False):
        super().configure(arm, warm)
        if getattr(self, 'phase', 'clean') == 'common':
            prefixes = ('common_', 'pool_', 'caption_', 'image_source.', 'clip_image.', 'clip_text.')
            for n, p in self.named_parameters():
                p.requires_grad_(n.startswith(prefixes) and n!='caption_mix_gate')

    def source_encode(self, bart_hidden, mask, image_wire, clip_image, clip_text):
        if image_wire.shape[1:] != (224, 16):
            raise ValueError('Fixed image payload')
        b = image_wire.shape[0]
        t = self.text.transmit(bart_hidden, mask)
        i = unit(image_wire+self.image_tx(image_wire))
        im = torch.cat((self.image_source(image_wire), self.clip_image(clip_image[:, None])), 1)
        tx = clip_text*0 if self.arm == 'J2_no_clip_text' else clip_text
        tm = self.clip_text(tx[:, None])
        iq = self.pool_image_queries.expand(b, -1, -1)
        tq = self.pool_text_queries.expand(b, -1, -1)
        ip = self.pool_norm_image(iq+self.pool_image(iq, im))
        tp = self.pool_norm_text(tq+self.pool_text(tq, tm))
        memory = torch.cat((ip, tp), 1)  # equal16/16 tokens; no extra source information
        queries = self.common_queries.expand(b, -1, -1)
        blocked = torch.tensor(source_mask(self.arm), device=memory.device)
        common = self.common_tx(self.common_norm(queries+self.common_attention(queries, memory, blocked)))
        common = torch.cat((unit(common[:, :16]), unit(common[:, 16:])), 1)
        return torch.cat((common.reshape(b, -1), i.reshape(b, -1), t), 1)

    def receive(self, observed, available, pilot_snr, reliability, intervention='normal'):
        out = super().receive(observed, available, pilot_snr, reliability, intervention)
        cwire = out['common_wire']
        factor = available.to(cwire).reshape(-1, 1, 1)
        common = self.common_rx(cwire)*factor
        if self.arm == 'J0_separate':
            common = common[:, 16:]
        query = self.caption_queries.expand(observed.shape[0], -1, -1)
        memory = self.caption_norm(self.caption_projection(query+self.caption_attention(query, common)))*factor
        out['common_text'] = dict(memory=memory,
            receiver_mask=torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device),
            available=available.bool())
        if intervention!='no_common_fusion':
            out['text']['memory']=(out['text']['memory']+torch.tanh(self.caption_mix_gate)*memory)*factor
            coarse=F.interpolate(out['coarse_latent'],size=out['latent'].shape[-2:],mode='bilinear',align_corners=False)
            base_coarse=F.interpolate(F.adaptive_avg_pool2d(out['latent'],(4,4)),
                size=out['latent'].shape[-2:],mode='bilinear',align_corners=False)
            out['latent']=(out['latent']+torch.tanh(self.coarse_gate)*(coarse-base_coarse))*factor.reshape(-1,1,1,1)
        return out

    def groups(self):
        g = super().groups()
        g['balanced_pool'] = ('pool_', 'image_source.', 'clip_image.', 'clip_text.')
        g['common_caption'] = ('caption_',)
        g['image_rx'] += ('coarse_gate',)
        g['text_fusion'] += ('caption_mix_gate',)
        if getattr(self, 'phase', 'clean') == 'common':
            return {n:g[n] for n in ('common_source', 'common_receiver', 'common_tasks', 'balanced_pool', 'common_caption')}
        return g


def masked_wire(wire, policy):
    out = wire.clone()
    for slot in policy['image_slots']:
        out[:, 512+16*slot:512+16*(slot+1)] = 0
    for slot in policy['text_slots']:
        out[:, 4096+64*slot:4096+64*(slot+1)] = 0
    return out
