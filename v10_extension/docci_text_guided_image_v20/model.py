"""Warm-started spatial codec. TX and RX have separate public interfaces."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from docci_reconstruction_v10.model import ImageBranch
from docci_longclip_joint_v8.model import unit

class Fusion(nn.Module):
    def __init__(self, channels, side_width):
        super().__init__()
        self.side_width = side_width
        self.side = nn.Sequential(nn.LayerNorm(side_width), nn.Linear(side_width,128))
        self.query = nn.Parameter(torch.randn(1,32,128)*.02)
        self.resample = nn.MultiheadAttention(128,4,batch_first=True,dropout=0.)
        self.image_norm = nn.LayerNorm(channels)
        self.image_q = nn.Linear(channels,128)
        self.attention = nn.MultiheadAttention(128,4,batch_first=True,dropout=0.)
        self.output = nn.Linear(128,channels)
        self.alpha = nn.Parameter(torch.tensor(.01))

    def forward(self, image, side):
        b,d,h,w = image.shape
        if side is None:
            # Same module/parameter count/attention operations, but only own-image
            # information. Fixed tiling adds no parameters or external semantics.
            tokens = F.adaptive_avg_pool2d(image,(4,8)).flatten(2).transpose(1,2)
            side = tokens.repeat(1,1,math.ceil(self.side_width/d))[...,:self.side_width]
        if side.ndim != 3 or side.shape[0] != b or side.shape[-1] != self.side_width:
            raise ValueError('Wrong fixed side-feature interface')
        source = self.side(side)
        query = self.query.expand(b,-1,-1)
        slots = query+self.resample(query,source,source,need_weights=False)[0]
        q = self.image_q(self.image_norm(image.flatten(2).transpose(1,2)))
        delta = self.output(self.attention(q,slots,slots,need_weights=False)[0])
        return image+torch.tanh(self.alpha)*delta.transpose(1,2).reshape(b,d,h,w)

class GuidedImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = ImageBranch(dict(image_size=512,image_width=8),'I2_shifted')
        self.tx32 = Fusion(96,512)
        self.tx16 = Fusion(128,512)
        self.rx32 = Fusion(96,768)
        self.rx64 = Fusion(64,768)

    def encode(self, source_latent, source_semantics=None):
        e = self.base.image_encoder
        middle = e.b(e.down(e.a(e.input(source_latent))))
        middle = self.tx32(middle,source_semantics)
        coarse = e.coarse[1](e.coarse[0](middle))
        coarse = self.tx16(coarse,source_semantics)
        return unit(e.wire(e.merge(middle+e.coarse[2](coarse)))).flatten(1)

    def receive(self, observed, received_memory, pilot_snr, reliability):
        # No source image/description, mask, original length, or rectangle here.
        if observed.ndim != 2 or observed.shape[1] != 8192 or not torch.isfinite(observed).all():
            raise ValueError('Expected finite public image payload')
        d = self.base.image_decoder
        x = observed.reshape(-1,8,32,32)
        cond = torch.stack((pilot_snr.clamp(-40,40)/40,reliability.clamp(0,1)),1)
        middle = d.a(d.input(x)+d.condition(cond)[:,:,None,None])
        middle = self.rx32(middle,received_memory)
        fine = d.c(d.up(d.b(middle+d.coarse(middle))))
        return d.output(self.rx64(fine,received_memory))

    def groups(self):
        return dict(image_encoder=('base.image_encoder.',),
                    image_decoder=('base.image_decoder.',),
                    tx_fusion=('tx32.','tx16.'), rx_fusion=('rx32.','rx64.'))

    def saved(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items()}

    def restore(self, state):
        own = self.state_dict()
        if set(own) != set(state) or any(own[k].shape != state[k].shape or
                not torch.isfinite(state[k]).all() for k in own):
            raise ValueError('Incomplete/malformed V20 checkpoint')
        self.load_state_dict(state,strict=True)

def select_sides(arm, source_semantics, received_memory):
    if arm not in ('A_self','B_tx_text','C_tx_rx_text'):
        raise ValueError('Unknown arm')
    return (None if arm=='A_self' else source_semantics,
            received_memory if arm=='C_tx_rx_text' else None)
