"""Source-independent token positions and mode-conditioned protection.

All receiver metadata is reconstructed from decoded mode + public dimensions.
No source caption length, top-k scores, or sender index list crosses the link.
"""
from __future__ import annotations
import torch
from torch import nn
from .channels import ReliabilityAdapter

TEXT_QUOTAS = (32, 128, 192)
MODE_NAMES = ('reconstruction', 'hybrid', 'generation')

def schedule(mode, device, common=32, image=256, text=77):
    t = TEXT_QUOTAS[int(mode)]
    i = 256-t
    if image != 256 or text != 77 or common != 32:
        raise ValueError('V10 protocol requires C32/I-pool256/T-pool77')
    # Spatially distributed deterministic positions, nested in a public order.
    # Odd multiplication permutes 0..255. Receiver needs no image-dependent index.
    image_idx = (torch.arange(256, device=device)*137) % 256
    text_idx = torch.arange(t, device=device) % text
    return image_idx[:i], text_idx

def metadata_from_modes(modes, common=32, image=256, text=77):
    device = modes.device
    ii, tt = [], []
    for mode in modes.detach().cpu().tolist():
        i, t = schedule(mode, device, common, image, text)
        ii.append(i)
        tt.append(t)
    tc = torch.tensor([len(t) for t in tt], device=device)
    return {'image_indices': ii, 'text_indices': tt, 'image_counts': 256-tc,
            'text_counts': tc, 'num_common': torch.tensor(common, device=device),
            'num_image': torch.tensor(image, device=device), 'num_text': torch.tensor(text, device=device),
            'predicted_text_quota': tc.float(), 'target_text_quota': tc.float(),
            'text_redundant_slots': (tc-text).clamp_min(0),
            'text_repetition_factor': tc.float()/tc.clamp(max=text)}

def unpack(packed, metadata):
    batch, _, width = packed.shape
    nc, ni, nt = [int(metadata[k]) for k in ('num_common', 'num_image', 'num_text')]
    image, text = packed.new_zeros(batch, ni, width), packed.new_zeros(batch, nt, width)
    for b, (ii, tt) in enumerate(zip(metadata['image_indices'], metadata['text_indices'])):
        image[b, ii] = packed[b, nc:nc+len(ii)]
        text[b].index_add_(0, tt, packed[b, nc+len(ii):nc+len(ii)+len(tt)])
        counts = packed.new_zeros(nt)
        counts.index_add_(0, tt, packed.new_ones(len(tt)))
        text[b] = text[b]/counts.clamp_min(1)[:, None]
    return packed[:, :nc], image, text

class DeterministicPacker(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame = None

    def pack(self, z_common, z_image, z_text, *unused, **ignored):
        if self.frame is None:
            raise RuntimeError('No physical frame prepared')
        rows = []
        for b, mode in enumerate(self.frame.tx_mode.detach().cpu().tolist()):
            ii, tt = schedule(mode, z_image.device)
            rows.append(torch.cat([z_common[b], z_image[b, ii], z_text[b, tt]], 0))
        return torch.stack(rows), metadata_from_modes(self.frame.rx_mode)

    unpack = staticmethod(unpack)

def amplitudes(modes, count, optical, uep):
    result = torch.ones(len(modes), count, device=modes.device)
    if count == 32 or not uep:
        return result
    if count != 288:
        raise ValueError('Only 288-token full or 32-token Common payload supported')
    # Frozen public table, not a claimed optimal allocation. Retrain equal-power
    # controls to measure benefit; finite nonzero private protection is preserved.
    weights = ((1.5, 1., 1.), (3., 1., 1.5), (6., 1., 2.5))
    for b, m in enumerate(modes.detach().cpu().tolist()):
        t = TEXT_QUOTAS[m]
        c, i, txt = weights[m]
        result[b, :32], result[b, 32:288-t], result[b, 288-t:] = c, i, txt
    result = result/result.mean(1, keepdim=True)
    return result if optical else result.sqrt()

def normalize_groups(tokens, modes):
    rows = []
    for b, m in enumerate(modes.detach().cpu().tolist()):
        ranges = ((0, 32),) if tokens.size(1) == 32 else ((0, 32), (32, 288-TEXT_QUOTAS[m]), (288-TEXT_QUOTAS[m], 288))
        groups = []
        for start, end in ranges:
            v = tokens[b, start:end].float()
            groups.append(v/v.square().mean().clamp_min(1e-8).sqrt())
        rows.append(torch.cat(groups, 0))
    return torch.stack(rows)

class PilotSemanticChannel(nn.Module):
    channel_type = 'v10_physical'
    def __init__(self, width=16, uep=True, adapter=True, interleave=True):
        super().__init__()
        self.adapter = ReliabilityAdapter(width)
        self.use_uep, self.use_adapter, self.interleave = uep, adapter, interleave
        self.frame = None

    def forward(self, latent, snr_db=None, standard_noise=None, channel_gain=None):
        if self.frame is None or standard_noise is not None or channel_gain is not None:
            raise ValueError('V10 channel owns physical noise/CSI; prepare a frame first')
        frame = self.frame
        x = normalize_groups(latent, frame.tx_mode)
        tx_a = amplitudes(frame.tx_mode, x.size(1), frame.optical, self.use_uep)
        rx_a = amplitudes(frame.rx_mode, x.size(1), frame.optical, self.use_uep)
        received, reliability = frame.payload(x, tx_a, rx_a, self.interleave)
        result = self.adapter(received, reliability) if self.use_adapter else received
        # A learned bias must not turn an erased packet into received evidence.
        return (result*frame.header_ok[:, None, None]).to(latent.dtype)
