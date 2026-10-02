"""Frozen A process-local frame hooks. No source text in the receiving frame."""
import numpy as np
import torch
from v10.channels import ReceiverCSI


class ReplayFrame:
    def __init__(self, profile, snr_db, batch, device, context, **kwargs):
        if batch != 1 or kwargs.get('fixed_mode') != 0 or kwargs.get('blocks', 1) != 1:
            raise ValueError('Require batch1, fixed reconstruction, one coherence block')
        self.context = context
        self.optical, self.blocks, self.pilots = False, 1, 256
        self.tx_mode = self.rx_mode = self.requested_mode = torch.zeros(1, dtype=torch.long, device=device)
        self.header_ok = torch.tensor([bool(context.get('available', True))], device=device)
        self.feedback_rejected = torch.zeros(1, dtype=torch.bool, device=device)
        estimate = float(context.get('probe_snr_db', 40.))
        if not np.isfinite(estimate):
            raise ValueError('Finite pilot SNR required, never offline physical SNR')
        self.routing_snr = torch.tensor([estimate], device=device).float().clamp(-40, 40)
        noise = torch.pow(10., -self.routing_snr[:, None]/10)
        self.receiver = ReceiverCSI(torch.ones((1, 1), dtype=torch.complex64, device=device),
                                   noise, noise/256, self.routing_snr[:, None])
        self.last_stats = dict(context.get('stats', {}))
        self.transport = None
        self.captured = None

    def payload(self, tokens, tx_amp, rx_amp, interleave=True):
        if torch.is_grad_enabled():
            raise ValueError('This final comparison is inference-only')
        if tokens.shape != (1, 288, 16):
            raise ValueError('Unexpected normalized semantic payload shape')
        self.captured = tokens[0].detach().cpu().numpy().copy()
        if self.context.get('capture', False):
            restored = self.captured
        else:
            # Fresh source encoding cannot leak through: receiver uses these values ONLY.
            restored = np.asarray(self.context['received'], np.float32)
        if restored.shape != (288, 16) or not np.isfinite(restored).all():
            raise ValueError('Invalid received feature array')
        received = torch.as_tensor(restored[None], device=tokens.device, dtype=tokens.dtype)
        reliability = float(self.context.get('reliability', 1.))
        if not 0 <= reliability <= 1:
            raise ValueError('Invalid public receiver reliability')
        return received, tokens.new_full((1, 288, 1), reliability)
