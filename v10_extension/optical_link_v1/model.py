"""Inference-only process-local optical LinkFrame replacement; A stays unchanged."""
import torch
from v10.channels import ReceiverCSI
from optical_link_v1.transport import OpticalTransport
from optical_link_v1.wire import send_tokens


class OpticalFrame:
    def __init__(self, profile, snr_db, batch, device, context, **kwargs):
        if batch != 1 or kwargs.get('fixed_mode') != 0 or kwargs.get('blocks', 1) != 1:
            raise ValueError('Require batch1/fixed reconstruction/one coherence block')
        self.context = context
        self.optical, self.blocks, self.pilots = False, 1, 256
        self.tx_mode = self.rx_mode = self.requested_mode = torch.zeros(1, dtype=torch.long, device=device)
        self.header_ok = torch.ones(1, dtype=torch.bool, device=device)
        self.feedback_rejected = torch.zeros(1, dtype=torch.bool, device=device)
        self.transport = OpticalTransport(context['config'], context['field_seed'], context['noise_seed'])
        estimated = self.transport.probe()
        self.routing_snr = torch.tensor([estimated], device=device).float().clamp(-40, 40)
        noise = torch.pow(10., -self.routing_snr[:, None]/10)
        self.receiver = ReceiverCSI(torch.ones((1, 1), dtype=torch.complex64, device=device),
                                   noise, noise/256, self.routing_snr[:, None])
        self.last_stats = {}

    def payload(self, tokens, tx_amp, rx_amp, interleave=True):
        if torch.is_grad_enabled() and tokens.requires_grad:
            raise ValueError('Physical optical V1 is inference-only, no hidden STE training')
        restored, self.last_stats = send_tokens(tokens[0].detach().cpu().numpy(), self.transport)
        self.last_stats['routing_probe_snr_db'] = self.transport.events[0]['estimated_snr_db']
        self.last_stats['routing_snr_clamped_db'] = float(self.routing_snr[0])
        ok = self.last_stats['feature_object_crc_accepted']
        self.header_ok.fill_(ok)
        return torch.as_tensor(restored[None], device=tokens.device, dtype=tokens.dtype), tokens.new_full((1, 288, 1), float(ok))
