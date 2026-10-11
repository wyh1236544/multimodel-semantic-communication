"""Differentiable single-shot coherent channel; source text never enters Frame RX."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from v10.channels import ReceiverCSI
from v10.protocol import normalize_groups
from fso_jscc_v1.core import observe, seed_for, kernels


class ResidualMap(nn.Module):
    def __init__(self, conditional=False):
        super().__init__()
        self.conditional = conditional
        self.first = nn.Linear(17 if conditional else 16, 32)
        self.last = nn.Linear(32, 16)
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def forward(self, x, reliability=None):
        features = torch.cat((x, reliability), -1) if self.conditional else x
        return x+.1*torch.tanh(self.last(F.gelu(self.first(features))))


class ContinuousChannel(nn.Module):
    channel_type = 'fso_jscc_single_shot'

    def __init__(self, old, config):
        super().__init__()
        self.adapter = old.adapter  # preserve the exact learned A receiver adapter
        self.use_adapter = old.use_adapter
        self.tx_map, self.rx_map = ResidualMap(), ResidualMap(True)
        self.register_buffer('rrc', torch.from_numpy(kernels(config))[None, None, :])
        self.config, self.frame = dict(config), None
        self.papr_penalty = None
        self.waveform_papr_db = None

    def forward(self, latent, snr_db=None, standard_noise=None, channel_gain=None):
        if self.frame is None or standard_noise is not None or channel_gain is not None:
            raise ValueError('Only the prepared public frame owns channel noise')
        frame = self.frame
        if latent.shape != (1, 288, 16):
            raise ValueError('Fixed C32/I224/T32 width16 required')
        x = normalize_groups(latent, frame.tx_mode)
        # No CSI/SNR input to TX. Normalize BEFORE channel; never after fading.
        x = normalize_groups(self.tx_map(x), frame.tx_mode)
        axes = x.reshape(1, -1, 2).transpose(1, 2)/math.sqrt(2)
        wave = F.conv_transpose1d(axes.reshape(2, 1, -1), self.rrc,
                                 stride=self.config['sps']).reshape(1, 2, -1)*math.sqrt(self.config['sps'])
        power = wave.square().sum(1)
        ratio = power.max(1).values/power.mean(1).clamp_min(1e-10)
        self.waveform_papr_db = 10*torch.log10(ratio.clamp_min(1e-10))
        # Smooth exceedance penalty; measurement is shaped-waveform peak/average.
        target = 10**(self.config['papr_target_db']/10)
        self.papr_penalty = (F.relu(power/power.mean(1, keepdim=True).clamp_min(1e-10)-target)**2).mean()
        received, reliability = frame.payload(x)
        result = self.adapter(received, reliability) if self.use_adapter else received
        result = self.rx_map(result, reliability)
        frame.last_stats.update(waveform_papr_db=float(self.waveform_papr_db.detach()[0]),
            shaped_waveform_samples=int(wave.shape[-1]),
            tx_payload_mean_symbol_energy=float((x.square().mean()).detach()),
            papr_is_payload_only_not_full_frame=True)
        return (result*frame.header_ok[:, None, None]).to(latent.dtype)


class Frame:
    def __init__(self, profile, snr_db, batch, device, context, config, **kwargs):
        if batch != 1 or kwargs.get('fixed_mode') != 0 or kwargs.get('blocks', 1) != 1:
            raise ValueError('V1 requires batch1, fixed public mode0, one slow-fading frame')
        self.context, self.config, self.device = context, config, device
        self.optical, self.blocks, self.pilots = False, 1, config['pilots']
        self.tx_mode = self.rx_mode = self.requested_mode = torch.zeros(1, dtype=torch.long, device=device)
        self.header_ok = torch.ones(1, dtype=torch.bool, device=device)
        self.feedback_rejected = torch.zeros(1, dtype=torch.bool, device=device)
        self.transport = context.get('transport')
        self.last_stats, self.captured = {}, None
        if self.transport is not None:
            estimated = self.transport.probe()
            residual, rel = 10**(-max(-40., min(40., estimated))/10), 1.
        else:
            self.gain, self.variance, self.estimate, residual, rel, estimated, self.details = observe(
                config, context['profile'], context['snr'], context['field_seed'], context['noise_seed'])
        self.routing_snr = torch.tensor([estimated], device=device).float().clamp(-40, 40)
        noise = torch.full((1, 1), max(residual, 1e-10), device=device)
        self.receiver = ReceiverCSI(torch.ones((1, 1), dtype=torch.complex64, device=device),
                                   noise, noise/self.pilots, self.routing_snr[:, None])
        self.residual, self.reliability = residual, rel

    def payload(self, tokens):
        self.captured = tokens.detach().cpu().numpy()[0].copy()
        if self.transport is not None:
            if torch.is_grad_enabled():
                raise ValueError('Detailed waveform evaluation is inference-only, no identity STE')
            values, rel, stats = self.transport.continuous(self.captured, 1, 'lmmse')
            self.header_ok.fill_(stats['analog_receiver_available'])
            self.last_stats = dict(stats)
            restored = torch.as_tensor(values[None], device=tokens.device, dtype=tokens.dtype)
        else:
            pair = tokens.float().reshape(1, -1, 2)
            symbol = torch.complex(pair[..., 0], pair[..., 1])/math.sqrt(2)
            rng = np.random.default_rng(seed_for(self.context['noise_seed'], 'payload'))
            z = (rng.standard_normal(symbol.shape)+1j*rng.standard_normal(symbol.shape))/math.sqrt(2)
            noise = torch.as_tensor(z, dtype=torch.complex64, device=tokens.device)
            coefficient = 1. if self.context['profile'] == 'clean' else np.conj(self.estimate)/(
                abs(self.estimate)**2+self.residual*(1+1/self.pilots))
            y = (symbol*complex(self.gain)+noise*math.sqrt(self.variance))*complex(coefficient)
            restored = torch.stack((y.real, y.imag), -1).reshape_as(tokens)*math.sqrt(2)
            rel = self.reliability
            mse = float((restored.detach()-tokens.detach()).square().mean())
            self.last_stats = dict(self.details, nominal_snr_db=self.context['snr'],
                pilot_snr_db=float(self.routing_snr[0]),
                true_postfade_snr_db=None if self.variance == 0 else self.context['snr']+10*math.log10(max(abs(self.gain)**2, 1e-30)),
                noise_variance_before_fade=self.variance, feature_mse=mse,
                payload_complex_symbols=2304, pilot_complex_symbols=self.pilots,
                forward_complex_symbols=2304+self.pilots, reverse_symbols=0,
                scheduled_seconds=(2304+self.pilots)/(self.config['fs_hz']/self.config['sps']),
                analog_repeats=1, receiver_available=True, reliability=float(rel),
                pre_fec_ber=None, post_fec_ber=None, raw_utf8_wire_bytes=0,
                proxy_omits_waveform_DAC_MZM_sync=True,
                channel_unit='coherent_complex_symbol; NOT differential IMDD intensity pair')
        return restored, tokens.new_full((1, 288, 1), float(rel))


ENCODER = ('base.multimodal_encoder.', 'base.bottleneck.')
RECEIVER = ('base.joint_decoder.', 'base.image_head.', 'base.text_head.',
            'base.text_direct_adapter.', 'base.text_direct_gate_logit')


def parameters(model):
    return [(n, p) for n, p in model.named_parameters() if n.startswith(ENCODER+RECEIVER+('base.channel.',))]


def stage(model, warm):
    eligible = {n for n, _ in parameters(model)}
    for n, p in model.named_parameters():
        p.requires_grad_(n in eligible and not (warm and n.startswith('base.multimodal_encoder.')))
    # Dropout off in ALL arms; eval does not disable gradient.
    model.eval()


def gradient_groups(named):
    groups = dict(source_encoder=0., tx_mapping=0., rx_mapping=0., decoder=0.)
    for name, p in named:
        if p.grad is None:
            continue
        if name.startswith('base.multimodal_encoder.'):
            key = 'source_encoder'
        elif name.startswith('base.channel.tx_map.'):
            key = 'tx_mapping'
        elif name.startswith('base.channel.rx_map.'):
            key = 'rx_mapping'
        elif name.startswith(RECEIVER):
            key = 'decoder'
        else:
            continue
        groups[key] += float(p.grad.detach().square().sum())
    return groups
