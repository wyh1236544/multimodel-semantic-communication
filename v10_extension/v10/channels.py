"""RF: unit-energy complex symbols, CN(0,N0); FSO: nonnegative pairs.

FSO nominal SNR=P_avg**2/noise_variance, responsivity=P_avg=1. Thermal
noise only; NOT a calibrated outdoor link budget or photon-counting model.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import nn

PROFILES = {
    'awgn': {'family': 'awgn'},
    'rayleigh': {'family': 'rayleigh'},
    'fso_weak': {'family': 'fso', 'alpha': 11.6, 'beta': 10.1, 'xi': 3.0, 'a0': .95, 'loss_db': 0.0},
    'fso_moderate': {'family': 'fso', 'alpha': 4.0, 'beta': 2.0, 'xi': 1.8, 'a0': .9, 'loss_db': 0.0},
    'fso_strong': {'family': 'fso', 'alpha': 2.1, 'beta': 1.2, 'xi': 1.0, 'a0': .8, 'loss_db': 0.0},
}

def complex_noise(shape, device):
    return torch.complex(torch.randn(shape, device=device), torch.randn(shape, device=device)) / math.sqrt(2)

def sample_gain(profile, shape, device):
    family = profile['family']
    if family == 'awgn':
        return torch.ones(shape, device=device, dtype=torch.complex64)
    if family == 'rayleigh':
        return complex_noise(shape, device)
    if family != 'fso':
        raise ValueError('Unknown family: ' + str(family))
    alpha, beta, xi = (float(profile[k]) for k in ('alpha', 'beta', 'xi'))
    a0, loss = float(profile.get('a0', 1)), float(profile.get('loss_db', 0))
    if min(alpha, beta, xi) <= 0 or not 0 < a0 <= 1 or loss < 0:
        raise ValueError('Invalid turbulence/pointing/path-loss parameters')
    aa, bb = torch.full(shape, alpha, device=device), torch.full(shape, beta, device=device)
    turbulence = torch.distributions.Gamma(aa, aa).sample() * torch.distributions.Gamma(bb, bb).sample()
    pointing = a0 * torch.rand(shape, device=device).clamp_min(1e-12).pow(1 / xi**2)
    return turbulence * pointing * 10**(-loss / 10)  # do not normalize away losses

def mode_codebook(device, length=64):
    if length < 8 or length & (length - 1):
        raise ValueError('Control length must be a power of two >=8')
    h = torch.ones(1, 1, device=device)
    while h.size(0) < length:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h[1:4]

def detect_mode(observed, book, threshold):
    correlations = observed.float() @ book.t() / book.size(1)
    values, indices = correlations.topk(2, dim=1)
    margin = (values[:, 0] - values[:, 1]) / observed.float().square().mean(1).sqrt().clamp_min(1e-8)
    return indices[:, 0], margin >= threshold, margin

@dataclass
class ReceiverCSI:
    gain: torch.Tensor
    noise_var: torch.Tensor
    estimation_var: torch.Tensor
    effective_snr_db: torch.Tensor

class LinkFrame:
    """One quasistatic frame; truth only for propagation and diagnostics.

    Multiple independent blocks mean actual independent coherence intervals.
    Default one block. No delay/Doppler/synchronization/retransmission model.
    """
    def __init__(self, profile, snr_db, batch, device, blocks=1, pilots=32,
                 feedback_snr_db=0., control_length=64, threshold=.2,
                 csi='pilot', equalizer='mmse', fixed_mode=None,
                 generation_threshold_db=2.5, reconstruction_threshold_db=8.):
        if blocks < 1 or pilots < 4 or not math.isfinite(float(snr_db)):
            raise ValueError('Invalid block/pilot/SNR configuration')
        if csi not in ('pilot', 'perfect') or equalizer not in ('mmse', 'zf'):
            raise ValueError('Unsupported CSI/equalizer')
        if not generation_threshold_db < reconstruction_threshold_db:
            raise ValueError('Mode thresholds are not ordered')
        self.profile, self.batch, self.device = dict(profile), batch, device
        self.blocks, self.pilots = int(blocks), int(pilots)
        self.optical = profile['family'] == 'fso'
        self.true_gain = sample_gain(profile, (batch, blocks), device)
        self.true_noise_var = torch.full((batch, blocks), 10**(-float(snr_db)/10), device=device)
        self.equalizer, self.csi_kind = equalizer, csi
        self.control_length = int(control_length)
        observed = self.propagate(torch.ones(batch, blocks, pilots, device=device))
        estimate = observed.mean(-1)
        variance = (observed-estimate[..., None]).abs().square().sum(-1)/(pilots-1)
        if csi == 'perfect':  # explicitly labelled oracle ablation
            estimate = self.true_gain
            variance = self.true_noise_var/(2 if self.optical else 1)
        variance = variance.clamp_min(1e-8)
        error = variance/pilots if csi == 'pilot' else torch.zeros_like(variance)
        power = (estimate.abs().square()-error).clamp_min(1e-10)
        snr = 10*torch.log10(power/(variance+error))
        self.receiver = ReceiverCSI(estimate, variance, error, snr.clamp(-40, 40))
        self.routing_snr = self.receiver.effective_snr_db.sort(1).values[:, (blocks-1)//4]
        self.requested_mode = torch.where(self.routing_snr <= generation_threshold_db, 2,
                                         torch.where(self.routing_snr >= reconstruction_threshold_db, 0, 1))
        if fixed_mode is not None:
            if fixed_mode not in (0, 1, 2):
                raise ValueError('fixed_mode must be 0,1,2')
            self.requested_mode.fill_(fixed_mode)
        book = mode_codebook(device, control_length)
        # A separate reverse RF-AWGN control link. Fixed-length, not ARQ.
        feedback = book[self.requested_mode]+torch.randn(batch, control_length, device=device)*10**(-float(feedback_snr_db)/20)
        tx_mode, ok, self.feedback_margin = detect_mode(feedback, book, threshold)
        self.tx_mode = torch.where(ok, tx_mode, torch.full_like(tx_mode, 2))
        self.feedback_rejected = ~ok
        # The forward header announces the sender's ACTUAL mode, over this link.
        header = book[self.tx_mode, None, :].expand(-1, blocks, -1)
        noisy_header = self.propagate(header)
        weight = self.receiver.gain.conj()/(self.receiver.noise_var+self.receiver.estimation_var)
        combined = (noisy_header*weight[..., None]).sum(1).real
        self.rx_mode, self.header_ok, self.header_margin = detect_mode(combined, book, threshold)
        self.rx_mode = torch.where(self.header_ok, self.rx_mode, torch.full_like(self.rx_mode, 2))
        self.last_stats = {}

    def propagate(self, values):
        if values.shape[:2] != (self.batch, self.blocks):
            raise ValueError('Propagation shape mismatch')
        if self.optical:
            if values.is_complex() or values.detach().abs().max() > 1.00001:
                raise ValueError('Optical input must be bounded and real')
            plus, minus = 1+values, 1-values
            noise = (torch.randn_like(values)-torch.randn_like(values))*self.true_noise_var.sqrt()[..., None]
            return (self.true_gain[..., None]*(plus-minus)+noise)/2
        return self.true_gain[..., None]*values+complex_noise(values.shape, values.device)*self.true_noise_var.sqrt()[..., None]

    def payload(self, tokens, token_amplitudes, receiver_amplitudes, interleave=True):
        batch, count, width = tokens.shape
        if batch != self.batch or width % 2 or count % self.blocks:
            raise ValueError('Invalid payload batch/width/coherence blocks')
        order = torch.arange(count, device=tokens.device)
        if interleave:
            order = order.view(-1, self.blocks).t().reshape(-1)
        inverse = torch.argsort(order)
        x = tokens.float()[:, order].reshape(batch, self.blocks, -1, width)
        tx_amp = token_amplitudes[:, order, None].reshape(batch, self.blocks, -1, 1)
        rx_amp = receiver_amplitudes[:, order, None].reshape(batch, self.blocks, -1, 1)
        if self.optical:
            bounded = torch.tanh(x)
            plus, minus = tx_amp*(1+bounded), tx_amp*(1-bounded)
            n = (torch.randn_like(x)-torch.randn_like(x))*self.true_noise_var.sqrt()[..., None, None]
            received = (self.true_gain[..., None, None]*(plus-minus)+n)/2
            used = count*width*2
            power = (plus+minus).mean((1, 2, 3))/2
            peak = torch.maximum(plus, minus).flatten(1).max(1).values
        else:
            cx = torch.view_as_complex(x.reshape(batch, self.blocks, -1, width//2, 2).contiguous())/math.sqrt(2)
            transmitted = cx*tx_amp
            received = self.true_gain[..., None, None]*transmitted+complex_noise(transmitted.shape, x.device)*self.true_noise_var.sqrt()[..., None, None]
            used = count*width//2
            power = transmitted.abs().square().mean((1, 2, 3))
            peak = transmitted.abs().square().flatten(1).max(1).values
        h = self.receiver.gain[..., None, None]*rx_amp
        variance = self.receiver.noise_var[..., None, None]
        uncertainty = self.receiver.estimation_var[..., None, None]*rx_amp.square()
        if self.equalizer == 'mmse':
            coefficient = h.conj()/(h.abs().square()+variance+uncertainty).clamp_min(1e-8)
        else:
            coefficient = h.conj()/h.abs().square().clamp_min(1e-6)
        restored = coefficient*received
        reliability = h.abs().square()/(h.abs().square()+variance+uncertainty).clamp_min(1e-8)
        if not self.optical:
            restored = torch.view_as_real(restored)*math.sqrt(2)
        restored = restored.reshape(batch, count, width)[:, inverse]*self.header_ok[:, None, None]
        reliability = reliability.expand(batch, self.blocks, count//self.blocks, 1).reshape(batch, count, 1)[:, inverse]
        pilot_uses = self.blocks*self.pilots*(2 if self.optical else 1)
        header_uses = self.blocks*self.control_length*(2 if self.optical else 1)
        self.last_stats = {
            'payload_uses': used, 'pilot_uses': pilot_uses, 'header_uses': header_uses,
            'forward_uses': used+pilot_uses+header_uses,
            'reverse_rf_real_uses': self.control_length,
            'physical_unit': 'optical_intensity_slot' if self.optical else 'rf_complex_symbol',
            'tx_average_power': power.detach(), 'tx_peak_power': peak.detach(),
            'estimated_snr_db': self.routing_snr.detach(),
            'estimated_outage': (self.routing_snr < -5).detach(),
            'header_rejected': (~self.header_ok).detach(),
            'header_undetected_error': (self.header_ok & (self.rx_mode != self.tx_mode)).detach(),
            'feedback_payload_error': (self.tx_mode != self.requested_mode).detach(),
        }
        return restored, reliability

class ReliabilityAdapter(nn.Module):
    """Zero-initialized receiver-only residual; no true CSI or family input."""
    def __init__(self, width=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width+2, 64), nn.GELU(), nn.Linear(64, width))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, received, reliability):
        # Erased packets are exactly zero: sqrt'(0) would produce 0*inf NaNs.
        rms = received.square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt().clamp(max=10)
        correction = self.net(torch.cat([received, reliability, rms], -1))
        return received+reliability*.25*torch.tanh(correction)
