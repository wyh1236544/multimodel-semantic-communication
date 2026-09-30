"""Minimal pilot-aided coherent-FSO field channel and constellation diagnostics.

The FSO Gamma-Gamma/pointing variable is *irradiance*, so field amplitude is
its square root. A random but constant carrier phase is estimated from pilots.
No waveform, polarization, shot noise, phase drift, LO impairment or hardware
calibration is implied. QPSK/16-QAM are hard constellation constraints on V10
latents, NOT a JPEG/bitstream/FEC separate-source-channel baseline.
"""
from __future__ import annotations

import math

import torch

from v10.channels import LinkFrame, ReceiverCSI, detect_mode, mode_codebook, sample_gain


def quantize_complex(values, constellation):
    if constellation == 'continuous':
        return values
    if constellation == 'qpsk':
        level = 1.0 / math.sqrt(2.0)
        re = torch.where(values.real >= 0, level, -level)
        im = torch.where(values.imag >= 0, level, -level)
    elif constellation == '16qam':
        scale = 1.0 / math.sqrt(10.0)
        boundary = 2.0 * scale
        re = torch.where(values.real.abs() < boundary, scale, 3.0 * scale) * torch.where(values.real >= 0, 1.0, -1.0)
        im = torch.where(values.imag.abs() < boundary, scale, 3.0 * scale) * torch.where(values.imag >= 0, 1.0, -1.0)
    else:
        raise ValueError('Unsupported constellation: ' + str(constellation))
    return torch.complex(re, im)


class CoherentFSOFrame(LinkFrame):
    """Use existing public packet protocol but coherent optical field physics."""

    def __init__(self, profile, snr_db, batch, device, blocks=1, pilots=32,
                 feedback_snr_db=0., control_length=64, threshold=.2,
                 csi='pilot', equalizer='mmse', fixed_mode=None,
                 generation_threshold_db=2.5, reconstruction_threshold_db=8.,
                 constellation='continuous'):
        if profile['family'] != 'fso':
            raise ValueError('Coherent FSO requires an FSO irradiance profile')
        if constellation not in ('continuous', 'qpsk', '16qam'):
            raise ValueError('Unsupported constellation')
        if blocks < 1 or pilots < 4 or not math.isfinite(float(snr_db)):
            raise ValueError('Invalid block/pilot/SNR configuration')
        if csi not in ('pilot', 'perfect') or equalizer not in ('mmse', 'zf'):
            raise ValueError('Unsupported CSI/equalizer')
        if generation_threshold_db >= reconstruction_threshold_db:
            raise ValueError('Mode thresholds are not ordered')
        self.profile, self.batch, self.device = dict(profile), batch, device
        self.blocks, self.pilots = int(blocks), int(pilots)
        self.optical = False  # Reuse the complex-field payload/equalizer path.
        self.constellation = constellation
        self.irradiance = sample_gain(profile, (batch, blocks), device)
        self.carrier_phase = (torch.rand(batch, blocks, device=device) * 2 - 1) * math.pi
        self.true_gain = torch.polar(self.irradiance.clamp_min(0).sqrt(), self.carrier_phase)
        self.true_noise_var = torch.full((batch, blocks), 10 ** (-float(snr_db) / 10), device=device)
        self.equalizer, self.csi_kind = equalizer, csi
        self.control_length = int(control_length)
        observed = self.propagate(torch.ones(batch, blocks, pilots, device=device))
        estimate = observed.mean(-1)
        variance = (observed - estimate[..., None]).abs().square().sum(-1) / (pilots - 1)
        if csi == 'perfect':
            estimate = self.true_gain
            variance = self.true_noise_var
        variance = variance.clamp_min(1e-8)
        error = variance / pilots if csi == 'pilot' else torch.zeros_like(variance)
        power = (estimate.abs().square() - error).clamp_min(1e-10)
        snr = 10 * torch.log10(power / (variance + error))
        self.receiver = ReceiverCSI(estimate, variance, error, snr.clamp(-40, 40))
        self.routing_snr = self.receiver.effective_snr_db.sort(1).values[:, (blocks - 1) // 4]
        self.requested_mode = torch.where(self.routing_snr <= generation_threshold_db, 2,
                                          torch.where(self.routing_snr >= reconstruction_threshold_db, 0, 1))
        if fixed_mode is not None:
            if fixed_mode not in (0, 1, 2):
                raise ValueError('fixed_mode must be 0, 1 or 2')
            self.requested_mode.fill_(fixed_mode)
        book = mode_codebook(device, control_length)
        feedback = book[self.requested_mode] + torch.randn(batch, control_length, device=device) * 10 ** (-float(feedback_snr_db) / 20)
        tx_mode, ok, self.feedback_margin = detect_mode(feedback, book, threshold)
        self.tx_mode = torch.where(ok, tx_mode, torch.full_like(tx_mode, 2))
        self.feedback_rejected = ~ok
        header = book[self.tx_mode, None, :].expand(-1, blocks, -1)
        noisy_header = self.propagate(header)
        weight = self.receiver.gain.conj() / (self.receiver.noise_var + self.receiver.estimation_var)
        combined = (noisy_header * weight[..., None]).sum(1).real
        self.rx_mode, self.header_ok, self.header_margin = detect_mode(combined, book, threshold)
        self.rx_mode = torch.where(self.header_ok, self.rx_mode, torch.full_like(self.rx_mode, 2))
        self.last_stats = {}

    def payload(self, tokens, token_amplitudes, receiver_amplitudes, interleave=True):
        batch, count, width = tokens.shape
        if width % 2:
            raise ValueError('Coherent field requires paired real features')
        transmitted = torch.view_as_complex(tokens.float().reshape(batch, count, width // 2, 2).contiguous()) / math.sqrt(2)
        constrained = quantize_complex(transmitted, self.constellation)
        mapped = torch.view_as_real(constrained).reshape(batch, count, width) * math.sqrt(2)
        received, reliability = super().payload(mapped, token_amplitudes, receiver_amplitudes, interleave)
        self.last_stats['physical_unit'] = 'coherent_optical_complex_symbol'
        self.last_stats['optical_irradiance_gain'] = self.irradiance.mean(1).detach()
        self.last_stats['quantization_mse'] = (tokens.float() - mapped).square().mean((1, 2)).detach()
        if self.constellation != 'continuous':
            equalized = torch.view_as_complex(received.reshape(batch, count, width // 2, 2).contiguous()) / math.sqrt(2)
            # The existing V10 MMSE equalizer attenuates the symbol by its
            # reliability. Undo that *known receiver-side* shrinkage before a
            # hard finite-alphabet decision; no true channel gain is used.
            unbiased = equalized / reliability.clamp_min(1e-6)
            decisions = quantize_complex(unbiased, self.constellation)
            decisions = torch.where(self.header_ok[:, None, None], decisions, torch.zeros_like(decisions))
            self.last_stats['constellation_symbol_error_rate'] = (decisions != constrained).float().mean((1, 2)).detach()
            received = torch.view_as_real(decisions).reshape(batch, count, width) * math.sqrt(2)
        return received.to(tokens.dtype), reliability
