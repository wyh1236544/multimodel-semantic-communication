"""Evaluation-only V10 adapter; model, checkpoint and old channels unchanged."""
import math
import numpy as np
import torch
from link_compare.coherent import CoherentFSOFrame
from physical_link.digital import (encode_source, decode_source, packet_size, fec_encode,
                                   fec_decode, fec_size, scramble, modulate, demodulate)
from physical_link.waveform import Pulse, preamble, transmit, receive


class WaveformFSOFrame(CoherentFSOFrame):
    def __init__(self, *args, variant='analog', rs=14e9, fs=64e9,
                 rolloff=.01, span=128, qbits=8, quantizer_limit=4.,
                 dac_bits=8, bandwidth=None, reference_rate=14e9, feedback_rate=1e6, **kwargs):
        if variant not in ('analog', '16qam_none', '16qam_conv', 'qpsk_conv'):
            raise ValueError('Unknown waveform variant')
        if kwargs.get('blocks', 1) != 1 or kwargs.get('csi', 'pilot') != 'pilot':
            raise ValueError('V2 supports one slow-fading block and pilot CSI only')
        self.variant, self.qbits, self.quantizer_limit = variant, qbits, quantizer_limit
        self.pulse = Pulse(fs, rs, rolloff, span)
        self.dac_bits, self.bandwidth = dac_bits, bandwidth
        if feedback_rate <= 0 or not math.isfinite(feedback_rate):
            raise ValueError('Invalid RF feedback symbol rate')
        self.feedback_rate = feedback_rate
        self.wave_seed = int(torch.randint(0, 2 ** 30, (1,)).item())
        self.propagation_calls = 0
        self.control_samples = 0
        self.control_symbols = 0
        self.control_energy = 0.
        self.reference_snr = float(args[1] if len(args) > 1 else kwargs['snr_db'])
        effective = self.reference_snr - 10 * math.log10(rs / reference_rate)
        args = list(args)
        if len(args) > 1:
            args[1] = effective
        else:
            kwargs['snr_db'] = effective
        super().__init__(*args, constellation='continuous', **kwargs)

    def rng(self, purpose, index=0):
        return np.random.default_rng(np.random.SeedSequence([self.wave_seed, purpose, index]))

    def propagate(self, values):
        """Probe and mode header EACH traverse RRC/DAC/FSO/matched filter."""
        training = preamble()
        outputs = []
        for b in range(self.batch):
            symbols = np.concatenate([training, values[b, 0].detach().cpu().numpy()])
            observed, diag, _ = transmit(symbols, self.pulse,
                complex(self.true_gain[b, 0].item()), float(self.true_noise_var[b, 0].item()),
                self.rng(10 + self.propagation_calls, b), self.dac_bits, bandwidth=self.bandwidth)
            matched, _ = receive(observed, self.pulse, len(symbols), training, raw=True)
            outputs.append(matched[len(training):])
        self.propagation_calls += 1
        self.control_samples += diag['awg_samples']
        self.control_symbols += len(symbols)
        self.control_energy += diag['waveform_energy_power_seconds']
        return torch.as_tensor(np.stack(outputs)[:, None], dtype=torch.complex64, device=values.device)

    def payload(self, tokens, token_amplitudes, receiver_amplitudes, interleave=True):
        batch, count, width = tokens.shape
        if batch != self.batch or width % 2:
            raise ValueError('Invalid semantic payload')
        training = preamble()
        reconstructed, reliabilities, records = [], [], []
        for b in range(batch):
            original = tokens[b].detach().float().cpu().numpy()
            if self.variant == 'analog':
                paired = original.reshape(-1, 2)
                symbols = (paired[:, 0] + 1j * paired[:, 1]) / math.sqrt(2)
                tx_amp = np.repeat(token_amplitudes[b].detach().cpu().numpy(), width // 2)
                symbols = symbols * tx_amp
                source_bits, coded_bits = 0, 0
                quant_mse, clipped = 0., 0.
            else:
                bits, quantized, clipped = encode_source(original, self.qbits, self.quantizer_limit)
                source_bits = len(bits)
                scrambled = bits ^ scramble(len(bits))
                encoded = fec_encode(scrambled) if self.variant.endswith('_conv') else scrambled
                order = 4 if self.variant == 'qpsk_conv' else 16
                symbols = modulate(encoded, order)
                coded_bits = len(encoded)
                quant_mse = float(np.mean((original.ravel() - quantized) ** 2))
            frame = np.concatenate([training, symbols])
            observed, txdiag, _ = transmit(frame, self.pulse,
                complex(self.true_gain[b, 0].item()), float(self.true_noise_var[b, 0].item()),
                self.rng(100, b), self.dac_bits, bandwidth=self.bandwidth)
            equalized, rxdiag = receive(observed, self.pulse, len(frame), training)
            payload = equalized[len(training):]
            variance = rxdiag['equalized_noise_variance']
            if self.variant == 'analog':
                rx_amp = np.repeat(receiver_amplitudes[b].detach().cpu().numpy(), width // 2)
                hpower = 1. / max(variance, 1e-10)
                reliability = rx_amp ** 2 * hpower / (rx_amp ** 2 * hpower + 1)
                if self.equalizer == 'mmse':
                    payload *= reliability
                payload /= np.maximum(rx_amp, 1e-6)
                restored = np.stack([payload.real, payload.imag], 1).ravel() * math.sqrt(2)
                reliability = reliability.reshape(count, width // 2).mean(1)[:, None]
                accepted, preber, postber = True, None, None
            else:
                llr = demodulate(payload, order, variance)[:coded_bits]
                decoded = fec_decode(llr, source_bits) if self.variant.endswith('_conv') else (llr < 0).astype(np.uint8)
                recovered = decoded ^ scramble(source_bits)
                restored, accepted = decode_source(recovered, original.size, self.qbits, self.quantizer_limit)
                reliability = np.full((count, 1), float(accepted))
                # TX references used ONLY for diagnostics after receive/decoding.
                preber = float(np.mean((llr < 0) != encoded))
                postber = float(np.mean(recovered != bits))
            header_ok = bool(self.header_ok[b].item())
            restored *= int(header_ok)
            reliability *= int(header_ok)
            reconstructed.append(restored.reshape(count, width))
            reliabilities.append(reliability)
            total_samples = self.control_samples + txdiag['awg_samples']
            sequential_seconds = total_samples / self.pulse.fs + self.control_length / self.feedback_rate
            records.append({**txdiag, **rxdiag,
                'crc_accepted': bool(accepted), 'pre_fec_ber': preber, 'post_fec_ber': postber,
                'quantization_mse': quant_mse, 'source_clip_fraction': clipped,
                'source_packet_bits': source_bits, 'coded_bits': coded_bits,
                'payload_uses': len(symbols), 'pilot_uses': self.pilots + 3 * len(training),
                'header_uses': self.control_length,
                'forward_uses': self.control_symbols + len(frame),
                'forward_awg_samples': total_samples, 'forward_seconds': total_samples / self.pulse.fs,
                'forward_energy_power_seconds': self.control_energy + txdiag['waveform_energy_power_seconds'],
                'reverse_seconds': self.control_length / self.feedback_rate,
                'sequential_link_seconds': sequential_seconds,
                'semantic_reals': original.size,
                'gross_qam_bps': (0 if self.variant == 'analog' else math.log2(order) * self.pulse.rs),
                'semantic_quantized_goodput_bps': (0 if self.variant == 'analog' else original.size * self.qbits / (total_samples / self.pulse.fs) * int(accepted and header_ok)),
                'sequential_semantic_goodput_bps': (0 if self.variant == 'analog' else original.size * self.qbits / sequential_seconds * int(accepted and header_ok)),
                'optical_irradiance_gain': float(self.irradiance[b, 0].item()),
                'reverse_rf_real_uses': self.control_length,
                'physical_unit': 'coherent_rrc_complex_symbol',
                'tx_average_power': txdiag['symbol_average_energy'],
                'tx_peak_power': float(np.max(np.abs(frame) ** 2)),
                'estimated_outage': rxdiag['estimated_snr_db'] < -5,
                'header_rejected': not header_ok,
                'header_undetected_error': header_ok and bool(self.rx_mode[b] != self.tx_mode[b]),
                'feedback_payload_error': bool(self.tx_mode[b] != self.requested_mode[b]),
                'effective_nominal_snr_db': float(-10 * math.log10(self.true_noise_var[b, 0].item())),
                'forward_sample_rate': self.pulse.fs, 'forward_symbol_rate': self.pulse.rs,
                'digital_uep': 'none_fixed_code_rate' if self.variant != 'analog' else 'public_token_power',
            })
        # Existing eval consumes batch=1 scalars. Keep batch support for tensors.
        self.last_stats = {}
        for key in records[0]:
            vals = [r[key] for r in records]
            if batch == 1:
                self.last_stats[key] = vals[0]
            elif isinstance(vals[0], (int, float, bool)):
                self.last_stats[key] = torch.tensor(vals, device=tokens.device)
            else:
                self.last_stats[key] = vals[0]
        return (torch.as_tensor(np.stack(reconstructed), device=tokens.device, dtype=tokens.dtype),
                torch.as_tensor(np.stack(reliabilities), device=tokens.device, dtype=tokens.dtype))
