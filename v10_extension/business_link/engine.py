"""NumPy packet transport. TX references are diagnostics, never RX inputs.

One slow-fading optical field is shared by probe, text, features and retries.
All bursts include an RRC preamble, fixed-length wire packet and guard interval.
Optional ARQ explicitly assumes error-free, charged RF ACKs; it is NOT optical
ACK simulation. The default is one attempt, with no ARQ/ACK assumption.
"""
import hashlib
import math
import struct
import zlib
import numpy as np
from physical_link.digital import (bytes_bits, bits_bytes, fec_encode, fec_decode,
                                   scramble, modulate, demodulate)
from physical_link.waveform import Pulse, preamble, transmit, receive

POLICIES = ('analog', '16qam_none', '16qam_conv', '16qam_repeat2', 'qpsk_conv')
WIRE = struct.Struct('>4sBHHH')  # magic, kind, sequence, total packets, body length
KINDS = {'text': 1, 'features': 2, 'mode': 3}


def identity_seed(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:16], 16)


def field_draw(profile, seed):
    """Irradiance != field amplitude. No normalization of pointing/path loss."""
    rng = np.random.default_rng(seed)
    family = profile['family']
    if family == 'awgn':
        return 1.+0j, 1.
    if family == 'rayleigh':
        gain = (rng.standard_normal()+1j*rng.standard_normal())/math.sqrt(2)
        return gain, float(abs(gain)**2)
    if family != 'fso':
        raise ValueError('Unknown channel family')
    alpha, beta, xi = [float(profile[k]) for k in ('alpha', 'beta', 'xi')]
    a0, loss = float(profile.get('a0', 1)), float(profile.get('loss_db', 0))
    if min(alpha, beta, xi) <= 0 or not 0 < a0 <= 1 or loss < 0:
        raise ValueError('Invalid FSO parameters')
    irradiance = (rng.gamma(alpha, 1/alpha)*rng.gamma(beta, 1/beta)*a0
                  *max(rng.random(), 1e-12)**(1/xi**2)*10**(-loss/10))
    return math.sqrt(irradiance)*np.exp(1j*rng.uniform(-math.pi, math.pi)), float(irradiance)


def wire_packet(body, kind, sequence, total, capacity):
    if kind not in KINDS or not 0 <= sequence < total <= 65535 or len(body) > capacity:
        raise ValueError('Invalid wire packet')
    padded = WIRE.pack(b'BL03', KINDS[kind], sequence, total, len(body)) + body
    padded += bytes(capacity-len(body))
    return padded + struct.pack('>I', zlib.crc32(padded) & 0xffffffff)


def read_wire(data, kind, capacity):
    """CRC and received header only. No original bytes/length/sequence here."""
    if len(data) != WIRE.size+capacity+4:
        return None
    body, crc = data[:-4], data[-4:]
    if zlib.crc32(body) & 0xffffffff != struct.unpack('>I', crc)[0]:
        return None
    magic, type_id, seq, total, length = WIRE.unpack(body[:WIRE.size])
    if (magic != b'BL03' or type_id != KINDS[kind] or not 0 <= seq < total
            or length > capacity or any(body[WIRE.size+length:])):
        return None
    return seq, total, body[WIRE.size:WIRE.size+length]


def order_for(policy):
    return 4 if policy == 'qpsk_conv' else 16


class Transport:
    def __init__(self, settings, gain, noise_var, seed):
        self.settings = dict(settings)
        self.fs = float(settings.get('fs', 64e9))
        self.sps = int(settings.get('sps', 4))
        if self.sps < 2 or float(settings.get('sps', 4)) != self.sps:
            raise ValueError('sps must be an integer >=2')
        self.pulse = Pulse(self.fs, self.fs/self.sps,
                           float(settings.get('rolloff', .1)), int(settings.get('span', 32)))
        self.gain, self.noise_var, self.seed = complex(gain), float(noise_var), int(seed)
        self.capacity = int(settings.get('packet_bytes', 512))
        self.guard = int(settings.get('guard_samples', 256))
        self.attempts = int(settings.get('max_attempts', 1))
        self.feedback_rate = float(settings.get('feedback_rate', 1e6))
        self.distance = float(settings.get('distance_m', 1000))
        if (not 1 <= self.capacity <= 4096 or self.guard < 0 or not 1 <= self.attempts <= 4
                or not math.isfinite(self.noise_var) or self.noise_var < 0
                or not np.isfinite(self.gain) or not math.isfinite(self.distance)
                or self.distance < 0 or not math.isfinite(self.feedback_rate) or self.feedback_rate <= 0):
            raise ValueError('Invalid packet/noise/link budget')
        if self.attempts > 1 and not settings.get('ideal_ack_arq', False):
            raise ValueError('ARQ requires explicit ideal_ack_arq assumption')
        self.events, self.constellations = [], []
        self.used_policy_by_kind = {}

    def burst(self, symbols, purpose, sequence=0, attempt=0, policy='probe'):
        training = preamble(256)
        frame = np.concatenate([training, symbols])
        rng = np.random.default_rng(identity_seed(self.seed, purpose, sequence, attempt))
        observed, tx, _ = transmit(frame, self.pulse, self.gain, self.noise_var, rng,
                                  int(self.settings.get('dac_bits', 8)),
                                  bandwidth=self.settings.get('bandwidth'))
        eq, rx = receive(observed, self.pulse, len(frame), training)
        if not np.isfinite(eq).all():
            raise FloatingPointError('Nonfinite matched/equalized waveform')
        data = eq[len(training):]
        evm2 = float(np.mean(np.abs(data-symbols)**2)/max(np.mean(np.abs(symbols)**2), 1e-12)) if len(symbols) else None
        event = dict(tx, **rx, purpose=purpose, sequence=sequence, attempt=attempt,
                     policy=policy, guard_samples=self.guard, pilot_symbols=len(training),
                     data_symbols=len(symbols), evm_squared=evm2,
                     scheduled_seconds=(tx['awg_samples']+self.guard)/self.fs,
                     pre_errors=0, pre_bits=0, post_errors=0, post_bits=0,
                     useful_bytes_accepted=0, crc_accepted=None, ack_seconds=0., arq_wait_seconds=0.)
        self.events.append(event)
        if len(symbols) and not any(c['purpose'] == purpose for c in self.constellations):
            n = min(len(symbols), 2048)
            self.constellations.append({'purpose': purpose, 'policy': policy,
                'tx': [[float(z.real), float(z.imag)] for z in symbols[:n]],
                'rx': [[float(z.real), float(z.imag)] for z in data[:n]]})
        return data, event

    def probe(self):
        _, diag = self.burst(np.empty(0, complex), 'probe')
        return diag['estimated_snr_db']

    def transfer(self, data, kind, policy):
        if policy == 'analog' or policy not in POLICIES or kind not in KINDS:
            raise ValueError('Digital transfer requires a digital policy and known kind')
        data = bytes(data)
        self.used_policy_by_kind[kind] = policy
        total = max(1, math.ceil(len(data)/self.capacity))
        received, received_total = {}, None
        first_event = len(self.events)
        for seq in range(total):
            body = data[seq*self.capacity:(seq+1)*self.capacity]
            bits = bytes_bits(wire_packet(body, kind, seq, total, self.capacity))
            whitened = bits ^ scramble(len(bits))
            coded = whitened if policy.endswith('_none') else fec_encode(whitened)
            repeated = policy == '16qam_repeat2'
            on_wire = np.repeat(coded, 2) if repeated else coded
            symbols = modulate(on_wire, order_for(policy))
            for attempt in range(self.attempts):
                eq, event = self.burst(symbols, kind, seq, attempt, policy)
                llr = demodulate(eq, order_for(policy), event['equalized_noise_variance'])[:len(on_wire)]
                hard = (llr < 0).astype(np.uint8)
                soft = llr.reshape(-1, 2).sum(1) if repeated else llr
                decoded = ((soft < 0).astype(np.uint8) if policy.endswith('_none')
                           else fec_decode(soft, len(bits))) ^ scramble(len(bits))
                parsed = read_wire(bits_bytes(decoded), kind, self.capacity)
                # Sender references used AFTER receiver decoding, for BER only.
                event.update(pre_errors=int(np.count_nonzero(hard != on_wire)), pre_bits=len(on_wire),
                             post_errors=int(np.count_nonzero(decoded != bits)), post_bits=len(bits),
                             crc_accepted=parsed is not None,
                             coded_padding_bits=(-len(on_wire)) % int(math.log2(order_for(policy))))
                if self.attempts > 1:
                    event['ack_seconds'] = 64/self.feedback_rate
                    event['arq_wait_seconds'] = 2*self.distance/299792458.
                if parsed is not None:
                    rx_seq, rx_total, rx_body = parsed
                    if received_total is None:
                        received_total = rx_total
                    if rx_total == received_total and rx_seq not in received:
                        received[rx_seq] = rx_body
                        event['useful_bytes_accepted'] = len(rx_body)
                    break  # ideal ACK reflects CRC status only, never exact TX equality
        ok = (received_total is not None and set(received) == set(range(received_total)))
        reconstructed = b''.join(received[i] for i in range(received_total)) if ok else b''
        events = self.events[first_event:]
        return reconstructed, bool(ok), {'sent_bytes': len(data), 'received_bytes': len(reconstructed),
            'complete': bool(ok), 'packets_planned': total, 'packet_attempts': len(events),
            'crc_failures': sum(not e['crc_accepted'] for e in events),
            'undetected_object_error': bool(ok and reconstructed != data)}

    def analog(self, values, tx_amp, rx_amp, equalizer='mmse'):
        original = np.asarray(values, float)
        pairs = original.reshape(-1, 2)
        symbols = (pairs[:, 0]+1j*pairs[:, 1])/math.sqrt(2)*tx_amp
        payload, event = self.burst(symbols, 'features', policy='analog')
        variance = event['equalized_noise_variance']
        reliability = rx_amp**2/(rx_amp**2+variance)
        if equalizer == 'mmse':
            payload = payload*reliability
        payload /= np.maximum(rx_amp, 1e-6)
        restored = np.stack([payload.real, payload.imag], 1).ravel()*math.sqrt(2)
        return restored, reliability, event

    def totals(self):
        e = self.events
        summed = {k: sum(x[k] for x in e) for k in (
            'awg_samples', 'guard_samples', 'pilot_symbols', 'data_symbols',
            'pre_errors', 'pre_bits', 'post_errors', 'post_bits',
            'scheduled_seconds', 'waveform_seconds', 'ack_seconds', 'arq_wait_seconds',
            'waveform_energy_power_seconds', 'useful_bytes_accepted')}
        summed['link_seconds_without_mode_feedback'] = (summed['scheduled_seconds']+summed['ack_seconds']
            +summed['arq_wait_seconds']+self.distance/299792458.)
        summed['pre_fec_ber'] = summed['pre_errors']/summed['pre_bits'] if summed['pre_bits'] else None
        summed['post_fec_ber'] = summed['post_errors']/summed['post_bits'] if summed['post_bits'] else None
        packets = [v for v in e if v['crc_accepted'] is not None]
        summed['packet_error_rate'] = sum(not v['crc_accepted'] for v in packets)/len(packets) if packets else None
        summed['packet_attempts'] = len(packets)
        summed['crc_failures'] = sum(not v['crc_accepted'] for v in packets)
        summed['estimated_snr_db'] = float(np.mean([v['estimated_snr_db'] for v in e]))
        num = sum(v['evm_squared']*v['data_symbols'] for v in e if v['evm_squared'] is not None)
        den = sum(v['data_symbols'] for v in e if v['evm_squared'] is not None)
        summed['evm_rms'] = math.sqrt(num/den) if den else None
        summed['dac_clip_fraction_max'] = max(v['dac_clip_fraction'] for v in e)
        return summed
