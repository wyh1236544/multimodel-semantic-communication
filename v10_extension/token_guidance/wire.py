"""Exact existing waveform/CRC/FEC transport, WITHOUT the V3 raw UTF8 branch."""
import hashlib
import math
import numpy as np
from business_link.engine import Transport, field_draw
from physical_link.digital import encode_source, decode_source, bits_bytes, bytes_bits
from token_guidance.spec import SETTINGS


def quantized(values):
    values = np.asarray(values, np.float32)
    bits, reference, clipped = encode_source(values, SETTINGS['qbits'], SETTINGS['quantizer_limit'])
    restored, ok = decode_source(bits, values.size, SETTINGS['qbits'], SETTINGS['quantizer_limit'])
    if not ok:
        raise RuntimeError('Locally encoded object failed its own CRC')
    return bits_bytes(bits), restored.reshape(values.shape), float(clipped)


def send(values, profile, snr, field_seed, noise_seed, clean=False, prepared_transport=None):
    """RX reads only received headers/CRC; TX values used for offline diagnostics."""
    values = np.asarray(values, np.float32)
    if values.shape != (288, 16) or not np.isfinite(values).all():
        raise ValueError('Need finite C32/I224/T32 x width16')
    data, reference, clipped = quantized(values)
    if clean:
        restored, ok, transport = reference, True, None
        stats = dict(estimated_snr_db=None, pre_fec_ber=None, post_fec_ber=None,
                     packet_error_rate=None, pre_errors=0, pre_bits=0, post_errors=0,
                     post_bits=0, scheduled_seconds=None, awg_samples=None,
                     optical_irradiance_gain=None, field_identity='ideal_quantized_no_channel')
    else:
        gain, irradiance = field_draw(profile, field_seed)
        transport = prepared_transport
        if transport is None:
            transport = Transport(SETTINGS, gain, 10**(-float(snr)/10), noise_seed)
            transport.probe()
        elif transport.gain != gain or transport.seed != noise_seed or len(transport.events) != 1:
            raise ValueError('Prepared probe must match payload field/noise and contain one event')
        received, packets_ok, record = transport.transfer(data, 'features', '16qam_repeat2')
        flat, ok = (decode_source(bytes_bits(received), values.size, SETTINGS['qbits'],
                                 SETTINGS['quantizer_limit']) if packets_ok
                    else (np.zeros(values.size), False))
        restored = flat.reshape(values.shape)*int(ok)
        stats = dict(transport.totals(), feature_packets=record,
                     optical_irradiance_gain=irradiance,
                     field_identity='{:.17g},{:.17g}'.format(gain.real, gain.imag))
    stats.update(feature_object_crc_accepted=bool(ok), semantic_reals=values.size,
                 semantic_tokens=288, common_tokens=32, image_tokens=224, text_tokens=32,
                 semantic_body_bytes=4608, feature_object_wire_bytes=len(data),
                 raw_utf8_wire_bytes=0, policy='16qam_repeat2', fixed_mode='reconstruction',
                 payload_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                 source_clip_fraction=clipped,
                 source_quantization_mse=float(np.mean((reference-values)**2)),
                 symbol_rate_baud=16e9, gross_modulation_bps=64e9,
                 awg_sample_rate=64e9, sps=4, rolloff=.1,
                 bandwidth_hz=17.6e9,
                 crc_note='whole_semantic_object_erased_on_any_packet_or_object_crc_failure')
    return np.asarray(restored, np.float32), stats, transport
