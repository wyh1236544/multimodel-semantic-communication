"""A-compatible C32/I224/T32 payload; CRC erasure retained, no UTF8 side channel."""
import hashlib
import numpy as np
from physical_link.digital import decode_source, bytes_bits
from token_guidance.wire import quantized


def send_tokens(values, transport):
    values = np.asarray(values, np.float32)
    if values.shape != (288, 16) or not np.isfinite(values).all():
        raise ValueError('Expected finite C32/I224/T32 x width16')
    data, reference, clipped = quantized(values)
    policy = transport.optical_config['waveform']['policy']
    received, packets_ok, packets = transport.transfer(data, 'features', policy)
    restored, object_ok = (decode_source(bytes_bits(received), values.size, 8, 4.) if packets_ok
                           else (np.zeros(values.size), False))
    stats = dict(transport.totals(), feature_packets=packets,
        feature_object_crc_accepted=bool(object_ok), semantic_reals=values.size,
        semantic_tokens=288, common_tokens=32, image_tokens=224, text_tokens=32,
        semantic_body_bytes=4608, feature_object_wire_bytes=len(data), raw_utf8_wire_bytes=0,
        policy=policy, fixed_mode='reconstruction',
        payload_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
        source_clip_fraction=clipped, source_quantization_mse=float(np.mean((reference-values)**2)),
        crc_note='whole_semantic_object_erased_on_any_packet_or_object_crc_failure')
    seconds = stats['scheduled_seconds']
    stats.update(semantic_body_offered_bps=4608*8/seconds,
                 semantic_body_goodput_bps=4608*8*int(object_ok)/seconds,
                 symbol_rate_baud=transport.pulse.rs,
                 gross_modulation_bps=transport.pulse.rs*(2 if policy == 'qpsk_conv' else 4),
                 awg_sample_rate=transport.fs, sps=transport.sps,
                 rolloff=transport.pulse.alpha, occupied_complex_bandwidth_hz=(1+transport.pulse.alpha)*transport.pulse.rs)
    return np.asarray(restored, np.float32).reshape(values.shape)*int(object_ok), stats
