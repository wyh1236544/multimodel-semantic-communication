"""Evaluation-only adapter: preserve the V10 checkpoint and packing layout."""
import math
import numpy as np
import torch
from business_link.engine import Transport, identity_seed, field_draw, order_for
from physical_link.digital import encode_source, decode_source, bits_bytes, bytes_bits
from v10.channels import ReceiverCSI, mode_codebook, detect_mode


class BusinessFrame:
    def __init__(self, profile, snr_db, batch, device, blocks=1, pilots=32,
                 feedback_snr_db=0., control_length=64, threshold=.2,
                 csi='pilot', equalizer='mmse', fixed_mode=0,
                 generation_threshold_db=2.5, reconstruction_threshold_db=8.,
                 settings=None, context=None):
        if batch != 1 or blocks != 1 or csi != 'pilot':
            raise ValueError('V3 requires batch=1, one quasistatic field and pilot CSI')
        if equalizer not in ('mmse', 'zf') or not math.isfinite(float(snr_db)):
            raise ValueError('Invalid SNR/equalizer')
        if not generation_threshold_db < reconstruction_threshold_db:
            raise ValueError('Invalid mode thresholds')
        self.settings, self.context = dict(settings), dict(context)
        self.policy = self.context['policy']
        self.device, self.batch, self.blocks = device, batch, blocks
        self.optical, self.equalizer = False, equalizer  # complex coherent field, not IM/DD
        self.pilots, self.control_length = 256, control_length
        self.received_text, self.text_ok, self.fusion_used = '', False, False
        gain, irradiance = field_draw(profile, self.context['field_seed'])
        self.field_identity = '{:.17g},{:.17g}'.format(gain.real, gain.imag)
        self.irradiance = irradiance
        # Fixed average symbol power; fixed receiver noise PSD referenced to 16 GBd.
        rs = self.settings.get('fs', 64e9)/self.settings.get('sps', 4)
        noise_var = 10**(-float(snr_db)/10)*(rs/16e9)
        self.transport = Transport(self.settings, gain, noise_var, self.context['noise_seed'])
        estimated = self.transport.probe()
        self.routing_snr = torch.tensor([estimated], dtype=torch.float32, device=device).clamp(-40, 40)
        # Normalized receiver CSI proxy from measured pilot SNR, never true gain/noise.
        measured_noise = torch.pow(10., -self.routing_snr[:, None]/10)
        self.receiver = ReceiverCSI(torch.ones((1, 1), dtype=torch.complex64, device=device),
                                    measured_noise, measured_noise/256, self.routing_snr[:, None])
        requested = 2 if estimated <= generation_threshold_db else (0 if estimated >= reconstruction_threshold_db else 1)
        self.mode_feedback_seconds = 0.
        self.feedback_rejected = torch.zeros(1, dtype=torch.bool, device=device)
        if fixed_mode is not None:
            if fixed_mode not in (0, 1, 2):
                raise ValueError('Invalid fixed mode')
            requested, tx_mode, rx_mode, ok = fixed_mode, fixed_mode, fixed_mode, True
            self.mode_protocol = 'public_fixed_mode_no_feedback_or_mode_packet'
        else:
            # Receiver asks for a mode over the old noisy RF Hadamard channel.
            book = mode_codebook(device, control_length)
            rng = np.random.default_rng(identity_seed(self.context['noise_seed'], 'mode-feedback'))
            noise = torch.as_tensor(rng.standard_normal((1, control_length)), dtype=torch.float32, device=device)
            observed = book[requested][None]+noise*10**(-feedback_snr_db/20)
            detected, accepted, _ = detect_mode(observed, book, threshold)
            tx_mode = int(detected[0]) if bool(accepted[0]) else 2
            self.feedback_rejected = ~accepted
            self.mode_feedback_seconds = control_length/self.transport.feedback_rate
            # The actual TX mode traverses the optical bitstream with CRC and FEC.
            header_policy = 'qpsk_conv' if order_for(self.policy) == 4 else '16qam_repeat2'
            header, ok, _ = self.transport.transfer(bytes([tx_mode]), 'mode', header_policy)
            ok = bool(ok and len(header) == 1 and header[0] in (0, 1, 2))
            rx_mode = header[0] if ok else 2
            self.mode_protocol = 'noisy_rf_request_plus_crc_fec_optical_mode_announcement'
        self.requested_mode = torch.tensor([requested], device=device)
        self.tx_mode, self.rx_mode = torch.tensor([tx_mode], device=device), torch.tensor([rx_mode], device=device)
        self.header_ok = torch.tensor([ok], dtype=torch.bool, device=device)
        self.mode_header_ok = bool(ok)
        self.last_stats = {}

    def payload(self, tokens, token_amplitudes, receiver_amplitudes, interleave=True):
        if tokens.size(0) != 1 or tokens.size(1) not in (32, 288) or tokens.size(2) != 16:
            raise ValueError('V3 supports one V10 fixed-width payload at a time')
        original = tokens[0].detach().float().cpu().numpy()
        count, width = original.shape
        self.payload_sha256 = __import__('hashlib').sha256(original.tobytes()).hexdigest()
        # Text serves both the unchanged joint source encoder and its own wire task.
        # No original caption is exposed to the receiver fusion hook.
        text_bytes = self.context['source_text'].encode('utf-8')
        text_policy = '16qam_conv' if self.policy == 'analog' else self.policy
        received, text_ok, text_record = self.transport.transfer(text_bytes, 'text', text_policy)
        try:
            self.received_text = received.decode('utf-8') if text_ok else ''
            self.text_ok = bool(text_ok)
        except UnicodeDecodeError:
            self.received_text, self.text_ok = '', False
        bits, quantized, clipped = encode_source(original, self.settings.get('qbits', 8),
                                                  self.settings.get('quantizer_limit', 4.))
        if self.policy == 'analog':
            tx_amp = np.repeat(token_amplitudes[0].detach().cpu().numpy(), width//2)
            rx_amp = np.repeat(receiver_amplitudes[0].detach().cpu().numpy(), width//2)
            restored, reliability, _ = self.transport.analog(original, tx_amp, rx_amp, self.equalizer)
            reliability = reliability.reshape(count, width//2).mean(1)[:, None]
            feature_ok, quant_mse, feature_bytes = True, 0., None
            feature_record = {'complete': True, 'sent_bytes': None, 'crc_failures': None}
        else:
            feature_bytes = bits_bytes(bits)
            received_features, packets_ok, feature_record = self.transport.transfer(feature_bytes, 'features', self.policy)
            restored, feature_ok = decode_source(bytes_bits(received_features), original.size,
                self.settings.get('qbits', 8), self.settings.get('quantizer_limit', 4.)) if packets_ok else (np.zeros(original.size), False)
            reliability = np.full((count, 1), float(feature_ok))
            quant_mse = float(np.mean((original.ravel()-quantized)**2))
        usable_features = bool(feature_ok and self.mode_header_ok)
        # Adapter bias must not manufacture evidence from a CRC-erased feature object.
        self.header_ok.fill_(usable_features)
        restored = restored.reshape(count, width)*int(usable_features)
        reliability *= int(usable_features)
        total = self.transport.totals()
        flight_seconds = self.transport.distance/299792458.
        mode_roundtrip = 2*flight_seconds if self.mode_protocol.startswith('noisy_rf') else 0.
        link_seconds = total['link_seconds_without_mode_feedback']+self.mode_feedback_seconds+mode_roundtrip
        self.last_stats = dict(total, semantic_reals=original.size,
            semantic_float32_reference_bytes=original.size*4,
            semantic_quantized_body_bytes=(math.ceil(original.size*self.settings.get('qbits', 8)/8) if feature_bytes is not None else None),
            feature_object_wire_bytes=len(feature_bytes) if feature_bytes is not None else None,
            text_utf8_bytes=len(text_bytes), received_text_utf8_bytes=len(self.received_text.encode('utf-8')),
            text_crc_accepted=self.text_ok, text_exact_match_diagnostic=(self.received_text == self.context['source_text'] and self.text_ok),
            feature_object_crc_accepted=usable_features, mode_header_accepted=self.mode_header_ok,
            feedback_rejected=bool(self.feedback_rejected[0]), mode_protocol=self.mode_protocol,
            mode_feedback_seconds=self.mode_feedback_seconds, sequential_link_seconds=link_seconds,
            one_way_propagation_seconds=flight_seconds, mode_roundtrip_seconds=mode_roundtrip,
            serialization_and_control_seconds=link_seconds-flight_seconds,
            source_quantization_mse=quant_mse, source_clip_fraction=(0. if self.policy == 'analog' else clipped),
            optical_irradiance_gain=self.irradiance, field_identity=self.field_identity,
            payload_sha256=self.payload_sha256,
            gross_modulation_bps=(None if self.policy == 'analog' else math.log2(order_for(self.policy))*self.transport.pulse.rs),
            symbol_rate_baud=self.transport.pulse.rs, awg_sample_rate=self.transport.fs,
            sps=self.transport.sps, rolloff=self.transport.pulse.alpha,
            rrc_occupied_bandwidth_hz=(1+self.transport.pulse.alpha)*self.transport.pulse.rs,
            physical_unit='single_polarization_coherent_rrc_iq',
            digital_uep='none_uniform_fixed_code' if self.policy != 'analog' else 'existing_public_token_power_table',
            actual_digital_policies=self.transport.used_policy_by_kind,
            header_undetected_error=bool(self.mode_header_ok and int(self.tx_mode[0]) != int(self.rx_mode[0])),
            digital_ber_scope=('text_and_mode_only' if self.policy == 'analog' else 'text_mode_and_semantic_packets'),
            crc_note=('analog_features_have_no_crc' if self.policy == 'analog' else 'whole_feature_object_erased_if_any_packet_or_object_crc_fails'),
            text_packet_record=text_record, feature_packet_record=feature_record)
        return (torch.as_tensor(restored[None], device=tokens.device, dtype=tokens.dtype),
                torch.as_tensor(reliability[None], device=tokens.device, dtype=tokens.dtype))


def receiver_text_hook(model, original_forward, weight):
    """Zero-shot diagnostic late fusion; no new trained/aligned text projection.

    Failed text transfer follows the EXACT old forward. CRC-success text is
    locally encoded and passed through the existing trained refiner in a second
    branch. A public fixed weight blends the two receiver image-token branches.
    """
    if not 0 <= weight <= 1:
        raise ValueError('Fusion weight outside [0,1]')

    def forward(image_tokens, text_tokens, reliability=None):
        baseline, gate = original_forward(image_tokens, text_tokens, reliability=reliability)
        frame = model.last_frame
        if weight == 0 or not frame.text_ok or not frame.received_text:
            return baseline, gate
        encoded = model.base.text_encoder([frame.received_text], image_tokens.device)
        valid = encoded['attention_mask'][0].bool()
        memory = encoded['tokens'][:, valid].to(image_tokens.dtype)
        if memory.size(-1) != image_tokens.size(-1):
            raise ValueError('Received-text projection is not decoder hidden width')
        assisted, _ = original_forward(image_tokens, memory,
                                      reliability=image_tokens.new_ones((1, 1)))
        frame.fusion_used = True
        return baseline*(1-weight)+assisted*weight, gate
    return forward
