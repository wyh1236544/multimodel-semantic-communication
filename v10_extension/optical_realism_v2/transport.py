"""Fixed-workpoint SI optics + costed pilots; inherits only packet protocol/totals."""
import copy
import math
import numpy as np
from business_link.engine import identity_seed
from business_link.engine import POLICIES, KINDS, wire_packet, read_wire, order_for
from physical_link.digital import bytes_bits, bits_bytes, fec_encode, fec_decode, scramble, modulate, demodulate
from optical_link_v1.transport import OpticalTransport
from optical_link_v1.physics import power, ratio_db, dbm_to_w
from final_compare_v1.transport import regulated_config, BudgetTransport
from physical_link.waveform import preamble, dac, convolve
from optical_realism_v2.hardware import optical_config, tx_impair, interpolate
from optical_realism_v2.physics import propagate
from optical_realism_v2.dsp import frame_symbols, receive


def papr(value):
    return 10*math.log10(max(float(np.max(abs(value)**2)),1e-30)/max(power(value),1e-30))


class RealisticTransport(OpticalTransport):
    def __init__(self, config, field_seed, noise_seed, target_dbm, slot_samples, laser_cap_dbm, hardware):
        self.hardware = copy.deepcopy(hardware)
        if config['waveform']['dac_bits'] != 8 or config['receiver']['adc_bits'] != 12 or config['modulator']['model'] != 'iq_mzm':
            raise ValueError('Expected DAC8/ADC12 and passive IQ-MZM')
        super().__init__(optical_config(config, hardware),field_seed,noise_seed)
        self.slot_samples, self.target_dbm = int(slot_samples),float(target_dbm)
        # Fixed public calibration waveform. No payload, RX power, true h or SNR.
        reference, _, _, _, _ = frame_symbols(preamble(1024),hardware)
        wave = self.pulse.shape(reference)*math.sqrt(self.pulse.sps)
        converted, _, _ = dac(wave,self.sps,self.settings['dac_bits'])
        # Calibrate deterministic IQ mismatch; random converter noise is not a control input.
        calibration = dict(hardware,dac_effective_bits=8.)
        converted, _ = tx_impair(converted,calibration,np.random.default_rng(0))
        self.optical_config, self.fixed_laser_dbm = regulated_config(converted,self.optical_config,target_dbm,laser_cap_dbm)
        c = self.optical_config
        # Passive IQ-MZM |field|^2 <= laser*insertion; EDFA gain upper bound.
        self.peak_signal_bound_w = dbm_to_w(self.fixed_laser_dbm)*10**(-c['modulator']['insertion_loss_db']/10)
        if c['tx_edfa']['enabled']: self.peak_signal_bound_w *= 10**(c['tx_edfa']['gain_db']/10)
        self.peak_signal_bound_w *= 10**(-c['path']['tx_optics_loss_db']/10)

    def burst(self, symbols, purpose, sequence=0, attempt=0, policy='probe'):
        symbols = np.asarray(symbols,complex)
        frame, data_at, pilot_at, _, _ = frame_symbols(symbols,self.hardware)
        wave = self.pulse.shape(frame)*math.sqrt(self.pulse.sps)
        used = sum(e['awg_samples']+e['guard_samples'] for e in self.events)
        if used+len(wave)+self.guard > self.slot_samples: raise ValueError('Charged pilot/time budget exceeded BEFORE TX')
        if (used+len(wave)+self.guard)/self.fs > self.optical_config['turbulence']['coherence_time_s']:
            raise ValueError('Slot exceeds slow-fading coherence time')
        rng = np.random.default_rng(identity_seed(self.seed,purpose,sequence,attempt))
        converted, _, clip = dac(wave,self.sps,self.settings['dac_bits'])
        converted, drive_clip = tx_impair(converted,self.hardware,rng)
        adc, signal, noise, physical = propagate(converted,self.optical_config,self.path_draw,rng,self.phase_state,self.hardware)
        data, raw, estimated, at, transform, data_at = receive(np.pad(adc,(7,self.guard)),self.pulse,len(symbols),self.hardware)
        # Simulation truth diagnostics only, projected AFTER the RX-selected corrections.
        def diagnose(x):
            mf = convolve(np.pad(x,(7,self.guard)),self.pulse.kernels[0][::-1])
            return interpolate(mf,at)*transform
        s, n = diagnose(signal), diagnose(noise)
        active = data_at if len(data_at) else pilot_at
        useful = frame[active]
        coefficient = np.vdot(useful,s[active])/max(np.vdot(useful,useful).real,1e-30)
        signal_energy = float(abs(coefficient)**2*np.sum(abs(useful)**2))
        noise_energy = float(np.sum(abs(n[active])**2))
        distortion = float(np.sum(abs(s[active]-coefficient*useful)**2))
        evm2 = float(power(data-symbols)/max(power(symbols),1e-30)) if len(symbols) else None
        windows = lambda x: [papr(x[j:j+4096]) for j in range(0,len(x)-4095,4096)]
        event = dict(physical,**estimated,purpose=purpose,sequence=sequence,attempt=attempt,policy=policy,
            awg_samples=len(wave),guard_samples=self.guard,waveform_seconds=len(wave)/self.fs,
            scheduled_seconds=(len(wave)+self.guard)/self.fs,pilot_symbols=len(pilot_at),data_symbols=len(symbols),
            tracking_pilot_symbols=len(pilot_at)-256,evm_squared=evm2,
            sample_average_power=power(converted),sample_peak_power=float(np.max(abs(converted)**2)),
            waveform_energy_power_seconds=float(np.sum(abs(converted)**2)/self.fs),normalized_dsp_power_not_watts=True,
            dac_clip_fraction=clip,driver_clip_fraction=drive_clip,papr_pre_dac_db=papr(wave),papr_post_driver_db=papr(converted),
            papr_window_samples=4096,papr_pre_fixed_windows_db=windows(wave),papr_post_fixed_windows_db=windows(converted),
            physical_symbol_snr_db=ratio_db(signal_energy,noise_energy),snr_signal_energy_a2=signal_energy,
            snr_noise_energy_a2=noise_energy,clean_distortion_energy_a2=distortion,
            snr_reference='offline_RX_selected_timing_phase_projection_optical_shot_thermal_noise_only; converter_noise_and_distortion_in_EVM',
            launch_target_signal_dbm=self.target_dbm,fixed_laser_dbm=self.fixed_laser_dbm,
            simulator_lo_offset_hz=self.hardware['frequency_offset_hz'],
            simulator_adc_clock_ppm=self.hardware['adc_clock_ppm'],
            cfo_sign_note='RX complex CFO is TX minus LO; expected negative of configured LO offset, diagnostic only',
            launch_peak_signal_bound_w=self.peak_signal_bound_w,
            launch_signal_energy_j=physical['power_stages']['launch']['signal_w']*len(wave)/self.fs,
            pre_errors=0,pre_bits=0,post_errors=0,post_bits=0,useful_bytes_accepted=0,
            crc_accepted=None,ack_seconds=0.,arq_wait_seconds=0.)
        self.events.append(event)
        if len(symbols) and not any(c['purpose']==purpose for c in self.constellations):
            pairs = lambda a: [[float(z.real),float(z.imag)] for z in a[:2048]]
            self.constellations.append(dict(purpose=purpose,policy=policy,tx=pairs(symbols),rx=pairs(data)))
        return data,event

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
                parsed = read_wire(bits_bytes(decoded), kind, self.capacity) if event['receiver_locked'] else None
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

    def totals(self):
        result = super().totals()
        used = result['awg_samples']+result['guard_samples']
        cap = self.peak_signal_bound_w*self.slot_samples/self.fs
        if used > self.slot_samples or result['launch_signal_energy_j'] > cap*(1+1e-8):
            raise ValueError('Physical time/peak-envelope energy bound violated')
        result.update(slot_samples=self.slot_samples,used_samples=used,slot_seconds=self.slot_samples/self.fs,
            unused_slot_samples=self.slot_samples-used,launch_target_signal_dbm=self.target_dbm,
            fixed_laser_dbm=self.fixed_laser_dbm,launch_peak_signal_bound_w=self.peak_signal_bound_w,
            signal_energy_budget_j=cap,resource_feasible=True,
            same_resource_contract='same fixed laser/MZM/EDFA, peak envelope, bandwidth and slot cap; actual mean launch power and spent energy may differ',
            launch_target_note='public QPSK calibration reference, NOT actual equalized per-burst launch power',
            receiver_lock_failures=sum(not e['receiver_locked'] for e in self.events),
            receiver_all_bursts_locked=all(e['receiver_locked'] for e in self.events),
            tracking_pilot_symbols=sum(e['tracking_pilot_symbols'] for e in self.events),
            fixed_workpoint_no_per_payload_power_control=True,bit_rate_for_continuous_features_undefined=True,
            timing_note='optical sample scheduling+one L/c; buffered receiver, NN and CPU simulation wall time are NOT measured hardware latency',
            hardware_preset=self.hardware['preset'],hardware_calibrated=False)
        return result

    def continuous(self, values, repeats, equalizer):
        received,reliability,stats = BudgetTransport.continuous(self,values,repeats,equalizer)
        if not self.events[-1]['receiver_locked']:
            received[:] = 0; reliability = 0.
        stats.update(analog_reliability=reliability,analog_receiver_available=self.events[-1]['receiver_locked'])
        return received,reliability,stats
