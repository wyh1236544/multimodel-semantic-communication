"""Reuse the historical bit protocol/FEC, replace ONLY the sampled optical link."""
import math
import numpy as np
from business_link.engine import Transport, identity_seed
from physical_link.waveform import preamble, dac
from physical_link.digital import modulate
from optical_link_v1.physics import draw_path, propagate, power, ratio_db, w_to_dbm


def receive_si(observed, pulse, count, training, mode, step, order=16):
    """Volts in; public preamble, constellation and clock only. No oracle h/SNR."""
    reference = pulse.shape(training)
    scores = [abs(np.vdot(reference, observed[i:i+len(reference)]))
              if i+len(reference) <= len(observed) else -1 for i in range(65)]
    start = int(np.argmax(scores))
    matched = pulse.match(observed, count, start)
    slope = 0.
    if mode == 'pilot_cfo_dd':
        angles = np.unwrap(np.angle(matched[:len(training)]*np.conj(training)))
        slope = float(np.polyfit(np.arange(len(training)), angles, 1)[0])
    corrected = matched*np.exp(-1j*slope*np.arange(count))
    pilot = corrected[:len(training)]
    estimate = np.vdot(training, pilot)/np.vdot(training, training).real
    residual = pilot-estimate*training
    # Tiny SI currents must not use the historical normalized variance floor 1e-10.
    variance = max(float(np.sum(np.abs(residual)**2)/(len(training)-2)), 1e-30)
    gain_power = max(abs(estimate)**2, 1e-30)
    eq = corrected*np.conj(estimate)/gain_power
    if mode == 'pilot_cfo_dd' and step:
        # Public nearest-QAM decision-directed phase loop. Decisions can be wrong;
        # no transmitted data are supplied to this loop or used to resolve slips.
        width = int(math.log2(order))
        constellation = modulate(np.array([int(v) for i in range(order) for v in format(i, '0{}b'.format(width))], np.uint8), order)
        phase = 0.
        for ix in range(len(training), count):
            value = eq[ix]*np.exp(-1j*phase)
            decision = constellation[np.argmin(np.abs(value-constellation)**2)]
            phase += step*float(np.angle(value*np.conj(decision)))
            eq[ix] = value
    pilot_signal = gain_power*float(np.sum(np.abs(training)**2))
    pilot_noise = variance*len(training)
    return eq, matched, dict(sync_sample=start, estimated_snr_db=ratio_db(pilot_signal, pilot_noise),
        equalized_noise_variance=variance/gain_power, estimated_cfo_hz=slope*pulse.rs/(2*math.pi),
        estimated_gain_real=float(estimate.real), estimated_gain_imag=float(estimate.imag),
        pilot_signal_energy_v2=pilot_signal, pilot_noise_energy_v2=pilot_noise)


class OpticalTransport(Transport):
    def __init__(self, config, field_seed, noise_seed):
        self.optical_config = config
        self.path_draw = draw_path(config, np.random.default_rng(field_seed))
        self.field_seed = int(field_seed)
        w, path = config['waveform'], config['path']
        settings = dict(fs=w['fs_hz'], sps=w['sps'], rolloff=w['rolloff'], span=w['span'],
            dac_bits=w['dac_bits'], packet_bytes=w['packet_bytes'], guard_samples=w['guard_samples'],
            max_attempts=1, distance_m=path['distance_m'])
        super().__init__(settings, complex(self.path_draw['field_real'], self.path_draw['field_imag']), 0., noise_seed)
        self.reference_power = None
        self.phase_state = dict(tx=0., lo=0., time_s=0.)

    def burst(self, symbols, purpose, sequence=0, attempt=0, policy='probe'):
        training = preamble(256)
        frame = np.concatenate([training, np.asarray(symbols, complex)])
        wave = self.pulse.shape(frame)*math.sqrt(self.pulse.sps)
        converted, _, clip = dac(wave, self.pulse.sps, self.settings['dac_bits'])
        elapsed = sum(e['scheduled_seconds'] for e in self.events)+(len(wave)+self.guard)/self.fs
        if elapsed > self.optical_config['turbulence']['coherence_time_s']:
            raise ValueError('Frame exceeds configured coherence time; constant-block model invalid')
        rng = np.random.default_rng(identity_seed(self.seed, purpose, sequence, attempt))
        adc, signal, noise, physical = propagate(converted, self.optical_config, self.path_draw, rng, self.phase_state)
        # Capture delay is unknown to receiver. It is NOT a free oracle timing input.
        observed = np.pad(adc, (7, self.guard))
        rx = self.optical_config['receiver']
        eq, raw, estimated = receive_si(observed, self.pulse, len(frame), training,
                                       rx['carrier_recovery'], rx['dd_phase_step'],
                                       4 if policy == 'qpsk_conv' else 16)
        if not np.isfinite(eq).all():
            raise FloatingPointError('Nonfinite optical matched/equalized waveform')
        data = eq[len(training):]
        # Offline truth uses the same sample location chosen by RX, no CSI input.
        start = estimated['sync_sample']
        s = self.pulse.match(np.pad(signal, (7, self.guard)), len(frame), start)
        n = self.pulse.match(np.pad(noise, (7, self.guard)), len(frame), start)
        active = slice(len(training), None) if len(symbols) else slice(None)
        useful = frame[active]
        # Projection defines useful signal; deterministic distortion is NOT noise.
        coefficient = np.vdot(useful, s[active])/np.vdot(useful, useful).real
        signal_energy = float(abs(coefficient)**2*np.sum(np.abs(useful)**2))
        noise_energy = float(np.sum(np.abs(n[active])**2))
        clean_distortion = float(np.sum(np.abs(s[active]-coefficient*useful)**2))
        evm2 = float(np.mean(np.abs(data-symbols)**2)/max(power(symbols), 1e-30)) if len(symbols) else None
        event = dict(physical, **estimated, purpose=purpose, sequence=sequence, attempt=attempt,
            policy=policy, awg_samples=len(wave), guard_samples=self.guard,
            waveform_seconds=len(wave)/self.fs, scheduled_seconds=(len(wave)+self.guard)/self.fs,
            pilot_symbols=len(training), data_symbols=len(symbols), evm_squared=evm2,
            sample_average_power=power(converted), sample_peak_power=float(np.max(np.abs(converted)**2)),
            waveform_energy_power_seconds=float(np.sum(np.abs(converted)**2)/self.fs),
            normalized_dsp_power_not_watts=True, dac_clip_fraction=clip,
            physical_symbol_snr_db=ratio_db(signal_energy, noise_energy),
            physical_snr_status='defined' if signal_energy>0 and noise_energy>0 else 'no_signal' if signal_energy==0 else 'noise_free',
            snr_signal_energy_a2=signal_energy, snr_noise_energy_a2=noise_energy,
            clean_distortion_energy_a2=clean_distortion,
            snr_reference='matched_filter_symbol_samples_useful_projection_noise_only_offline',
            pre_errors=0, pre_bits=0, post_errors=0, post_bits=0, useful_bytes_accepted=0,
            crc_accepted=None, ack_seconds=0., arq_wait_seconds=0.)
        self.events.append(event)
        if len(symbols) and not any(c['purpose'] == purpose for c in self.constellations):
            count = min(len(symbols), 2048)
            norm = max(math.sqrt(estimated['estimated_gain_real']**2+estimated['estimated_gain_imag']**2), 1e-15)
            pairs = lambda a: [[float(z.real), float(z.imag)] for z in a[:count]]
            self.constellations.extend([
                dict(purpose=purpose+'_before_gain_phase_equalization', policy=policy, tx=pairs(symbols), rx=pairs(raw[len(training):]/norm)),
                dict(purpose=purpose, policy=policy, tx=pairs(symbols), rx=pairs(data))])
        return data, event

    def totals(self):
        totals = super().totals()
        events = self.events
        signal = sum(e['snr_signal_energy_a2'] for e in events)
        noise = sum(e['snr_noise_energy_a2'] for e in events)
        pilot_signal = sum(e['pilot_signal_energy_v2'] for e in events)
        pilot_noise = sum(e['pilot_noise_energy_v2'] for e in events)
        totals.update(physical_symbol_snr_db=ratio_db(signal, noise),
            estimated_snr_db=ratio_db(pilot_signal, pilot_noise),
            snr_aggregation='ratio_of_summed_linear_signal_and_noise_energies_not_mean_db',
            snr_signal_energy_a2=signal, snr_noise_energy_a2=noise,
            pilot_signal_energy_v2=pilot_signal, pilot_noise_energy_v2=pilot_noise,
            snr_reference=events[0]['snr_reference'], nominal_snr_db=None,
            snr_is_simulation_not_hardware_measurement=True, path_draw=self.path_draw,
            optical_irradiance_gain=self.path_draw['coupled_irradiance_gain'],
            adc_clip_fraction_max=max(e['adc_clip_fraction'] for e in events),
            pd_saturation_fraction_max=max(e['pd_saturation_fraction'] for e in events))
        # Optical active-burst time-weighted means: convert W to dBm AFTER pooling.
        duration = sum(e['waveform_seconds'] for e in events)
        stages = {}
        for name in events[0]['power_stages']:
            values = {k:sum(e['power_stages'][name][k]*e['waveform_seconds'] for e in events)/duration
                      for k in ('signal_w', 'ase_w', 'total_w')}
            values.update(signal_dbm=w_to_dbm(values['signal_w']), total_dbm=w_to_dbm(values['total_w']))
            stages[name] = values
        totals['power_stages'] = stages
        totals['received_optical_signal_dbm'] = stages['detector_input']['signal_dbm']
        totals['received_optical_total_dbm'] = stages['detector_input']['total_dbm']
        ref_nm = self.optical_config['receiver']['osnr_reference_bandwidth_nm']
        # Each event OSNR can be pooled through its linear reference noise power.
        weighted_noise = sum(e['osnr_reference_noise_w']*e['waveform_seconds'] for e in events)/duration
        totals.update(osnr_db=ratio_db(stages['detector_input']['signal_w'], weighted_noise) if weighted_noise else None,
                      osnr_reference_bandwidth_nm=ref_nm,
                      sndr_from_evm_db=-20*math.log10(max(totals['evm_rms'], 1e-150)) if totals['evm_rms'] is not None else None,
                      launch_signal_energy_j=sum(e['power_stages']['launch']['signal_w']*e['waveform_seconds'] for e in events),
                      timing_note='scheduled optical samples+guards; link also adds one L/c; neural wall time separate',
                      power_note='active burst power; guard/idle ASE and device electrical energy not modeled')
        return totals
