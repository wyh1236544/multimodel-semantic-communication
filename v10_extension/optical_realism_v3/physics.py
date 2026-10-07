"""V3 sampled SI propagation, adapted from preserved optical_link_v1 physics.
Only the ADC sampling/noise stage changes here; all SI power/ASE formulas retained.
Diagnostic noise-only branch excludes converter noise (included in received EVM).
"""
import math
import numpy as np
from optical_link_v1.physics import (power, modulate_field, dbm_to_w, w_to_dbm,
    phase_walk, amplifier, filtered, complex_white, quantize_axes, filter_enbw,
    response, ratio_db, Q_ELECTRON, C_LIGHT)
from optical_realism_v3.hardware import sample_clock, extra_converter_noise, interpolate


def propagate(converted, config, path_draw, rng, phase_state=None, hardware=None):
    """Return ADC volts plus independent offline diagnostic signal/noise currents.

    Signal/LO are single-polarization. Two independent ASE polarizations are
    retained for optical power and shot noise; only the LO-aligned one beats.
    Receiver never sees the diagnostic branches or simulator path draw.
    """
    fs = config['waveform']['fs_hz']
    l, p, rx = config['laser'], config['path'], config['receiver']
    n = len(converted)
    sig = modulate_field(converted, config, fs)
    after_mzm = power(sig)
    phase_state = phase_state if phase_state is not None else dict(tx=0., lo=0., time_s=0.)
    tx_phase = phase_state['tx']+phase_walk(n, l['linewidth_hz'], fs, rng)
    sig *= np.exp(1j*tx_phase)
    tx = amplifier(after_mzm, 0., config['tx_edfa'], fs, l['wavelength_m'])
    sig *= math.sqrt(tx['gain_linear'])
    launch_transmission = 10**(-p['tx_optics_loss_db']/10)
    sig *= math.sqrt(launch_transmission)
    ps_launch = power(sig)
    h = complex(path_draw['field_real'], path_draw['field_imag'])
    sig *= h
    tx_psd = tx['added_ase_psd_w_per_hz_per_pol']*launch_transmission*abs(h)**2
    amp_rx = amplifier(power(sig), tx_psd, config['rx_edfa'], fs, l['wavelength_m'])
    sig *= math.sqrt(amp_rx['gain_linear'])
    tx_psd *= amp_rx['gain_linear']
    rx_psd = amp_rx['added_ase_psd_w_per_hz_per_pol']
    cutoff = rx['optical_filter_full_bandwidth_hz']/2
    sig = filtered(sig, fs, cutoff)
    tx_ase = filtered(complex_white(rng, n, tx_psd*fs), fs, cutoff)
    rx_ase = filtered(complex_white(rng, n, rx_psd*fs), fs, cutoff)
    orth_ase = filtered(complex_white(rng, n, (tx_psd+rx_psd)*fs), fs, cutoff)
    lo_walk = phase_state['lo']+phase_walk(n, rx['lo_linewidth_hz'], fs, rng)
    lo_phase = lo_walk+2*math.pi*rx['frequency_offset_hz']*(phase_state['time_s']+np.arange(n)/fs)
    idle_s = config['waveform']['guard_samples']/fs
    phase_state.update(tx=float(tx_phase[-1]+rng.standard_normal()*math.sqrt(2*math.pi*l['linewidth_hz']*idle_s)),
                       lo=float(lo_walk[-1]+rng.standard_normal()*math.sqrt(2*math.pi*rx['lo_linewidth_hz']*idle_s)),
                       time_s=phase_state['time_s']+n/fs+idle_s)
    lo = math.sqrt(dbm_to_w(rx['lo_power_dbm']))*np.exp(1j*lo_phase)
    observed_field = sig+tx_ase+rx_ase
    # Lossless 90-degree hybrid: sum of its four port powers equals Ps+Plo.
    ports = [(observed_field+lo)/2, (observed_field-lo)/2,
             (observed_field+1j*lo)/2, (observed_field-1j*lo)/2]
    r = rx['responsivity_a_per_w']
    currents = [r*(np.abs(port)**2+np.abs(orth_ase)**2/4) for port in ports]
    saturation = float(np.mean(np.any(np.stack(currents)>rx['pd_saturation_a'], axis=0)))
    clipped = [np.minimum(current, rx['pd_saturation_a']) for current in currents]
    ideal_iq = r*sig*np.conj(lo)
    linear_noisy_iq = r*observed_field*np.conj(lo)
    limited_iq = clipped[0]-clipped[1]+1j*(clipped[2]-clipped[3])
    # Per-PD one-sided shot PSD=2q(I+Id); sampled real variance=q(I+Id)Fs.
    shots = [rng.standard_normal(n)*np.sqrt(Q_ELECTRON*(v+rx['dark_current_a'])*fs)
             if rx['shot_noise'] else np.zeros(n) for v in currents]
    shot_iq = shots[0]-shots[1]+1j*(shots[2]-shots[3])
    # Thermal density is ONE-sided, per photodiode; each PD variance=i_n^2 Fs/2.
    thermals = [rng.standard_normal(n)*rx['thermal_density_a_per_sqrt_hz_per_pd']*math.sqrt(fs/2) for _ in range(4)]
    thermal_iq = thermals[0]-thermals[1]+1j*(thermals[2]-thermals[3])
    components = dict(tx_ase=r*tx_ase*np.conj(lo), rx_ase=r*rx_ase*np.conj(lo),
                      shot=shot_iq, thermal=thermal_iq)
    electrical = rx['electrical_bandwidth_hz']
    signal_current = filtered(ideal_iq, fs, electrical)
    noise_current = filtered(linear_noisy_iq-ideal_iq+shot_iq+thermal_iq, fs, electrical)
    actual = filtered(limited_iq+shot_iq+thermal_iq, fs, electrical)*rx['tia_v_per_a']
    # Sampling-clock/jitter acts BEFORE ADC quantization, not after ideal recovery.
    sampled, clock_at = sample_clock(actual, hardware, fs, rng)
    sampled = extra_converter_noise(sampled, rx['adc_fullscale_v'], rx['adc_bits'],
                                    hardware['adc_effective_bits'], rng)
    adc, adc_clip = quantize_axes(sampled, rx['adc_bits'], rx['adc_fullscale_v'])
    signal_current = interpolate(signal_current, clock_at)
    noise_current = interpolate(noise_current, clock_at)
    enbw = filter_enbw(fs, cutoff)
    ref_hz = C_LIGHT/l['wavelength_m']**2*rx['osnr_reference_bandwidth_nm']*1e-9
    stages = {
        'laser': dict(signal_w=dbm_to_w(l['power_dbm']), ase_w=0.),
        'after_mzm': dict(signal_w=after_mzm, ase_w=0.),
        'after_tx_edfa': dict(signal_w=tx['signal_output_w'], ase_w=tx['ase_output_w']),
        'launch': dict(signal_w=ps_launch, ase_w=tx['ase_output_w']*launch_transmission),
        'rx_aperture': dict(signal_w=ps_launch*path_draw['aperture_irradiance_gain'],
                            ase_w=tx['ase_output_w']*launch_transmission*path_draw['aperture_irradiance_gain']),
        'rx_coupled': dict(signal_w=ps_launch*abs(h)**2,
                          ase_w=2*tx['added_ase_psd_w_per_hz_per_pol']*launch_transmission*abs(h)**2*fs),
        'after_rx_edfa': dict(signal_w=amp_rx['signal_output_w'], ase_w=amp_rx['ase_output_w']),
        'detector_input': dict(signal_w=power(sig), ase_w=2*(tx_psd+rx_psd)*enbw)}
    for values in stages.values():
        values.update(total_w=values['signal_w']+values['ase_w'],
                      signal_dbm=w_to_dbm(values['signal_w']),
                      total_dbm=w_to_dbm(values['signal_w']+values['ase_w']))
    reference_enbw = float(np.mean(response(32768, fs, cutoff)[np.abs(np.fft.fftfreq(32768, 1/fs)) <= ref_hz/2]**2)*ref_hz)
    osnr_noise = 2*(tx_psd+rx_psd)*reference_enbw
    diagnostics = dict(power_stages=stages, tx_edfa=tx, rx_edfa=amp_rx,
        osnr_db=ratio_db(power(sig), osnr_noise) if osnr_noise else None,
        osnr_reference_bandwidth_nm=rx['osnr_reference_bandwidth_nm'],
        osnr_reference_bandwidth_hz=ref_hz, optical_filter_enbw_hz=enbw,
        osnr_reference_noise_w=osnr_noise,
        detector_ase_psd_w_per_hz_per_pol=tx_psd+rx_psd,
        pd_saturation_fraction=saturation, adc_clip_fraction=adc_clip,
        analog_signal_current_power_a2=power(signal_current),
        analog_noise_current_power_a2=power(noise_current),
        analog_noise_only_snr_db=ratio_db(power(signal_current), power(noise_current)),
        receiver_noise_components_a2={k:power(filtered(v, fs, electrical)) for k,v in components.items()},
        ideal_hybrid_power_error_w=float(np.max(np.abs(sum(np.abs(v)**2 for v in ports)-(np.abs(observed_field)**2+np.abs(lo)**2)))))
    return adc, signal_current, noise_current, diagnostics

