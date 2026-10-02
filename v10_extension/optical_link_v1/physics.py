"""Original component model: sqrt(W) fields, two ASE polarizations, A currents.

The EDFA is a flat-band, average total-output-limited black box, NOT erbium rate
equations or a measured 3-dB saturation curve. All noise is created at its own
component. No preset SNR, source-dependent waveform normalization or oracle CSI.
"""
import math
import numpy as np

H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.
Q_ELECTRON = 1.602176634e-19


def dbm_to_w(value):
    return 1e-3*10**(float(value)/10)


def w_to_dbm(value):
    return 10*math.log10(float(value)/1e-3) if value > 0 else None


def power(x):
    return float(np.mean(np.abs(x)**2))


def ratio_db(signal, noise):
    # JSON cannot represent +/-infinity. Null explicitly denotes absent signal
    # or noiseless diagnostic, rather than inventing a giant finite "real SNR".
    if signal <= 0 or noise <= 0:
        return None
    return 10*(math.log10(float(signal))-math.log10(float(noise)))


def response(count, fs, cutoff):
    # Assumed smooth zero-phase fourth-order amplitude filter. A transfer-function
    # model, not measured hardware/group delay. Zero padding reduces wraparound.
    freq = np.fft.fftfreq(count, 1/fs)
    return 1/np.sqrt(1+(freq/cutoff)**8)


def filtered(x, fs, cutoff):
    size = 1 << (2*len(x)-1).bit_length()
    return np.fft.ifft(np.fft.fft(x, size)*response(size, fs, cutoff))[:len(x)]


def filter_enbw(fs, cutoff):
    # Integral over the sampled two-sided frequency interval, in Hz.
    return float(np.mean(response(32768, fs, cutoff)**2)*fs)


def complex_white(rng, count, variance):
    return (rng.standard_normal(count)+1j*rng.standard_normal(count))*math.sqrt(variance/2)


def gamma_parameters(turb, distance, wavelength):
    if turb['mode'] == 'disabled':
        return None, None, None
    if turb['mode'] == 'manual_gamma_gamma':
        return float(turb['alpha']), float(turb['beta']), None
    s = 1.23*turb['cn2_m_minus_2_over_3']*(2*math.pi/wavelength)**(7/6)*distance**(11/6)
    if s == 0:
        return None, None, 0.
    if s > 10:
        raise ValueError('Rytov variance >10: outside this declared plane-wave screen')
    alpha = 1/math.expm1(.49*s/(1+1.11*s**(6/5))**(7/6))
    beta = 1/math.expm1(.51*s/(1+.69*s**(6/5))**(5/6))
    return alpha, beta, s


def aperture_fraction(beam_radius, aperture_radius, offset):
    """Circular Gaussian aperture integral, no small-aperture approximation.

    Analytically integrate y, then Gauss-Legendre integrate x=a*sin(theta).
    Includes BOTH aligned collection and pointing loss; never add them again.
    """
    if min(beam_radius, aperture_radius) <= 0 or offset < 0:
        raise ValueError('Invalid aperture geometry')
    if offset == 0:
        return -math.expm1(-2*(aperture_radius/beam_radius)**2)
    nodes, weights = np.polynomial.legendre.leggauss(128)
    # Focus quadrature on the beam footprint even if aperture/beam radius is huge.
    lower, upper = max(-aperture_radius, offset-8*beam_radius), min(aperture_radius, offset+8*beam_radius)
    if lower >= upper:
        return 0.
    theta = nodes*math.pi/2
    midpoint, halfwidth = (lower+upper)/2, (upper-lower)/2
    x = midpoint+halfwidth*np.sin(theta)
    ymax = np.sqrt(np.maximum(aperture_radius**2-x*x, 0.))
    erf = np.array([math.erf(math.sqrt(2)*y/beam_radius) for y in ymax])
    value = math.sqrt(2/math.pi)/beam_radius*np.sum(weights*math.pi/2*halfwidth*np.cos(theta)*erf
                    *np.exp(-2*((x-offset)/beam_radius)**2))
    return min(max(float(value), 0.), 1.)


def draw_path(config, rng):
    p, t, point = config['path'], config['turbulence'], config['pointing']
    alpha, beta, rytov = gamma_parameters(t, p['distance_m'], config['laser']['wavelength_m'])
    scintillation = float(rng.gamma(alpha, 1/alpha)*rng.gamma(beta, 1/beta)) if alpha else 1.
    geometry = {}
    if point['mode'] == 'manual_a0_xi':
        collection = point['a0']*max(float(rng.random()), 1e-300)**(1/point['xi']**2)
        geometry.update(a0=point['a0'], xi=point['xi'])
    else:
        distance = p['distance_m']
        w = math.hypot(point['beam_waist_m'], distance*point['divergence_half_angle_rad'])
        a = point['receiver_aperture_diameter_m']/2
        x = distance*(point['boresight_x_rad']+point['jitter_per_axis_rad']*rng.standard_normal())
        y = distance*(point['boresight_y_rad']+point['jitter_per_axis_rad']*rng.standard_normal())
        collection = aperture_fraction(w, a, math.hypot(x, y))
        v = math.sqrt(math.pi)*a/(math.sqrt(2)*w)
        a0_approx = math.erf(v)**2
        # Equivalent-width/xi are diagnostics of the Gaussian approximation ONLY.
        weq2 = w*w*math.sqrt(math.pi)*math.erf(v)/(2*v*math.exp(-v*v)) if v*v < 700 else None
        sigma = distance*point['jitter_per_axis_rad']
        xi = math.sqrt(weq2)/(2*sigma) if sigma and weq2 is not None else None
        geometry.update(beam_radius_at_rx_m=w, aperture_radius_m=a, offset_x_m=x,
                        offset_y_m=y, offset_radius_m=math.hypot(x, y),
                        a0=aperture_fraction(w, a, 0), a0_gaussian_approx=a0_approx,
                        xi=xi, approximation_w_over_a=w/a,
                        pointing_method='numerical_circular_aperture_not_a0_xi_approx')
    atmosphere_loss = p['attenuation_db_per_km']*p['distance_m']/1000
    atm = 10**(-atmosphere_loss/10)
    aperture_gain = atm*collection*scintillation
    coupled_gain = aperture_gain*10**(-p['rx_optics_loss_db']/10)*p['coupling_efficiency']
    phase = float(rng.uniform(-math.pi, math.pi))
    return dict(geometry, alpha=alpha, beta=beta, rytov_variance=rytov,
        gamma_gamma_irradiance=scintillation, atmosphere_loss_db=atmosphere_loss,
        atmosphere_transmission=atm, aperture_pointing_collection=collection,
        aperture_irradiance_gain=aperture_gain, coupled_irradiance_gain=coupled_gain,
        field_phase_rad=phase, field_real=math.sqrt(coupled_gain)*math.cos(phase),
        field_imag=math.sqrt(coupled_gain)*math.sin(phase),
        temporal_model='independent_frame_snapshot_constant_within_coherence_time')


def amplifier(psignal, input_ase_psd_per_pol, config, fs, wavelength):
    if not config['enabled']:
        return dict(gain_linear=1., gain_db=0., added_ase_psd_w_per_hz_per_pol=0.,
                    signal_output_w=psignal, ase_output_w=2*input_ase_psd_per_pol*fs,
                    output_limited=False)
    photon = H_PLANCK*C_LIGHT/wavelength
    nf = 10**(config['noise_figure_db']/10)
    incoming = psignal+2*input_ase_psd_per_pol*fs
    cap = dbm_to_w(config['max_total_output_dbm'])
    if incoming > cap:
        raise ValueError('EDFA input exceeds output limit; use a VOA/config change, not negative gain')
    requested = 10**(config['gain_db']/10)
    target = cap if config['control'] == 'gain' else min(cap, dbm_to_w(config['target_total_output_dbm']))
    if config['control'] == 'total_power' and incoming > target:
        raise ValueError('EDFA cannot attenuate to a target below its input')
    limit_gain = (target+photon*fs)/(incoming+nf*photon*fs)
    gain = max(1., min(requested, limit_gain))
    ase = (nf*gain-1)*photon/2 if gain > 1 else 0.
    return dict(gain_linear=gain, gain_db=10*math.log10(gain),
        added_ase_psd_w_per_hz_per_pol=ase, signal_output_w=gain*psignal,
        ase_output_w=2*(gain*input_ase_psd_per_pol+ase)*fs,
        output_limited=gain < requested*(1-1e-12),
        control=config['control'], noise_figure_db=config['noise_figure_db'],
        total_output_limit_w=target, saturation_model='average_total_power_cap_within_sampled_optical_band')


def phase_walk(count, linewidth, fs, rng):
    return np.cumsum(rng.standard_normal(count))*math.sqrt(2*math.pi*linewidth/fs) if linewidth else np.zeros(count)


def quantize_axes(value, bits, fullscale):
    clip = float(np.mean((np.abs(value.real)>fullscale)|(np.abs(value.imag)>fullscale)))
    levels = (1 << bits)-1
    def axis(x):
        return (np.rint((np.clip(x/fullscale, -1, 1)+1)*levels/2)*2/levels-1)*fullscale
    return axis(value.real)+1j*axis(value.imag), clip


def modulate_field(converted, config, fs):
    m, l = config['modulator'], config['laser']
    drive = filtered(converted/4*m['drive_peak_v'], fs, m['bandwidth_hz'])
    if m['model'] == 'iq_mzm':
        i = np.cos(math.pi/2*(drive.real+m['bias_i_v'])/m['vpi_v'])
        q = np.cos(math.pi/2*(drive.imag+m['bias_q_v'])/m['vpi_v'])
    else:
        i = math.pi/2*drive.real/m['vpi_v']
        q = math.pi/2*drive.imag/m['vpi_v']
    # Two lossless 50/50 couplers (split then recombine). The unused output port
    # carries the remaining energy; even biased/imperfect quadrature stays passive.
    field = math.sqrt(dbm_to_w(l['power_dbm'])*10**(-m['insertion_loss_db']/10))/2
    return field*(i+np.exp(1j*(math.pi/2+m['quadrature_error_rad']))*q)


def propagate(converted, config, path_draw, rng, phase_state=None):
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
    adc, adc_clip = quantize_axes(actual, rx['adc_bits'], rx['adc_fullscale_v'])
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
