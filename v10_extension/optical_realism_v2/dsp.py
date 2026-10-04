"""Buffered pilot-only timing/CFO/phase recovery for arbitrary complex payloads.

No nearest-QAM decisions, TX data, true delay/clock/phase/channel coefficients.
Public frame layout and nominal 4 SPS only. Block-clock/phase fits need buffering;
this implementation does not claim a real-time FPGA receiver or causal latency.
"""
import math
import numpy as np
from physical_link.waveform import preamble, convolve
from optical_realism_v2.hardware import interpolate


def layout(count, h):
    if type(count) is not int or count < 0: raise ValueError('Invalid payload length')
    pilot_at = list(range(256)); pilot = list(preamble(256)); groups = [(0, 256)]
    data_at = []; pos = 256
    for offset in range(0, count, h['pilot_interval']):
        size = min(h['pilot_interval'], count-offset)
        data_at.extend(range(pos, pos+size)); pos += size
        pilot_at.extend(range(pos, pos+h['pilot_block']))
        pilot.extend(preamble(h['pilot_block']))
        groups.append((len(pilot)-h['pilot_block'], len(pilot)))
        pos += h['pilot_block']
    return np.asarray(data_at, int), np.asarray(pilot_at, int), np.asarray(pilot), groups, pos


def frame_symbols(symbols, h):
    data_at, pilot_at, pilot, groups, count = layout(len(symbols), h)
    frame = np.zeros(count, complex); frame[data_at] = symbols; frame[pilot_at] = pilot
    return frame, data_at, pilot_at, pilot, groups


def receive(observed, pulse, count, h):
    """Only public receiver settings in h are read; impairment truth is not used."""
    data_at, pilot_at, pilot, groups, length = layout(count, h)
    if pulse.q != 1: raise ValueError('Integer nominal SPS required')
    mf = convolve(observed, pulse.kernels[0][::-1])
    nominal = 2*pulse.half+np.arange(256)*pulse.sps
    candidates = np.arange(0, h['timing_search_samples']+.01, .125)
    y = interpolate(mf, candidates[:,None]+nominal[None,:])
    # Short correlations tolerate CFO; a whole-preamble coherent sum would fail.
    z = (y*np.conj(pilot[:256])).reshape(len(candidates),16,16)
    score = np.sum(abs(z.sum(axis=2))**2,axis=1)/(16*np.sum(abs(y)**2,axis=1)+1e-300)
    start = float(candidates[np.argmax(score)])
    pilot0 = interpolate(mf, nominal+start)*np.conj(pilot[:256])
    slope = float(np.angle(np.sum(pilot0[1:]*np.conj(pilot0[:-1]))))
    # Fit residual slope after adjacent-lag coarse CFO acquisition.
    residual = np.unwrap(np.angle(pilot0*np.exp(-1j*slope*np.arange(256))))
    slope += float(np.polyfit(np.arange(256),residual,1)[0])
    centers, offsets = [127.5], [start]
    for lo, hi in groups[1:]:
        ix = pilot_at[lo:hi]; known = pilot[lo:hi]
        predicted = offsets[-1]
        if len(centers)>1:
            predicted += (offsets[-1]-offsets[-2])/(centers[-1]-centers[-2])*(ix.mean()-centers[-1])
        # Bound predicted drift by the public capture range, not simulator ppm.
        predicted = float(np.clip(predicted,start-h['timing_max_ppm']*1e-6*pulse.sps*ix.mean()-1,
                                  start+h['timing_max_ppm']*1e-6*pulse.sps*ix.mean()+1))
        shifts = predicted+np.arange(-h['timing_tracking_radius_samples'],h['timing_tracking_radius_samples']+.01,.125)
        samples = interpolate(mf,2*pulse.half+pulse.sps*ix[None,:]+shifts[:,None])
        product = samples*np.conj(known)*np.exp(-1j*slope*ix)
        metric = abs(product.sum(axis=1))**2/(len(ix)*np.sum(abs(samples)**2,axis=1)+1e-300)
        centers.append(float(ix.mean())); offsets.append(float(shifts[np.argmax(metric)]))
    if len(centers)>2:
        timing_slope, timing_intercept = np.polyfit(centers,offsets,1)
        timing_slope = float(np.clip(timing_slope,-h['timing_max_ppm']*1e-6*pulse.sps,h['timing_max_ppm']*1e-6*pulse.sps))
    else:
        timing_slope, timing_intercept = 0., start
    at = 2*pulse.half+timing_intercept+(pulse.sps+timing_slope)*np.arange(length)
    raw = interpolate(mf,at)
    corrected = raw*np.exp(-1j*slope*np.arange(length))
    # Per-block even pilots fit; odd pilots measure residual (not unbiased SNR).
    gain, phase_centers = [], []
    for lo,hi in groups:
        use = np.arange(lo,hi,2)
        gain.append(np.mean(corrected[pilot_at[use]]*np.conj(pilot[use])))
        phase_centers.append(float(pilot_at[use].mean()))
    gain = np.asarray(gain)
    phases = np.unwrap(np.angle(gain))
    phase_jumps = np.angle(gain[1:]*np.conj(gain[:-1]))
    phase = np.interp(np.arange(length),phase_centers,phases)
    amplitude = np.interp(np.arange(length),phase_centers,abs(gain))
    eq = corrected*np.exp(-1j*phase)/np.maximum(amplitude,1e-15)
    hold = np.arange(1,len(pilot),2)
    variance = max(float(np.mean(abs(eq[pilot_at[hold]]-pilot[hold])**2)),1e-12)
    snr = -10*math.log10(variance)
    reasons = []
    if float(np.max(score)) < h['minimum_sync_score']: reasons.append('preamble_correlation')
    if snr < h['minimum_pilot_snr_db']: reasons.append('pilot_residual')
    if len(phase_jumps) and np.max(abs(phase_jumps)) > h['maximum_pilot_phase_jump_rad']:
        reasons.append('pilot_phase_ambiguity')
    if abs(timing_slope) >= h['timing_max_ppm']*1e-6*pulse.sps*.999:
        reasons.append('clock_capture_boundary')
    locked = not reasons
    data = eq[data_at] if locked else np.zeros(count,complex)
    diagnostic = dict(sync_sample=float(timing_intercept),sync_score=float(np.max(score)),
        receiver_locked=locked,receiver_rejection_reasons=reasons,
        estimated_snr_db=snr,equalized_noise_variance=variance if locked else max(variance,1e6),
        estimated_cfo_hz=slope*pulse.rs/(2*math.pi),estimated_clock_ppm=timing_slope/pulse.sps*1e6,
        estimated_gain_real=float(gain[0].real),estimated_gain_imag=float(gain[0].imag),
        pilot_signal_energy_v2=float(np.sum(amplitude[pilot_at[hold]]**2)),
        pilot_noise_energy_v2=float(np.sum(abs(corrected[pilot_at[hold]]-
            amplitude[pilot_at[hold]]*np.exp(1j*phase[pilot_at[hold]])*pilot[hold])**2)),
        phase_jump_max_rad=float(np.max(abs(phase_jumps))) if len(phase_jumps) else 0.,
        receiver_algorithm='buffered_pilot_only_fractional_timing_CFO_phase_no_QAM_decisions',
        pilot_estimate_note='held-out phase-fit pilots; timing acquisition uses all pilots; residual includes distortion and is not unbiased thermal SNR')
    # RX-selected sample locations/corrections returned for offline diagnostics only.
    transform = np.exp(-1j*(slope*np.arange(length)+phase))
    return data, raw[data_at], diagnostic, at, transform, data_at
