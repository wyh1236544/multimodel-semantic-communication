"""Public-pilot bounded acquisition, joint clock fit and validated phase tracking."""
import math
import numpy as np
from physical_link.waveform import preamble, convolve
from optical_realism_v3.hardware import interpolate


def layout(count, h):
    if type(count) is not int or count < 0: raise ValueError("Invalid payload length")
    pilot_at = list(range(256)); pilot = list(preamble(256)); groups = [(0, 256)]
    data_at = []; pos = 256
    for offset in range(0, count, h["pilot_interval"]):
        size = min(h["pilot_interval"], count-offset)
        data_at.extend(range(pos, pos+size)); pos += size
        pilot_at.extend(range(pos, pos+h["pilot_block"]))
        pilot.extend(preamble(h["pilot_block"]))
        groups.append((len(pilot)-h["pilot_block"], len(pilot))); pos += h["pilot_block"]
    return np.asarray(data_at, int), np.asarray(pilot_at, int), np.asarray(pilot), groups, pos


def frame_symbols(symbols, h):
    data_at, pilot_at, pilot, groups, count = layout(len(symbols), h)
    frame = np.zeros(count, complex); frame[data_at] = symbols; frame[pilot_at] = pilot
    return frame, data_at, pilot_at, pilot, groups


def coherent_grid(y, known, index, frequencies, rs):
    cross = (y*np.conj(known)) @ np.exp(-2j*np.pi*index[:, None]*frequencies[None, :]/rs)
    return abs(cross)**2/(len(index)*np.sum(abs(y)**2, axis=-1, keepdims=True)+1e-300)


def acquire(mf, pulse, pilot, h):
    fit = np.arange(0, 256, 2)
    nominal = 2*pulse.half+fit*pulse.sps
    delays = np.arange(0, h["timing_search_samples"]+.01, .25)
    y = interpolate(mf, delays[:, None]+nominal[None, :])
    cap = h["cfo_capture_hz"]
    frequencies = np.linspace(-cap, cap, int(math.ceil(2*cap/h["cfo_grid_hz"]))+1)
    score = coherent_grid(y, pilot[fit], fit, frequencies, pulse.rs)
    ti, fi = np.unravel_index(np.argmax(score), score.shape)
    delay, frequency = float(delays[ti]), float(frequencies[fi])
    step = float(frequencies[1]-frequencies[0])
    for _ in range(2):
        frequencies = np.clip(frequency+np.linspace(-step, step, 17), -cap, cap)
        delays = np.clip(delay+np.arange(-.25, .251, .125), 0, h["timing_search_samples"])
        y = interpolate(mf, delays[:, None]+nominal[None, :])
        score = coherent_grid(y, pilot[fit], fit, frequencies, pulse.rs)
        ti, fi = np.unravel_index(np.argmax(score), score.shape)
        delay, frequency = float(delays[ti]), float(frequencies[fi]); step /= 8
    return delay, frequency, float(score[ti, fi])


def clock_fit(mf, pulse, pilot_at, pilot, groups, start, slope, h):
    if len(groups) < 3: return start, 0., None
    selected = np.unique(np.linspace(0, len(groups)-1, min(32, len(groups))).astype(int))
    use = []
    for j in selected:
        lo, hi = groups[j]
        even = np.arange(lo, hi, 2)
        use.append(even[np.linspace(0, len(even)-1, 16).astype(int)])
    use = np.asarray(use); ix, known = pilot_at[use], pilot[use]
    def scan(delays, ppms):
        delays, ppms = np.meshgrid(delays, ppms, indexing="ij")
        at = (2*pulse.half+delays.ravel()[:, None, None]
              +pulse.sps*(1+ppms.ravel()[:, None, None]*1e-6)*ix[None])
        y = interpolate(mf, at)
        z = y*np.conj(known)[None]*np.exp(-1j*slope*ix)[None]
        scores = (abs(z.sum(-1))**2/(16*np.sum(abs(y)**2, -1)+1e-300)).mean(-1)
        i = int(np.argmax(scores))
        return float(delays.ravel()[i]), float(ppms.ravel()[i]), float(scores[i])
    delay, ppm, score = scan(start+np.arange(-.5, .501, .25),
                            np.linspace(-h["timing_max_ppm"], h["timing_max_ppm"], 41))
    delay, ppm, score = scan(delay+np.arange(-.25, .251, .125),
                            np.clip(ppm+np.arange(-10, 10.01, 1), -h["timing_max_ppm"], h["timing_max_ppm"]))
    return delay, ppm, score


def receive(observed, pulse, count, h):
    """No source payload, simulator CFO/clock/phase or fading truth is read."""
    if pulse.q != 1: raise ValueError("Integer nominal SPS required")
    data_at, pilot_at, pilot, groups, length = layout(count, h)
    mf = convolve(observed, pulse.kernels[0][::-1])
    start, frequency, acquisition_score = acquire(mf, pulse, pilot, h)
    slope = 2*math.pi*frequency/pulse.rs
    delay, ppm, clock_score = clock_fit(mf, pulse, pilot_at, pilot, groups, start, slope, h)
    at = 2*pulse.half+delay+pulse.sps*(1+ppm*1e-6)*np.arange(length)
    raw = interpolate(mf, at)
    corrected = raw*np.exp(-1j*slope*np.arange(length))
    gains, centers, counts = [], [], []
    for lo, hi in groups:
        use = np.arange(lo, hi, 2)
        gains.append(np.mean(corrected[pilot_at[use]]*np.conj(pilot[use])))
        centers.append(float(pilot_at[use].mean())); counts.append(len(use))
    gains, centers, counts = np.asarray(gains), np.asarray(centers), np.asarray(counts)
    if len(gains) > 2:
        cross = gains[1:]*np.conj(gains[:-1]); spacing = np.diff(centers)
        grid = np.linspace(-h["cfo_grid_hz"]*2, h["cfo_grid_hz"]*2, 129)
        values = np.real(cross[:, None]*np.exp(-2j*np.pi*spacing[:, None]*grid[None]/pulse.rs)).sum(0)
        residual = float(grid[np.argmax(values)])
        frequency = float(np.clip(frequency+residual, -h["cfo_capture_hz"], h["cfo_capture_hz"]))
        slope = 2*math.pi*frequency/pulse.rs
        corrected = raw*np.exp(-1j*slope*np.arange(length))
        gains = np.asarray([np.mean(corrected[pilot_at[np.arange(lo, hi, 2)]]
                         *np.conj(pilot[np.arange(lo, hi, 2)])) for lo, hi in groups])
    if len(gains) > 1:
        spacing = float(np.median(np.diff(centers)))
        distance = abs(centers[:, None]-centers[None, :])/max(spacing, 1)
        radius = h["phase_smoothing_blocks"]//2
        weights = np.exp(-.5*(distance/max(radius, 1))**2)*(distance <= radius+.01)
        weights *= counts[None]
        smooth = (weights @ gains)/np.maximum(weights.sum(1), 1)
    else: smooth = gains
    phases = np.unwrap(np.angle(smooth))
    phase = np.interp(np.arange(length), centers, phases)
    fit = np.arange(0, len(pilot), 2)
    gain = float(abs(np.mean(corrected[pilot_at[fit]]*np.conj(pilot[fit])
                             *np.exp(-1j*phase[pilot_at[fit]]))))
    amplitude = np.full(length, max(gain, 1e-15))
    eq = corrected*np.exp(-1j*phase)/amplitude
    hold = np.arange(1, len(pilot), 2)
    variance = max(float(np.mean(abs(eq[pilot_at[hold]]-pilot[hold])**2)), 1e-12)
    snr = -10*math.log10(variance)
    validation = np.arange(1, 256, 2); pv = eq[pilot_at[validation]]
    sync_score = float(abs(np.sum(pv*np.conj(pilot[validation])))**2/
                       (len(validation)*np.sum(abs(pv)**2)+1e-300))
    block_errors = [np.mean(eq[pilot_at[np.arange(lo+1, hi, 2)]]
                           *np.conj(pilot[np.arange(lo+1, hi, 2)]))-1 for lo, hi in groups]
    floor = float(np.mean(abs(np.asarray(block_errors))**2))
    jumps = np.angle(smooth[1:]*np.conj(smooth[:-1]))
    reasons = []
    if sync_score < h["minimum_sync_score"]: reasons.append("preamble_validation")
    if snr < h["minimum_pilot_snr_db"]: reasons.append("pilot_residual")
    if len(jumps) and np.max(abs(jumps)) > h["maximum_pilot_phase_jump_rad"]:
        reasons.append("pilot_phase_ambiguity")
    if abs(ppm) >= h["timing_max_ppm"]*.999: reasons.append("clock_capture_boundary")
    if abs(frequency) >= h["cfo_capture_hz"]*.99: reasons.append("cfo_capture_boundary")
    locked = not reasons; pre_gate = eq[data_at]
    data = pre_gate.copy() if locked else np.zeros(count, complex)
    diagnostic = dict(sync_sample=float(delay), sync_score=sync_score,
        acquisition_fit_coherence=acquisition_score, clock_fit_coherence=clock_score,
        receiver_locked=locked, receiver_rejection_reasons=reasons,
        estimated_snr_db=snr, equalized_noise_variance=variance if locked else max(variance, 1e6),
        pilot_residual_variance_before_gate=variance, pilot_correlated_error_floor=floor,
        estimated_cfo_hz=frequency, estimated_clock_ppm=ppm,
        estimated_gain_real=float(gains[0].real), estimated_gain_imag=float(gains[0].imag),
        pilot_signal_energy_v2=float(np.sum(amplitude[pilot_at[hold]]**2)),
        pilot_noise_energy_v2=float(np.sum(abs(corrected[pilot_at[hold]]-
            amplitude[pilot_at[hold]]*np.exp(1j*phase[pilot_at[hold]])*pilot[hold])**2)),
        phase_jump_max_rad=float(np.max(abs(jumps))) if len(jumps) else 0.,
        receiver_algorithm="bounded_coherent_acquisition_joint_clock_smoothed_pilots_no_QAM_decisions",
        pilot_estimate_note="even pilots fit timing/CFO/phase; odd validate; distortion/finite-fit bias remains",
        synchronization_capture_hz=h["cfo_capture_hz"],
        pre_gate_constellation_preserved=True, reliability_is_not_certified_delivery=True)
    transform = np.exp(-1j*(slope*np.arange(length)+phase))
    return data, pre_gate, diagnostic, at, transform, data_at
