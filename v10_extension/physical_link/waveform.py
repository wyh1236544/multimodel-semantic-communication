"""Sampled coherent I/Q link at a fixed DAC/ADC clock, no KK.

Direct fractional-phase RRC synthesis evaluates the pulse at actual AWG sample
times. 14 GBaud at 64 GSa/s therefore has EXACTLY 32/7 samples/symbol, not a
rounded integer. RX assumes a shared symbol clock; frame start is public-preamble
correlation at sample precision. No carrier-frequency or clock drift is modeled.
"""
from fractions import Fraction
import math
import numpy as np


def rrc(t, alpha):
    t = np.asarray(t, dtype=float)
    if alpha == 0:
        return np.sinc(t)
    out = np.empty_like(t)
    zero = np.isclose(t, 0., atol=1e-12)
    singular = np.isclose(np.abs(t), 1 / (4 * alpha), atol=1e-12)
    regular = ~(zero | singular)
    u = t[regular]
    out[regular] = (np.sin(math.pi * u * (1 - alpha)) + 4 * alpha * u * np.cos(math.pi * u * (1 + alpha))) / (math.pi * u * (1 - (4 * alpha * u) ** 2))
    out[zero] = 1 + alpha * (4 / math.pi - 1)
    out[singular] = alpha / math.sqrt(2) * ((1 + 2 / math.pi) * math.sin(math.pi / (4 * alpha)) + (1 - 2 / math.pi) * math.cos(math.pi / (4 * alpha)))
    return out


def convolve(a, b):
    size = len(a) + len(b) - 1
    n = 1 << (size - 1).bit_length()
    return np.fft.ifft(np.fft.fft(a, n) * np.fft.fft(b, n))[:size]


class Pulse:
    def __init__(self, fs=64e9, rs=14e9, alpha=.01, span=128):
        if not all(math.isfinite(v) and v > 0 for v in (fs, rs)) or not 0 <= alpha <= 1:
            raise ValueError('Invalid waveform rates/rolloff')
        if span < 8 or span > 512 or span % 2 or fs < (1 + alpha) * rs:
            raise ValueError('Invalid RRC span or sampling/bandwidth constraint')
        ratio = Fraction(fs / rs).limit_denominator(128)
        if abs(float(ratio) - fs / rs) > 1e-9:
            raise ValueError('Clock ratio must be rational with denominator <=128')
        self.fs, self.rs, self.alpha, self.span = fs, rs, alpha, span
        self.p, self.q = ratio.numerator, ratio.denominator
        self.sps = float(ratio)
        self.half = math.ceil(span * self.sps / 2)
        offsets = np.arange(-self.half, self.half + 1)
        self.kernels = []
        for phase in range(self.q):
            h = rrc((offsets - phase / self.q) / self.sps, alpha)
            self.kernels.append(h / math.sqrt(np.sum(h * h)))

    def indices(self, count):
        numerator = np.arange(count, dtype=np.int64) * self.p
        return numerator // self.q, numerator % self.q

    def shape(self, symbols):
        symbols = np.asarray(symbols, dtype=complex)
        if not len(symbols):
            raise ValueError('Empty waveform')
        starts, phases = self.indices(len(symbols))
        wave = np.zeros(int(starts[-1]) + 2 * self.half + 1, dtype=complex)
        for phase, h in enumerate(self.kernels):
            indices = np.flatnonzero(phases == phase)
            if not len(indices):
                continue
            impulse = np.zeros(int(starts[-1]) + 1, dtype=complex)
            impulse[starts[indices]] = symbols[indices]
            wave += convolve(impulse, h)
        return wave

    def match(self, wave, count, start=0):
        starts, phases = self.indices(count)
        received = np.zeros(count, dtype=complex)
        for phase, h in enumerate(self.kernels):
            indices = np.flatnonzero(phases == phase)
            if not len(indices):
                continue
            filtered = convolve(wave, h[::-1])
            at = start + starts[indices] + 2 * self.half
            if at.max() >= len(filtered):
                raise ValueError('Truncated received frame')
            received[indices] = filtered[at]
        return received


def preamble(length=256):
    rng = np.random.default_rng(731)
    return ((2 * rng.integers(0, 2, length) - 1) + 1j * (2 * rng.integers(0, 2, length) - 1)) / math.sqrt(2)


def dac(wave, sps, bits=8):
    if bits not in (8, 10, 12, 16):
        raise ValueError('Unsupported DAC resolution')
    fullscale = 4.  # PUBLIC fixed scale, not source-dependent or rate-dependent
    normalized = wave / fullscale
    clipped = float(np.mean((np.abs(normalized.real) > 1) | (np.abs(normalized.imag) > 1)))
    levels = (1 << bits) - 1
    def axis(x):
        return np.rint((np.clip(x, -1, 1) + 1) * levels / 2) * 2 / levels - 1
    codes = axis(normalized.real) + 1j * axis(normalized.imag)
    return codes * fullscale, codes, clipped


def transmit(symbols, pulse, gain, noise_var, rng, dac_bits=8, delay=7, bandwidth=None):
    """TX diagnostics remain OUTSIDE receive(); h is only used in propagation."""
    # Unit-energy discrete RRC coefficients become fixed-average-power samples.
    # At fixed Fs, thermal sample-noise PSD is unchanged across Rs:
    # Nsample = (Rs/Rref)*Nref * (Fs/Rs) = Nref*Fs/Rref.
    wave = pulse.shape(symbols) * math.sqrt(pulse.sps)
    converted, codes, clipped = dac(wave, pulse.sps, dac_bits)
    frontend = converted
    if bandwidth is not None:
        if not math.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError('Invalid assumed analog bandwidth')
        # Assumed smooth zero-phase low-pass, NOT measured AWG/MZM response.
        freq = np.fft.fftfreq(len(converted), 1 / pulse.fs)
        response = 1 / np.sqrt(1 + (freq / bandwidth) ** 8)
        frontend = np.fft.ifft(np.fft.fft(converted) * response)
    observed = np.pad(gain * frontend, (delay, 0))
    noise = (rng.standard_normal(len(observed)) + 1j * rng.standard_normal(len(observed))) / math.sqrt(2)
    observed += noise * math.sqrt(noise_var * pulse.sps)
    diagnostic = {'awg_samples': len(wave), 'waveform_seconds': len(wave) / pulse.fs,
                  'dac_clip_fraction': clipped, 'dac_fullscale': 4.,
                  'sample_average_power': float(np.mean(np.abs(converted) ** 2)),
                  'sample_peak_power': float(np.max(np.abs(converted) ** 2)),
                  'waveform_energy_power_seconds': float(np.sum(np.abs(converted) ** 2) / pulse.fs),
                  'symbol_average_energy': float(np.mean(np.abs(symbols) ** 2))}
    return observed, diagnostic, codes


def receive(observed, pulse, count, training, search=32, raw=False):
    """Only public training, received waveform, length and public clock enter RX."""
    reference = pulse.shape(training)
    # Finite frame-start search, unknown delay; no TX delay parameter at RX.
    scores = [abs(np.vdot(reference, observed[i:i + len(reference)]))
              if i + len(reference) <= len(observed) else -1 for i in range(search + 1)]
    start = int(np.argmax(scores))
    matched = pulse.match(observed, count, start)
    pilot = matched[:len(training)]
    estimate = np.vdot(training, pilot) / np.vdot(training, training).real
    residual = pilot - estimate * training
    variance = max(float(np.mean(np.abs(residual) ** 2)), 1e-10)
    power = abs(estimate) ** 2
    received = matched * np.conj(estimate) / max(power, 1e-10)
    diagnostic = {'sync_sample': start, 'estimated_snr_db': float(10 * np.log10(max(power, 1e-12) / variance)),
                  'equalized_noise_variance': variance / max(power, 1e-10)}
    return (matched if raw else received), diagnostic
