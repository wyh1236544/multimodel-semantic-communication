"""Explicit accounting for a normalized fixed-power rate sweep.

Rates are assumed symbols/second, not measured hardware throughput. At fixed
received signal power and white thermal-noise spectral density, noise variance
scales with receiver bandwidth, here assumed proportional to symbol rate.
"""
from __future__ import annotations

import math


def effective_snr_db(reference_snr_db, symbol_rate, reference_rate):
    if not all(math.isfinite(float(x)) for x in (reference_snr_db, symbol_rate, reference_rate)):
        raise ValueError('SNR and symbol rates must be finite')
    if symbol_rate <= 0 or reference_rate <= 0:
        raise ValueError('Symbol rates must be positive')
    return float(reference_snr_db) - 10.0 * math.log10(float(symbol_rate) / float(reference_rate))


def serialization(forward_uses, forward_rate, reverse_uses, reverse_rate):
    if min(forward_uses, forward_rate, reverse_rate) <= 0 or reverse_uses < 0:
        raise ValueError('Invalid serialization budget')
    forward_seconds = float(forward_uses) / float(forward_rate)
    reverse_seconds = float(reverse_uses) / float(reverse_rate)
    return {
        'forward_seconds': forward_seconds,
        'reverse_seconds': reverse_seconds,
        'sequential_link_seconds': forward_seconds + reverse_seconds,
        'ideal_sequential_pairs_per_second': 1.0 / (forward_seconds + reverse_seconds),
    }
