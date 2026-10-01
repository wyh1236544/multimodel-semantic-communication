"""Predeclared quick screen, not a final paper protocol. Python 3.8 compatible."""
VERSION = 'V10-token-guidance-quick-v1'
VARIANTS = {'A': (False, False), 'B': (True, False),
            'C': (False, True), 'D': (True, True), 'OLD': (False, True)}
DIAGNOSTICS = ('tx_wrong', 'rx_wrong', 'rx_zero')
SETTINGS = dict(fs=64e9, sps=4, rolloff=.1, span=32, qbits=8,
                quantizer_limit=4., dac_bits=8, packet_bytes=512,
                guard_samples=256, max_attempts=1, distance_m=1000.)
TRAIN_CONDITIONS = (('awgn', 20.), ('fso_moderate', 20.),
                    ('awgn', 30.), ('fso_moderate', 30.))


def donor_map(ids):
    ids = list(map(int, ids))
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError('Need at least two unique image IDs for real mismatching')
    return dict(zip(ids, ids[1:]+ids[:1]))


def conditions(smoke=False):
    if smoke:
        return [('quantized_clean', 'awgn', 40.), ('physical', 'awgn', 20.)]
    return [('quantized_clean', 'awgn', 40.)]+[
        ('physical', p, s) for p in ('fso_moderate', 'fso_strong')
        for s in (0., 10., 20.)]


def expected_rows(samples, smoke=False):
    return samples*len(conditions(smoke))*(len(VARIANTS)+len(DIAGNOSTICS))
