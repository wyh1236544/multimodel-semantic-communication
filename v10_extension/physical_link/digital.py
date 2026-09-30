"""Actual source bits, Gray QAM, CRC32 and terminated soft-Viterbi FEC.

NumPy only. No semantic truth is passed to the receiver. The wire format and
quantizer range are public fixed configuration, not per-image side information.
"""
import math
import struct
import zlib
import numpy as np

HEADER = struct.Struct('>4sIBB')
POLYS = (0o171, 0o133)
BLOCK = 1024


def bytes_bits(data):
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_bytes(bits):
    if len(bits) % 8:
        raise ValueError('Byte alignment required')
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()


def packet_size(count, qbits):
    return (HEADER.size + 4) * 8 + math.ceil(count * qbits / 8) * 8


def encode_source(values, qbits=8, limit=4.):
    x = np.asarray(values, dtype=float).reshape(-1)
    if qbits not in (4, 6, 8) or limit <= 0 or not np.isfinite(x).all():
        raise ValueError('Invalid quantizer input')
    levels = (1 << qbits) - 1
    indices = np.rint((np.clip(x, -limit, limit) + limit) * levels / (2 * limit)).astype(np.uint8)
    bits = ((indices[:, None] >> np.arange(qbits - 1, -1, -1)) & 1).astype(np.uint8).ravel()
    bits = np.pad(bits, (0, (-len(bits)) % 8))
    body = HEADER.pack(b'SC02', len(x), qbits, 0) + bits_bytes(bits)
    packet = bytes_bits(body + struct.pack('>I', zlib.crc32(body) & 0xffffffff))
    quantized = indices.astype(float) * (2 * limit / levels) - limit
    return packet, quantized, float(np.mean(np.abs(x) > limit))


def decode_source(bits, count, qbits=8, limit=4.):
    """On ANY format/CRC failure return erasure, never repair using TX truth."""
    if len(bits) != packet_size(count, qbits):
        return np.zeros(count), False
    data = bits_bytes(bits)
    body, crc = data[:-4], data[-4:]
    if zlib.crc32(body) & 0xffffffff != struct.unpack('>I', crc)[0]:
        return np.zeros(count), False
    if HEADER.unpack(body[:HEADER.size]) != (b'SC02', count, qbits, 0):
        return np.zeros(count), False
    q = bytes_bits(body[HEADER.size:])[:count * qbits].reshape(count, qbits)
    idx = (q * (1 << np.arange(qbits - 1, -1, -1))).sum(1)
    return idx.astype(float) * (2 * limit / ((1 << qbits) - 1)) - limit, True


def scramble(size):
    # A fixed public additive scrambler; not encryption or a shared payload.
    return np.random.default_rng(1907).integers(0, 2, size, dtype=np.uint8)


def alphabet(order):
    if order not in (4, 16):
        raise ValueError('Only QPSK and 16-QAM implemented')
    width = int(math.log2(order))
    labels = ((np.arange(order)[:, None] >> np.arange(width - 1, -1, -1)) & 1).astype(np.uint8)
    axis = np.array([-1., 1.]) if order == 4 else np.array([-3., -1., 3., 1.])
    half = width // 2
    weights = 1 << np.arange(half - 1, -1, -1)
    symbols = (axis[labels[:, :half] @ weights] + 1j * axis[labels[:, half:] @ weights]) / math.sqrt(2 if order == 4 else 10)
    return symbols, labels


def modulate(bits, order):
    points, labels = alphabet(order)
    width = labels.shape[1]
    padded = np.pad(np.asarray(bits, dtype=np.uint8), (0, (-len(bits)) % width))
    index = padded.reshape(-1, width) @ (1 << np.arange(width - 1, -1, -1))
    return points[index]


def demodulate(symbols, order, variance):
    """Max-log bit LLR: positive means bit zero. Receiver variance only."""
    points, labels = alphabet(order)
    dist = np.abs(np.asarray(symbols)[:, None] - points[None, :]) ** 2
    llr = np.stack([dist[:, labels[:, i] == 1].min(1) - dist[:, labels[:, i] == 0].min(1)
                    for i in range(labels.shape[1])], axis=1)
    return (llr / max(float(variance), 1e-8)).ravel()


def trellis():
    dest = np.arange(64)
    previous = np.stack([(dest & 31) << 1, ((dest & 31) << 1) | 1], axis=1)
    registers = previous | ((dest[:, None] >> 5) << 6)
    outputs = np.array([[[[bin(int(r) & p).count('1') & 1 for p in POLYS]
                         for r in row] for row in registers]])[0]
    return previous, outputs


def fec_size(size, block=BLOCK):
    return math.ceil(size / block) * (block + 6) * 2


def fec_encode(bits, block=BLOCK):
    bits = np.asarray(bits, dtype=np.uint8)
    data = np.pad(bits, (0, (-len(bits)) % block)).reshape(-1, block)
    data = np.pad(data, ((0, 0), (0, 6)))
    state = np.zeros(len(data), dtype=np.int64)
    output = np.empty((len(data), block + 6, 2), dtype=np.uint8)
    lookup = np.array([[bin(r & p).count('1') & 1 for p in POLYS] for r in range(128)], dtype=np.uint8)
    for t in range(block + 6):
        register = state | (data[:, t].astype(np.int64) << 6)
        output[:, t] = lookup[register]
        state = register >> 1
    return output.ravel()


def fec_decode(llr, size, block=BLOCK):
    if len(llr) != fec_size(size, block):
        raise ValueError('Wrong coded frame length')
    observations = np.asarray(llr, dtype=float).reshape(-1, block + 6, 2)
    previous, outputs = trellis()
    sign = 1. - 2. * outputs
    batch = len(observations)
    metric = np.full((batch, 64), np.inf)
    metric[:, 0] = 0.
    paths = np.empty((block + 6, batch, 64), dtype=np.uint8)
    for t in range(block + 6):
        branch = -.5 * np.einsum('bj,dcj->bdc', observations[:, t], sign)
        candidates = metric[:, previous] + branch
        selected = candidates.argmin(2)
        paths[t] = selected
        metric = np.take_along_axis(candidates, selected[..., None], 2)[..., 0]
        metric -= metric.min(1, keepdims=True)
    state = np.zeros(batch, dtype=np.int64)  # terminated zero state
    decoded = np.empty((batch, block + 6), dtype=np.uint8)
    rows = np.arange(batch)
    for t in range(block + 5, -1, -1):
        decoded[:, t] = state >> 5
        state = previous[state, paths[t, rows, state]]
    return decoded[:, :block].ravel()[:size]
