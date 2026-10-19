"""Actual image bytes/CRC/QC-LDPC/Gray QPSK or 16QAM, all overhead charged.

LDPC tables/decoder are reused unchanged from V19. Header uses QPSK R1/2,
two coded copies. Body gain is transmitted INSIDE its protected header.
"""
import io
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import zlib
import numpy as np
from PIL import Image
RESAMPLE = getattr(Image, 'Resampling', Image)
from docci_digital_compare_v19.ldpc import code
from docci_digital_compare_v19.digital import bits, byte_data, add_crc, strip_crc, qpsk, demod, permutation
from docci_pretrained_checks_v22.core import noise, sha

HEADER = struct.Struct('>4sBBIf')
PROFILES = ('qpsk12', 'qpsk34', 'qam16_12', 'qam16_34')
TOTAL, PILOTS, HEAD = 4128, 32, 648
QAM_LABELS = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1).astype(np.uint8)
GRAY = np.array([-3, -1, 3, 1])
QAM_POINTS = (GRAY[QAM_LABELS[:, 0]*2+QAM_LABELS[:, 1]]+
              1j*GRAY[QAM_LABELS[:, 2]*2+QAM_LABELS[:, 3]])/math.sqrt(10)

def spec(profile):
    if profile not in PROFILES:
        raise ValueError('Unknown digital profile')
    return code(profile[-2:]), 4 if profile.startswith('qam16') else 2

def allocation(length, profile):
    ld, bps = spec(profile)
    blocks = math.ceil((length+4)*8/ld.k)
    available = (TOTAL-PILOTS-HEAD)*bps
    if length < 0 or blocks*648 > available:
        raise ValueError('Image bytes do not fit; never truncate')
    return dict(blocks=blocks, padding=blocks*ld.k-(length+4)*8,
                coded_bits=blocks*648, body_bits=available, bps=bps)

def capacity(profile):
    ld, bps = spec(profile)
    return ((TOTAL-PILOTS-HEAD)*bps//648)*ld.k//8-4

def modulate(b, bps):
    if bps == 2:
        return qpsk(b)
    b = np.asarray(b, np.uint8).reshape(-1, 4)
    return QAM_POINTS[b @ np.array([8, 4, 2, 1])]

def llrs(y, n0, bps, gain=1.):
    if not math.isfinite(gain) or not .25 < gain < 4:
        raise ValueError('Invalid decoded body gain')
    if bps == 2:
        return demod(y/gain, n0/gain**2)
    d = np.abs(np.asarray(y)[:, None]-gain*QAM_POINTS[None])**2
    return np.stack([(d[:, QAM_LABELS[:, k] == 1].min(1)-
                      d[:, QAM_LABELS[:, k] == 0].min(1))/n0 for k in range(4)], 1).ravel()

def encode(payload, kind, profile):
    if kind not in (0, 1) or not isinstance(payload, bytes):
        raise ValueError('Public source codec required')
    a = allocation(len(payload), profile)
    ld, bps = spec(profile)
    packet = bits(add_crc(payload))
    info = np.pad(packet, (0, a['padding'])).reshape(a['blocks'], ld.k)
    coded = ld.encode(info).ravel()
    ix = permutation(len(coded))[np.arange(a['body_bits']) % len(coded)]
    body = modulate(coded[ix], bps)
    gain = float(np.float32(1/math.sqrt(float(np.mean(np.abs(body)**2)))))
    hb = bits(add_crc(HEADER.pack(b'V22I', 1, kind, len(payload), gain)))
    h = code('12').encode(np.pad(hb, (0, 324-len(hb)))[None]).ravel()
    frame_bits = np.concatenate((np.zeros(PILOTS*2, np.uint8), np.tile(h, 2), coded[ix]))
    frame = np.concatenate((qpsk(frame_bits[:(PILOTS+HEAD)*2]), gain*body))
    if len(frame) != TOTAL or abs(float(np.mean(np.abs(frame)**2))-1) > 2e-6:
        raise RuntimeError('Equal total frame energy violated')
    return frame, dict(payload_bytes=len(payload), total_symbols=TOTAL,
        image_payload_symbols=4096, pilots=PILOTS, header_symbols=HEAD,
        header_bits=len(hb), crc_bits=32, body_gain=gain,
        repeated_body_bits=a['body_bits']-a['coded_bits'], **a), dict(bits=frame_bits, packet=packet)

def receive(y, profile):
    """No original, true length, TX gain or packet bits are RX arguments."""
    y = np.asarray(y)
    if y.shape != (TOTAL,) or not np.isfinite(y).all():
        raise ValueError('Invalid observed frame')
    pilot = qpsk(np.zeros(PILOTS*2, np.uint8))
    n0 = max(float(np.mean(np.abs(y[:PILOTS]-pilot)**2)), 1e-8)
    result = dict(status='HEADER_REJECT', payload=b'', kind=None, estimated_noise_power=n0,
                  header_crc=False, packet_crc=False, decoded_bits=None, gain=None)
    hi = code('12').decode(demod(y[PILOTS:PILOTS+HEAD], n0).reshape(2, 648).sum(0)[None], 40, .8, 180)[0]
    try:
        magic, version, kind, length, gain = HEADER.unpack(strip_crc(byte_data(hi[0, :144])))
        if magic != b'V22I' or version != 1 or kind not in (0, 1):
            raise ValueError('Header syntax')
        a = allocation(length, profile)
        if not math.isfinite(gain) or not .25 < gain < 4:
            raise ValueError('Body scale')
    except (ValueError, struct.error):
        return result
    result.update(header_crc=True, gain=gain, status='BODY_REJECT')
    ld, bps = spec(profile)
    ix = permutation(a['coded_bits'])[np.arange(a['body_bits']) % a['coded_bits']]
    combined = np.bincount(ix, weights=llrs(y[PILOTS+HEAD:], n0, bps, gain), minlength=a['coded_bits'])
    decoded = ld.decode(combined.reshape(a['blocks'], 648), 40, .8, a['padding'])[0].ravel()[:(length+4)*8]
    result['decoded_bits'] = decoded
    try:
        payload = strip_crc(byte_data(decoded))
    except ValueError:
        return result
    result.update(status='ACCEPT', packet_crc=True, payload=payload, kind=kind)
    return result

def channel(x, eid, snr, repeat):
    e = noise(eid, snr, repeat, 'image', 8192).reshape(-1, 2)
    p = noise(eid, snr, repeat, 'image_pilot', 64).reshape(-1, 2)
    n = np.concatenate(((p[:, 0]+1j*p[:, 1]), (e[:, 0]+1j*e[:, 1])))/math.sqrt(2)
    return x+n

def binary(path, expected):
    p = Path(path).resolve(strict=True)
    if sha(p) != expected:
        raise ValueError('Pinned BPG executable differs')
    return str(p)

def bpg_encode(image, q, executable, scratch):
    with tempfile.TemporaryDirectory(dir=str(scratch), prefix='v22_bpg_') as d:
        src, dst = Path(d)/'input.png', Path(d)/'output.bpg'
        image.save(src)
        subprocess.run([executable, '-q', str(q), '-o', str(dst), str(src)],
                       check=True, timeout=60, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return dst.read_bytes()

def reconstruct(payload, kind, decoder=None, scratch=None):
    if kind == 0:
        with Image.open(io.BytesIO(payload)) as im:
            if im.format != 'JPEG' or max(im.size) > 512:
                raise ValueError('Invalid JPEG geometry')
            return im.convert('RGB').resize((512, 512), RESAMPLE.BICUBIC)
    if decoder is None:
        raise ValueError('Explicit BPG decoder required; no JPEG fallback')
    with tempfile.TemporaryDirectory(dir=str(scratch), prefix='v22_bpg_rx_') as d:
        src, dst = Path(d)/'input.bpg', Path(d)/'output.png'
        src.write_bytes(payload)
        subprocess.run([decoder, '-o', str(dst), str(src)], check=True, timeout=60,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with Image.open(dst) as im:
            if max(im.size) > 512:
                raise ValueError('Invalid BPG geometry')
            return im.convert('RGB').resize((512, 512), RESAMPLE.BICUBIC)

def source_encode(image, profile, codec, scratch, encoder=None, budget=lambda: None):
    """Fixed rate-only selection, no SNR/PSNR/evaluation-quality search.

    Prefer largest public canvas resolution, then highest fitting JPEG quality
    / lowest BPG QP. Downsampling is explicit and is NOT a native-resolution BPG claim.
    """
    limit = capacity(profile)
    attempted = 0
    for size in (512, 256, 128, 64, 32):
        view = image.resize((size, size), RESAMPLE.LANCZOS)
        qs = range(95, 0, -1) if codec == 'jpeg' else range(52)
        for q in qs:
            budget()
            if codec == 'jpeg':
                f = io.BytesIO()
                view.save(f, format='JPEG', quality=q, optimize=True, subsampling=2)
                payload = f.getvalue()
            elif codec == 'bpg' and encoder:
                payload = bpg_encode(view, q, encoder, scratch)
            else:
                raise ValueError('Requested codec unavailable; never relabel JPEG as BPG')
            attempted += 1
            if len(payload) <= limit:
                return payload, dict(codec=codec, encoded_size=size, quality_or_QP=q,
                                     byte_limit=limit, candidates_tested=attempted, resized=size != 512)
    raise ValueError('NO_FIT: no registered complete source bitstream fits')
