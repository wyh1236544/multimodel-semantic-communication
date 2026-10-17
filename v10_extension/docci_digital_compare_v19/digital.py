"""Real compressed bytes -> protected header/CRC -> LDPC -> unit-energy QPSK.

The receiver takes only observed symbols and public configuration/profile.
Source strings and transmitted lengths are evaluation-only, not RX arguments.
"""
import bz2
import hashlib
import lzma
import math
import struct
import zlib
import numpy as np
from docci_digital_compare_v19.ldpc import code

HEADER=struct.Struct('>4sBBII')
CODECS=('raw','zlib','bz2','lzma')
MAX_RAW=16384

def bits(data):return np.unpackbits(np.frombuffer(data,dtype=np.uint8))
def byte_data(x):
    if len(x)%8:raise ValueError('Unaligned bytes')
    return np.packbits(np.asarray(x,dtype=np.uint8)).tobytes()
def add_crc(x):return x+struct.pack('>I',zlib.crc32(x)&0xffffffff)
def strip_crc(x):
    if len(x)<4 or zlib.crc32(x[:-4])&0xffffffff!=struct.unpack('>I',x[-4:])[0]:raise ValueError('CRC failure')
    return x[:-4]
def compressions(text):
    raw=text.encode('utf-8')
    if len(raw)>MAX_RAW:raise ValueError('Bounded full description required')
    return [raw,zlib.compress(raw,9),bz2.compress(raw,compresslevel=9),lzma.compress(raw,preset=6)]
def decompress(kind,payload,size):
    if not 0<=kind<len(CODECS) or not 0<=size<=MAX_RAW:raise ValueError('Bounded source header')
    if kind==0:out=payload
    elif kind==1:
        obj=zlib.decompressobj();out=obj.decompress(payload,size+1)
        if not obj.eof or obj.unused_data or obj.unconsumed_tail:raise ValueError('Zlib framing')
    elif kind==2:
        obj=bz2.BZ2Decompressor();out=obj.decompress(payload,max_length=size+1)
        if not obj.eof or obj.unused_data:raise ValueError('BZ2 framing')
    else:
        obj=lzma.LZMADecompressor(memlimit=64*2**20);out=obj.decompress(payload,max_length=size+1)
        if not obj.eof or obj.unused_data:raise ValueError('LZMA framing')
    if len(out)!=size:raise ValueError('Decoded source length mismatch')
    return out.decode('utf-8',errors='strict')
def rate_of(profile):
    if profile not in ('qpsk_ldpc12','qpsk_ldpc34'):raise ValueError('Public profile differs')
    return profile[-2:]
def permutation(n):return np.random.Generator(np.random.PCG64(1907+n)).permutation(n)
def allocation(payload_bytes,profile,c):
    ld=code(rate_of(profile));n=math.ceil((payload_bytes+4)*8/ld.k)
    budget=(c['symbols']-c['pilot_symbols']-c['header_symbols'])*2
    if not 0<n or n*648>budget:raise ValueError('Source exceeds fixed public frame; no silent truncation')
    return dict(blocks=n,info_bits=n*ld.k,coded_bits=n*648,body_symbols=budget//2,
                body_repeat_floor=budget//(n*648),body_extra_repeated_bits=budget%(n*648),
                fec_padding_bits=n*ld.k-(payload_bytes+4)*8)
def qpsk(x):
    q=np.asarray(x,dtype=np.uint8).reshape(-1,2).astype(float)
    return ((1-2*q[:,0])+1j*(1-2*q[:,1]))/math.sqrt(2)
def demod(y,n0):
    if not math.isfinite(n0) or n0<=0:raise ValueError('Positive estimated noise power required')
    return np.stack((y.real,y.imag),axis=1).ravel()*(2*math.sqrt(2)/n0)
def encode(text,profile,c):
    candidates=compressions(text);kind=min(range(len(candidates)),key=lambda i:len(candidates[i]))
    payload=candidates[kind];a=allocation(len(payload),profile,c);ld=code(rate_of(profile))
    h=bits(add_crc(HEADER.pack(b'V19T',1,kind,len(candidates[0]),len(payload))))
    header_info=np.pad(h,(0,324-len(h)))[None]
    header=np.tile(code('12').encode(header_info).ravel(),2)
    packet=bits(add_crc(payload));info=np.pad(packet,(0,a['fec_padding_bits'])).reshape(a['blocks'],ld.k)
    coded=ld.encode(info).ravel();ix=permutation(len(coded))[np.arange(a['body_symbols']*2)%len(coded)]
    pilot=np.zeros(c['pilot_symbols']*2,dtype=np.uint8)
    frame=np.concatenate((pilot,header,coded[ix]))
    signal=qpsk(frame)
    if len(signal)!=c['symbols'] or not np.allclose(np.abs(signal)**2,1):raise RuntimeError('Frame resource/power invariant')
    return signal,dict(codec=CODECS[kind],compression_id=kind,raw_bytes=len(candidates[0]),compressed_bytes=len(payload),
        compression_lengths={k:len(v) for k,v in zip(CODECS,candidates)},
        header_payload_bits=len(h),header_public_padding_bits=324-len(h),header_coded_bits=1296,
        pilot_symbols=c['pilot_symbols'],header_symbols=c['header_symbols'],total_symbols=len(signal),
        crc_body_bits=32,profile=profile,**a),dict(frame_bits=frame,packet_bits=packet,body_info=info)
def receive(observed,profile,c):
    y=np.asarray(observed,dtype=complex)
    if y.shape!=(c['symbols'],) or not np.isfinite(y).all():raise ValueError('Observed frame geometry')
    pilot=qpsk(np.zeros(c['pilot_symbols']*2,dtype=np.uint8))
    estimate=max(float(np.mean(np.abs(y[:len(pilot)]-pilot)**2)),1e-8)
    result=dict(received_caption='',packet_crc_pass=False,header_crc_pass=False,reason=None,
                estimated_noise_power=estimate,decoded_payload_bits=None,body_decoder=None)
    # No source size, source mask, payload bits or TX metadata is available here.
    start=c['pilot_symbols'];end=start+c['header_symbols']
    hl=demod(y[start:end],estimate).reshape(2,648).sum(0)[None]
    h,diag=code('12').decode(hl,c['ldpc_iterations'],c['min_sum_alpha'],180)
    result['header_decoder']=diag
    try:
        head=strip_crc(byte_data(h[0,:144]))
        magic,version,kind,size,length=HEADER.unpack(head)
        if magic!=b'V19T' or version!=1 or kind>=4 or size>MAX_RAW or length>MAX_RAW+1024:raise ValueError('Header format')
        a=allocation(length,profile,c);result['header_crc_pass']=True
    except (ValueError,struct.error):result['reason']='HEADER_REJECT';return result
    ld=code(rate_of(profile));ix=permutation(a['coded_bits'])[np.arange(a['body_symbols']*2)%a['coded_bits']]
    llr=np.bincount(ix,weights=demod(y[end:],estimate),minlength=a['coded_bits']).reshape(a['blocks'],648)
    info,diag=ld.decode(llr,c['ldpc_iterations'],c['min_sum_alpha'],a['fec_padding_bits'])
    decoded=info.ravel()[:(length+4)*8];result.update(body_decoder=diag,decoded_payload_bits=decoded)
    try:
        compressed=strip_crc(byte_data(decoded));text=decompress(kind,compressed,size)
    except (ValueError,UnicodeError,zlib.error,lzma.LZMAError,EOFError,OSError):
        result['reason']='BODY_REJECT';return result
    result.update(received_caption=text,packet_crc_pass=True,reason='ACCEPT')
    return result
def transmit(signal,snr,epsilon):
    x=np.asarray(signal);e=np.asarray(epsilon,dtype=float).reshape(-1,2)
    n=(e[:,0]+1j*e[:,1])/math.sqrt(2);y=x+n
    power=float(np.mean(np.abs(x)**2));pn=float(np.mean(np.abs(n)**2))
    return y,dict(nominal_snr_db=snr,realized_snr_db=None if snr is None else 10*math.log10(power/pn),
                  signal_power=power,noise_power=pn,post_noise_normalization=False,
                  epsilon_sha256=hashlib.sha256(np.asarray(epsilon,dtype=np.float32).tobytes()).hexdigest())
def evaluate_packet(text,profile,c,snr,epsilon):
    signal,a,private=encode(text,profile,c);y,stats=transmit(signal,snr,epsilon)
    rx=receive(y,profile,c);decoded=rx.pop('decoded_payload_bits')
    # Post-FEC BER is conditional on a decoded header; never call a missing value zero.
    wrong=None if decoded is None or len(decoded)!=len(private['packet_bits']) else int(np.count_nonzero(decoded!=private['packet_bits']))
    raw_hard=(demod(y,1)<0).astype(np.uint8)
    return dict(**rx,**stats,allocation=a,pre_fec_errors=int(np.count_nonzero(raw_hard!=private['frame_bits'])),
        pre_fec_bits=len(raw_hard),post_fec_errors=wrong,
        post_fec_bits=0 if wrong is None else len(private['packet_bits']),
        undetected_wrong_packet=rx['packet_crc_pass'] and rx['received_caption']!=text),y
