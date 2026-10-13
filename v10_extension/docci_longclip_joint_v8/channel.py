"""Explicit differentiable FSO symbol proxy and separate full optical DSP eval."""
import copy
import math
import numpy as np
from pathlib import Path
from docci_longclip_joint_v8.core import geometry,read
from fso_jscc_v1.core import seed_for
from bart_latent_codec_v4.channel import proxy_numpy
from fso_fluctuating_v2.core import environment,paths
from optical_realism_v3.transport import RealisticTransport
from optical_realism_v3.reliability import combine
from optical_realism_v3.spec import sampled_count


def physical_config():
    c=read(Path(__file__).resolve().parents[1]/'fso_fluctuating_v2/config.json')
    wanted={'weak_p06','moderate_p18','moderate_p21','strong_p18'}
    c['conditions']=[v for v in c['conditions'] if v['id'] in wanted]
    return c


def resource(c,physical=None):
    physical=physical or physical_config(); cases,plan,_=environment(physical)
    w=copy.deepcopy(cases['weak']['waveform'])
    w.update(pilot_interval=plan['hardware']['pilot_interval'],pilot_block=plan['hardware']['pilot_block'])
    if (w['fs_hz'],w['sps'],w['rolloff'])!=(64e9,4,.1): raise ValueError('Fixed AWG64GSa/s/4SPS/RRC0.1 required')
    g=geometry(c); n=sampled_count(0,w)+sampled_count(g['payload_complex_symbols'],w)
    if any(n/w['fs_hz']>case['turbulence']['coherence_time_s'] for case in cases.values()): raise ValueError('New frame exceeds frozen-within-frame fading coherence')
    return dict(g,scheduled_samples=n,scheduled_seconds=n/w['fs_hz'],slot_samples=n,
        occupied_complex_bandwidth_hz=(1+w['rolloff'])*w['fs_hz']/w['sps'],
        includes_probe_pilots_preambles_filter_tails_and_guards=True,
        scope='NEW V8 frame, not old3072 budget; same cap/settings not equal actual energy')


def training_receive(wire,c,step,eid,noisy):
    import torch
    if not noisy:
        return wire,torch.ones(wire.shape[0],device=wire.device,dtype=torch.bool),wire.new_full((wire.shape[0],),40.),wire.new_ones(wire.shape[0]),None
    rng=np.random.default_rng(seed_for(c['seed'],'v8-condition',step,eid))
    profile=tuple(sorted(c['proxy_profiles']))[int(rng.integers(0,len(c['proxy_profiles'])))]
    snr=float(rng.choice(c['proxy_snr_db']))
    zero=np.zeros(wire.shape[1],np.float32)
    noise,coefficient,_,stats=proxy_numpy(zero,c,profile,snr,seed_for(c['seed'],'v8-field',step,eid),seed_for(c['seed'],'v8-noise',step,eid))
    pair=wire.reshape(wire.shape[0],-1,2); z=torch.complex(pair[...,0],pair[...,1])/math.sqrt(2)
    z=z*torch.tensor(coefficient,device=wire.device,dtype=torch.complex64)
    received=torch.stack((z.real,z.imag),-1).reshape_as(wire)*math.sqrt(2)+torch.as_tensor(noise,device=wire.device)[None]
    estimated=stats['pilot_snr_db']; rel=1/(1+10**(-max(-40.,min(40.,estimated))/10))
    stats.update(pilot_estimate_numeric_floor=estimated<=-100.,reliability_scope='pilot-SNR heuristic, not calibrated delivery',
        actual_optical_power=None,temporal_waveform_DSP_not_in_training=True)
    return received,torch.ones(wire.shape[0],device=wire.device,dtype=torch.bool),wire.new_full((wire.shape[0],),estimated),wire.new_full((wire.shape[0],),rel),stats


def send(c,physical,condition,path,seed,wire):
    cases,plan,_=environment(physical); res=resource(c,physical)
    t=RealisticTransport(cases[condition['weather']],seed_for(c['seed'],'unused-v8-fade'),seed,
        condition['launch_signal_dbm'],res['slot_samples'],plan['laser_max_dbm'],plan['hardware'])
    t.path_draw=copy.deepcopy(path); pilot=float(t.probe())
    x=np.asarray(wire,np.float32)
    if x.shape!=(geometry(c)['total_reals'],) or not np.isfinite(x).all(): raise ValueError('Invalid V8 full payload')
    z=(x[::2]+1j*x[1::2])/math.sqrt(2)
    observed,event=t.burst(z,'features',policy='analog')
    restored,reliability=combine(observed[None],event['pilot_residual_variance_before_gate'],event['pilot_correlated_error_floor'],event['receiver_locked'],'lmmse')
    y=np.stack((restored.real,restored.imag),-1).reshape(-1)*math.sqrt(2)
    if not np.isfinite(y).all(): raise FloatingPointError('Nonfinite physical RX')
    stats=dict(t.totals(),receiver_available=bool(event['receiver_locked']),pilot_snr_db=pilot,
        analog_reliability=float(reliability['analog_reliability']),resources=res,payload_complex_symbols=len(z),
        analog_repeats=1,raw_utf8_wire_bytes=0,pre_fec_ber=None,post_fec_ber=None,analog_feature_mse=float(np.mean((x-y)**2)))
    return y.astype(np.float32),stats,t
