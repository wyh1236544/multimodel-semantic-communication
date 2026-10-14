"""Real independent branch frames; no zero-padded joint-frame accounting."""
import copy
import math
import numpy as np
from docci_modal_v9.core import geometry
from docci_longclip_joint_v8.channel import physical_config
from bart_latent_codec_v4.channel import proxy_numpy
from fso_jscc_v1.core import seed_for
from fso_fluctuating_v2.core import environment
from optical_realism_v3.spec import sampled_count
from optical_realism_v3.transport import RealisticTransport
from optical_realism_v3.reliability import combine


def resource(c,branch,physical=None):
    physical=physical or physical_config(); cases,plan,_=environment(physical); w=copy.deepcopy(cases['weak']['waveform'])
    w.update(pilot_interval=plan['hardware']['pilot_interval'],pilot_block=plan['hardware']['pilot_block'])
    if (w['fs_hz'],w['sps'],w['rolloff'])!=(64e9,4,.1): raise ValueError('AWG64G/4SPS/RRC.1 fixed')
    g=geometry(c,branch); samples=sampled_count(0,w)+sampled_count(g['payload_complex_symbols'],w)
    if any(samples/w['fs_hz']>case['turbulence']['coherence_time_s'] for case in cases.values()): raise ValueError('Independent frame exceeds coherence')
    return dict(g,slot_samples=samples,scheduled_seconds=samples/w['fs_hz'],includes_probe_pilots_preambles_tails_guards=True,
        occupied_complex_bandwidth_hz=(1+w['rolloff'])*w['fs_hz']/w['sps'],same_resource_contract='within-branch stages share caps; image/text budgets differ')


def noisy(wire,c,update,ids):
    import torch
    received=[]; stats=[]
    for i,eid in enumerate(ids):
        rng=np.random.default_rng(seed_for(c['seed'],'v9-condition',update,eid)); profile=tuple(sorted(c['proxy_profiles']))[int(rng.integers(2))]; snr=float(rng.choice(c['proxy_snr_db']))
        zero,coefficient,_,record=proxy_numpy(np.zeros(wire.shape[1],np.float32),c,profile,snr,seed_for(c['seed'],'v9-field',update,eid),seed_for(c['seed'],'v9-noise',update,eid))
        pair=wire[i].reshape(-1,2); z=torch.complex(pair[:,0],pair[:,1])/math.sqrt(2)
        z=z*torch.tensor(coefficient,device=wire.device,dtype=torch.complex64)
        y=torch.stack((z.real,z.imag),-1).reshape(-1)*math.sqrt(2)+torch.as_tensor(zero,device=wire.device)
        received.append(y); stats.append(record)
    y=torch.stack(received); snr=y.new_tensor([r['pilot_snr_db'] for r in stats]); rel=(1+10**(-snr.clamp(-40,40)/10)).reciprocal()
    return y,torch.ones(wire.shape[0],device=wire.device,dtype=torch.bool),snr,rel,stats


def clean(wire): return wire,wire.new_ones(wire.shape[0],dtype=__import__('torch').bool),wire.new_full((wire.shape[0],),40.),wire.new_ones(wire.shape[0])


def send(c,branch,physical,condition,path,seed,wire):
    cases,plan,_=environment(physical); res=resource(c,branch,physical)
    t=RealisticTransport(cases[condition['weather']],seed_for(c['seed'],'unused-v9-fade'),seed,condition['launch_signal_dbm'],res['slot_samples'],plan['laser_max_dbm'],plan['hardware'])
    t.path_draw=copy.deepcopy(path); pilot=float(t.probe()); x=np.asarray(wire,np.float32)
    if x.shape!=(res['total_reals'],) or not np.isfinite(x).all(): raise ValueError('Independent modality payload size differs')
    z=(x[::2]+1j*x[1::2])/math.sqrt(2); observed,event=t.burst(z,'features',policy='analog')
    restored,reliability=combine(observed[None],event['pilot_residual_variance_before_gate'],event['pilot_correlated_error_floor'],event['receiver_locked'],'lmmse')
    y=np.stack((restored.real,restored.imag),-1).reshape(-1)*math.sqrt(2)
    if not np.isfinite(y).all(): raise FloatingPointError('Nonfinite physical receiver')
    stats=dict(t.totals(),receiver_available=bool(event['receiver_locked']),pilot_snr_db=pilot,analog_reliability=float(reliability['analog_reliability']),
        resources=res,payload_complex_symbols=len(z),analog_repeats=1,raw_utf8_wire_bytes=0,pre_fec_ber=None,post_fec_ber=None,
        analog_feature_mse=float(np.mean((x-y)**2)))
    return y.astype(np.float32),stats,t
