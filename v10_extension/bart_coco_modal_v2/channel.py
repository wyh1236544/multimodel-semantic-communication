"""Actual compact R=1 frames; no padded full-frame/sliced old-bank resource fiction."""
import copy
import math
import numpy as np
from optical_realism_v3.transport import RealisticTransport
from optical_realism_v3.reliability import combine
from optical_realism_v3.spec import sampled_count
from fso_fluctuating_v2.core import environment,as_complex,seed_for
from bart_coco_modal_v2.core import geometry,indices


class ModalTransport(RealisticTransport):
    def __init__(self,*args,mode,**kwargs):
        self.mode=mode; super().__init__(*args,**kwargs)

    def continuous(self,values,repeats=1,equalizer='lmmse'):
        x=np.asarray(values,np.float32); n=len(indices(self.mode)); g=geometry(self.mode)
        if x.shape!=(n,16) or not np.isfinite(x).all() or type(repeats) is not int or repeats!=1 or equalizer!='lmmse':
            raise ValueError('Public compact mode, R=1 and LMMSE required')
        source=as_complex(x); received,event=self.burst(source,'features',policy='analog')
        y,observed=combine(received[None,:],event['pilot_residual_variance_before_gate'],
            event['pilot_correlated_error_floor'],event['receiver_locked'],equalizer)
        restored=np.stack((y.real,y.imag),-1).reshape(n,16)*math.sqrt(2)
        if not np.isfinite(restored).all(): raise FloatingPointError('Nonfinite compact features')
        event.update(observed)
        stats=dict(self.totals(),mode=self.mode,layout=g,analog_receiver_available=bool(event['receiver_locked']),
            analog_reliability=float(observed['analog_reliability']),analog_repeats=1,
            analog_source_complex_symbols=g['complex_symbols'],analog_source_reals=g['reals'],
            payload_complex_symbols=g['complex_symbols'],raw_utf8_wire_bytes=0,pre_fec_ber=None,post_fec_ber=None,
            feature_object_crc_accepted=None,analog_equalizer='lmmse',
            analog_feature_mse=float(np.mean((restored-x)**2)),
            analog_combined_estimated_snr_db=-10*math.log10(max(observed['analog_effective_combined_variance'],1e-12)),
            crc_note='No binary analog CRC/BER; lock is not a correct-recovery guarantee')
        return restored.astype(np.float32),float(observed['analog_reliability']),stats


def resource(c,mode):
    cases,plan,cap=environment(c); w=copy.deepcopy(cases['weak']['waveform'])
    w.update(pilot_interval=plan['hardware']['pilot_interval'],pilot_block=plan['hardware']['pilot_block'])
    count=sampled_count(0,w)+sampled_count(geometry(mode)['complex_symbols'],w)
    if count>cap['slot_samples']: raise ValueError('Compact mode exceeds shared slot cap')
    return dict(geometry(mode),scheduled_samples=count,scheduled_seconds=count/w['fs_hz'],
        shared_slot_samples=cap['slot_samples'],fs_hz=w['fs_hz'],sps=w['sps'],symbol_rate_baud=w['fs_hz']/w['sps'],rolloff=w['rolloff'],
        note='Probe, preambles, tracking pilots, pulse tails and guards charged; shared peak/slot caps, actual energy differs')


def transport(c,condition,path,seed,mode):
    cases,plan,cap=environment(c); resource(c,mode)
    t=ModalTransport(cases[condition['weather']],seed_for(c['seed'],'unused-modal-fade'),seed,
        condition['launch_signal_dbm'],cap['slot_samples'],plan['laser_max_dbm'],plan['hardware'],mode=mode)
    t.path_draw=copy.deepcopy(path); return t


def clean(wire):
    import torch
    return wire,wire.new_ones(wire.shape[0]),wire.new_full((wire.shape[0],),40.),torch.ones(wire.shape[0],device=wire.device,dtype=torch.bool)


def papr(wire,c):
    import torch
    import torch.nn.functional as F
    from fso_jscc_v1.core import kernels
    axes=(wire.reshape(1,-1,2)/math.sqrt(2)).transpose(1,2).reshape(2,1,-1)
    taps=torch.as_tensor(kernels(c),device=wire.device).reshape(1,1,-1)
    wave=F.conv_transpose1d(axes,taps,stride=c['sps']).reshape(1,2,-1)
    power=wave.square().sum(1); relative=power/power.mean(1,keepdim=True).clamp_min(1e-8)
    return F.relu(relative-10**(c['papr_target_db']/10)).square().mean()
