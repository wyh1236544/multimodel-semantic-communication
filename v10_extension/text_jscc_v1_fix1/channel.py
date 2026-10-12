"""New short-frame waveform calibration. Never slice/relabel the old 2304 bank."""
import copy
import math
from pathlib import Path
import numpy as np
from optical_realism_v3.transport import RealisticTransport
from optical_realism_v3.reliability import combine
from optical_realism_v3.spec import sampled_count
from fso_fluctuating_v2.core import (environment,paths,seed_for,as_complex,fit_operator,
                                    apply_operator,OFFSETS)


class TextTransport(RealisticTransport):
    def continuous(self,values,repeats=1,equalizer='lmmse'):
        x=np.asarray(values,np.float32)
        if x.shape!=(48,16) or not np.isfinite(x).all() or repeats!=1 or equalizer!='lmmse':
            raise ValueError('Only finite48x16 text payload, R=1,LMMSE')
        source=as_complex(x)
        received,event=self.burst(source,'features',policy='analog')
        y,observed=combine(received[None,:],event['pilot_residual_variance_before_gate'],
            event['pilot_correlated_error_floor'],event['receiver_locked'],equalizer)
        restored=np.stack((y.real,y.imag),-1).reshape(48,16)*math.sqrt(2)
        if not np.isfinite(restored).all(): raise FloatingPointError('Nonfinite short frame')
        event.update(observed)
        stats=dict(self.totals(),analog_receiver_available=bool(event['receiver_locked']),
            analog_reliability=float(observed['analog_reliability']),analog_repeats=1,
            analog_source_complex_symbols=384,analog_source_reals=768,
            payload_complex_symbols=384,raw_utf8_wire_bytes=0,pre_fec_ber=None,post_fec_ber=None,
            analog_feature_mse=float(np.mean((restored-x)**2)),
            crc_note='Continuous features: no binary CRC/BER; synchronization is not delivery proof',
            layout='text-only48x16; no image/private/common-image payload',
            analog_equalizer='lmmse',analog_combined_estimated_snr_db=-10*math.log10(max(observed['analog_effective_combined_variance'],1e-12)))
        return restored.astype(np.float32),float(observed['analog_reliability']),stats


def short_resource(channel_config):
    cases,plan,_=environment(channel_config)
    w=copy.deepcopy(next(iter(cases.values()))['waveform'])
    w.update(pilot_interval=plan['hardware']['pilot_interval'],pilot_block=plan['hardware']['pilot_block'])
    count=sampled_count(0,w)+sampled_count(384,w)
    return dict(slot_samples=count,slot_seconds=count/w['fs_hz'],payload_complex_symbols=384,
                symbol_rate_baud=w['fs_hz']/w['sps'],fs_hz=w['fs_hz'],sps=w['sps'],rolloff=w['rolloff'],
                note='Probe+payload preambles/tracking pilots/pulse tails/guards counted; not simply 1/6 old frame duration')


def transport(c,condition,path,seed):
    cases,plan,_=environment(c); resource=short_resource(c)
    t=TextTransport(cases[condition['weather']],seed_for(c['seed'],'unused_text_initial_draw'),seed,
        condition['launch_signal_dbm'],resource['slot_samples'],plan['laser_max_dbm'],plan['hardware'])
    t.path_draw=copy.deepcopy(path)
    return t


def public_payload(seed):
    rng=np.random.default_rng(seed); x=rng.standard_normal((48,16))
    if seed%3==1: x=np.clip(x,-1.5,1.5)
    if seed%3==2: x=rng.laplace(size=x.shape)
    return (x/max(float(np.mean(x*x)),1e-8)**.5).astype(np.float32)


def wave(c,condition,path,seed,payload):
    t=transport(c,condition,path,seed); snr=t.probe()
    y,rel,stats=t.continuous(payload)
    return y,rel,float(snr),stats,t


def fidelity_summary(rows,c):
    groups={}; passed=True
    for condition in c['conditions']:
        rr=[r for r in rows if r['condition']==condition['id']]
        usable=[r for r in rr if r['available_fit'] and r['available_fresh']]
        disagreement=float(np.mean([r['available_fit']!=r['available_fresh'] for r in rr])) if rr else 1.
        actual=float(np.mean([r['actual_evm2'] for r in usable])) if usable else None
        predicted=float(np.mean([r['predicted_evm2'] for r in usable])) if usable else None
        error=abs(predicted-actual)/max(actual,1e-8) if usable else None
        good=len(usable)>=2 and error<=c['surrogate_evm_relative_tolerance'] and disagreement<=c['surrogate_acceptance_difference_tolerance']
        groups[condition['id']]=dict(frames=len(rr),jointly_available=len(usable),actual_evm2=actual,
            predicted_evm2=predicted,relative_mean_evm_error=error,availability_disagreement=disagreement,passed=bool(good))
        passed=passed and good
    return dict(passed=bool(passed),groups=groups,rows=rows,
                tolerance_is_engineering_gate_not_hardware_certification=True)


def calibrate(c,cfg,folder,atomic_json,budget):
    folder.mkdir(exist_ok=False); records=[]; coefficients=[]; residuals=[]; fresh=[]
    for condition in c['conditions']:
        for partition,count in (('train',cfg['bank_train_frames']),('holdout',cfg['bank_holdout_frames'])):
            weather=paths(c,condition['weather'],count,('text-short-calibration',partition,condition['weather']))
            for i,path in enumerate(weather):
                budget(); seed=seed_for(cfg['seed'],'short-calibration',condition['id'],partition,i)
                x=public_payload(seed); y,rel,snr,stats,_=wave(c,condition,path,seed,x)
                coef,residual=fit_operator(x,y); ix=len(records)
                records.append(dict(index=ix,partition=partition,condition=condition['id'],path=path,
                    available=stats['analog_receiver_available'],probe_snr_db=snr,reliability=rel,stats=stats))
                coefficients.append(coef); residuals.append(residual)
                if partition=='holdout':
                    ns=seed_for(seed,'fresh'); xx=public_payload(ns); yy,_,_,ss,_=wave(c,condition,path,ns,xx)
                    replay=np.roll(residual,int(ns%384))*np.exp(1j*(ns%6283)/1000)
                    pred=apply_operator(as_complex(xx),coef,replay)
                    fresh.append(dict(condition=condition['id'],calibration_index=ix,
                        available_fit=records[-1]['available'],available_fresh=ss['analog_receiver_available'],
                        actual_evm2=float(np.mean(abs(as_complex(yy)-as_complex(xx))**2)),
                        predicted_evm2=float(np.mean(abs(pred-as_complex(xx))**2))))
                if len(records)%8==0: print('SHORT CALIBRATION frames={} condition={} partition={}'.format(len(records),condition['id'],partition),flush=True)
    proof=fidelity_summary(fresh,c)
    proof['passed']=proof['passed'] and all(any(r['available'] for r in records if r['partition']=='train' and r['condition']==q['id']) for q in c['conditions'])
    with (folder/'bank.npz').open('xb') as f: np.savez_compressed(f,coefficients=np.asarray(coefficients),residuals=np.asarray(residuals))
    atomic_json(folder/'bank.json',dict(schema='text-short-waveform-bank-v1',payload_complex_symbols=384,
        public_waveforms_only=True,model='five-tap widely-linear fit+correlated stochastic replay; not hardware differentiable',
        records=records,channel_config=c,fidelity=proof))
    if not proof['passed']: raise RuntimeError('Short-frame calibration fidelity failed; NO FSO training, preserve records')
    return dict(records=records,coefficients=np.asarray(coefficients),residuals=np.asarray(residuals))


def proxy(wire,bank,index,seed):
    import torch
    r=bank['records'][index]
    if r['partition']!='train': raise ValueError('Holdout may not train neural model')
    if wire.shape[1:]!=(48,16): raise ValueError('Wrong short proxy shape')
    pair=wire.reshape(wire.size(0),384,2); x=torch.complex(pair[:,:,0],pair[:,:,1])/math.sqrt(2)
    cols=[]
    for v in (x,x.conj()):
        for shift in OFFSETS:
            z=torch.cat((v[:,-shift:],v.new_zeros(v.size(0),-shift)),-1) if shift<0 else torch.cat((v.new_zeros(v.size(0),shift),v[:,:-shift]),-1) if shift>0 else v
            cols.append(z)
    matrix=torch.stack(cols+[torch.ones_like(x)],-1)
    rng=np.random.default_rng(seed_for(seed,'short-residual'))
    replay=np.roll(bank['residuals'][index],int(rng.integers(384)))*np.exp(1j*rng.uniform(-math.pi,math.pi))
    noise=torch.as_tensor(replay,device=x.device,dtype=x.dtype)
    y=matrix@torch.as_tensor(bank['coefficients'][index],device=x.device,dtype=x.dtype)+noise
    restored=torch.stack((y.real,y.imag),-1).reshape_as(wire)*math.sqrt(2)
    return restored,float(r['reliability']),float(r['probe_snr_db']),bool(r['available'])


def check_current(model,items,bank,c,cfg,stage,device,budget):
    import torch
    from text_jscc_v1_fix1.model import tensors
    rows=[]
    with torch.no_grad():
        for condition in c['conditions']:
            selected=[r for r in bank['records'] if r['partition']=='holdout' and r['condition']==condition['id']][:cfg['fidelity_frames']]
            for i,r in enumerate(selected):
                budget(); item=items[i%len(items)]; ids,lens=tensors([item],device)
                x=model.encode(ids,lens)[0].cpu().numpy(); ns=seed_for(cfg['seed'],'text-current-fidelity',stage,condition['id'],i)
                y,_,_,stats,_=wave(c,condition,r['path'],ns,x)
                rng=np.random.default_rng(seed_for(ns,'short-residual'))
                residual=np.roll(bank['residuals'][r['index']],int(rng.integers(384)))*np.exp(1j*rng.uniform(-math.pi,math.pi))
                pred=apply_operator(as_complex(x),bank['coefficients'][r['index']],residual)
                rows.append(dict(stage=stage,condition=condition['id'],calibration_index=r['index'],source_image_id=item['image_id'],
                    source_split='train',available_fit=r['available'],available_fresh=stats['analog_receiver_available'],
                    actual_evm2=float(np.mean(abs(as_complex(y)-as_complex(x))**2)),
                    predicted_evm2=float(np.mean(abs(pred-as_complex(x))**2))))
    return dict(fidelity_summary(rows,c),stage=stage)
