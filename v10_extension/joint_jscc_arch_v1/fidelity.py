"""Fresh full-waveform holdout checks for the CURRENT learned symbol distribution."""
import math
import numpy as np
from fso_fluctuating_v2.core import transport,as_complex,apply_operator,seed_for


def check(model,items,bank,channel_config,config,stage,device):
    import torch
    rows=[]
    with torch.no_grad():
        for condition in channel_config['conditions']:
            selected=[r for r in bank['records'] if r['partition']=='holdout' and r['condition']==condition['id']]
            selected=selected[:config['fidelity_frames_per_condition']]
            for ix,r in enumerate(selected):
                item=items[ix%len(items)]
                x=model.source_encode(item['latent'].to(device),item['text_features'].to(device),item['text_mask'].to(device))
                payload=x[0].cpu().numpy()
                ns=seed_for(config['seed'],'architecture-fidelity',stage,condition['id'],ix)
                t=transport(channel_config,condition,r['path'],ns); t.probe()
                actual,_,stats=t.continuous(payload,1,'lmmse')
                rng=np.random.default_rng(seed_for(ns,'residual'))
                residual=np.roll(bank['residuals'][r['index']],int(rng.integers(2304)))*np.exp(1j*rng.uniform(-math.pi,math.pi))
                predicted=apply_operator(as_complex(payload),bank['coefficients'][r['index']],residual)
                rows.append(dict(stage=stage,condition=condition['id'],calibration_index=r['index'],
                    source_image_id=item['image_id'],source_split='train',noise_seed=ns,
                    available_fit=bool(r['available']),available_fresh=bool(stats['analog_receiver_available']),
                    actual_evm2=float(np.mean(abs(as_complex(actual)-as_complex(payload))**2)),
                    predicted_evm2=float(np.mean(abs(predicted-as_complex(payload))**2)),
                    actual_launch_signal_energy_j=float(stats['launch_signal_energy_j'])))
    groups={}; passed=True
    for c in channel_config['conditions']:
        rr=[r for r in rows if r['condition']==c['id']]
        usable=[r for r in rr if r['available_fit'] and r['available_fresh']]
        difference=float(np.mean([r['available_fit']!=r['available_fresh'] for r in rr]))
        actual=float(np.mean([r['actual_evm2'] for r in usable])) if usable else None
        predicted=float(np.mean([r['predicted_evm2'] for r in usable])) if usable else None
        relative=abs(predicted-actual)/max(actual,1e-8) if usable else None
        good=len(usable)>=2 and relative<=channel_config['surrogate_evm_relative_tolerance'] and difference<=channel_config['surrogate_acceptance_difference_tolerance']
        groups[c['id']]=dict(frames=len(rr),both_available=len(usable),relative_mean_evm_error=relative,
                            availability_disagreement=difference,passed=bool(good))
        passed=passed and good
    return dict(stage=stage,passed=bool(passed),rows=rows,groups=groups,
                source_split='train',public_bank_partition='holdout',no_gradient_or_threshold_tuning=True,
                tolerances_are_engineering_gate_not_hardware_validation=True)

