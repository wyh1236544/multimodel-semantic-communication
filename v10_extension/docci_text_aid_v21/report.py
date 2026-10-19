"""Independent CPU matrix, channel, schedule, identity and paired-effect checks."""
import argparse
import html
import json
import math
from pathlib import Path
import sys
import numpy as np
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_text_aid_v21.core import (SCHEMA,CACHE_SHA,PARENTS,PILOT_ARMS,
    source_check,config,choose,read,rows,sha,array_sha,atomic,seed,order,impairment,results,EVAL_SHA,check_anchor)
from docci_text_guided_image_v20.channel import sample

def telemetry(s,eid,snr,phase,epoch=0,kind='image'):
    value=seed(42,'v20-AWGN',phase,epoch,eid,kind,snr,0)
    noise=sample(8192 if kind=='image' else 18432,snr,value)
    pilot=sample(64,snr,seed(value,'pilot'))
    pn=float(np.mean(noise.astype(np.float64)**2))
    n0=float(np.mean(pilot.astype(np.float64)**2))
    if (s['noise_seed']!=value or s['noise_sha256']!=array_sha(noise)
            or s['pilot_sha256']!=array_sha(pilot) or s['nominal_snr_db']!=snr
            or s['post_noise_renormalization']):
        raise ValueError('Wrong channel/noise identity')
    def close(a,b):
        return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)
    if not close(s['noise_power'],pn) or not close(s['pilot_noise_variance'],n0):
        raise ValueError('Wrong channel noise power')
    if snr is None:
        if s['estimated_snr_db'] is not None or s['realized_snr_db'] is not None:
            raise ValueError('Clean is not a measured40dB link')
    else:
        if not close(s['estimated_snr_db'],max(-40.,min(40.,-10*math.log10(max(n0,1e-12))))) or not close(
                s['realized_snr_db'],10*math.log10(s['signal_power']/pn)):
            raise ValueError('Wrong estimated/realized SNR')

def verify(root,full=False):
    root=Path(root)
    c=read(root/'config.json');lineage=read(root/'lineage.json')
    expected=config(c['action'],c['profile'],c['smoke'])
    if set(c)!=set(expected)|{'selected'} or any(c[k]!=v for k,v in expected.items()):
        raise ValueError('Unregistered experiment config')
    sizes=dict(train=c['diagnostic_train'] if c['action']=='diagnose' else c['train_samples'],
               dev=c['diagnostic_dev'] if c['action']=='diagnose' else c['dev_samples'])
    if (set(c['selected'])!=set(sizes) or any(len(c['selected'][g])!=n for g,n in sizes.items())
            or len({r['example_id'] for group in c['selected'].values() for r in group})!=sum(sizes.values())):
        raise ValueError('Selected cohort sizes/duplicate IDs differ')
    if any(r['cohort']!=g or r['official_split']!='train' or r['input_truncated']
           for g,group in c['selected'].items() for r in group):
        raise ValueError('Selected evaluation/TRAIN cohort crossed')
    if (lineage['cache_index_sha256']!=CACHE_SHA or lineage['parents']!=PARENTS
            or lineage['cache_regenerated'] or not lineage['strict_determinism']):
        raise ValueError('Wrong lineage')
    from docci_text_guided_image_v20_fix1.compat import validate_gate
    validate_gate(lineage['CUDA_gate'])
    origins={r['example_id']:r for g in c['selected'].values() for r in g}
    if full:
        old=Path(lineage['old_root'])
        if sha(old/'cache/index.json')!=CACHE_SHA or choose(read(old/'cache/index.json'),c)!=c['selected']:
            raise ValueError('Selected cohort differs from frozen full cache')
        for arm,h in PARENTS.items():
            if sha(old/('train_'+arm)/'last.pth')!=h:
                raise ValueError('Actual original parent changed')
    if not read(root/'RUN_FINISHED.json')['training_and_evaluation_finished']:
        raise ValueError('Stage is partial, not finished')
    data=[]
    for folder in ([root] if c['action']=='diagnose' else [root/a for a in PILOT_ARMS]):
        d=read(folder/'DONE.json');rr=rows(folder/'rows.jsonl')
        if d['rows']!=len(rr) or d['rows_sha256']!=sha(folder/'rows.jsonl') or not d['frozen_models_unchanged']:
            raise ValueError('Rows completion/hash/frozen audit differs')
        if c['smoke']:
            arms=PARENTS if c['action']=='diagnose' else ('A_self',)
            for arm in arms:
                anchor=read(folder/('ANCHOR_'+arm+'.json'))
                if (anchor['reference_rows_sha256']!=EVAL_SHA[arm] or len(anchor['rows'])!=2
                        or not anchor['actual_saved_weight_replay'] or not anchor['exact_TX_and_observation']):
                    raise ValueError('Required real anchor replay missing')
                for a in anchor['rows']:check_anchor(a['actual'],a['reference'])
        for r in rr:
            origin=origins[r['example_id']]
            if any(r[a]!=origin[b] for a,b in (('source_image_sha256','image_sha256'),('source_text_sha256','text_sha256')) if b in origin):
                raise ValueError('Wrong evaluation source identity')
            if set(r['trace'])!={'tx32','tx16','rx32','rx64'}:
                raise ValueError('Missing fusion telemetry')
            for t in r['trace'].values():
                if any(not math.isfinite(t[k]) for k in ('alpha','tanh_alpha','feature_RMS','residual_RMS','residual_relative_RMS')):
                    raise ValueError('Nonfinite fusion trace')
                if not math.isclose(t['tanh_alpha'],math.tanh(t['alpha']),abs_tol=1e-6):
                    raise ValueError('Fusion gate telemetry differs')
            telemetry(r['image_stats'],r['example_id'],r['snr_db'],
                'v21-diagnose' if c['action']=='diagnose' else 'v21-pilot-eval')
            if r.get('source_occupancy_passed_to_receiver',False):
                raise ValueError('Gold occupancy leaked into receiver')
            if r.get('text_stats') is not None:
                if c['action']=='diagnose':
                    text_snr=None if r['control'].startswith('rx_clean') else r['snr_db']
                    text_id=r['donor_id'] if r['control'] in ('rx_shuffled','rx_clean_shuffled') else r['example_id']
                    phase='v21-diagnose'
                else:
                    text_snr=None if r['control'].startswith('clean') else r['snr_db']
                    text_id=r['donor_id'] if 'shuffled' in r['control'] else r['example_id']
                    phase='v21-pilot-eval'
                telemetry(r['text_stats'],text_id,text_snr,phase,kind='text')
            if r['mask_sha256']!=array_sha(impairment(c,r['example_id'])):
                raise ValueError('Mask identity differs')
            for field,digest in (('image_file','image_sha256'),('source_image_file','source_view_sha256')):
                if r[field]:
                    name=Path(r[field])
                    if name.is_absolute() or '..' in name.parts or sha(folder/name)!=r[digest]:
                        raise ValueError('Saved real PNG changed/unsafe')
        data.extend(rr)
    if c['action']=='pilot':
        histories={}
        for arm in PILOT_ARMS:
            folder=root/arm;identity=read(folder/'identity.json');done=read(folder/'TRAIN_COMPLETE.json')
            history=rows(folder/'history.jsonl');histories[arm]=history
            audit=read(folder/'learning_audit.json')
            changed=sorted(n for n in audit['initial'] if audit['initial'][n]!=audit['final'][n])
            if (audit['initial']!=read(root/'self/learning_audit.json')['initial'] or
                    changed!=sorted(done['changed']) or any(not n.startswith(('rx32.','rx64.')) for n in changed)
                    or audit['frozen_backbones_before']!=audit['frozen_backbones_after']):
                raise ValueError('Frozen/initial state hash audit differs')
            if (len(history)!=c['updates'] or done['updates']!=c['updates']
                    or identity['parent_sha256']!=PARENTS['A_self'] or identity['config']!=c
                    or not done['checkpoint_roundtrip'] or not done['frozen_image_codec_TX_unchanged']
                    or not done['frozen_backbones_unchanged'] or not done['changed']
                    or not done['nonzero_gradients'] or not identity['only_RX_fusion_trains']
                    or any(not n.startswith(('rx32.','rx64.')) for n in done['changed'])):
                raise ValueError('Pilot optimization/frozen boundary differs')
            if full and sha(folder/'last.pth')!=done['checkpoint_sha256']:
                raise ValueError('Pilot saved weights changed')
            for step,((epoch,batch),r) in enumerate(zip(order(c),history),1):
                own=[c['selected']['train'][i]['example_id'] for i,j in batch]
                donors=[c['selected']['train'][j]['example_id'] for i,j in batch]
                if (r['update']!=step or r['epoch']!=epoch or
                        [o['example_id'] for o in r['observations']]!=own or
                        [o['donor_id'] for o in r['observations']]!=donors):
                    raise ValueError('Pilot order/TRAIN donor identity differs')
                for o in r['observations']:
                    s=None if seed(42,'v21-train-snr',epoch,o['example_id'])%4==0 else 0
                    if o['snr_db']!=s or o['mask_sha256']!=array_sha(impairment(c,o['example_id'])):
                        raise ValueError('Training SNR/mask differs')
                    telemetry(o['image_stats'],o['example_id'],s,'v21-pilot-train',epoch)
                    telemetry(o['text_stats'],o['donor_id'] if arm=='donor' else o['example_id'],None,'v21-pilot-train',epoch,'text')
                    loss=o['pixel_MSE']+.01*o['latent_MSE']+.05*o['LPIPS']
                    if not math.isclose(loss,o['loss'],rel_tol=2e-6,abs_tol=2e-7):
                        raise ValueError('Pilot loss differs')
                if not math.isclose(r['loss'],float(np.mean([o['loss'] for o in r['observations']])),rel_tol=2e-6):
                    raise ValueError('Pilot mean objective differs')
                lr=c['learning_rate']*(.1+.9*(1-(step-1)/c['updates']))
                if (not math.isclose(r['learning_rate'],lr,rel_tol=1e-12) or not math.isfinite(r['gradient_norm'])
                        or r['total_updates']!=c['updates']):
                    raise ValueError('Learning schedule/nonfinite gradient differs')
                ref=histories['self'][step-1]
                for o,b in zip(r['observations'],ref['observations']):
                    if any(o[k]!=b[k] for k in ('example_id','donor_id','snr_db','tx_sha256','observed_sha256','mask_sha256','image_stats')):
                        raise ValueError('Arm-specific image/source/noise resources')
            if identity['trainable_parameters']!=read(root/'self/identity.json')['trainable_parameters']:
                raise ValueError('Unequal trainable parameter counts')
    out=results(c,data)
    out.update(verification='Independent CPU recomputation; not neural/LPIPS replay',
        actual_arrays_and_weights_verified=full,rows=len(data))
    return out

def render(root,data):
    c=read(root/'config.json')
    parts=['<!doctype html><meta charset="utf-8"><title>V21 actual results</title>',
        '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto}img{width:240px}pre{white-space:pre-wrap}.grid{display:flex;flex-wrap:wrap;gap:16px}figure{margin:0;max-width:260px}</style>',
        '<h1>V21：真实权重诊断 / 文本辅助小实验</h1>',
        '<p>clean＝无噪声，SNR不适用；0＝名义AWGN 0dB。missing25额外擦除不是FSO。可靠文本是特权诊断，真实噪声文本另列。未审核事实，不宣布语义成功。</p>',
        '<pre>'+html.escape(json.dumps(data,ensure_ascii=False,indent=2))+'</pre><div class="grid">']
    for folder in ([root] if c['action']=='diagnose' else [root/a for a in PILOT_ARMS]):
        for r in rows(folder/'rows.jsonl'):
            if r['image_file']:
                for field,label in (('source_image_file','原图'),('image_file',r['control'])):
                    path=(folder/r[field]).relative_to(root).as_posix()
                    parts.append('<figure><img src="{}"><figcaption>{}</figcaption></figure>'.format(
                        html.escape(path,quote=True),html.escape('{} {} {} SNR={} {} PSNR={:.3f} LPIPS={:.4f}'.format(
                        r['arm'],r['cohort'],r['example_id'],r['snr_db'],label,r['psnr'],r['LPIPS']))))
    return ''.join(parts)+ '</div>'

def main():
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--full',action='store_true');a=p.parse_args()
    source_check();root=Path(a.root)
    out=verify(root,a.full)
    if (root/'VERIFIED.json').exists() or (root/'index.html').exists():
        raise FileExistsError('Existing report preserved; collection can verify read-only without rewriting it')
    atomic(root/'VERIFIED.json',out)
    with (root/'index.html').open('x',encoding='utf-8') as f:
        f.write(render(root,out))
    print('V21 VERIFIED rows={} numerical_screen={} facts=PENDING'.format(out['rows'],out['numeric_screen_passed']),flush=True)

if __name__=='__main__':
    main()
