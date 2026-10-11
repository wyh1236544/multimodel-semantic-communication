"""Matrix/resource/gradient audit and explicitly separated fallback contributions."""
import argparse
import html
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
EXT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXT))
from vae_tail_ablation_v1 import VERSION, ARMS
from vae_tail_ablation_v1.core import digest, seed_for, check_history_pair, tail_prefixes, visual_positions
from fso_jscc_v1.report import safe_file, read_rows


def verify(directory,complete=False):
    d=Path(directory); design=json.loads((d/'design.json').read_text()); cfg=design['config']
    if design['version']!=VERSION or tuple(design['arms'])!=ARMS or design['split']!='validation': raise ValueError('Wrong experiment identity')
    ids=design['validation_ids']
    if len(ids)!=cfg['val_samples'] or len(set(ids))!=len(ids) or set(ids)&set(design['train_ids']): raise ValueError('Invalid split/IDs')
    if design['primary_validation_ids']!=ids[cfg['primary_validation_offset']:]: raise ValueError('Primary validation subset changed')
    prefixes=tuple(design['vae_tail_prefixes'])
    if prefixes!=tail_prefixes(int(prefixes[0].split('.')[-2])+1): raise ValueError('Unexpected tail selection')
    checkpoint=json.loads((d/'checkpoint_load.json').read_text())
    candidate=checkpoint['candidate_parameters']
    tail={n for n in candidate if n.startswith(prefixes)}
    if not tail or checkpoint['vae_tail_prefixes']!=list(prefixes): raise ValueError('Missing/different tail parameters')
    if checkpoint['vae_tail_parameters']!=sum(candidate[n] for n in tail): raise ValueError('Wrong tail parameter count')
    semantic=sum(v for n,v in candidate.items() if n not in tail)
    if checkpoint['semantic_parameters']!=semantic or checkpoint['trainable_per_arm']!={ARMS[0]:0,ARMS[1]:semantic,ARMS[2]:sum(candidate.values())}: raise ValueError('Wrong trainable budget')
    if not checkpoint['CLIP_and_VAE_encoder_frozen']: raise ValueError('Backbone unfreeze outside scope')
    expected={(a,i,c['id'],r) for a in ARMS for i in ids for c in cfg['conditions'] for r in range(cfg['draws'])}
    expected|={(a,i,'clean',0) for a in ARMS for i in ids}
    rows=read_rows(d/'samples.jsonl'); seen=set(); paired=defaultdict(list)
    events=defaultdict(list)
    for e in read_rows(d/'events.jsonl'): events[(e['arm'],e['image_id'],e['condition'],e['draw'])].append(e)
    for r in rows:
        key=(r['arm'],r['image_id'],r['condition'],r['draw'])
        if key in seen: raise ValueError('Duplicate sample')
        seen.add(key)
        if r['source_sha256']!=design['validation_source_sha256'][str(r['image_id'])]: raise ValueError('Changed source identity')
        if r['frame_index']!=ids.index(r['image_id']): raise ValueError('Temporal order changed')
        if r['analog_repeats']!=1 or r['raw_utf8_wire_bytes']!=0 or r['pre_fec_ber'] is not None or r['post_fec_ber'] is not None: raise ValueError('Hidden repeat/text oracle/analog BER')
        for k in ('psnr','ssim','normalized_wer','cer','token_acc','common_to_text_cosine'):
            if not math.isfinite(r[k]): raise ValueError('Nonfinite metric')
        paired[key[1:]].append((r['noise_seed'],r['weather_path'],r['source_sha256']))
        for k in ('image_file','source_image_file','constellation_file'):
            if k in r and digest(safe_file(d,r[k]))!=r[k+'_sha256']: raise ValueError('Changed visualization')
        if ('image_file' in r)!=(r['frame_index'] in visual_positions(cfg)): raise ValueError('Changed preselected image coverage')
        ev=events.get(key,[])
        if r['condition']=='clean':
            if ev or r['payload_complex_symbols']!=2304 or not r['receiver_available']: raise ValueError('Wrong clean control')
        else:
            if len(ev)!=2 or r['event_count']!=2 or [e['purpose'] for e in ev]!=['probe','features']: raise ValueError('Unexpected transmissions')
            if ev[1]['data_symbols']!=2304: raise ValueError('Changed feature budget')
            samples=sum(e['awg_samples']+e['guard_samples'] for e in ev)
            if samples!=r['awg_samples']+r['guard_samples'] or samples>design['environment'][2]['slot_samples']: raise ValueError('Wrong time resource')
            if r['receiver_available']!=ev[1]['receiver_locked']: raise ValueError('Erasure mismatch')
            if r['weather_path']['temporal_model'].startswith('independent'): raise ValueError('Missing temporal FSO')
            if any(e['launch_signal_energy_j']>e['launch_peak_signal_bound_w']*e['waveform_seconds']*(1+1e-8) for e in ev): raise ValueError('Peak-envelope energy bound')
    if seen!=expected or len(rows)!=len(expected) or set(events)!={k for k in expected if k[2]!='clean'}: raise ValueError('Incomplete/extra matrix or events')
    if any(len(v)!=3 or any(x!=v[0] for x in v) for v in paired.values()): raise ValueError('Unpaired noise/weather/source')
    histories=[]
    for arm in ARMS:
        t=d/('train_'+arm); record=json.loads((t/'COMPLETE.json').read_text())
        trained=arm!=ARMS[0]
        if record['checkpoint_roundtrip_verified']!=trained: raise ValueError('Saved delta roundtrip not verified')
        if record['arm']!=arm or record['steps']!=(design['train_forward_steps'] if trained else 0) or record['optimizer_steps']!=(design['optimizer_steps'] if trained else 0): raise ValueError('Unequal training budget')
        if trained:
            if min(record['gradient_steps'][k] for k in ('source_encoder','tx_mapping','rx_mapping','decoder'))<=0: raise ValueError('No gradient for required module')
            for prefix in ('base.multimodal_encoder.','base.channel.tx_map.','base.channel.rx_map.','base.joint_decoder.'):
                if sum(v for k,v in record['parameter_changes'].items() if k.startswith(prefix))<=0: raise ValueError('Required parameter unchanged')
            history=read_rows(t/'history.jsonl')
            if len(history)!=design['train_forward_steps'] or [r['forward_step'] for r in history]!=list(range(1,len(history)+1)): raise ValueError('Training history incomplete')
            if any(r['vae_tail_trainable']!=(arm==ARMS[2]) for r in history): raise ValueError('Wrong per-step tail eligibility')
            histories.append(history)
            if complete and (t/'last_delta.pth').exists() and digest(t/'last_delta.pth')!=record['checkpoint_sha256']: raise ValueError('Changed checkpoint')
        if set(record['parameter_changes'])!=set(candidate) or not record['frozen_parameters_unchanged']: raise ValueError('Incomplete frozen/parameter audit')
        changes=record['parameter_changes']
        if any(not math.isfinite(v) or v<0 for v in changes.values()): raise ValueError('Nonfinite parameter change')
        delta=sum(changes[n] for n in tail)
        if record['vae_tail_parameter_change']!=delta or record['vae_tail_gradient_steps']!=record['gradient_steps']['vae_tail']: raise ValueError('Tail audit differs')
        if arm==ARMS[2]:
            if delta<=0 or record['vae_tail_gradient_steps']<=0: raise ValueError('Tail did not actually train')
        elif delta!=0 or record['vae_tail_gradient_steps']!=0: raise ValueError('Frozen tail changed')
        if not trained and any(changes.values()): raise ValueError('Frozen U0 changed')
        for k in ('training_peak_allocated_gib','training_peak_reserved_gib'):
            if record[k] is not None and (not math.isfinite(record[k]) or record[k]<0): raise ValueError('Invalid memory evidence')
    check_history_pair(*histories)
    diagnostic=read_rows(d/'diagnostics.jsonl'); found=set(); diagnostic_pairs=defaultdict(dict)
    for r in diagnostic:
        key=(r['arm'],r['image_id'])
        if key in found: raise ValueError('Duplicate diagnostic')
        found.add(key); diagnostic_pairs[r['image_id']][r['arm']]=r
        if r['source_sha256']!=design['validation_source_sha256'][str(r['image_id'])] or not r['direct_reference_not_mathematical_upper_bound']: raise ValueError('Wrong VAE reference identity')
        for k in ('direct_vae_psnr','direct_vae_ssim','fixed_U0_semantic_latent_psnr','fixed_U0_semantic_latent_ssim'):
            if not math.isfinite(r[k]): raise ValueError('Nonfinite VAE diagnostic')
        for k in ('direct_vae_file','fixed_latent_file'):
            if k in r and digest(safe_file(d,r[k]))!=r[k+'_sha256']: raise ValueError('Changed diagnostic image')
            if (k in r)!=(ids.index(r['image_id']) in visual_positions(cfg)): raise ValueError('Missing diagnostic image')
    if found!={(a,i) for a in ARMS for i in ids}: raise ValueError('Incomplete VAE reference matrix')
    for records in diagnostic_pairs.values():
        for key in ('cached_vae_latent_sha256','fixed_reference_latent_sha256'):
            if len({r[key] for r in records.values()})!=1: raise ValueError('Unpaired reference latent')
        for key in ('direct_vae_psnr','direct_vae_ssim','fixed_U0_semantic_latent_psnr','fixed_U0_semantic_latent_ssim'):
            if abs(records[ARMS[0]][key]-records[ARMS[1]][key])>1e-5: raise ValueError('Frozen decoder direct reference changed')
    identity=json.loads((d/'run_identity.json').read_text())
    if identity['version']!=VERSION: raise ValueError('Wrong runtime version')
    if digest(d/'calibration_summary.json')!=identity['calibration_sha256']: raise ValueError('Changed public calibration evidence')
    for n,sha in identity['new_source_sha256'].items():
        if Path(n).name!=n or digest(Path(__file__).parent/n)!=sha: raise ValueError('New source differs from executed code')
    for n,sha in identity['calibration_source_sha256'].items():
        if Path(n).is_absolute() or '..' in Path(n).parts or digest(EXT/n)!=sha: raise ValueError('Dependency source drift')
    for n,sha in identity['source_sha256'].items():
        if Path(n).is_absolute() or '..' in Path(n).parts or digest(EXT/n)!=sha: raise ValueError('Model/runtime source drift')
    if complete:
        end=json.loads((d/'COMPLETE.json').read_text())
        if end['version']!=VERSION or end['rows']!=len(rows) or identity['status']!='complete' or (d/'FAILED.json').exists(): raise ValueError('Not complete')
        required={'design.json','samples.jsonl','events.jsonl','checkpoint_load.json','run_identity.json',
            'summary.json','REPORT.md','report.html','quality_timeline.svg','calibration_summary.json','diagnostics.jsonl'}
        required.update('train_'+arm+'/COMPLETE.json' for arm in ARMS)
        required.update('train_'+arm+'/history.jsonl' for arm in ARMS[1:])
        if not required <= set(end['result_sha256']): raise ValueError('Completion manifest missing essential evidence')
        for n,sha in end['result_sha256'].items():
            if digest(safe_file(d,n))!=sha: raise ValueError('Changed result file')
        for arm in ARMS[1:]:
            n='train_'+arm+'/last_delta.pth'
            if end['checkpoint_sha256'].get(n)!=json.loads((d/('train_'+arm)/'COMPLETE.json').read_text())['checkpoint_sha256']: raise ValueError('Checkpoint hash evidence differs')
    return rows


def contrast(rows,conditions,resamples,image_ids=None):
    pairs={}
    for r in rows:
        if r['condition'] in conditions and r['arm'] in ARMS[1:] and (image_ids is None or r['image_id'] in image_ids):
            pairs.setdefault((r['image_id'],r['condition'],r['draw']),{})[r['arm']]=r
    matched=[(v[ARMS[1]],v[ARMS[2]]) for v in pairs.values()]
    def stat(subset):
        if not subset: return dict(mean=None,ci95=None,trials=0,images=0)
        groups=defaultdict(list)
        for a,b in subset: groups[a['image_id']].append(b['psnr']-a['psnr'])
        x=np.asarray([np.mean(groups[i]) for i in sorted(groups)])
        rng=np.random.default_rng(seed_for('v2-bootstrap',conditions,len(subset)))
        boot=x[rng.integers(len(x),size=(resamples,len(x)))].mean(1)
        return dict(mean=float(x.mean()),ci95=[float(v) for v in np.quantile(boot,[.025,.975])],trials=len(subset),images=len(x),
            confidence_scope='conditional image resampling; fixed selected weather trajectories, single training replicate, NOT uncertainty across weather or training seeds')
    both=[p for p in matched if p[0]['receiver_available'] and p[1]['receiver_available']]
    neither=[p for p in matched if not p[0]['receiver_available'] and not p[1]['receiver_available']]
    discordant=[p for p in matched if p not in both and p not in neither]
    # Decomposition uses trial-weighted means (not the image-weighted conditional
    # means); contributions add exactly to the all-trial gain, without hiding fail.
    contrib=lambda v: sum(b['psnr']-a['psnr'] for a,b in v)/len(matched) if matched else None
    return dict(all_frames=stat(matched),both_available=stat(both),both_unavailable_fallback=stat(neither),
        discordant_availability=stat(discordant),trial_weighted_contributions=dict(
            received=contrib(both),fallback=contrib(neither),discordant=contrib(discordant)),
        outage_U1=float(np.mean([not a['receiver_available'] for a,b in matched])),
        outage_U2=float(np.mean([not b['receiver_available'] for a,b in matched])))


def report(directory):
    d=Path(directory); rows=verify(d); design=json.loads((d/'design.json').read_text()); cfg=design['config']
    groups={}
    for condition in ['clean']+[c['id'] for c in cfg['conditions']]:
        for arm in ARMS:
            data=[r for r in rows if r['arm']==arm and r['condition']==condition]
            usable=[r for r in data if r['receiver_available']]
            mean=lambda k,subset=data:float(np.mean([r[k] for r in subset])) if subset else None
            events=[e for e in read_rows(d/'events.jsonl') if e['arm']==arm and e['condition']==condition and e['purpose']=='features']
            groups[condition+'/'+arm]=dict(psnr_all=mean('psnr'),psnr_available=mean('psnr',usable),ssim_all=mean('ssim'),
                normalized_wer=mean('normalized_wer'),raw_cer=mean('cer'),common_to_text_cosine=mean('common_to_text_cosine'),
                outage=1-len(usable)/len(data),psnr_p10=float(np.quantile([r['psnr'] for r in data],.1)),
                projection_noise_snr_db=float(np.mean([e['physical_symbol_snr_db'] for e in events])) if events else None,
                pilot_snr_db=float(np.mean([e['estimated_snr_db'] for e in events])) if events else None,
                waveform_energy_j=mean('launch_signal_energy_j') if events else None,
                scheduled_seconds=mean('scheduled_seconds'),
                joint_delivery_fraction=float(np.mean([r['receiver_available'] and r['psnr']>=20 and r['normalized_wer']<=.1 for r in data])))
    primary=contrast(rows,cfg['primary_conditions'],cfg['bootstrap_resamples'],design['primary_validation_ids'])
    diagnostic=read_rows(d/'diagnostics.jsonl')
    diagnostic_groups={a:{k:float(np.mean([r[k] for r in diagnostic if r['arm']==a]))
        for k in ('direct_vae_psnr','direct_vae_ssim','fixed_U0_semantic_latent_psnr','fixed_U0_semantic_latent_ssim')} for a in ARMS}
    reference_pairs={i:{r['arm']:r for r in diagnostic if r['image_id']==i} for i in design['validation_ids']}
    def reference_contrast(key):
        differences=np.asarray([v[ARMS[2]][key]-v[ARMS[1]][key] for i,v in sorted(reference_pairs.items())])
        rng=np.random.default_rng(seed_for('tail-reference',key))
        boot=differences[rng.integers(len(differences),size=(cfg['bootstrap_resamples'],len(differences)))].mean(1)
        return dict(mean=float(differences.mean()),ci95=list(map(float,np.quantile(boot,[.025,.975]))),images=len(differences),exploratory=True)
    training={a:json.loads((d/('train_'+a)/'COMPLETE.json').read_text()) for a in ARMS}
    parameter_info=json.loads((d/'checkpoint_load.json').read_text())
    value=dict(version=VERSION,groups=groups,primary=primary,
        primary_validation_ids=design['primary_validation_ids'],diagnostics=diagnostic_groups,
        direct_vae_U2_minus_U1=reference_contrast('direct_vae_psnr'),
        fixed_latent_U2_minus_U1=reference_contrast('fixed_U0_semantic_latent_psnr'),
        training={a:{k:r[k] for k in ('wall_seconds','training_peak_allocated_gib','training_peak_reserved_gib','gpu_name','vae_tail_gradient_steps','vae_tail_parameter_change')} for a,r in training.items()},
        parameters={k:parameter_info[k] for k in ('semantic_parameters','vae_tail_parameters','trainable_per_arm')},
        clean_contrast=contrast(rows,['clean'],cfg['bootstrap_resamples']),
        all_validation_primary_conditions_exploratory=contrast(rows,cfg['primary_conditions'],cfg['bootstrap_resamples']),
        conditions={c['id']:contrast(rows,[c['id']],cfg['bootstrap_resamples']) for c in cfg['conditions']},
        thresholds_note='joint_delivery: receiver_available AND PSNR>=20 AND normalized_WER<=0.1; diagnostic, not universal perceptual threshold',
        conclusion_boundary='Local VAE-tail eligibility effect in existing JOINT model; no digital baseline, no proof joint>separate, no hardware measurement; direct VAE reference not theoretical ceiling')
    with (d/'summary.json').open('x',encoding='utf-8') as f: json.dump(value,f,indent=2,allow_nan=False)
    lines=['# Partial VAE decoder ablation V1','',
        'U0=frozen V2 B2; U1=FSO fine tune with frozen VAE; U2=same plus trainable terminal VAE decoder. CLIP/VAE encoder remain frozen.',
        'All frames retained. Primary: BOTH-AVAILABLE U2-U1 on validation positions32-63, predeclared low-quality conditions. Fallback gain is not received semantic information.',
        'One training replicate; image intervals conditional on fixed weather paths. Same peak/slot caps do not imply equal actual energy.',
        '', '| Condition / arm | PSNR all | PSNR available | outage | WER |', '|---|---:|---:|---:|---:|']
    for k,g in groups.items():
        available='NA' if g['psnr_available'] is None else '{:.3f}'.format(g['psnr_available'])
        lines.append('| {} | {:.3f} | {} | {:.1%} | {:.3f} |'.format(k,g['psnr_all'],available,g['outage'],g['normalized_wer']))
    lines+=['','Primary contrast:','', '```json',json.dumps(primary,indent=2),'```','',
        'VAE references use identical cached posterior samples and identical frozen-U0 semantic latents; neither is a theoretical upper bound.',
        'No automatic enlargement if fallback alone improves or direct/clean reconstruction degrades. This experiment does not repair the CLIP text recovery bottleneck.',
        '', 'Parameter/runtime/diagnostic evidence:', '```json',json.dumps({k:value[k] for k in ('parameters','training','diagnostics','direct_vae_U2_minus_U1','fixed_latent_U2_minus_U1')},indent=2),'```']
    with (d/'REPORT.md').open('x') as f: f.write('\n'.join(lines)+'\n')
    # Timeline, not an artificial monotone SNR curve: weather is correlated.
    colors={ARMS[0]:'#64748b',ARMS[1]:'#0072B2',ARMS[2]:'#D55E00'}
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{}" viewBox="0 0 1000 {}">'.format(90+190*len(cfg['conditions']),90+190*len(cfg['conditions'])),
        '<rect width="100%" height="100%" fill="white"/><text x="40" y="25" font-size="18">PSNR over fluctuating FSO time · draw 0 · R=1</text>',
        '<text x="40" y="48" font-size="13">Gray=U0 frozen; blue=U1 frozen VAE; orange=U2 VAE tail. Open circles: rejected (fallback).</text>']
    for ci,condition in enumerate(cfg['conditions']):
        top=90+190*ci
        svg.append('<text x="40" y="{}" font-size="14">{}</text>'.format(top-8,condition['id']))
        for level in (10,20,30):
            y=top+140-level*3.5
            svg.append('<path d="M80 {}H940" stroke="#ddd"/><text x="40" y="{}" font-size="12">{}dB</text>'.format(y,y+4,level))
        for arm in ARMS:
            data=sorted([r for r in rows if r['condition']==condition['id'] and r['draw']==0 and r['arm']==arm],key=lambda r:r['frame_index'])
            points=[(80+860*r['frame_index']/max(cfg['val_samples']-1,1),top+140-max(0.,min(40.,r['psnr']))*3.5,r) for r in data]
            dash={ARMS[0]:'2 4',ARMS[1]:'8 4',ARMS[2]:'none'}[arm]
            svg.append('<polyline fill="none" stroke="{}" stroke-dasharray="{}" points="{}"/>'.format(colors[arm],dash,' '.join('{:.2f},{:.2f}'.format(x,y) for x,y,r in points)))
            for x,y,r in points:
                svg.append('<circle cx="{:.2f}" cy="{:.2f}" r="3" stroke="{}" fill="{}"/>'.format(x,y,colors[arm],colors[arm] if r['receiver_available'] else 'white'))
        svg.append('<text x="80" y="{}" font-size="12">0 ms</text><text x="850" y="{}" font-size="12">{:.2f} ms</text>'.format(top+162,top+162,(cfg['val_samples']-1)*cfg['frame_spacing_s']*1000))
    svg.append('</svg>')
    with (d/'quality_timeline.svg').open('x') as f: f.write(''.join(svg))
    cards=[]
    ordered=sorted(rows,key=lambda r:(r['condition'],r['draw'],r['image_id'],ARMS.index(r['arm'])))
    for r in ordered:
        if 'image_file' not in r: continue
        cards.append('<article><h3>{} · {} · {} · draw {}</h3><img src="{}"><p>PSNR {:.2f}; SSIM {:.3f}; WER {:.3f}; receiver available {}</p><p>TX reference (evaluation only): {}</p><p>Received: {}</p></article>'.format(
            r['image_id'],r['arm'],r['condition'],r['draw'],html.escape(r['image_file']),r['psnr'],r['ssim'],r['normalized_wer'],r['receiver_available'],
            html.escape(r['source_caption_evaluation_only']),html.escape(r['received_caption'])))
    references=[]
    for r in diagnostic:
        if 'direct_vae_file' not in r: continue
        references.append('<article><h3>{} {}</h3><p>Direct VAE {:.2f}dB</p><img src="{}"><p>Identical U0 latent {:.2f}dB</p><img src="{}"></article>'.format(r['image_id'],r['arm'],r['direct_vae_psnr'],html.escape(r['direct_vae_file']),r['fixed_U0_semantic_latent_psnr'],html.escape(r['fixed_latent_file'])))
    content='<meta charset="utf-8"><style>body{font:16px sans-serif;margin:30px}main{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}article{border:1px solid #ddd;padding:12px}img{max-width:100%;image-rendering:auto}article img{width:256px}p{overflow-wrap:anywhere}</style><h1>Partial VAE decoder: image and text</h1><p>R=1. Failures retained. CLIP/VAE encoder frozen. No source-text receiver bypass. Simulator, NOT hardware measurement.</p><img src="quality_timeline.svg"><pre>'+html.escape('\n'.join(lines))+'</pre><h2>Direct VAE and fixed-latent references (not upper bounds)</h2><main>'+''.join(references)+'</main><h2>Actual image/text reconstruction</h2><main>'+''.join(cards)+'</main>'
    with (d/'report.html').open('x') as f: f.write(content)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('directory'); p.add_argument('--verify-only',action='store_true')
    a=p.parse_args(); verify(a.directory,complete=True)
    print('Verified full matrix, paired training/weather/latents, exact tail/frozen eligibility, resources and hashes.')
