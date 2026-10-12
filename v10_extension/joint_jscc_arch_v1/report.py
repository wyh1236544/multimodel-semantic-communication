"""Independent matrix, schedule, actual-update, fidelity and resource verification."""
import argparse
import html
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from joint_jscc_arch_v1 import VERSION,ARMS,TRAINED_ARMS
from joint_jscc_arch_v1.core import (digest,expected_keys,check_pair_histories,bootstrap,arm_options)
from fso_jscc_v1.report import safe_file,read_rows


def validate_row(row):
    if row['analog_repeats']!=1 or row['raw_utf8_wire_bytes']!=0 or row['payload_complex_symbols']!=2304:
        raise ValueError('Hidden repeat/original text side channel/changed budget')
    if row['pre_fec_ber'] is not None or row['post_fec_ber'] is not None:
        raise ValueError('Analog symbols do not have a fictitious bit BER')
    for key in ('psnr','ssim','normalized_wer','covered_normalized_wer','cer','waveform_papr_db'):
        if not math.isfinite(row[key]): raise ValueError('Nonfinite metric '+key)
    if row['neural_input_capacity_tokens'] < 77 or row['covered_content_tokens'] > row['neural_input_capacity_tokens']-2:
        raise ValueError('Wrong text capacity/reference')


def verify(directory,complete=False):
    d=Path(directory); design=json.loads((d/'design.json').read_text()); c=design['config']
    if design['version']!=VERSION or tuple(design['arms'])!=ARMS or design['split']!='validation':
        raise ValueError('Wrong experiment identity')
    ids=design['validation_ids']
    if len(ids)!=c['val_samples'] or len(set(ids))!=len(ids) or set(ids)&set(design['train_ids']): raise ValueError('Invalid split')
    expected=expected_keys(ARMS,ids,design['conditions'],c['draws'])
    rows=read_rows(d/'samples.jsonl'); seen=set(); pairs=defaultdict(list); events=defaultdict(list)
    for e in read_rows(d/'events.jsonl'):
        events[(e['arm'],e['image_id'],e['condition'],e['draw'])].append(e)
    for r in rows:
        validate_row(r); key=(r['arm'],r['image_id'],r['condition'],r['draw'])
        if key in seen: raise ValueError('Duplicate sample')
        seen.add(key)
        if r['source_sha256']!=design['validation_source_sha256'][str(r['image_id'])] or r['frame_index']!=ids.index(r['image_id']):
            raise ValueError('Changed source/order')
        pairs[key[1:]].append((r['noise_seed'],r['weather_path'],r['source_sha256']))
        if r['arm']!='B0_legacy' and r['neural_input_capacity_tokens']!=c['max_text_tokens']:
            raise ValueError('New arms have unequal source text capacity')
        for name in ('image_file','source_image_file','constellation_file'):
            if name in r and digest(safe_file(d,r[name]))!=r[name+'_sha256']: raise ValueError('Changed visualization')
        ev=events.get(key,[])
        if r['condition']=='clean':
            if ev or not r['receiver_available']: raise ValueError('Invalid clean control')
        else:
            if len(ev)!=2 or r['event_count']!=2 or [e['purpose'] for e in ev]!=['probe','features']:
                raise ValueError('Unexpected transmissions')
            if ev[1]['data_symbols']!=2304 or r['receiver_available']!=ev[1]['receiver_locked']:
                raise ValueError('Payload/availability mismatch')
            count=sum(e['awg_samples']+e['guard_samples'] for e in ev)
            if count!=r['awg_samples']+r['guard_samples'] or count>design['environment'][2]['slot_samples']:
                raise ValueError('Uncharged waveform/time resources')
            for e in ev:
                if e['launch_signal_energy_j']>e['launch_peak_signal_bound_w']*e['waveform_seconds']*(1+1e-8):
                    raise ValueError('Peak-envelope energy limit')
    if seen!=expected or len(rows)!=len(expected) or set(events)!={k for k in expected if k[2]!='clean'}:
        raise ValueError('Incomplete/extra matrix/events')
    if any(len(v)!=len(ARMS) or any(x!=v[0] for x in v) for v in pairs.values()): raise ValueError('Unpaired weather/noise/source')
    histories=[]; records={}
    counts=json.loads((d/'checkpoint_load.json').read_text())['parameter_counts']
    calibration=json.loads((d/'calibration_summary.json').read_text())
    bank_records={r['index']:r for r in calibration['records']}
    for arm in TRAINED_ARMS:
        folder=d/('train_'+arm); rec=json.loads((folder/'COMPLETE.json').read_text()); records[arm]=rec
        if rec['arm']!=arm or rec['steps']!=design['train_forward_steps'] or rec['optimizer_steps']!=design['optimizer_steps']:
            raise ValueError('Unequal training budget')
        if not rec['checkpoint_roundtrip_verified'] or not rec['frozen_backbones'] or not rec['initial_core_verified']:
            raise ValueError('Missing checkpoint/frozen verification')
        required=['source','text_compression','ar','tx','rx','receiver']+(['conditioning','tx_spatial','rx_spatial'] if arm_options(arm)['modern'] else [])+(['common_coarse'] if arm_options(arm)['auxiliary'] else [])
        if rec['required_groups']!=required or any(rec['parameter_group_changes'][k]<=0 for k in required): raise ValueError('No actual required parameter update')
        frozen_prefixes=('modern_tx.','modern_rx.','spatial_tx.','spatial_rx.','rx_condition.','missing_token')
        eligible={n for n in counts if arm_options(arm)['modern'] or not n.startswith(frozen_prefixes)}
        if rec['trainable_parameters']!=sum(counts[n] for n in eligible) or rec['registered_parameters']!=sum(counts.values()):
            raise ValueError('Parameter count/eligibility evidence differs')
        covered=set(rec['observed_nonzero_gradient_names'])
        if not covered<=eligible or rec['observed_nonzero_gradient_parameters']!=sum(counts[n] for n in covered):
            raise ValueError('Observed gradient coverage evidence differs')
        if set(rec['parameter_changes'])!=set(counts) or any(rec['parameter_changes'][n]!=0 for n in set(counts)-eligible):
            raise ValueError('Ineligible parameter changed')
        history=read_rows(folder/'history.jsonl'); histories.append(history)
        if len(history)!=design['train_forward_steps'] or [r['forward_step'] for r in history]!=list(range(1,len(history)+1)):
            raise ValueError('Incomplete training history')
        if any(r['image_id'] not in design['train_ids'] for r in history): raise ValueError('Non-training image used')
        for r in history:
            ix=r['bank_index']
            if r['phase']=='clean' and ix is not None: raise ValueError('Noisy frame in clean phase')
            if bool(r['anchor'])!=(ix is None): raise ValueError('Wrong clean anchor')
            if ix is None and not r['task_loss_applied']: raise ValueError('Masked clean supervision')
            if ix is not None:
                record=bank_records[ix]
                if record['partition']!='train' or bool(r['task_loss_applied'])!=bool(record['available']):
                    raise ValueError('Holdout used for gradients or changed failure mask')
        phase_counts={p:sum(r['phase']==p for r in history) for p in ('clean','fso')}
        if phase_counts!={'clean':c['clean_epochs']*c['train_samples'],'fso':c['noise_epochs']*c['train_samples']}:
            raise ValueError('Missing clean/noisy phase')
        if any(bool(r['auxiliary_applied']) != (arm_options(arm)['auxiliary'] and bool(r['task_loss_applied'])) for r in history):
            raise ValueError('Wrong auxiliary loss arm')
        for stage in ('initial','clean','fso'):
            proof=json.loads((folder/('fidelity_'+stage+'.json')).read_text())
            if proof['stage']!=stage or not proof['passed'] or proof['source_split']!='train' or proof['public_bank_partition']!='holdout':
                raise ValueError('Current-symbol fidelity gate missing/failed')
            if set(proof['groups'])!={x['id'] for x in design['channel_config']['conditions']}:
                raise ValueError('Incomplete fidelity conditions')
            if any(not g['passed'] or g['both_available']<2 for g in proof['groups'].values()): raise ValueError('False fidelity pass')
            fidelity_keys=set()
            for r in proof['rows']:
                key=(r['condition'],r['calibration_index'])
                if key in fidelity_keys or r['source_image_id'] not in design['train_ids']: raise ValueError('Invalid fidelity image/duplicate frame')
                fidelity_keys.add(key); bank_row=bank_records[r['calibration_index']]
                if bank_row['partition']!='holdout' or bank_row['condition']!=r['condition'] or bool(bank_row['available'])!=bool(r['available_fit']):
                    raise ValueError('Wrong fidelity partition/condition/availability')
                if not all(math.isfinite(r[k]) and r[k]>=0 for k in ('actual_evm2','predicted_evm2','actual_launch_signal_energy_j')):
                    raise ValueError('Invalid raw fidelity measurement')
            for condition in design['channel_config']['conditions']:
                rr=[r for r in proof['rows'] if r['condition']==condition['id']]
                if len(rr)!=c['fidelity_frames_per_condition']: raise ValueError('Incomplete fidelity frames')
                usable=[r for r in rr if r['available_fit'] and r['available_fresh']]
                if len(usable)<2: raise ValueError('Not enough real received fidelity frames')
                actual=float(np.mean([r['actual_evm2'] for r in usable])); predicted=float(np.mean([r['predicted_evm2'] for r in usable]))
                relative=abs(predicted-actual)/max(actual,1e-8)
                disagreement=float(np.mean([r['available_fit']!=r['available_fresh'] for r in rr]))
                if relative>design['channel_config']['surrogate_evm_relative_tolerance'] or disagreement>design['channel_config']['surrogate_acceptance_difference_tolerance']:
                    raise ValueError('Fidelity falsely marked passed')
        if complete:
            for phase,key in [('fso','checkpoint_sha256'),('clean','clean_checkpoint_sha256')]:
                file=folder/(phase+'_delta.pth')
                if file.exists() and digest(file)!=rec[key]: raise ValueError('Changed checkpoint')
    check_pair_histories(histories)
    if records['B3_common']['trainable_parameters']!=records['B4_separate']['trainable_parameters'] or records['B3_common']['registered_parameters']!=records['B4_separate']['registered_parameters']:
        raise ValueError('Joint/separate parameter scale differs')
    clean=read_rows(d/'clean_stage.jsonl')
    if len(clean)!=len(TRAINED_ARMS)*len(ids) or {(r['arm'],r['image_id']) for r in clean}!={(a,i) for a in TRAINED_ARMS for i in ids}:
        raise ValueError('Missing clean-stage diagnosis')
    frozen=json.loads((d/'frozen_verification.json').read_text())
    if not all(frozen.values()): raise ValueError('Historical/backbone change or test usage')
    identity=json.loads((d/'run_identity.json').read_text())
    if identity['version']!=VERSION or digest(d/'calibration_summary.json')!=identity['calibration_sha256']:
        raise ValueError('Changed provenance/calibration')
    for n,sha in identity['new_source_sha256'].items():
        if Path(n).name!=n or digest(Path(__file__).parent/n)!=sha: raise ValueError('New source drift')
    for mapping in ('source_sha256','calibration_source_sha256'):
        for n,sha in identity[mapping].items():
            if Path(n).is_absolute() or '..' in Path(n).parts or digest(EXT/n)!=sha: raise ValueError('Dependency source drift')
    if complete:
        end=json.loads((d/'COMPLETE.json').read_text())
        if end['version']!=VERSION or end['rows']!=len(rows) or identity['status']!='complete' or (d/'FAILED.json').exists():
            raise ValueError('Run not complete')
        essential={'design.json','samples.jsonl','events.jsonl','clean_stage.jsonl','frozen_verification.json','checkpoint_load.json',
                   'run_identity.json','summary.json','REPORT.md','report.html','gain_summary.svg','calibration_summary.json'}
        essential.update('train_'+a+'/'+n for a in TRAINED_ARMS for n in ('COMPLETE.json','history.jsonl','fidelity_initial.json','fidelity_clean.json','fidelity_fso.json'))
        if not essential<=set(end['result_sha256']): raise ValueError('Completion ledger lacks evidence')
        for n,sha in end['result_sha256'].items():
            if digest(safe_file(d,n))!=sha: raise ValueError('Changed result artifact')
        for arm in TRAINED_ARMS:
            for phase,key in [('fso','checkpoint_sha256'),('clean','clean_checkpoint_sha256')]:
                if end['checkpoint_sha256'].get('train_'+arm+'/'+phase+'_delta.pth')!=records[arm][key]: raise ValueError('Wrong checkpoint ledger')
    return rows


def contrast(rows,arms,conditions,metric,resamples,available=False):
    index={(r['arm'],r['image_id'],r['condition'],r['draw']):r for r in rows}
    by_image=defaultdict(list); trials=0
    for key,a in index.items():
        arm,i,c,draw=key
        if arm!=arms[0] or c not in conditions: continue
        b=index[(arms[1],i,c,draw)]
        if available and not (a['receiver_available'] and b['receiver_available']): continue
        by_image[i].append(b[metric]-a[metric]); trials+=1
    result=bootstrap([np.mean(by_image[i]) for i in sorted(by_image)],(arms,conditions,metric,available),resamples)
    result.update(arms=list(arms),metric=metric,trials=trials,both_available=available,
                  exploratory=True,multiplicity_adjusted=False)
    return result


def report(directory):
    d=Path(directory); rows=verify(d); design=json.loads((d/'design.json').read_text()); cfg=design['config']
    groups={}
    for condition in ['clean']+[c['id'] for c in design['conditions']]:
        for arm in ARMS:
            rr=[r for r in rows if r['arm']==arm and r['condition']==condition]
            mean=lambda key:float(np.mean([r[key] for r in rr]))
            groups[condition+'/'+arm]=dict(psnr=mean('psnr'),ssim=mean('ssim'),covered_wer=mean('covered_normalized_wer'),
                full_wer=mean('normalized_wer'),outage=float(np.mean([not r['receiver_available'] for r in rr])),
                papr_db=mean('waveform_papr_db'),truncated_fraction=float(np.mean([r['input_truncated'] for r in rr])),
                psnr_p10=float(np.quantile([r['psnr'] for r in rr],.1)),images=len({r['image_id'] for r in rr}),trials=len(rr))
    comparisons={}
    for label,pair in [('codec_bundle',('B1_text','B2_codec')),('common_task',('B2_codec','B3_common')),('joint',('B4_separate','B3_common'))]:
        for metric in ('psnr','ssim','covered_normalized_wer','normalized_wer'):
            for subset,av in [('all',False),('both_available',True)]:
                name=label+'/'+metric+'/'+subset
                comparisons[name]=contrast(rows,pair,design['primary_conditions'],metric,cfg['bootstrap_resamples'],av)
        # Same images and all failed frames: compare noisy improvement to own clean improvement.
        idx={(r['arm'],r['image_id'],r['condition'],r['draw']):r for r in rows}
        margin=[]
        for iid in design['validation_ids']:
            noisy=np.mean([idx[(pair[1],iid,c,r)]['psnr']-idx[(pair[0],iid,c,r)]['psnr'] for c in design['primary_conditions'] for r in range(cfg['draws'])])
            clean=idx[(pair[1],iid,'clean',0)]['psnr']-idx[(pair[0],iid,'clean',0)]['psnr']
            margin.append(noisy-clean)
        comparisons[label+'/noisy_minus_clean']=bootstrap(margin,(label,'margin'),cfg['bootstrap_resamples'])
    runtime={a:json.loads((d/('train_'+a)/'COMPLETE.json').read_text()) for a in TRAINED_ARMS}
    summary=dict(version=VERSION,groups=groups,comparisons=comparisons,training=runtime,
        clean_stage=read_rows(d/'clean_stage.jsonl'),
        interpretation='positive PSNR is better; NEGATIVE WER is better. B0 text capacity differs and is NOT a matched AR-only causal contrast.',
        limitations=design['limitations'],no_analog_digital_superiority_test=True)
    (d/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False))
    lines=['# Joint JSCC architecture V1 screening','',
        'Frozen VAE/CLIP; whole-text chunks up to configured capacity -> T32; causal AR decode; cross-token JSCC/RX public conditioning; raw-wire-C tasks.',
        'B0 is a same-posterior-mean legacy rerun (77 source tokens), NOT a same-capacity AR-only comparison. B1-4 use identical new source capacity.',
        'One transmission, 2304 complex payload symbols; all pilots/guards/outages retained. Same slot/peak limits NOT equal actual energy.',
        'B2-B1 measures codec + reliable-receiver BUNDLE, not either component alone. B3-B2 isolates Common task loss. B3-B4 is an equal-parameter masked neural joint/separate comparison; FLOPs differ.','',
        '| Condition/arm | PSNR | SSIM | Covered WER | Full-source WER | Outage |','|---|---:|---:|---:|---:|---:|']
    for name,g in groups.items():
        lines.append('| {} | {:.3f} | {:.4f} | {:.3f} | {:.3f} | {:.1%} |'.format(name,g['psnr'],g['ssim'],g['covered_wer'],g['full_wer'],g['outage']))
    lines+=['','Paired image-cluster contrasts (fixed weather/one training replicate; exploratory, no global multiplicity claim):','',
            '```json',json.dumps(comparisons,indent=2),'```','',
            'Common diagnostic decodes only C received in a FULL frame before private-symbol RX mixing. It is NOT proof of a low-bandwidth Common-only link.',
            'Clean stage and final clean controls are distinct. A short screen does not establish convergence; historical validation is not a final test.',
            'CLIP chunks have <=75 content tokens plus BOS/EOS; input truncation above total configured capacity is explicitly reported.',
            'Training CE uses teacher forcing; reported text comes from actual autoregressive generation without source text at RX.',
            'See report.html for all preselected image/text examples, including failures.']
    (d/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Small two-metric forest plots, independent axes/units; WER gain is negative.
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="470"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif">',
         '<text x="24" y="30" font-size="21">Architecture screen: matched low-quality-link contrasts</text>',
         '<text x="24" y="55" font-size="13">95% conditional image-cluster intervals; one training replicate, fixed weather paths. Not final test.</text>']
    for panel,metric in enumerate(('psnr','covered_normalized_wer')):
        names=['codec_bundle','common_task','joint']; numbers=[comparisons[n+'/'+metric+'/all'] for n in names]
        scale=1 if metric=='psnr' else 100
        bound=max(.1,max(abs(v)*scale for number in numbers for v in number['ci95']))*1.2
        left=30+panel*550; origin=left+260; span=210
        title='PSNR change (dB): right is better' if metric=='psnr' else 'Covered WER change (pp): left is better'
        svg.append('<text x="{}" y="95" font-size="15">{}</text>'.format(left,title))
        svg.append('<path d="M{} 115V375" stroke="#64748b" stroke-dasharray="4 4"/>'.format(origin))
        for j,(name,number) in enumerate(zip(names,numbers)):
            y=150+j*95; lo,hi=[origin+v*scale/bound*span for v in number['ci95']]; mid=origin+number['mean']*scale/bound*span
            svg.append('<text x="{}" y="{}" font-size="14">{}</text><path d="M{} {}H{}" stroke="#0072B2" stroke-width="3"/><circle cx="{}" cy="{}" r="5" fill="#D55E00"/><text x="{}" y="{}" font-size="12">{:+.3f} [{:+.3f}, {:+.3f}]</text>'.format(
                left,y-20,name,lo,y,hi,mid,y,left,y+25,number['mean']*scale,number['ci95'][0]*scale,number['ci95'][1]*scale))
        svg.append('<text x="{}" y="400" font-size="12">-{:.3f}</text><text x="{}" y="400" font-size="12">0</text><text x="{}" y="400" font-size="12">+{:.3f}</text>'.format(origin-span,bound,origin,origin+span-40,bound))
    svg.append('<text x="24" y="446" font-size="13">Both-available, clean-normalized margins, full WER, failures and exact resources are retained in summary.json.</text></g></svg>')
    (d/'gain_summary.svg').write_text(''.join(svg))
    esc=html.escape; cards=[]
    for condition in ['clean']+[c['id'] for c in design['conditions']]:
        for draw in range(1 if condition=='clean' else cfg['draws']):
            for iid in design['validation_ids'][:cfg['save_images']]:
                data=[r for r in rows if r['condition']==condition and r['draw']==draw and r['image_id']==iid]
                cards.append('<h2>{} · image {} · draw {}</h2><p>Original text (evaluation only): {}</p><div class="grid"><article><h3>Source</h3><img src="{}"></article>'.format(esc(condition),iid,draw,esc(data[0]['source_caption_evaluation_only']),esc(data[0]['source_image_file'])))
                for r in data:
                    cards.append('<article><h3>{}</h3><img src="{}"><p>PSNR {:.2f} · SSIM {:.3f} · available {}</p><p>Covered WER {:.3f} · full WER {:.3f}</p><p>{}</p><small>Common received: {}</small></article>'.format(
                        esc(r['arm']),esc(r['image_file']),r['psnr'],r['ssim'],r['receiver_available'],r['covered_normalized_wer'],r['normalized_wer'],esc(r['received_caption']),esc(r.get('common_caption','not measured'))))
                cards.append('</div>')
    (d/'report.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Joint JSCC architecture V1</title><style>body{font-family:system-ui;margin:24px;line-height:1.5}.grid{display:grid;grid-template-columns:repeat(6,minmax(160px,1fr));gap:16px;overflow:auto}article{background:#f1f5f9;padding:12px}img{width:100%;image-rendering:auto}small{overflow-wrap:anywhere}</style><h1>Joint JSCC architecture V1</h1><p>Preselected examples; all saved conditions and failures. B0 capacity77, B1–4 identical new capacity. Analog R=1. Teacher text is never supplied to the receiver.</p><p><a href="REPORT.md">Full report</a> · <a href="summary.json">Exact statistics</a></p><img style="max-width:1100px" src="gain_summary.svg">'+''.join(cards)+'</html>')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('directory'); p.add_argument('--verify-only',action='store_true'); a=p.parse_args()
    if a.verify_only: verify(a.directory,complete=True); print('Verified full matrix, equal-parameter control, actual updates, source/fidelity/resource and result hashes.')
    else: verify(a.directory,complete=True); report(a.directory)
