"""Strict matrix/protocol checks and paired, conditional single-seed evidence."""
import argparse
import html
import json
import math
from pathlib import Path
import sys
import numpy as np
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from bidirectional_rx_v7 import VERSION,ARMS
from bidirectional_rx_v7.core import read,lines,load,plan,geometry,mask_policy,variants,digest,array_hash,metrics,summarize,paired_ci,seal
from clip_bart_joint_v5.channel import resource


def verify(root,sealed=True):
    root=Path(root).resolve(strict=True); identity=read(root/'run_identity.json'); d=read(root/'design.json'); c=identity['config']; schedule=plan(c,identity['mode'])
    if identity['version']!=VERSION or identity['status']!='complete' or (root/'FAILED.json').exists(): raise ValueError('No successful execution')
    if any(f.is_symlink() for f in root.rglob('*')): raise ValueError('Symlink records')
    if identity['schedule']!=schedule or d['schedule']!=schedule or d['config']!=c or d['geometry']!=geometry() or d['arms']!=list(ARMS): raise ValueError('Design differs')
    if identity['TRAIN8_weights_loaded'] or identity['text_global_milestone']!=9000 or not d['source_frozen'] or not d['common_not_retrained']: raise ValueError('Wrong source training/init')
    if not read(root/'initial_preservation.json')['passed'] or not all(read(root/'frozen_verification.json')[k] for k in ('passed','backbones_unchanged','old_weights_unchanged','original_data_unchanged')): raise ValueError('Frozen/init check failed')
    ti=d['image_train_view']['image_ids']; vi=d['image_validation_view']['image_ids']
    if len(ti)!=schedule['train_samples'] or len(vi)!=schedule['val_samples'] or len(set(ti))!=len(ti) or len(set(vi))!=len(vi) or set(ti)&set(vi): raise ValueError('Split/ID mismatch')
    initial=read(root/'initial_state_hashes.json'); sizes=set()
    for phase in schedule['phases']:
        paired=None
        for arm in ARMS:
            folder=root/arm/phase; hist=lines(folder/'history.jsonl'); done=read(folder/'TRAIN_COMPLETE.json'); audit=read(folder/'learning_audit.json')
            total=schedule['epochs']*len(ti)
            if done['steps']!=total or done['optimizer_updates']!=(total+c['grad_accum']-1)//c['grad_accum'] or done['initial_hashes']!=initial or [r['step'] for r in hist]!=list(range(1,total+1)): raise ValueError('Training budget/init differs')
            if not audit['passed'] or not audit['frozen_unchanged'] or set(audit['groups'])!={'image_input','text_input','image_attention','text_attention','image_output','text_output','image_gate','text_gate'}: raise ValueError('Learning audit failed')
            if not all(v['tensors'] and v['nonzero_gradient'] and v['changed'] for v in audit['groups'].values()): raise ValueError('Missing actual learning')
            sizes.add(audit['registered_parameters'])
            forward=read(folder/'forward_consistency.json'); reload=read(folder/'reload_audit.json')
            if max(forward['errors'].values())>c['forward_tolerance'] or not reload['passed'] or not reload['actual_disk_reload'] or reload['output_max_error']>c['forward_tolerance']: raise ValueError('Forward/reload failed')
            if (folder/'last.pth').exists() and digest(folder/'last.pth')!=done['checkpoint_sha256']: raise ValueError('Weights changed')
            keys=[]
            for j,r in enumerate(hist):
                epoch,pos=divmod(j,len(ti)); order=np.random.default_rng(c['seed']+epoch).permutation(len(ti)); iid=ti[int(order[pos])]
                noisy=phase=='fso' and j%4!=0
                if r['image_id']!=iid or r['phase']!=phase or r['noisy']!=noisy or bool(r['proxy'])!=noisy or r['private_mask']!=mask_policy(c,j,iid): raise ValueError('Train order/noise/drop changed')
                if r['wire_sha256']!=d['source_wire_hashes'][str(iid)] or not math.isfinite(r['loss']): raise ValueError('Training source/loss invalid')
                if noisy and (len(r['proxy'])!=1 or r['proxy'][0]['payload_complex_symbols']!=3072 or r['proxy'][0]['analog_repeats']!=1): raise ValueError('Training resource changed')
                keys.append((iid,r['wire_sha256'],r['observed_sha256'],r['pilot_snr_db'],r['reliability']))
            if paired is not None and paired!=keys: raise ValueError('Arms saw different received damage')
            paired=keys
    if len(sizes)!=1: raise ValueError('Registered adapter size differs')
    folder=root/'evaluation'; ed=read(folder/'eval_design.json'); res=resource(d['channel_config'])
    conditions=['clean','clean_drop_image','clean_drop_text']+[v['id'] for v in d['channel_config']['conditions']]
    if ed['conditions']!=conditions or ed['variants']!=[list(v) for v in variants(schedule['phases'])] or ed['resources']!=res or ed['validation_ids']!=vi or ed['channel_config']!=d['channel_config'] or not ed['same_physical_waveform_all_decoders']: raise ValueError('Evaluation design changed')
    frames=lines(folder/'frames.jsonl'); frame_map={}; expected_frames={(cid,iid) for cid in conditions for iid in vi}
    for f in frames:
        key=f['condition'],f['image_id']
        if key not in expected_frames or key in frame_map: raise ValueError('Frame duplicate/unknown')
        name=f['frame_file']; relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0]!='frames': raise ValueError('Unsafe frame path')
        with np.load(folder/name,allow_pickle=False) as z:
            if set(z.files)!={'observed','available','pilot_snr_db','reliability'} or z['observed'].shape!=(6144,) or not np.isfinite(z['observed']).all(): raise ValueError('Cache source leak/layout')
            if array_hash(z['observed'])!=f['observed_sha256'] or bool(z['available'])!=f['receiver_available'] or float(z['pilot_snr_db'])!=f['pilot_snr_db'] or float(z['reliability'])!=f['reliability']: raise ValueError('Frame cache mismatch')
        if f['condition'].startswith('clean'):
            if f['stats'] is not None or f['pilot_snr_db']!=40. or not f['receiver_available']: raise ValueError('Clean sentinel treated as measured')
            if f['condition']=='clean' and f['observed_sha256']!=d['source_wire_hashes'][str(f['image_id'])]: raise ValueError('Clean wire differs')
            step={'clean_drop_image':1,'clean_drop_text':2}.get(f['condition'])
            expected_mask=None if step is None else mask_policy(c,step,f['image_id'])
            if f['synthetic_private_mask']!=expected_mask: raise ValueError('Synthetic mask differs')
        else:
            st=f['stats']
            if st['resources']!=res or st['used_samples']!=res['scheduled_samples'] or st['payload_complex_symbols']!=3072 or st['analog_repeats']!=1 or st['raw_utf8_wire_bytes']!=0 or st['pre_fec_ber'] is not None: raise ValueError('Hidden or wrong physical resource')
            if st['receiver_available']!=f['receiver_available'] or abs(st['pilot_snr_db']-f['pilot_snr_db'])>1e-4 or f['synthetic_private_mask'] is not None: raise ValueError('Physical observation differs')
        frame_map[key]=f
    if set(frame_map)!=expected_frames: raise ValueError('Missing full received frame matrix')
    events=lines(folder/'events.jsonl'); counts={}
    for e in events:
        k=e['condition'],e['image_id']
        if k not in expected_frames or k[0].startswith('clean'): raise ValueError('Unknown physical event')
        counts[k]=counts.get(k,0)+1
    if counts!={k:2 for k in expected_frames if not k[0].startswith('clean')}: raise ValueError('Probe/features physical events missing')
    rows=lines(folder/'samples.jsonl'); expected={(arm,phase,control,cid,iid) for arm,phase,control in variants(schedule['phases']) for cid,iid in expected_frames}; seen=set()
    for r in rows:
        key=r['arm'],r['phase'],r['control'],r['condition'],r['image_id']
        if key not in expected or key in seen or r['source_split']!='validation' or r['geometry']!=geometry(): raise ValueError('Sample matrix/layout differs')
        seen.add(key); frame=frame_map[(r['condition'],r['image_id'])]
        if r['frame_file']!=frame['frame_file'] or r['observed_sha256']!=frame['observed_sha256'] or r['stats']!=frame['stats'] or r['pilot_snr_db']!=frame['pilot_snr_db']: raise ValueError('Decoders saw different waveform')
        if r['receiver_failure']==frame['receiver_available'] or r['receiver_failure'] and r['received_caption']: raise ValueError('Unavailable receiver leaked text')
        if r['source_sha256']!=d['image_validation_view']['source_sha256'][str(r['image_id'])]: raise ValueError('Evaluation target changed')
        if any(r.get(k)!=v for k,v in metrics(r['source_text'],r['covered_text'],r['received_caption']).items()): raise ValueError('Text metric mismatch')
        if any(not math.isfinite(r[k]) for k in ('mse','psnr','ssim')) or abs(r['psnr']+10*math.log10(max(r['mse'],1e-12)))>1e-4: raise ValueError('Image metric mismatch')
        if r['control'].startswith('shuffle_'):
            donor=vi[(vi.index(r['image_id'])+1)%len(vi)]
            if r['donor_image_id']!=donor: raise ValueError('Wrong actual received donor')
        elif r['donor_image_id'] is not None: raise ValueError('Unexpected donor')
        if r['control']!='normal' and not r['receiver_intervention_not_new_transmission']: raise ValueError('Intervention resource claim')
        if r['image_id'] in vi[:min(c['save_images'],len(vi))]:
            from PIL import Image
            for field in ('image_file','source_image_file'):
                path=Path(r[field])
                if path.is_absolute() or '..' in path.parts or path.parts[0]!='images' or not (folder/path).is_file(): raise ValueError('Missing/unsafe example')
            if r['receiver_failure']:
                with Image.open(folder/r['image_file']) as im:
                    if np.asarray(im).any(): raise ValueError('Unavailable frame leaked image')
    if seen!=expected: raise ValueError('Missing decoded/control matrix')
    for name,sha in identity['source_sha256'].items():
        if Path(name).name!=name or digest(root/'source_snapshot'/name)!=sha: raise ValueError('Runtime source drift')
    if sealed:
        done=read(root/'COMPLETE.json')
        if not done['execution_only_not_quality_success'] or done['files']!=seal(root): raise ValueError('Sealed records drift')
    return dict(passed=True,rows=len(rows),frames=len(frames),parameters=next(iter(sizes)),mode=identity['mode'])


def finish(root):
    from v10.io import atomic_json
    root=Path(root); verified=verify(root,False); d=read(root/'design.json'); rows=lines(root/'evaluation/samples.jsonl')
    groups={}
    for arm,phase,control in variants(d['schedule']['phases']):
        for cid in read(root/'evaluation/eval_design.json')['conditions']:
            rr=[r for r in rows if (r['arm'],r['phase'],r['control'],r['condition'])==(arm,phase,control,cid)]
            groups['/'.join((arm,phase,control,cid))]=dict(text=summarize(rr),psnr_db=float(np.mean([r['psnr'] for r in rr])),ssim=float(np.mean([r['ssim'] for r in rr])),
                raw_exact=sum(r['source_text']==r['received_caption'] for r in rr),outages=sum(r['receiver_failure'] for r in rr),samples=len(rr))
    comparisons={}
    physical=[v['id'] for v in d['channel_config']['conditions'] if v['weather']=='moderate']
    for phase in d['schedule']['phases']:
        for arm in ARMS[1:]:
            for scope,target in (('primary_moderate_fso',physical),('clean',['clean']),('synthetic_private_drop',['clean_drop_image','clean_drop_text'])):
                comparisons[arm+'/'+phase+'/'+scope]=paired_ci(rows,arm,phase,target)
    noise_comparisons={}
    if d['schedule']['phases']==['clean','fso']:
        for arm in ARMS:
            view=[]
            for r in rows:
                if r['arm']==arm and r['control']=='normal':
                    view.append(dict(r,arm='R0_self' if r['phase']=='clean' else 'R3_bidirectional',phase='paired'))
            noise_comparisons[arm]=paired_ci(view,'R3_bidirectional','paired',physical)
    summary=dict(verification=verified,groups=groups,comparisons=comparisons,noise_training_fso_minus_clean=noise_comparisons,
        primary='R3 normal minus R0 normal, same phase, predeclared moderate physical conditions',
        limitations=['one seed, reused small validation; not independent test','training symbol proxy is not full waveform channel',
            'registered parameter count matched; active compute may differ','C/source frozen: cannot claim learned shared source semantics or compression savings',
            'no diffusion, no oracle received source, no channel bits/BER for continuous symbols'],
        perceptual_metrics=dict(lpips=None,fid=None,reason='not installed/evaluated'))
    atomic_json(root/'summary.json',summary)
    paragraphs=['# V7 bidirectional receiver screening','Execution completion is NOT a positive quality result.',
        'Compare R1/R2/R3 with R0 in the SAME phase; compare clean vs FSO training separately. B0 is a new live hybrid frozen reference, not a historical V6 replay.',
        'Clean pilot 40dB is a conditioning sentinel, NOT measured optical SNR. Drop conditions are synthetic, NOT physical outages.',
        'Primary paired comparisons (PSNR positive / WER negative is better):',json.dumps(comparisons,ensure_ascii=False,indent=2),
        'Only conditional image-cluster intervals are reported. Original source text/images in examples are evaluation targets, never receiver inputs.']
    with (root/'REPORT.md').open('x',encoding='utf-8') as f: f.write('\n\n'.join(paragraphs)+'\n')
    page=['<!doctype html><meta charset="utf-8"><title>V7 actual image/text recovery</title><style>body{font-family:system-ui;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:10px;vertical-align:top}img{width:128px}pre{white-space:pre-wrap}small{color:#666}</style>',
        '<h1>V7：相同接收波形的图文恢复对照</h1><p>原图和原文仅供评价；无扩散、无原文旁路。低质量帧失败也计入结果。关闭/置零/打乱辅助不增加传输资源。</p>',
        '<h2>按条件汇总</h2><table><tr><th>配置/训练/控制/条件</th><th>PSNR</th><th>WER</th><th>逐字相同</th><th>失锁</th></tr>']
    for k,v in groups.items():
        page.append('<tr><td>{}</td><td>{:.3f}</td><td>{:.3f}</td><td>{}/{}</td><td>{}</td></tr>'.format(html.escape(k),v['psnr_db'],v['text']['full_wer'],v['raw_exact'],v['samples'],v['outages']))
    page.append('</table><h2>固定前几张样例：所有条件，无挑选</h2>')
    for cid in read(root/'evaluation/eval_design.json')['conditions']:
        page.append('<h3>'+html.escape(cid)+'</h3><table><tr><th>评价原图/原文</th><th>配置</th><th>实际恢复</th><th>实际文字</th></tr>')
        for r in rows:
            if r['condition']!=cid or r['control']!='normal' or 'image_file' not in r: continue
            page.append('<tr><td><img src="evaluation/{}"><p>{}</p></td><td>{}/{}<br>ID{}<br>PSNR {:.2f}, WER {:.3f}<br>{}</td><td><img src="evaluation/{}"></td><td>{}</td></tr>'.format(
                html.escape(r['source_image_file']),html.escape(r['source_text']),html.escape(r['arm']),html.escape(r['phase']),r['image_id'],r['psnr'],r['normalized_wer'],
                '失锁' if r['receiver_failure'] else '正常接收',html.escape(r['image_file']),html.escape(r['received_caption'])))
        page.append('</table>')
    page.append('<h2>配对增益与限制</h2><pre>'+html.escape(json.dumps(comparisons,ensure_ascii=False,indent=2))+'</pre>')
    with (root/'actual_image_text.html').open('x',encoding='utf-8') as f: f.write('\n'.join(page))
    atomic_json(root/'COMPLETE.json',dict(version=VERSION,verification=verified,execution_only_not_quality_success=True,files=seal(root)))
    verify(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root'); a=p.parse_args(); print(json.dumps(verify(a.root),indent=2))
