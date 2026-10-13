"""Independent NumPy record verification; execution is not evidence of benefit."""
import hashlib
import html
import json
import math
from pathlib import Path
import sys
import numpy as np
from PIL import Image
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from docci_longclip_joint_v8.core import read,digest,atomic_json,jsonlines,geometry
from docci_longclip_joint_v8.metrics import metrics,summarize,paired

CONTROLS=('normal','zero_common','fusion_off','shuffle_common','shuffle_image','shuffle_text')


def under(root,name):
    p=Path(name)
    if p.is_absolute() or not p.parts or any(x in ('.','..') for x in p.parts): raise ValueError('Unsafe record-relative path')
    at=Path(root)
    for part in p.parts:
        at=at/part
        if at.is_symlink(): raise ValueError('Symlink record rejected')
    return at


def close(a,b):
    if type(a) is bool or type(b) is bool: return a is b
    if isinstance(a,(float,int)) and isinstance(b,(float,int)): return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=2e-6,abs_tol=2e-5)
    return a==b


def verify_controls(root,design):
    rows=jsonlines(root/'prefit/validation_controls.jsonl'); expected={x['example_id'] for x in design['validation_view']['records']}
    source={r['example_id']:r for r in read(root/'docci_view.json')['validation']}; seen=set(); groups={k:[] for k in ('matched','zero','shuffled')}
    for row in rows:
        key=(row['example_id'],row['control'])
        if key in seen or row['example_id'] not in expected or row['control'] not in groups or row['source_split']!='qual_dev': raise ValueError('Invalid prefit controls matrix')
        seen.add(key)
        if row['input_truncated'] is not False or row['source_text']!=source[row['example_id']]['text']: raise ValueError('Wrong full-source text')
        for k,v in metrics(row['source_text'],row['source_text'],row['received_caption']).items():
            if not close(row[k],v): raise ValueError('Prefit recomputed metric differs '+k)
        groups[row['control']].append(row)
    if seen!={(eid,k) for eid in expected for k in groups}: raise ValueError('Missing prefit control')
    groups={k:summarize(v) for k,v in groups.items()}; c=design['config']; matched=groups['matched']['full_wer']
    passed=matched<=c['text_wer_gate'] and min(groups[k]['full_wer']-matched for k in ('zero','shuffled'))>=c['memory_margin_gate'] and groups['matched']['capacity_stop_fraction']==0
    gate=read(root/'prefit/text_gate.json')
    if bool(gate['passed'])!=bool(passed) or not close(gate['full_wer'],matched): raise ValueError('Declared prefit gate differs')
    return groups,bool(passed)


def verify(root):
    root=Path(root).resolve(strict=True); identity=read(root/'run_identity.json'); design=read(root/'design.json'); c=design['config']; g=geometry(c)
    if identity['version']!='docci-longclip-joint-v8' or identity['config']!=c or identity['schedule']!=design['schedule']: raise ValueError('Run identity differs')
    if identity['status'] not in ('quality_stop','finished_execution'): raise ValueError('Run not in verifiable terminal state')
    for name,sha in identity['source_sha256'].items():
        if digest(under(root/'source_snapshot',name))!=sha: raise ValueError('Source snapshot differs')
    if read(root/'frozen_verification.json')['passed'] is not True: raise ValueError('Frozen/source identity check absent')
    if read(root/'checkpoint_load.json')['longclip_sha256']!='ca2face3716d60473a4e284e5be2cf0a1f3a4ad9954a6a651d878e814cc0c894': raise ValueError('Wrong Long-CLIP-B')
    tv=design['train_view']['records']; vv=design['validation_view']['records']; source=read(root/'docci_view.json')
    tr={r['example_id'] for r in tv}; vr={r['example_id'] for r in vv}
    if len(tr)!=design['schedule']['train_samples'] or len(vr)!=design['schedule']['val_samples'] or tr&vr: raise ValueError('DOCCI counts/split overlap')
    if any(r['official_split']!='train' for r in tv) or any(r['official_split']!='qual_dev' for r in vv) or design['test_used'] is not False: raise ValueError('Wrong DOCCI official split')
    if any(not 2<=r['source_tokens']<=c['max_text_tokens'] or r['clip_chunks']!=math.ceil(r['clip_tokens']/246) for r in tv+vv): raise ValueError('Full token coverage differs')
    for prefix in ('prefit/text','prefit/image'):
        if read(root/prefix/'learning_audit.json')['passed'] is not True: raise ValueError('Prefit actual learning failed')
        done=read(root/prefix/'TRAIN_COMPLETE.json')
        weight=root/prefix/'last.pth'
        if weight.exists() and digest(weight)!=done['checkpoint_sha256']: raise ValueError('Live prefit weight changed')
    groups,gate=verify_controls(root,design)
    result=dict(mode=identity['mode'],image_size=c['image_size'],geometry=g,text_prefit_passed=gate,text_prefit=groups,
        status=identity['status'],wall_seconds=identity['wall_seconds'],claims_not_implied_by_completion=True,local_vs_server_scope='records verified, no hardware calibration proof')
    if identity['status']=='quality_stop':
        if gate or (root/'evaluation').exists(): raise ValueError('Quality stop inconsistent')
        return result,[]
    if identity['mode']!='smoke' and not gate: raise ValueError('Formal joint run bypassed text gate')
    for arm in ('separate','joint'):
        for stage in ('clean','fso'):
            folder=root/arm/stage; audit=read(folder/'learning_audit.json'); done=read(folder/'TRAIN_COMPLETE.json')
            if audit['passed'] is not True or not all(v['changed'] and v['nonzero_gradient'] and v['parameters']>0 for v in audit['groups'].values()): raise ValueError('Required communication group did not learn')
            if done['arm']!=arm or done['stage']!=stage: raise ValueError('Checkpoint provenance differs')
            if (folder/'last.pth').exists() and digest(folder/'last.pth')!=done['checkpoint_sha256']: raise ValueError('Live checkpoint drift')
    ev=root/'evaluation'; matrix=read(ev/'matrix.json'); done=read(ev/'EVAL_COMPLETE.json'); rows=jsonlines(ev/'rows.jsonl')
    if matrix['controls']!=list(CONTROLS) or matrix['arms']!=['separate','joint'] or matrix['stages']!=['clean','fso'] or set(matrix['example_ids'])!=vr: raise ValueError('Declared eval matrix differs')
    expected={(a,s,cond,eid,k) for a in matrix['arms'] for s in matrix['stages'] for cond in matrix['conditions'] for eid in vr for k in CONTROLS}
    if done['rows']!=len(expected) or done['rows_sha256']!=digest(ev/'rows.jsonl'): raise ValueError('Eval completion differs')
    vsource={r['example_id']:r for r in source['validation']}; vreceipt={r['example_id']:r for r in vv}; seen=set(); frames={}; paired_rows={}
    for row in rows:
        key=tuple(row[k] for k in ('arm','stage','condition','example_id','control'))
        if key in seen or key not in expected: raise ValueError('Missing/duplicate/unknown evaluation key')
        seen.add(key); eid=row['example_id']; reference=vsource[eid]; receipt=vreceipt[eid]
        if row['official_split']!='qual_dev' or row['input_truncated'] is not False or row['image_size']!=c['image_size']: raise ValueError('Incorrect source coverage')
        if row['source_text']!=reference['text'] or row['source_text_sha256']!=reference['text_sha256'] or row['source_image_sha256']!=receipt['image_sha256']: raise ValueError('Changed source target')
        if row['source_tokens']!=receipt['source_tokens'] or row['evaluation_rectangle']!=receipt['rectangle']: raise ValueError('Token/rectangle provenance differs')
        for k,v in metrics(row['source_text'],row['source_text'],row['received_caption']).items():
            if not close(row[k],v): raise ValueError('Full-source metric mismatch '+k)
        if not all(math.isfinite(row[k]) for k in ('mse','psnr','ssim')) or not 0<=row['mse']<=1 or not -1.0001<=row['ssim']<=1.0001:
            raise ValueError('Invalid image metrics')
        if not math.isclose(row['psnr'],-10*math.log10(max(row['mse'],1e-12)),rel_tol=2e-6,abs_tol=1e-4): raise ValueError('PSNR/MSE differ')
        fn=row['frame_file']; file=under(ev,fn)
        if fn not in frames:
            if digest(file)!=row['frame_sha256']: raise ValueError('Received frame changed')
            with np.load(file,allow_pickle=False) as z:
                if set(z.files)!={'observed','available','pilot_snr_db','reliability'}: raise ValueError('Forbidden clean source/oracle in RX frame')
                x=z['observed']; snr=float(z['pilot_snr_db']); rel=float(z['reliability']); ok=bool(z['available'])
                if x.dtype!=np.float32 or x.shape!=(g['total_reals'],) or not np.isfinite(x).all() or not math.isfinite(snr) or not 0<=rel<=1: raise ValueError('Invalid public RX frame')
                frames[fn]=dict(sha=digest(file),array_sha=hashlib.sha256(x.tobytes()).hexdigest(),snr=snr,rel=rel,ok=ok)
        frame=frames[fn]; stats=row['stats']
        if frame['sha']!=row['frame_sha256'] or frame['array_sha']!=row['received_sha256'] or not close(frame['snr'],stats['pilot_snr_db']) or not close(frame['rel'],stats['analog_reliability']) or frame['ok']!=stats['receiver_available']: raise ValueError('Received public CSI/frame differs')
        if row['receiver_failure']!= (not frame['ok']) or (not frame['ok'] and row['received_caption']!=''): raise ValueError('Receiver failures not retained')
        if stats['payload_complex_symbols']!=g['payload_complex_symbols'] or stats['analog_repeats']!=1 or stats['raw_utf8_wire_bytes']!=0 or stats['resources']!=matrix['resources']: raise ValueError('Charged public resource differs')
        if row['condition']!='clean':
            if stats['resource_feasible'] is not True or stats['slot_samples']!=matrix['resources']['slot_samples'] or stats['used_samples']>stats['slot_samples']: raise ValueError('Physical cap exceeded')
            if stats['launch_signal_energy_j']>stats['signal_energy_budget_j']*(1+1e-8): raise ValueError('Optical energy exceeded')
        if row['control']=='normal': paired_rows[key[:4]]=row
        if row['control'].startswith('shuffle_') and (row['donor_example_id']==eid or row['donor_example_id'] not in vr): raise ValueError('Invalid received donor')
        image=row['restored_image_file']
        if image:
            path=under(ev,image)
            if digest(path)!=row['restored_image_sha256']: raise ValueError('Recovered PNG changed')
            a=np.asarray(Image.open(path).convert('RGB'),np.float64)/255; b=np.asarray(Image.open(ev/'images'/('source__'+eid+'.png')).convert('RGB'),np.float64)/255
            if a.shape!=(c['image_size'],c['image_size'],3) or b.shape!=a.shape: raise ValueError('Recovered/source image size differs')
            l,t,r,bottom=row['evaluation_rectangle']; mse=float(np.mean((a[t:bottom,l:r]-b[t:bottom,l:r])**2))
            tolerance=(1/255.)*(2*math.sqrt(row['mse'])+1/255.)+1e-7
            if abs(mse-row['mse'])>tolerance: raise ValueError('PNG reconstruction does not support reported floating MSE')
    if seen!=expected: raise ValueError('Incomplete held-out matrix')
    for row in rows:
        own=paired_rows[(row['arm'],row['stage'],row['condition'],row['example_id'])]
        if any(row[k]!=own[k] for k in ('frame_file','frame_sha256','tx_wire_sha256','noise_seed')): raise ValueError('Counterfactual did not reuse same frame')
        if row['donor_example_id']:
            donor=paired_rows[(row['arm'],row['stage'],row['condition'],row['donor_example_id'])]
            if row['donor_received_sha256']!=donor['received_sha256']: raise ValueError('Shuffled donor is not actual received wire')
    summaries={}; gains={}
    for arm in matrix['arms']:
        for stage in matrix['stages']:
            for cond in matrix['conditions']:
                for control in CONTROLS:
                    chosen=[r for r in rows if (r['arm'],r['stage'],r['condition'],r['control'])==(arm,stage,cond,control)]
                    summaries['/'.join((arm,stage,cond,control))]=dict(summarize(chosen),psnr=float(np.mean([r['psnr'] for r in chosen])),ssim=float(np.mean([r['ssim'] for r in chosen])),
                        receiver_failure_fraction=float(np.mean([r['receiver_failure'] for r in chosen])))
    for stage in matrix['stages']:
        for cond in matrix['conditions']:
            a=[paired_rows[('separate',stage,cond,eid)] for eid in matrix['example_ids']]
            b=[paired_rows[('joint',stage,cond,eid)] for eid in matrix['example_ids']]
            if any(x['noise_seed']!=y['noise_seed'] for x,y in zip(a,b)): raise ValueError('Across-arm channel noise pairing differs')
            gains[stage+'/'+cond]=dict(psnr_joint_minus_separate_db=paired([y['psnr']-x['psnr'] for x,y in zip(a,b)],c['seed']),
                wer_separate_minus_joint=paired([x['normalized_wer']-y['normalized_wer'] for x,y in zip(a,b)],c['seed']))
    result.update(rows=len(rows),received_frames=len(frames),summaries=summaries,paired_gains=gains,
        conclusions_scope='exploratory DOCCI qual_dev, single training seed; no claim that larger inputs or Long-CLIP cause improvement',
        matched='same native source/cohort, registered parameters, C/I/T budget, path/noise seeds and hardware caps',
        unmatched='active attention compute and actual waveform/energy; clean vs FSO total training steps; old-vs-new data/resolution',
        continuous_vs_digital='no digital baseline in this architecture diagnostic; cannot claim advantage over JPEG/FEC/QAM')
    return result,rows


def render(root,result,rows):
    title='DOCCI + Long-CLIP V8'; parts=['<!doctype html><meta charset="utf-8"><title>'+title+'</title><style>body{font:16px system-ui;margin:28px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd;padding:12px;vertical-align:top}img{max-width:260px;width:100%}.text{white-space:pre-wrap;max-width:420px}small{color:#555}</style><h1>'+title+'</h1>']
    parts.append('<p>'+html.escape(result['status'])+' / '+str(result['image_size'])+' pixels. Official DOCCI qual_dev; full description. No diffusion; R=1. Execution success is not a benefit claim.</p>')
    if not rows:
        parts.append('<h2>Long-text prefit stopped</h2><p>Joint training was not started. Preserve the diagnostics and extend/revise the private codec in a NEW run.</p><pre>'+html.escape(json.dumps(result['text_prefit'],ensure_ascii=False,indent=2))+'</pre>')
        controlrows=jsonlines(root/'prefit/validation_controls.jsonl')
        parts.append('<table><tr><th>Full original</th><th>Control</th><th>Recovered</th><th>WER</th></tr>')
        for r in controlrows[:24]: parts.append('<tr><td class="text">'+html.escape(r['source_text'])+'</td><td>'+r['control']+'</td><td class="text">'+html.escape(r['received_caption'])+'</td><td>{:.3f}</td></tr>'.format(r['normalized_wer']))
        parts.append('</table>')
    else:
        matrix=read(root/'evaluation/matrix.json'); shown=matrix['example_ids'][:2]
        parts.append('<p>First two fixed validation examples shown, not selected by PSNR. Image metrics exclude letterbox padding. Continuous payload has no defined BER or bit rate. PHY assumptions are not hardware-calibrated.</p>')
        parts.append('<table><tr><th>Original DOCCI image/text</th><th>Model/channel</th><th>Recovered image</th><th>Full recovered text</th></tr>')
        for row in rows:
            if row['control']!='normal' or row['example_id'] not in shown: continue
            parts.append('<tr><td><img src="evaluation/images/source__'+row['example_id']+'.png"><div class="text">'+html.escape(row['source_text'])+'</div></td><td>'+html.escape('/'.join(row[k] for k in ('arm','stage','condition')))+'<br>PSNR={:.2f} dB<br>SSIM={:.3f}<br>WER={:.3f}<br>Pilot SNR={:.2f} dB<br>lock={}</td><td><img src="evaluation/{}"></td><td class="text">{}</td></tr>'.format(row['psnr'],row['ssim'],row['normalized_wer'],row['stats']['pilot_snr_db'],not row['receiver_failure'],html.escape(row['restored_image_file']),html.escape(row['received_caption'])))
        parts.append('</table><h2>Paired joint gains (positive is better)</h2><pre>'+html.escape(json.dumps(result['paired_gains'],ensure_ascii=False,indent=2))+'</pre>')
        parts.append('<h2>All controls and metrics</h2><pre>'+html.escape(json.dumps(result['summaries'],ensure_ascii=False,indent=2))+'</pre>')
    (root/'actual_image_text.html').write_text(''.join(parts),encoding='utf-8')


def finish(root):
    root=Path(root); result,rows=verify(root)
    atomic_json(root/'summary.json',result); render(root,result,rows)
    marker='QUALITY_STOP.json' if result['status']=='quality_stop' else 'SMOKE_COMPLETE.json' if result['mode']=='smoke' else 'COMPLETE.json'
    atomic_json(root/marker,dict(version='docci-longclip-joint-v8',summary_sha256=digest(root/'summary.json'),
        scope='engineering execution/memory only' if result['mode']=='smoke' else 'quality stop, NOT joint completion' if result['status']=='quality_stop' else 'verified exploratory validation; NOT paper proof',
        text_prefit_passed=result['text_prefit_passed'],geometry=result['geometry']))
    return result


if __name__=='__main__':
    if len(sys.argv)!=2: raise SystemExit('report.py PROJECT_RUN_DIR')
    r=finish(Path(sys.argv[1])); print(json.dumps({k:r[k] for k in ('mode','status','image_size','text_prefit_passed','wall_seconds')},indent=2))
