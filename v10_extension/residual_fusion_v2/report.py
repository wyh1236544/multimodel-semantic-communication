"""Independent raw-row/paired receipt/gradient/diagnostic/ledger verification."""
import argparse
from collections import defaultdict
import html
import json
import math
import sys
from pathlib import Path
import numpy as np
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from residual_fusion_v2 import VERSION,ARMS,TRAINED_ARMS
from residual_fusion_v2.core import digest,check_receipt_group,diagnostic_decision,paired_bootstrap,augmentation,donor_index,parameter_counts
from fso_jscc_v1.report import safe_file,read_rows
from final_compare_v1.metrics import caption_metrics


def verify_diagnostics(d,design):
    c=design['config']; root=d/'text_diagnostics'; gate=json.loads((root/'gate.json').read_text())
    rows=read_rows(root/'samples.jsonl'); ids=design['train_ids'][:c['diagnostic_samples']]; means={}
    for label in ('D0_precompression','D1_T32_clean'):
        rr=[r for r in rows if r['label']==label]
        if len(rr)!=len(ids) or {r['image_id'] for r in rr}!=set(ids): raise ValueError('Incomplete/duplicate diagnosis')
        rec=json.loads((root/label/'COMPLETE.json').read_text())
        if rec['updates']!=c['diagnostic_updates'] or rec['train_ids']!=ids or not rec['checkpoint_roundtrip_verified'] or not rec['changed_names'] or not rec['observed_gradient_names']:
            raise ValueError('Incomplete actual diagnostic training')
        history=read_rows(root/label/'history.jsonl')
        if len(history)!=c['diagnostic_updates'] or [r['step'] for r in history]!=list(range(1,len(history)+1)) or any(r['image_id'] not in ids or not math.isfinite(r['text_ce']) for r in history):
            raise ValueError('Invalid diagnostic history')
        means[label]=float(np.mean([r['normalized_wer'] for r in rr]))
        if abs(rec['actual_covered_wer']-means[label])>1e-12: raise ValueError('False diagnostic score')
    h0=read_rows(root/'D0_precompression/history.jsonl'); h1=read_rows(root/'D1_T32_clean/history.jsonl')
    if [(r['image_id'],r['schedule_seed']) for r in h0]!=[(r['image_id'],r['schedule_seed']) for r in h1]: raise ValueError('Unpaired text diagnosis')
    rec0=json.loads((root/'D0_precompression/COMPLETE.json').read_text()); rec1=json.loads((root/'D1_T32_clean/COMPLETE.json').read_text())
    if rec0['initial_parameter_sha256']!=rec1['initial_parameter_sha256']: raise ValueError('Different diagnostic decoder initialization')
    expected=diagnostic_decision(means,c['diagnostic_wer_gate'],design['smoke'])
    for k,v in expected.items():
        if gate[k]!=v: raise ValueError('False gate decision '+k)
    for r in rows:
        if r['source_split']!='train' or r['image_id'] not in ids or r['source_text_at_receiver']:
            raise ValueError('Nontrain/original-text diagnostic input')
        calculated=caption_metrics(r['covered_reference_evaluation_only'],r['received_caption'])
        if abs(calculated['normalized_wer']-r['normalized_wer'])>1e-12: raise ValueError('Incorrect generated WER')
        generated=r['generated_token_ids']; tok=design['tokenizer_ids']
        if not generated or generated[0]!=tok['bos'] or len(generated)-1!=r['generated_tokens'] or len(generated)>128:
            raise ValueError('Incorrect generation length/prefix')
        if r['stopped_by_eos']!=(generated[-1]==tok['eos']) or r['stopped_by_capacity']==r['stopped_by_eos']:
            raise ValueError('Incorrect termination marker')
        if r['stopped_by_capacity'] and len(generated)!=128: raise ValueError('Premature capacity marker')
    actual={(r['image_id'],r['condition']) for r in rows if r['label']=='D2_T32_FSO'}
    if actual!={(i,c) for i in ids for c in design['diagnostic_conditions']} or sum(r['label']=='D2_T32_FSO' for r in rows)!=len(actual):
        raise ValueError('Incomplete/noisy diagnostic matrix')
    return gate


def verify(directory,complete=True,status=None):
    d=Path(directory); design=json.loads((d/'design.json').read_text()); c=design['config']
    identity=json.loads((d/'run_identity.json').read_text()); status=status or identity['status']
    if design['version']!=VERSION or identity['version']!=VERSION or status not in ('complete','diagnostic_stop') or identity['status']!=status:
        raise ValueError('Wrong result identity/status')
    if tuple(design['arms'])!=ARMS or design['split']!='validation' or set(design['train_ids'])&set(design['validation_ids']): raise ValueError('Wrong split/arms')
    if len(design['validation_ids'])!=c['val_samples'] or len(set(design['validation_ids']))!=c['val_samples']: raise ValueError('Invalid validation IDs')
    gate=verify_diagnostics(d,design)
    if status=='diagnostic_stop' and (gate['proceed'] or design['smoke']): raise ValueError('False diagnostic stop')
    if status=='complete' and not gate['proceed']: raise ValueError('Quality gate bypassed')
    frozen=json.loads((d/'frozen_verification.json').read_text())
    if set(frozen)!={'all_B4_parameters_unchanged','all_legacy_parameters_unchanged','all_B4_gradients_none','all_legacy_gradients_none','no_test_split_used'} or not all(frozen.values()):
        raise ValueError('Frozen backbone/source contract violated')
    if digest(d/'calibration_summary.json')!=identity['calibration_sha256']: raise ValueError('Changed calibration')
    for n,sha in identity['new_source_sha256'].items():
        if Path(n).name!=n or digest(Path(__file__).parent/n)!=sha: raise ValueError('Source drift')
    expected_source={f.name for f in Path(__file__).parent.iterdir() if f.suffix in ('.py','.json','.md')}
    if set(identity['new_source_sha256'])!=expected_source: raise ValueError('Incomplete new source identity')
    manifest=json.loads((Path(__file__).parent/'MANIFEST.json').read_text())
    for n,sha in manifest['required_sources'].items():
        if digest(safe_file(EXT.parent,n))!=sha: raise ValueError('Historical required dependency drift')
    for n,sha in identity['source_sha256'].items():
        if Path(n).is_absolute() or '..' in Path(n).parts or digest(EXT/n)!=sha: raise ValueError('Dependency source drift')
    rows=[]
    if status=='diagnostic_stop':
        if (d/'COMPLETE.json').exists() or (d/'samples.jsonl').exists() or any(d.glob('train_R*')): raise ValueError('Stopped diagnosis masquerades as complete main run')
    else:
        proof=json.loads((d/'fidelity_fixed_B4.json').read_text()); bank=json.loads((d/'calibration_summary.json').read_text())
        records={r['index']:r for r in bank['records']}; cc=design['channel_config']
        if not proof['passed'] or proof['stage']!='fixed_B4': raise ValueError('Missing fixed payload fidelity')
        for condition in cc['conditions']:
            rr=[r for r in proof['rows'] if r['condition']==condition['id']]
            if len(rr)!=design['architecture_config']['fidelity_frames_per_condition']: raise ValueError('Incomplete fidelity')
            if len({r['calibration_index'] for r in rr})!=len(rr): raise ValueError('Duplicate fidelity frame')
            for r in rr:
                bankrow=records[r['calibration_index']]
                if r['source_image_id'] not in design['train_ids'] or bankrow['partition']!='holdout' or bankrow['condition']!=r['condition'] or bankrow['available']!=r['available_fit']:
                    raise ValueError('Invalid fidelity split/availability')
                if any(not math.isfinite(r[k]) or r[k]<0 for k in ('actual_evm2','predicted_evm2','actual_launch_signal_energy_j')): raise ValueError('Invalid fidelity metric')
            good=[r for r in rr if r['available_fit'] and r['available_fresh']]
            actual=np.mean([r['actual_evm2'] for r in good]); predicted=np.mean([r['predicted_evm2'] for r in good])
            if len(good)<2 or abs(actual-predicted)/max(actual,1e-8)>cc['surrogate_evm_relative_tolerance'] or np.mean([r['available_fit']!=r['available_fresh'] for r in rr])>cc['surrogate_acceptance_difference_tolerance']:
                raise ValueError('False fidelity pass')
        initial=json.loads((d/'residual_initial.json').read_text()); counts=initial['parameter_counts']; train_records=[]; histories=[]
        expected_parameters=2*256*c['residual_hidden']+4*c['residual_hidden']**2+97+4*c['residual_hidden']
        if counts!=parameter_counts(256,c['residual_hidden']) or sum(counts.values())!=expected_parameters: raise ValueError('Unexpected residual parameter scale')
        for arm in TRAINED_ARMS:
            root=d/('train_'+arm); rec=json.loads((root/'COMPLETE.json').read_text()); train_records.append(rec)
            if rec['arm']!=arm or rec['steps']!=design['train_forward_steps'] or rec['optimizer_steps']!=design['optimizer_steps'] or not rec['checkpoint_roundtrip_verified']:
                raise ValueError('Unequal/incomplete training')
            if rec['trainable_parameters']!=sum(counts.values()) or rec['initialization_sha256']!=initial['initial_parameter_sha256']:
                raise ValueError('Different parameter count/initialization')
            if set(rec['parameter_changes'])!=set(counts) or not set(rec['observed_gradient_names'])<=set(counts): raise ValueError('Invalid update coverage')
            if any(not math.isfinite(v) or v<0 for v in rec['parameter_changes'].values()): raise ValueError('Invalid parameter change evidence')
            for prefix in ('query.','memory.','attention.','gate.','output.'):
                if sum(v for n,v in rec['parameter_changes'].items() if n.startswith(prefix))<=0 or not any(n.startswith(prefix) for n in rec['observed_gradient_names']):
                    raise ValueError('No actual required update/gradient')
            history=read_rows(root/'history.jsonl'); histories.append(history)
            if len(history)!=rec['steps'] or [r['forward_step'] for r in history]!=list(range(1,rec['steps']+1)): raise ValueError('Missing training history')
            for r in history:
                if r['image_id'] not in design['train_ids'] or r['augmentation'] not in ('correct','zero','shuffle'): raise ValueError('Invalid training ID/augmentation')
                pos=(r['forward_step']-1)%c['train_samples']
                if r['augmentation']!=augmentation(c,r['phase'],r['epoch']-1,pos): raise ValueError('Changed deterministic augmentation')
                if any(not math.isfinite(r[k]) for k in ('pixel_mse','base_pixel_mse','latent_mse','loss','gate')): raise ValueError('Nonfinite history')
                if r['augmentation']=='shuffle' and r['donor_id']!=design['train_ids'][donor_index(design['train_ids'].index(r['image_id']),len(design['train_ids']))]: raise ValueError('Wrong distinct donor')
                if r['augmentation']!='shuffle' and r['donor_id'] is not None: raise ValueError('Unexpected donor')
                if r['bank_index'] is not None and (records[r['bank_index']]['partition']!='train' or r['task_loss_applied']!=records[r['bank_index']]['available']): raise ValueError('Calibration holdout/mask leakage')
                if r['phase']=='clean' and r['bank_index'] is not None: raise ValueError('Noisy clean training')
                if r['bank_index'] is None and not r['task_loss_applied']: raise ValueError('Masked clean loss')
            if {p:sum(r['phase']==p for r in history) for p in ('clean','fso')}!={'clean':c['train_samples']*c['clean_epochs'],'fso':c['train_samples']*c['noise_epochs']}: raise ValueError('Wrong phase budget')
        fields=('phase','epoch','image_id','noise_seed','bank_index','augmentation','donor_id','task_loss_applied','optimizer_step')
        if [tuple(r[k] for k in fields) for r in histories[0]]!=[tuple(r[k] for k in fields) for r in histories[1]]: raise ValueError('Unpaired training schedule')
        rows=read_rows(d/'samples.jsonl'); groups=defaultdict(list)
        for r in rows:
            groups[(r['image_id'],r['condition'],r['draw'])].append(r)
            if r['source_sha256']!=design['validation_source_sha256'][str(r['image_id'])]: raise ValueError('Source mismatch')
            for k in ('psnr','ssim','normalized_wer','covered_normalized_wer','fusion_gate'):
                if not math.isfinite(r[k]): raise ValueError('Nonfinite recovery metric')
            if r['pre_fec_ber'] is not None or r['post_fec_ber'] is not None: raise ValueError('Fictitious analog BER')
            for name in ('image_file','source_image_file','constellation_file'):
                if name in r and digest(safe_file(d,r[name]))!=r[name+'_sha256']: raise ValueError('Changed visualization')
        expected={(i,'clean',0) for i in design['validation_ids']}|{(i,c['id'],draw) for i in design['validation_ids'] for c in design['conditions'] for draw in range(design['config']['draws'])}
        if set(groups)!=expected or len(rows)!=len(expected)*len(ARMS): raise ValueError('Incomplete main matrix')
        ev=defaultdict(list)
        for e in read_rows(d/'events.jsonl'): ev[e['frame_key']].append(e)
        expected_events=set()
        for key,rr in groups.items():
            check_receipt_group(rr); ref=rr[0]; frame=ref['frame_key']
            calculated=caption_metrics(ref['source_caption_evaluation_only'],ref['received_caption'])
            if any(abs(calculated['normalized_wer']-r['normalized_wer'])>1e-12 for r in rr): raise ValueError('Invalid actual text metric')
            joint=next(r for r in rr if r['arm']=='R1_joint'); zero=next(r for r in rr if r['arm']=='R1_zero_text'); shuffled=next(r for r in rr if r['arm']=='R1_shuffled_text')
            baseline=next(r for r in rr if r['arm']=='B4_frozen')
            for r in rr:
                if r['arm'] in ('R1_off','R1_zero_text') and (r['psnr']!=baseline['psnr'] or r['ssim']!=baseline['ssim']): raise ValueError('Disabled/zero-text recovery differs from baseline')
            if not ref['receiver_available'] and any(r['latent_sha256']!=baseline['latent_sha256'] or r['fusion_gate']!=0 for r in rr): raise ValueError('Unavailable frame used nonzero fusion')
            empty=np.zeros((1,48,256),dtype=np.float32)
            empty_sha=__import__('hashlib').sha256(str(empty.dtype).encode()+json.dumps(list(empty.shape)).encode()+empty.tobytes()).hexdigest()
            if zero['fusion_memory_sha256']!=empty_sha or zero['latent_sha256']!=baseline['latent_sha256']: raise ValueError('Zero text must recover exact frozen baseline')
            donor=design['validation_ids'][donor_index(design['validation_ids'].index(key[0]),len(design['validation_ids']))]
            donor_joint=next(r for r in groups[(donor,key[1],key[2])] if r['arm']=='R1_joint')
            if shuffled['fusion_donor_id']!=donor or shuffled['fusion_memory_sha256']!=donor_joint['fusion_memory_sha256']: raise ValueError('Batch-one shuffled memory not a real donor')
            if key[1]=='clean':
                if frame in ev or not ref['receiver_available']: raise ValueError('Invalid clean control')
            else:
                expected_events.add(frame); ee=ev[frame]
                if len(ee)!=2 or [e['purpose'] for e in ee]!=['probe','features'] or ee[1]['data_symbols']!=2304 or ee[1]['receiver_locked']!=ref['receiver_available']: raise ValueError('Hidden/missing transmission')
                samples=sum(e['awg_samples']+e['guard_samples'] for e in ee)
                energy=sum(e['launch_signal_energy_j'] for e in ee)
                seconds=samples/cc['fs_hz']
                for r in rr:
                    if samples!=r['awg_samples']+r['guard_samples'] or abs(energy-r['launch_signal_energy_j'])>max(1e-24,abs(energy)*1e-10): raise ValueError('Different/uncharged actual resources')
                    if abs(r['scheduled_seconds']-seconds)>1e-18: raise ValueError('Uncharged time/different decoder duration')
                if samples>design['environment'][2]['slot_samples']: raise ValueError('Slot budget exceeded')
                for e in ee:
                    if abs(e['waveform_seconds']-e['awg_samples']/cc['fs_hz'])>1e-18: raise ValueError('Wrong event waveform duration')
                    if e['launch_signal_energy_j']>e['launch_peak_signal_bound_w']*e['waveform_seconds']*(1+1e-8): raise ValueError('Peak envelope exceeded')
        if set(ev)!=expected_events: raise ValueError('Extra physical events')
    if complete:
        marker='COMPLETE.json' if status=='complete' else 'DIAGNOSTIC_COMPLETE.json'; end=json.loads((d/marker).read_text())
        if end['version']!=VERSION or end['status']!=status or (d/'FAILED.json').exists(): raise ValueError('False completion')
        essential={'design.json','run_identity.json','checkpoint_load.json','calibration_summary.json','frozen_verification.json',
            'summary.json','REPORT.md','report.html','text_diagnostics/gate.json','text_diagnostics/samples.jsonl'}
        essential.update('text_diagnostics/'+label+'/'+name for label in ('D0_precompression','D1_T32_clean') for name in ('COMPLETE.json','history.jsonl'))
        if status=='complete':
            essential.update({'samples.jsonl','events.jsonl','residual_initial.json','fidelity_fixed_B4.json'})
            essential.update('train_'+a+'/'+n for a in TRAINED_ARMS for n in ('COMPLETE.json','history.jsonl'))
        if not essential<=set(end['result_sha256']): raise ValueError('Incomplete evidence ledger')
        for n,sha in end['result_sha256'].items():
            if digest(safe_file(d,n))!=sha: raise ValueError('Changed result file')
        for n,sha in end['checkpoint_sha256'].items():
            file=safe_file(d,n)
            if file.exists() and digest(file)!=sha: raise ValueError('Changed checkpoint')
        for label in ('D0_precompression','D1_T32_clean'):
            name='text_diagnostics/'+label+'/decoder_delta.pth'; rec=json.loads((d/'text_diagnostics'/label/'COMPLETE.json').read_text())
            if end['checkpoint_sha256'].get(name)!=rec['checkpoint_sha256']: raise ValueError('Wrong diagnostic checkpoint ledger')
        if status=='complete':
            for arm in TRAINED_ARMS:
                rec=json.loads((d/('train_'+arm)/'COMPLETE.json').read_text())
                if end['checkpoint_sha256'].get('train_'+arm+'/last_delta.pth')!=rec['checkpoint_sha256']: raise ValueError('Wrong residual checkpoint ledger')
    return rows


def report(directory,status):
    d=Path(directory); design=json.loads((d/'design.json').read_text()); c=design['config']; esc=html.escape
    gate=json.loads((d/'text_diagnostics/gate.json').read_text()); dr=read_rows(d/'text_diagnostics/samples.jsonl')
    summary=dict(version=VERSION,status=status,diagnostic_gate=gate,diagnostic_only_text_weights=True,
        interpretation='positive image difference is better; WER lower is better; all main captions use the SAME frozen B4 decoder',
        limitations=design['limitations'])
    cards=['<h1>接收端残差融合 V2</h1><p>状态：'+esc(status)+'</p><p>诊断权重不替代正式文字头。原文只供监督/评价，主接收端无原文旁路。先看诊断，再看同一接收帧上的融合结果。</p>']
    if status=='diagnostic_stop':
        cards.append('<h2>文字定位门禁未通过：正式融合实验未启动</h2><p>请返回本记录包定位问题；不要自行放宽阈值或称为模型效果已验证。</p>')
    else:
        rows=read_rows(d/'samples.jsonl'); summary['groups']={}
        for cond in ['clean']+[r['id'] for r in design['conditions']]:
            for arm in ARMS:
                rr=[r for r in rows if r['arm']==arm and r['condition']==cond]
                summary['groups'][cond+'/'+arm]={k:float(np.mean([r[k] for r in rr])) for k in ('psnr','ssim','normalized_wer','covered_normalized_wer','fusion_gate')}
        summary['comparisons']={}
        for name,first,second in [('matched_joint','R0_image','R1_joint'),('over_frozen','B4_frozen','R1_joint'),
                ('correct_vs_zero','R1_zero_text','R1_joint'),('correct_vs_shuffle','R1_shuffled_text','R1_joint')]:
            for metric in ('psnr','ssim'):
                for av in (False,True):
                    summary['comparisons'][name+'/'+metric+('/available' if av else '/all')]=paired_bootstrap(rows,first,second,design['primary_conditions'],metric,c['bootstrap_resamples'],av)
            index={(r['arm'],r['image_id'],r['condition'],r['draw']):r for r in rows}
            margins=[]
            for i in design['validation_ids']:
                noisy=np.mean([index[(second,i,cond,draw)]['psnr']-index[(first,i,cond,draw)]['psnr'] for cond in design['primary_conditions'] for draw in range(c['draws'])])
                clean=index[(second,i,'clean',0)]['psnr']-index[(first,i,'clean',0)]['psnr']
                margins.append(float(noisy-clean))
            if margins:
                rng=np.random.default_rng(42); x=np.array(margins); samples=x[rng.integers(len(x),size=(c['bootstrap_resamples'],len(x)))].mean(1)
                summary['comparisons'][name+'/noisy_minus_clean']=dict(mean=float(x.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),images=len(x),confidence_scope='fixed traces/one training replicate')
        summary['training']={a:json.loads((d/('train_'+a)/'COMPLETE.json').read_text()) for a in TRAINED_ARMS}
        # A compact vector plot makes the four causal comparisons legible.
        names=('matched_joint','over_frozen','correct_vs_zero','correct_vs_shuffle')
        values=[summary['comparisons'][n+'/psnr/all'] for n in names]
        usable=[x for x in values if x['mean'] is not None]
        if usable:
            bound=max(.1,max(abs(v) for x in usable for v in x['ci95']))*1.2
            svg=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="430"><rect width="900" height="430" fill="white"/><g font-family="sans-serif" fill="#183048"><text x="24" y="30" font-size="21">Receiver residual gains: same physical receipts</text><text x="24" y="54" font-size="13">All frames; PSNR difference (dB), right is better. 95% conditional image-cluster intervals.</text><line x1="540" y1="80" x2="540" y2="370" stroke="#64748b" stroke-dasharray="4 4"/>']
            for j,(name,v) in enumerate(zip(names,values)):
                if v['mean'] is None: continue
                y=112+75*j; lo,hi=[540+x/bound*280 for x in v['ci95']]; mid=540+v['mean']/bound*280
                svg.append('<text x="24" y="{}" font-size="15">{}</text><line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#0072B2" stroke-width="3"/><circle cx="{}" cy="{}" r="5" fill="#D55E00"/><text x="24" y="{}" font-size="13">{:+.3f} [{:+.3f}, {:+.3f}]</text>'.format(y-8,name,lo,y,hi,y,mid,y,y+16,v['mean'],*v['ci95']))
            svg.append('<text x="260" y="389" font-size="13">-{:.3f}</text><text x="535" y="389" font-size="13">0</text><text x="782" y="389" font-size="13">+{:.3f}</text><text x="24" y="418" font-size="12">One trained replicate/fixed weather. Not a final test or transmitter-joint coding proof.</text></g></svg>'.format(bound,bound))
            (d/'gains.svg').write_text(''.join(svg)); cards.append('<h2>全帧图片差值；95%条件图片区间</h2><img style="max-width:900px" src="gains.svg">')
        cards.append('<details><summary>展开全部条件统计</summary><pre>'+esc(json.dumps(summary['comparisons'],indent=2))+'</pre></details>')
        for cond in ['clean']+[r['id'] for r in design['conditions']]:
            for draw in range(1 if cond=='clean' else c['draws']):
                for i in design['validation_ids'][:c['save_images']]:
                    rr=[r for r in rows if r['image_id']==i and r['condition']==cond and r['draw']==draw]
                    cards.append('<h2>{} · 图片{} · draw{}</h2><p>原文（评价用）：{}</p><p>实际文字（所有组相同）：{}</p><div class="grid"><article><h3>原图</h3><img src="{}"></article>'.format(esc(cond),i,draw,esc(rr[0]['source_caption_evaluation_only']),esc(rr[0]['received_caption']),esc(rr[0]['source_image_file'])))
                    for r in rr:
                        cards.append('<article><h3>{}</h3><img src="{}"><p>PSNR {:.3f} · SSIM {:.4f}<br>接收可用 {} · gate {:.3f}<br>文字WER {:.1%}</p></article>'.format(esc(r['arm']),esc(r['image_file']),r['psnr'],r['ssim'],r['receiver_available'],r['fusion_gate'],r['normalized_wer']))
                    cards.append('</div>')
    cards.append('<h2>文字定位：全部训练样例实际生成（不是正式验证成绩）</h2>')
    for r in dr:
        cards.append('<details><summary>{} · {} · 图片{} · WER {:.1%} · {}</summary><p>目标：{}</p><p>实际：{}</p><p>生成token数 {}；CE与WER口径不同。</p></details>'.format(esc(r['label']),esc(r['condition']),r['image_id'],r['normalized_wer'],esc(r['termination']),esc(r['covered_reference_evaluation_only']),esc(r['received_caption']),r['generated_tokens']))
    (d/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False))
    (d/'REPORT.md').write_text('# Residual fusion V2\n\nStatus: '+status+'\n\n'+json.dumps(summary,indent=2,ensure_ascii=False)+'\n\nSame actual received frames/charged events across decoders. First-stage receiver-only fusion is NOT a source-joint coding/continuous-vs-digital proof. Diagnostic overfit text weights are never production text decoders.\n')
    (d/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Residual Fusion V2</title><style>body{font:16px/1.6 system-ui;margin:28px;color:#183048}.grid{display:grid;grid-template-columns:repeat(7,minmax(160px,1fr));gap:12px;overflow:auto}article{padding:12px;background:#f2f6fa}img{width:100%}pre{white-space:pre-wrap;font-size:13px}details{border-top:1px solid #ccd7df;padding:8px}</style>'+''.join(cards))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('directory'); a=p.parse_args()
    rows=verify(a.directory); print('Verified {} main rows; paired receipts, actual updates, frozen source and diagnostic decision.'.format(len(rows)))
