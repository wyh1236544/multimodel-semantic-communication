"""Verify budgets, actual text outputs, negative gates and full evaluation matrix."""
import argparse
import html
import json
from pathlib import Path
import sys
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from clip_bart_joint_v6 import VERSION,ARMS
from clip_bart_joint_v6.core import digest,metrics,summarize,geometry,text_gate


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p): return [json.loads(v) for v in Path(p).read_text(encoding='utf-8').splitlines() if v.strip()]


def controls(p,ids,secondary_beam=False):
    # V4 text prefit writes a fourth, secondary beam decode. Joint controls do
    # not. Validate the entire declared matrix, but keep greedy-only gates.
    primary=('matched','zero','shuffled')
    declared=primary+('matched_beam4',) if secondary_beam else primary
    rows=lines(p); expected={(k,i) for k in declared for i in ids}; seen=set()
    for r in rows:
        key=(r['control'],r['image_id'])
        if key in seen: raise ValueError('Duplicate controls row: '+str(key))
        if key not in expected: raise ValueError('Unexpected controls row: '+str(key))
        if r['source_split']!='validation': raise ValueError('Controls source is not validation: '+str(key))
        seen.add(key)
        if any(r.get(k)!=v for k,v in metrics(r['source_text'],r['covered_text'],r['received_caption']).items()): raise ValueError('Text metrics differ')
        if r['control']=='shuffled' and (r.get('donor_image_id') not in ids or r['donor_image_id']==r['image_id']): raise ValueError('No real shuffle donor')
    if seen!=expected: raise ValueError('Missing controls: '+str(sorted(expected-seen)))
    return {k:summarize([r for r in rows if r['control']==k]) for k in primary}


def training(root,arm,phase,c):
    f=root/arm/phase; done=read(f/'TRAIN_COMPLETE.json'); hist=lines(f/'history.jsonl'); audit=read(f/'learning_audit.json')
    count=c['train_samples']*(c['common_epochs'] if phase=='common' else (c['clean_epochs'] if phase=='clean' else c['continue_epochs']))
    if done['steps']!=count or [r['step'] for r in hist]!=list(range(1,count+1)) or done['optimizer_updates']!=(count+c['grad_accum']-1)//c['grad_accum']:
        raise ValueError('Joint update budget differs')
    import numpy as np
    ids=read(root/'design.json')['image_train_view']['image_ids']
    for j,r in enumerate(hist):
        epoch,pos=divmod(j,len(ids)); order=np.random.default_rng(c['seed']+epoch).permutation(len(ids))
        if r['image_id']!=ids[int(order[pos])] or r['total_steps']!=count: raise ValueError('Paired data/order differs')
        noisy=phase=='fso' and j%c['clean_anchor_period']!=0
        from clip_bart_joint_v6.core import private_mask
        if r['private_mask']!=private_mask(c,j,r['image_id'],phase): raise ValueError('Training mask protocol differs')
        if r['noisy']!=noisy or bool(r['proxy'])!=noisy: raise ValueError('Phase/channel/anchor mismatch')
        if noisy and (len(r['proxy'])!=1 or r['proxy'][0]['payload_complex_symbols']!=3072 or r['proxy'][0]['analog_repeats']!=1): raise ValueError('Wrong training resources')
    required={'text_codec','common_source','common_receiver','common_tasks','image_tx','image_rx','text_fusion','balanced_pool','common_caption'}
    if phase=='common': required={'common_source','common_receiver','common_tasks','balanced_pool','common_caption'}
    if set(audit['groups'])!=required or not audit['passed'] or not audit['frozen_unchanged'] or any(not all(v[k] for k in ('tensors','nonzero_gradient','changed')) for v in audit['groups'].values()):
        raise ValueError('Required learning/frozen audit failed')
    if not read(f/'reload_audit.json')['passed']: raise ValueError('Actual reload missing')
    forward=read(f/'forward_consistency.json')
    if forward['tolerance']!=c['forward_tolerance'] or max(forward['errors'].values())>c['forward_tolerance']: raise ValueError('Forward mismatch')
    if (f/'last.pth').exists() and digest(f/'last.pth')!=done['checkpoint_sha256']: raise ValueError('Checkpoint modified')
    return read(f/'trainability.json')['registered_bridge_parameters']


def evaluation(folder,arm,phase,c,ids):
    ed=read(folder/'eval_design.json'); rows=lines(folder/'samples.jsonl')
    expected={(condition,control,iid) for condition in ed['conditions'] for control in ed['controls'] for iid in ids}
    if arm=='J1_joint': expected|={('clean','wrong_text_source',iid) for iid in ids}
    if ed['validation_ids']!=ids or ed['phase']!=phase or ed['arm']!=arm: raise ValueError('Evaluation source/arm mismatch')
    seen=set(); normal=[]
    from clip_bart_joint_v5.channel import resource
    res=resource(ed['channel_config'])
    if res!=ed['resources']: raise ValueError('Waveform resources differ')
    events=lines(folder/'events.jsonl')
    if len(events)!=2*len(ids)*(len(ed['conditions'])-1): raise ValueError('Missing full-frame events')
    for r in rows:
        key=(r['condition'],r['control'],r['image_id'])
        if key not in expected or key in seen or r['source_split']!='validation' or r['geometry']!=geometry(): raise ValueError('Evaluation matrix duplicate/missing/layout')
        seen.add(key)
        if any(r.get(k)!=v for k,v in metrics(r['source_text'],r['covered_text'],r['received_caption']).items()): raise ValueError('Actual text metric mismatch')
        import math
        if any(not math.isfinite(r[k]) for k in ('mse','psnr','ssim')) or abs(r['psnr']+10*math.log10(max(r['mse'],1e-12)))>1e-4: raise ValueError('Image metric inconsistency')
        if r['receiver_failure'] and r['received_caption']: raise ValueError('Unavailable frame leaked text')
        if r['condition']!='clean':
            st=r['stats']
            if st['used_samples']!=res['scheduled_samples'] or st['payload_complex_symbols']!=3072 or st['analog_repeats']!=1 or st['raw_utf8_wire_bytes']!=0:
                raise ValueError('Hidden/mismatched physical resource')
            if r['receiver_failure']==st['receiver_available']: raise ValueError('Receiver failure differs')
            if len([v for v in events if v['condition']==r['condition'] and v['image_id']==r['image_id']])!=2: raise ValueError('Probe/features pair missing')
        if r['control']=='normal': normal.append(r)
        if r['control']=='normal' and r['image_id'] in ids[:c['save_images']]:
            import numpy as np
            name=r.get('receiver_condition_file')
            if not name or Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Missing/unsafe received generation condition')
            with np.load(folder/name,allow_pickle=False) as z:
                if set(z.files)!={'common','received_caption','receiver_available','image_id','source_split','pilot_snr_db'} or z['common'].shape!=(32,16) or not np.isfinite(z['common']).all():
                    raise ValueError('Receiver condition schema/oracle leakage')
                if int(z['image_id'])!=r['image_id'] or str(z['source_split'])!='validation' or str(z['received_caption'])!=r['received_caption'] or bool(z['receiver_available'])==r['receiver_failure']:
                    raise ValueError('Received conditioning differs from actual row')
                expected_snr=20. if r['condition']=='clean' else r['stats']['pilot_snr_db']
                if not np.isfinite(z['pilot_snr_db']) or abs(float(z['pilot_snr_db'])-expected_snr)>1e-4:
                    raise ValueError('Received pilot SNR differs')
    if seen!=expected: raise ValueError('Missing evaluated row')
    return normal


def verify(root,sealed=True):
    root=Path(root).resolve(strict=True); identity=read(root/'run_identity.json'); design=read(root/'design.json'); c=identity['config']
    if identity['version']!=VERSION or identity['status'] not in ('screen_complete','smoke_complete') or (root/'FAILED.json').exists(): raise ValueError('No completed screen')
    if any(f.is_symlink() for f in root.rglob('*')): raise ValueError('Symlink records')
    if design['config']!=c or design['arms']!=list(ARMS) or design['geometry']!=geometry() or identity['TRAIN8_weights_loaded'] or design['TRAIN8_weights_loaded']:
        raise ValueError('Design/init/resource differs')
    ti=design['text_train_view']['image_ids']; vi=design['text_validation_view']['image_ids']
    if len(ti)!=c['text_train_samples'] or len(vi)!=c['val_samples'] or set(ti)&set(vi): raise ValueError('Text split count/overlap')
    tf=root/'text_prefit/clean'; done=read(tf/'TRAIN_COMPLETE.json'); hist=lines(tf/'history.jsonl'); audit=read(tf/'learning_audit.json')
    if done['updates']!=c['text_block_updates'] or [r['update'] for r in hist]!=list(range(1,c['text_block_updates']+1)): raise ValueError('Text prefit budget differs')
    from bart_latent_codec_v4.core import batch_indices
    for j,r in enumerate(hist):
        expected=[ti[i] for i in batch_indices(len(ti),identity['text_config']['batch_size'],j,c['seed'])]
        if r['batch_image_ids']!=expected: raise ValueError('Text prefit sources differ')
    if not audit['passed'] or not audit['frozen_unchanged'] or not read(tf/'reload_audit.json')['passed']: raise ValueError('Text prefit learning/reload failed')
    tg=text_gate(controls(root/'text_prefit/validation_controls.jsonl',vi,secondary_beam=True),c)
    if read(root/'text_prefit/validation_gate.json')!=tg: raise ValueError('Text prefit gate differs')
    check_text_stages(root,identity,design,c)
    statuses=read(root/'arm_status.json')
    if set(statuses)!=set(ARMS): raise ValueError('Missing arms')
    initial=None; registered=None
    for arm,s in statuses.items():
        folder=root/arm
        if not s['trained']:
            if tg['passed'] or identity['smoke'] or s['status']!='text_prefit_stop' or folder.exists(): raise ValueError('Wrong text-prefit stop')
            continue
        if not tg['passed'] and not identity['smoke']: raise ValueError('Joint training through failed text prefit')
        hashes=read(folder/'initial_state_hashes.json')
        if initial is not None and initial!=hashes: raise ValueError('Matched initialization differs')
        initial=hashes; training(root,arm,'common',c)
        controls(folder/'common_controls.jsonl',vi)
        controls(folder/'joint_common_controls.jsonl',vi)
        check_masked_validation(folder/'masked_validation.jsonl',vi,c)
        for name in ('common_controls.jsonl','joint_common_controls.jsonl'):
            if any(not r.get('common_only_receiver_diagnostic') or r.get('charged_payload_complex_symbols')!=3072 for r in lines(folder/name)):
                raise ValueError('Common-only scope/resource differs')
        count=training(root,arm,'clean',c)
        if registered is not None and registered!=count: raise ValueError('Registered model size differs')
        registered=count; iv=design['image_validation_view']['image_ids']; it=design['image_train_view']['image_ids']
        if len(iv)!=c['val_samples'] or len(it)!=c['train_samples'] or set(iv)&set(it) or iv!=vi: raise ValueError('Image/text validation source mismatch')
        cg=text_gate(controls(folder/'clean_controls.jsonl',iv),c)
        if read(folder/'clean_gate.json')!=cg or s['clean_passed']!=cg['passed']: raise ValueError('Clean gate differs')
        evaluation(folder/'eval_clean',arm,'clean',c,iv)
        if s['noise_trained']:
            if not identity['noise'] or not (cg['passed'] or identity['smoke']) or s['status']!='noise_evaluated': raise ValueError('Unauthorized noise continuation')
            for phase in ('clean_continue','fso'):
                training(root,arm,phase,c); controls(folder/(phase+'_controls.jsonl'),iv); evaluation(folder/('eval_'+phase),arm,phase,c,iv)
        elif s['status']!=('clean_passed' if cg['passed'] else 'clean_validation_stop') or (folder/'fso').exists(): raise ValueError('Wrong quality stop')
    if statuses['J1_joint']['trained']: check_generation_cache(root/'J1_joint/generation_train.npz',design,c)
    frozen=read(root/'frozen_verification.json')
    if not frozen['passed'] or not frozen['backbones_unchanged'] or not frozen['original_data_unchanged']: raise ValueError('Frozen verification missing')
    for name,sha in identity['source_sha256'].items():
        if Path(name).name!=name or digest(root/'source_snapshot'/name)!=sha: raise ValueError('Runtime source hash differs')
    if sealed:
        manifest=read(root/'RECORDS_MANIFEST.json'); marker='SMOKE_COMPLETE.json' if identity['smoke'] else 'SCREEN_COMPLETE.json'; done=read(root/marker)
        if done['manifest_sha256']!=digest(root/'RECORDS_MANIFEST.json') or not done['not_a_quality_success_marker']: raise ValueError('Completion marker differs')
        current={str(f.relative_to(root)):digest(f) for f in sorted(root.rglob('*')) if f.is_file() and f.suffix!='.pth' and f.name not in ('RECORDS_MANIFEST.json','SMOKE_COMPLETE.json','SCREEN_COMPLETE.json')}
        if current!=manifest['files']: raise ValueError('Missing/extra/drifted records')
    return dict(passed=True,text_prefit_passed=tg['passed'],registered_bridge_parameters=registered,smoke=identity['smoke'],arm_status=statuses)


def finish(root):
    from v10.io import atomic_json
    import statistics
    root=Path(root); verified=verify(root,False); statuses=read(root/'arm_status.json')
    summary=dict(verification=verified,text_prefit=read(root/'text_prefit/validation_gate.json'),
        text_milestones={str(u):read(root/'text_prefit'/('validation_gate.json' if u==read(root/'run_identity.json')['config']['text_updates'] else 'gate_u'+str(u)+'.json')) for u in read(root/'run_identity.json')['text_milestones']},
        arms={},comparisons={})
    samples=[]
    for arm,s in statuses.items():
        summary['arms'][arm]=dict(s)
        if s['trained']:
            summary['arms'][arm]['common_warm']=controls(root/arm/'common_controls.jsonl',read(root/'design.json')['text_validation_view']['image_ids'])
            summary['arms'][arm]['common_joint']=controls(root/arm/'joint_common_controls.jsonl',read(root/'design.json')['text_validation_view']['image_ids'])
            mr=lines(root/arm/'masked_validation.jsonl')
            summary['arms'][arm]['masked_validation']={}
            for control in ('normal','zero_common','no_common_fusion','shuffled_common'):
                group=[r for r in mr if r['control']==control]
                summary['arms'][arm]['masked_validation'][control]=dict(text=summarize(group),
                    psnr_mean=statistics.mean(r['psnr'] for r in group) if all('psnr' in r for r in group) else None,
                    scope='synthetic receiver-feature dropout, NOT real FSO')
        for path in sorted((root/arm).glob('eval_*/samples.jsonl')):
            entries=lines(path); samples.extend([(str(path.parent.relative_to(root)),r) for r in entries])
            summary['arms'][arm][path.parent.name]={}
            for condition in sorted({r['condition'] for r in entries}):
                normal=[r for r in entries if r['condition']==condition and r['control']=='normal']
                summary['arms'][arm][path.parent.name][condition]=dict(text=summarize(normal),
                    psnr_mean=statistics.mean(r['psnr'] for r in normal),psnr_sd=statistics.stdev(r['psnr'] for r in normal),
                    ssim_mean=statistics.mean(r['ssim'] for r in normal),receiver_failures=sum(r['receiver_failure'] for r in normal))
    for phase in ('clean','clean_continue','fso'):
        maps={arm:{(r['condition'],r['image_id']):r for _,r in samples if r['arm']==arm and r['phase']==phase and r['control']=='normal'} for arm in ARMS}
        if maps['J0_separate'] and maps['J1_joint']:
            if set(maps['J0_separate'])!=set(maps['J1_joint']): raise ValueError('Unpaired primary matrix')
            delta=[(maps['J1_joint'][k]['psnr']-maps['J0_separate'][k]['psnr'],maps['J1_joint'][k]['covered_wer']-maps['J0_separate'][k]['covered_wer']) for k in maps['J0_separate']]
            summary['comparisons'][phase]=dict(joint_minus_separate_psnr=statistics.mean(v[0] for v in delta),
                joint_minus_separate_wer=statistics.mean(v[1] for v in delta),rows=len(delta),
                scope='descriptive pooled fixed validation conditions; no training-seed or weather uncertainty')
    atomic_json(root/'summary.json',summary)
    page=['<!doctype html><meta charset="utf-8"><title>CLIP+BART联合V6实际恢复</title>',
        '<style>body{font:16px sans-serif;padding:24px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:8px;vertical-align:top}img{width:128px;height:128px}td{max-width:320px;overflow-wrap:anywhere}</style>',
        '<h1>V6实际图片与文字恢复</h1><p>无扩散、无原文接收旁路。联合/分开同3072复符号。保留全部错误和失锁；干预不是额外传输。</p>']
    page.append('<h2>文本3000→6000→9000：同一验证集，固定最后权重</h2><table><tr><th>总更新</th><th>WER</th><th>逐字完全一致比例</th><th>工程门槛</th></tr>')
    for u,g in summary['text_milestones'].items():
        page.append('<tr><td>{}</td><td>{:.4f}</td><td>{:.4f}</td><td>{}</td></tr>'.format(
            html.escape(u),g['groups']['matched']['covered_wer'],g['groups']['matched']['full_exact_fraction'],g['passed']))
    page.append('</table><details><summary>原始BART记忆离线诊断（不是实际传输）</summary><table><tr><th>ID/原文</th><th>诊断恢复</th><th>WER</th></tr>')
    for r in lines(root/'text_prefit/native_bart_oracle.jsonl'):
        page.append('<tr><td>{} / {}</td><td>{}</td><td>{:.3f}</td></tr>'.format(r['image_id'],
            html.escape(r['source_text']),html.escape(r['received_caption']),r['covered_wer']))
    page.append('</table></details>')
    page.append('<h2>先行文字泛化检查：全部验证输出</h2><table><tr><th>控制/ID</th><th>原文（评价用）</th><th>实际文字</th><th>WER/结束</th></tr>')
    for r in lines(root/'text_prefit/validation_controls.jsonl'):
        page.append('<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.3f}/{}</td></tr>'.format(
            html.escape('{} / {}'.format(r['control'],r['image_id'])),html.escape(r['source_text']),html.escape(r['received_caption']),
            r['covered_wer'],'容量上限' if r['stopped_by_capacity'] else '自然EOS'))
    page.append('</table>')
    for arm,state in statuses.items():
        if not state['trained']: continue
        for name in ('common_controls.jsonl','joint_common_controls.jsonl','masked_validation.jsonl'):
            page.append('<details><summary>{} / {}：全部验证文字与诊断图片</summary><p>Common-only仍按3072符号帧诊断；私有遮挡不是实际FSO。</p><table><tr><th>ID/控制</th><th>原文（评价）</th><th>实际恢复</th><th>图片</th><th>WER</th></tr>'.format(html.escape(arm),html.escape(name)))
            for r in lines(root/arm/name):
                img='<img src="{}">'.format(html.escape(arm+'/'+r['image_file'],quote=True)) if r.get('image_file') else ''
                page.append('<tr><td>{}/{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.3f}</td></tr>'.format(
                    r['image_id'],html.escape(r['control']),html.escape(r['source_text']),
                    html.escape(r['received_caption']),img,r['covered_wer']))
            page.append('</table></details>')
    if not samples: page.append('<p>文字泛化门禁未通过，未开始联合训练。详见summary.json及text_prefit/validation_controls.jsonl。</p>')
    page.append('<table><tr><th>分支/阶段/条件/控制/ID</th><th>输入原文（评价用）</th><th>实际恢复文字</th><th>图片（原图/恢复）</th><th>PSNR/SSIM/WER</th></tr>')
    for base,r in samples:
        images=''
        for key in ('source_image_file','image_file'):
            if key in r: images+='<img src="{}">'.format(html.escape(base+'/'+r[key],quote=True))
        page.append('<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.2f}/{:.3f}/{:.3f}</td></tr>'.format(
            html.escape('{} / {} / {} / {} / {}'.format(r['arm'],r['phase'],r['condition'],r['control'],r['image_id'])),
            html.escape(r['source_text']),html.escape(r['received_caption']),images,r['psnr'],r['ssim'],r['covered_wer']))
    page.append('</table>')
    with (root/'actual_image_text.html').open('x',encoding='utf-8') as f: f.write('\n'.join(page))
    with (root/'REPORT.md').open('x',encoding='utf-8') as f:
        f.write('# CLIP+BART joint V6 screening\n\nExecution completion is NOT proof of joint benefit. All results: summary.json and actual_image_text.html.\n\n')
        f.write(json.dumps(verified,ensure_ascii=False,indent=2)+'\n')
    files={str(f.relative_to(root)):digest(f) for f in sorted(root.rglob('*')) if f.is_file() and f.suffix!='.pth'}
    weights={str(f.relative_to(root)):dict(sha256=digest(f),bytes=f.stat().st_size) for f in sorted(root.rglob('*.pth'))}
    atomic_json(root/'RECORDS_MANIFEST.json',dict(version=VERSION,files=files,weights=weights))
    marker='SMOKE_COMPLETE.json' if read(root/'run_identity.json')['smoke'] else 'SCREEN_COMPLETE.json'
    atomic_json(root/marker,dict(version=VERSION,manifest_sha256=digest(root/'RECORDS_MANIFEST.json'),not_a_quality_success_marker=True))
    verify(root)


def _cli():
    p=argparse.ArgumentParser(); p.add_argument('root'); a=p.parse_args(); print(json.dumps(verify(a.root),indent=2))



def check_text_stages(root,identity,design,c):
    source=read(root/'parent_text_identity.json')
    expected=[3000,3000+c['text_block_updates'],c['text_updates']]
    if expected!=identity['text_milestones'] or not source['source_proof']['passed'] or source['source_proof']['smoke'] or source['source_updates']!=3000 or source['optimizer_restored']:
        raise ValueError('Parent/milestone continuation mismatch')
    milestones=read(root/'text_prefit/milestones.json')
    if milestones!=dict(global_updates=expected,fixed_last_checkpoint=True,no_validation_checkpoint_selection=True):
        raise ValueError('Text selection protocol differs')
    vi=design['text_validation_view']['image_ids']; ti=design['text_train_view']['image_ids']
    from bart_latent_codec_v4.core import batch_indices
    for u in expected[:-1]:
        cp=root/'text_prefit'/('validation_u'+str(u)+'.jsonl')
        gate=text_gate(controls(cp,vi,secondary_beam=True),c)
        if gate!=read(root/'text_prefit'/('gate_u'+str(u)+'.json')): raise ValueError('Milestone gate differs')
        if u!=3000:
            folder=root/'text_prefit'/('u'+str(u)); done=read(folder/'TRAIN_COMPLETE.json'); history=lines(folder/'history.jsonl')
            if done['updates']!=c['text_block_updates'] or [r['update'] for r in history]!=list(range(1,c['text_block_updates']+1)):
                raise ValueError('Intermediate text budget differs')
            for j,r in enumerate(history):
                if r['batch_image_ids']!=[ti[i] for i in batch_indices(len(ti),identity['text_config']['batch_size'],j,c['seed'])]:
                    raise ValueError('Intermediate text order differs')
            if not read(folder/'learning_audit.json')['passed'] or not read(folder/'reload_audit.json')['passed']:
                raise ValueError('Intermediate text learning/reload failed')
            if not read(folder/'learning_audit.json')['frozen_unchanged']:
                raise ValueError('Intermediate BART not frozen')
    oracle=lines(root/'text_prefit/native_bart_oracle.jsonl')
    if [r['image_id'] for r in oracle]!=vi or any(r['source_split']!='validation' or r['control']!='oracle_not_transmitted' or not r['original_source_memory_provided'] for r in oracle):
        raise ValueError('Native BART diagnostic source/label mismatch')
    for r in oracle:
        if any(r.get(k)!=v for k,v in metrics(r['source_text'],r['covered_text'],r['received_caption']).items()):
            raise ValueError('Native diagnostic text mismatch')


def check_generation_cache(path,design,c):
    import numpy as np
    import hashlib
    with np.load(path,allow_pickle=False) as cache:
        ids=design['image_train_view']['image_ids'][:c['generation_train_samples']]
        if set(cache.files)!={'common','received_caption','source_text_train_teacher','image_id','source_split','proxy_noisy','pilot_snr_db'} or cache['pilot_snr_db'].shape!=(len(ids),) or not np.isfinite(cache['pilot_snr_db']).all():
            raise ValueError('Generation TRAIN cache schema differs')
        if cache['image_id'].tolist()!=ids or set(cache['source_split'].tolist())!={'train'} or cache['common'].shape!=(len(ids),32,16) or not np.isfinite(cache['common']).all():
            raise ValueError('Generation cache is not fixed TRAIN received conditions')
        for iid,text in zip(ids,cache['source_text_train_teacher'].tolist()):
            if hashlib.sha256(text.encode()).hexdigest()!=design['image_train_view']['source_sha256'][str(iid)]:
                raise ValueError('Adapter TRAIN teacher differs from declared source')


def check_masked_validation(path,ids,c):
    from clip_bart_joint_v6.core import private_mask
    expected={(k,i) for k in ('normal','zero_common','no_common_fusion','shuffled_common') for i in ids}; seen=set()
    for r in lines(path):
        key=(r['control'],r['image_id'])
        if key not in expected or key in seen or r['source_split']!='validation' or r['geometry']!=geometry() or not r['not_a_physical_channel']:
            raise ValueError('Masked validation scope/matrix differs')
        seen.add(key)
        if r['private_mask']!=private_mask(c,3,r['image_id'],'clean'): raise ValueError('Masked validation schedule differs')
        if any(r.get(k)!=v for k,v in metrics(r['source_text'],r['covered_text'],r['received_caption']).items()):
            raise ValueError('Masked actual text metrics differ')
        if r['control']=='shuffled_common' and (r.get('donor_image_id') not in ids or r['donor_image_id']==r['image_id']):
            raise ValueError('Masked common donor invalid')
    if seen!=expected: raise ValueError('Masked validation missing rows')


if __name__=='__main__':
    _cli()
