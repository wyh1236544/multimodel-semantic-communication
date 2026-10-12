"""Recompute source-based metrics/gates/matrix and charge actual optical events."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from text_jscc_v1_fix1 import VERSION,ARMS
from text_jscc_v1_fix1.core import metrics,summarize,overfit_gate,bootstrap,seed_for,digest,file_under


def read(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def checked_text(rows,design):
    eos=design['tokenizer_ids']['eos']; bos=design['tokenizer_ids']['bos']
    for r in rows:
        expected=metrics(r['source_caption_evaluation_only'],r['covered_reference_evaluation_only'],r['received_caption'])
        for key,value in expected.items():
            observed=r[key]
            if isinstance(value,(int,float)) and not isinstance(value,bool):
                if not math.isfinite(observed) or abs(observed-value)>1e-9: raise ValueError('False text metric '+key)
            elif observed!=value: raise ValueError('False text metric '+key)
        if r['source_text_at_receiver'] or r['source_length_at_receiver'] or r['raw_utf8_wire_bytes']!=0:
            raise ValueError('Source/oracle text bypass')
        ids=r['generated_token_ids']
        if r['termination']=='receiver_rejected':
            if ids or r['received_caption'] or r['stopped_by_eos'] or r['stopped_by_capacity']: raise ValueError('False rejection output')
        else:
            if not ids or ids[0]!=bos or len(ids)>design['config']['max_tokens'] or r['generated_tokens']!=len(ids)-1:
                raise ValueError('False generated token capacity')
            ended=len(ids)>1 and ids[-1]==eos
            if ended!=r['stopped_by_eos'] or r['stopped_by_capacity']==ended:
                raise ValueError('False EOS/cap record')
            if r['stopped_by_capacity'] and len(ids)!=design['config']['max_tokens']:
                raise ValueError('False public capacity stop')
        if hashlib.sha256(r['source_caption_evaluation_only'].encode()).hexdigest()!=r['source_sha256']:
            raise ValueError('Wrong source text hash')


def groups_for(rows):
    groups=defaultdict(list)
    for r in rows: groups[r['condition']].append(r)
    return {k:summarize(v) for k,v in groups.items()}


def verify(folder,complete=True):
    folder=Path(folder); d=json.loads((folder/'design.json').read_text()); identity=json.loads((folder/'run_identity.json').read_text())
    if d['version']!=VERSION or identity['version']!=VERSION or d['split']!='validation': raise ValueError('Wrong run identity')
    status=identity['status']; c=d['config']; ids=d['train_ids']; val=d['validation_ids']
    if len(ids)!=c['train_samples'] or len(val)!=c['val_samples'] or len(ids)!=len(set(ids)) or len(val)!=len(set(val)) or set(ids)&set(val):
        raise ValueError('Wrong/overlapping source population')
    if d['payload_complex_symbols']!=384 or d['raw_utf8_wire_bytes']!=0 or d['analog_repeats']!=1:
        raise ValueError('Wrong text budget')
    if set(d['arms'])!=set(ARMS): raise ValueError('Wrong arm contract')
    diagnostics=read(folder/'legacy_diagnostics/samples.jsonl'); checked_text(diagnostics,d)
    expected={(arm,i,cond) for arm in ('D0_precompression','D1_compressed_clean') for i in ids[:c['diagnostic_samples']] for cond in ('matched','zero','shuffled')}
    if {(r['arm'],r['image_id'],r['condition']) for r in diagnostics}!=expected or len(diagnostics)!=len(expected): raise ValueError('Incomplete legacy diagnostic')
    diagnostic_initial=[]
    for arm in ('D0_precompression','D1_compressed_clean'):
        proof=json.loads((folder/'legacy_diagnostics'/arm/'proof.json').read_text())
        history=read(folder/'legacy_diagnostics'/arm/'history.jsonl')
        if len(history)!=c['diagnostic_updates'] or not proof['gradient_names'] or not proof['changed_names'] or not proof['checkpoint_reload_verified']:
            raise ValueError('Legacy diagnostic training proof missing')
        diagnostic_initial.append(proof['initial_sha256'])
    if not diagnostic_initial[0] or diagnostic_initial[0]!=diagnostic_initial[1]: raise ValueError('Unpaired diagnostic head initialization')
    native=read(folder/'native_overfit/samples.jsonl'); checked_text(native,d)
    expected={(i,cond) for i in ids[:c['diagnostic_samples']] for cond in ('matched','zero','shuffled')}
    if {(r['image_id'],r['condition']) for r in native}!=expected or len(native)!=len(expected): raise ValueError('Incomplete native diagnostic')
    ng=groups_for(native); actual=overfit_gate(ng,c); stored=json.loads((folder/'native_overfit/gate.json').read_text())
    if stored['passed']!=actual['passed'] or abs(stored['matched_wer']-actual['matched_wer'])>1e-9:
        raise ValueError('False native overfit gate')
    if stored['smoke_bypasses_quality_only']!=d['smoke']: raise ValueError('False smoke bypass')
    if status not in ('complete','overfit_stop','clean_validation_stop'): raise ValueError('No successful completion identity')
    if status=='overfit_stop':
        if actual['passed'] or d['smoke'] or (folder/'train_T0_clean').exists(): raise ValueError('Wrong overfit stop boundary')
    elif not actual['passed'] and not d['smoke']: raise ValueError('Failed native gate continued')
    if status!='overfit_stop':
        rows=read(folder/'clean_validation_samples.jsonl'); checked_text(rows,d)
        expected={(i,cond) for i in val for cond in ('matched','zero','shuffled')}
        if {(r['image_id'],r['condition']) for r in rows}!=expected or len(rows)!=len(expected): raise ValueError('Incomplete clean validation')
        cg=groups_for(rows); passed=cg['matched']['covered_wer']<=c['clean_validation_wer_gate'] and min(cg[k]['covered_wer']-cg['matched']['covered_wer'] for k in ('zero','shuffled'))>=c['memory_margin_gate']
        stored=json.loads((folder/'clean_validation_gate.json').read_text())
        if bool(passed)!=stored['passed'] or stored['smoke_bypasses_quality_only']!=d['smoke']: raise ValueError('False clean gate')
        if status=='clean_validation_stop' and (passed or d['smoke'] or (folder/'short_calibration').exists()): raise ValueError('Wrong clean stop boundary')
        if status=='complete' and not passed and not d['smoke']: raise ValueError('Failed clean gate continued')
    train_labels=['native_overfit']+([] if status=='overfit_stop' else ['T0_clean'])+(['T1_clean_continue','T2_fso'] if status=='complete' else [])
    training={}
    for label in train_labels:
        sub=folder/('native_overfit' if label=='native_overfit' else 'train_'+label)
        proof=json.loads((sub/'proof.json').read_text()); history=read(sub/'history.jsonl')
        epochs=c['clean_epochs'] if label=='T0_clean' else c['continue_epochs']
        expected_updates=c['diagnostic_updates'] if label=='native_overfit' else math.ceil(len(ids)/c['batch_size'])*epochs
        if len(history)!=expected_updates or proof['optimizer_steps']!=expected_updates or not proof['checkpoint_reload_verified']:
            raise ValueError('Wrong training update/reload proof')
        if any(value<=0 for value in proof['gradient_energy'].values()) or not proof['gradient_names'] or not any(v>0 for v in proof['parameter_changes'].values()):
            raise ValueError('Required trainable group never learned')
        for prefix in ('encoder.','compressor.','tx.','rx.','receiver.','language.decoder.'):
            if not any(n.startswith(prefix) for n in proof['gradient_names']) or not any(n.startswith(prefix) and value>0 for n,value in proof['parameter_changes'].items()):
                raise ValueError('No actual gradient/update in '+prefix)
        population=set(ids[:c['diagnostic_samples']] if label=='native_overfit' else ids)
        if any(x['image_id'] not in population for r in history for x in r['items']): raise ValueError('Validation/test source trained')
        if (sub/'last.pth').exists() and digest(sub/'last.pth')!=proof['checkpoint_sha256']: raise ValueError('Wrong real saved weight')
        training[label]=(proof,history)
    if status=='complete':
        a,b=training['T1_clean_continue'],training['T2_fso']
        if a[0]['initial_sha256']!=b[0]['initial_sha256'] or a[0]['trainable_parameters']!=b[0]['trainable_parameters']:
            raise ValueError('Unpaired continuation initialization/size')
        if [[x['image_id'] for x in r['items']] for r in a[1]]!=[[x['image_id'] for x in r['items']] for r in b[1]]:
            raise ValueError('Unpaired source order')
        from text_jscc_v1_fix1.channel import fidelity_summary
        bank=json.loads((folder/'short_calibration/bank.json').read_text())
        if bank['payload_complex_symbols']!=384 or not bank['public_waveforms_only']: raise ValueError('Old/invalid bank used')
        with np.load(folder/'short_calibration/bank.npz',allow_pickle=False) as arrays:
            n=len(bank['records'])
            if arrays['coefficients'].shape!=(n,11) or arrays['residuals'].shape!=(n,384): raise ValueError('Wrong short bank shape')
            if not np.isfinite(arrays['coefficients']).all() or not np.isfinite(arrays['residuals']).all(): raise ValueError('Nonfinite short bank')
        for label in ('T1_clean_continue','T2_fso'):
            for step in training[label][1]:
                for item in step['items']:
                    ix=item.get('bank_index')
                    if ix is not None and (label!='T2_fso' or type(ix) is not int or not 0<=ix<n or bank['records'][ix]['partition']!='train'):
                        raise ValueError('Invalid/holdout training replay')
        if not fidelity_summary(bank['fidelity']['rows'],d['channel_config'])['passed']: raise ValueError('Failed short calibration continued')
        for label in ('T0_clean','T1_clean_continue','T2_fso'):
            proof=json.loads((folder/('fidelity_'+label+'.json')).read_text())
            if not fidelity_summary(proof['rows'],d['channel_config'])['passed']: raise ValueError('Failed current-payload gate')
            if any(r['source_image_id'] not in ids or r['source_split']!='train' or bank['records'][r['calibration_index']]['partition']!='holdout' for r in proof['rows']): raise ValueError('Invalid fidelity source/partition')
        rows=read(folder/'samples.jsonl'); checked_text(rows,d)
        conditions=['clean']+[q['id'] for q in d['channel_config']['conditions']]
        expected={(arm,i,cond) for arm in ARMS for i in val for cond in conditions}
        if {(r['arm'],r['image_id'],r['condition']) for r in rows}!=expected or len(rows)!=len(expected): raise ValueError('Missing/duplicate final matrix')
        events=read(folder/'events.jsonl'); byframe=defaultdict(list)
        for e in events: byframe[e['frame_key']].append(e)
        expected_frames={r['frame_key'] for r in rows if r['condition']!='clean'}
        if set(byframe)!=expected_frames: raise ValueError('Missing/free optical events')
        for r in rows:
            if r['source_split']!='validation' or d['validation_source_sha256'][str(r['image_id'])]!=r['source_sha256']: raise ValueError('Wrong validation source')
            if r['payload_complex_symbols']!=384 or r['analog_repeats']!=1 or r['pre_fec_ber'] is not None or r['post_fec_ber'] is not None: raise ValueError('Changed symbol/repeat/BER contract')
            if not r['receiver_available'] and r['received_caption']: raise ValueError('Rejected frame hallucinated delivered text')
            if r['condition']=='clean': continue
            ev=byframe[r['frame_key']]
            if len(ev)!=2 or {e['purpose'] for e in ev}!={'probe','features'} or sum(e['data_symbols'] for e in ev)!=384:
                raise ValueError('Wrong costed short waveform')
            used=sum(e['awg_samples']+e['guard_samples'] for e in ev)
            duration=used/d['resource']['fs_hz']; energy=sum(e['launch_signal_energy_j'] for e in ev)
            if used>d['resource']['slot_samples'] or abs(r['scheduled_seconds']-duration)>1e-15 or abs(r['launch_signal_energy_j']-energy)>max(1e-24,energy*1e-9): raise ValueError('False time/energy accounting')
    else: rows=[]
    frozen=json.loads((folder/'frozen_verification.json').read_text())
    if not all(frozen.get(k) is True for k in ('old_model_unchanged','B4_unchanged','old_gradients_none','no_test_source_used')): raise ValueError('Missing frozen checks')
    if complete:
        marker='COMPLETE.json' if status=='complete' else 'DIAGNOSTIC_COMPLETE.json'
        end=json.loads((folder/marker).read_text())
        if end['version']!=VERSION or end['status']!=status: raise ValueError('False completion marker')
        for name,sha in dict(end['result_sha256'],**end['bank_sha256']).items():
            if digest(file_under(folder,name))!=sha: raise ValueError('Changed raw result '+name)
        if status!='complete' and (folder/'COMPLETE.json').exists(): raise ValueError('Diagnostic stop marked main complete')
    return rows


def report(folder):
    folder=Path(folder); d=json.loads((folder/'design.json').read_text()); identity=json.loads((folder/'run_identity.json').read_text())
    rows=read(folder/'samples.jsonl') if (folder/'samples.jsonl').exists() else []
    groups=defaultdict(list)
    for r in rows: groups[r['condition']+'/'+r['arm']].append(r)
    summary={key:dict(summarize(rr),outage=float(np.mean([not r['receiver_available'] for r in rr])),
        semantic_chunked_clip_cosine=float(np.mean([r['semantic_chunked_clip_cosine'] for r in rr]))) for key,rr in groups.items()}
    paired={}
    if rows:
        index={(r['arm'],r['image_id'],r['condition']):r for r in rows}
        primary=d['channel_config']['primary_conditions']
        for first,second in (('T1_clean_continue','T2_fso'),('L0_frozen_B4','T2_fso')):
            values=[np.mean([index[(second,i,c)]['covered_wer']-index[(first,i,c)]['covered_wer'] for c in primary]) for i in d['validation_ids']]
            paired[second+' minus '+first]=bootstrap(values,seed_for('text-paired',first,second),d['config']['bootstrap_resamples'])
    data=dict(version=VERSION,status=identity['status'],smoke=d['smoke'],groups=summary,paired_covered_wer=paired,
        comparison_note='T1/T2 same initialization/size/updates; L0 versus new text system is whole-architecture reference, not equal-parameter causal comparison',
        semantic_metric_note='Evaluation-only token-weighted chunked frozen CLIP text cosine; related representation family, not independent semantic correctness',
        constraints=d['limits'])
    from v10.io import atomic_json
    atomic_json(folder/'summary.json',data)
    body=['# 文字单模态JSCC筛选','', '状态：'+identity['status']+('（短预检，不代表正式质量通过）' if d['smoke'] else ''),'',
          '所有正式物理帧只发送48×16实数＝384复符号，单次发送，无图像载荷、无原文旁路。',
          '先检查干净恢复和接收信息依赖；未通过则停止，不降低门槛。旧CLIP/B4与原图文模型保持冻结。',
          'CLIP tokenizer会规范大小写/空白，文字容量128token；不声称字节无损。完整原文WER和覆盖文本WER分开。',
          '样本标准差描述样本差异；配对95%区间仅按图片聚类，一次训练与固定天气，不覆盖总体训练/天气变异。','',
          '| 条件/组 | 覆盖WER↓ | 完整WER↓ | BLEU2↑ | 语义余弦代理↑ | 失锁 |', '|---|---:|---:|---:|---:|---:|']
    for key,g in summary.items(): body.append('| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.1%} |'.format(key,g['covered_wer'],g['full_wer'],g['bleu2'],g['semantic_chunked_clip_cosine'],g['outage']))
    body+=['','配对比较（负值表示第二组WER更低）：','',json.dumps(paired,ensure_ascii=False,indent=2),'',d['limits']]
    (folder/'REPORT.md').write_text('\n'.join(body)+'\n')
    sections=[]
    for name,title in [('legacy_diagnostics/samples.jsonl','旧CLIP压缩前/后诊断'),('native_overfit/samples.jsonl','新文字网络少样本诊断'),('clean_validation_samples.jsonl','独立干净验证与记忆干预'),('samples.jsonl','最终完整波形图文对照（仅文字输出）')]:
        if not (folder/name).exists(): continue
        rr=read(folder/name); cards=[]
        # Every recorded sentence is shown, not only best cases.
        for r in rr:
            constellation='<p><a href="'+html.escape(r['constellation_file'])+'">接收星座图</a></p>' if r.get('constellation_file') else ''
            cards.append('<details><summary>{} · {} · ID{} · WER {:.1%}</summary><p>原文：{}</p><p>实际恢复：{}</p><p>覆盖WER {:.1%}；BLEU2 {:.3f}；停机 {}</p>{}</details>'.format(html.escape(r['arm']),html.escape(r['condition']),r['image_id'],r['covered_wer'],html.escape(r['source_caption_evaluation_only']),html.escape(r['received_caption']),r['covered_wer'],r['bleu2'],html.escape(r['termination']),constellation))
        sections.append('<section><h2>'+title+'</h2>'+''.join(cards)+'</section>')
    table='<table><tr><th>条件/组</th><th>覆盖WER</th><th>完整WER</th><th>BLEU2</th><th>语义余弦代理</th><th>失锁</th></tr>'+''.join('<tr><td>{}</td><td>{:.1%}</td><td>{:.1%}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.1%}</td></tr>'.format(html.escape(k),g['covered_wer'],g['full_wer'],g['bleu2'],g['semantic_chunked_clip_cosine'],g['outage']) for k,g in summary.items())+'</table>'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>文字单模态JSCC</title><style>body{font:16px/1.6 system-ui;margin:32px;color:#1b2e3b}main{max-width:1150px;margin:auto}section{margin:24px 0}details{padding:10px;border-bottom:1px solid #ddd}summary{cursor:pointer}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}p{white-space:pre-wrap}pre{white-space:pre-wrap}a{color:#005b99}</style><main><h1>文字单模态JSCC</h1><p>状态：'+html.escape(identity['status'])+'</p><p>全部恢复句子保留，包括失败、重复和截断；短预检不能冒充正式质量通过。</p>'+table+'<pre>'+html.escape('\n'.join(body[:10]))+'</pre>'+''.join(sections)+'</main></html>'
    (folder/'report.html').write_text(page)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('folder'); a=p.parse_args(); verify(a.folder); print('Text-only raw records/gates/matrix/resources verified.')
