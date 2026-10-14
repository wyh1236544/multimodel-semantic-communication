"""Independent matrix/frame/metrics/global-cursor verifier, once before collection."""
import argparse
import hashlib
import html
import math
from pathlib import Path
import sys
import numpy as np
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
from docci_text_continue_v11.core import (FOLDER,read,jsonlines,digest,load,parent,schedule,probes,
    choose_indices,phase_lr,geometry,joined_ids,gate,atomic_json)
from docci_text_continue_v11.metrics import metrics,summary
from docci_reconstruction_v10.channel import resource


def close(a,b): return math.isclose(float(a),float(b),rel_tol=1e-7,abs_tol=1e-8)


def file(root,name):
    name=Path(name)
    if name.is_absolute() or any(x in ('.','..') for x in name.parts): raise ValueError('Foreign record path')
    path=root
    for part in name.parts:
        path=path/part
        if path.is_symlink(): raise ValueError('Symlink record')
    if not path.is_file(): raise FileNotFoundError(str(path))
    return path


def evaluation(root,c,declared,receipts):
    root=Path(root); matrix=read(root/'matrix.json'); done=read(root/'EVAL_COMPLETE.json'); rows=jsonlines(root/'rows.jsonl')
    ids=declared['example_ids']; expected={(eid,control) for eid in ids for control in ('matched','zero','shuffled')}
    if matrix['example_ids']!=ids or matrix['controls']!=['matched','zero','shuffled'] or matrix['conditions']!=['clean']:
        raise ValueError('Evaluation cohort/control matrix differs')
    if matrix['stage']!=declared['stage'] or matrix['cohort']!=declared['cohort'] or matrix['source_count_or_mask_at_RX'] or matrix['Common_used']:
        raise ValueError('Wrong evaluation stage/RX scope')
    if matrix['resources']!=resource(c,'T1_chunks') or done['rows']!=len(rows) or len(rows)!=len(expected): raise ValueError('Rows/resources differ')
    if done['rows_sha256']!=digest(root/'rows.jsonl') or done['matrix_sha256']!=digest(root/'matrix.json'): raise ValueError('Evaluation hash differs')
    seen=set(); frames={}
    for row in rows:
        key=row['example_id'],row['control']
        if key not in expected or key in seen: raise ValueError('Duplicate/foreign control')
        seen.add(key); receipt=receipts[row['example_id']]
        if row['stage']!=declared['stage'] or row['cohort']!=declared['cohort'] or row['condition']!='clean' or row['branch']!='text' or row['arm']!='T1_chunks' or row['noise_seed'] is not None: raise ValueError('Row stage differs')
        if row['official_split']!=('train' if declared['cohort']=='diagnostic_train' else 'qual_dev'): raise ValueError('Test/source split contamination')
        if row['source_text_sha256']!=receipt['text_sha256'] or hashlib.sha256(row['source_text'].encode()).hexdigest()!=receipt['text_sha256'] or row['source_tokens']!=receipt['source_tokens'] or row['input_truncated']:
            raise ValueError('Full original source differs')
        calculated=metrics(row['source_text'],row['received_caption'])
        for k,v in calculated.items():
            if not close(row[k],v) if isinstance(v,(int,float)) else row[k]!=v: raise ValueError('Recomputed metric differs '+k)
        sequences=row['chunk_generated_token_ids']
        if len(sequences)!=9 or any(not seq or seq[0]!=2 or len(seq)>65 or any(type(t) is not int or not 0<=t<50265 for t in seq) for seq in sequences): raise ValueError('Public generation layout differs')
        joined,cap=joined_ids(sequences)
        if joined!=row['generated_token_ids'] or cap!=row['stopped_by_capacity'] or row['receiver_failure']: raise ValueError('Full BPE/EOS reassembly differs')
        frame=file(root,row['frame_file'])
        if digest(frame)!=row['frame_sha256']: raise ValueError('Frame hash differs')
        with np.load(frame,allow_pickle=False) as z:
            if set(z.files)!={'observed','available','pilot_snr_db','reliability'}: raise ValueError('Source information entered public NPZ')
            wire=z['observed'].copy()
            if not bool(z['available']) or float(z['pilot_snr_db'])!=40. or float(z['reliability'])!=1.: raise ValueError('Clean sentinel differs')
        h=hashlib.sha256(wire.tobytes()).hexdigest()
        if wire.shape!=(18432,) or wire.dtype!=np.float32 or not np.isfinite(wire).all() or abs(float(np.mean(wire**2))-1)>1e-4 or h!=row['received_sha256'] or h!=row['tx_wire_sha256']: raise ValueError('Actual wire budget/power/hash differs')
        stats=row['stats']
        if stats['resources']!=matrix['resources'] or stats['payload_complex_symbols']!=9216 or stats['analog_repeats']!=1 or stats['raw_utf8_wire_bytes']!=0 or not stats['receiver_available'] or stats['pilot_snr_db']!=40. or stats['analog_reliability']!=1. or not stats['clean_sentinel_not_measured_snr'] or stats['pre_fec_ber'] is not None or stats['post_fec_ber'] is not None: raise ValueError('Hidden resource/channel change')
        value=(row['frame_sha256'],h)
        if row['example_id'] in frames and frames[row['example_id']]!=value: raise ValueError('Controls used different frames')
        frames[row['example_id']]=value
    if seen!=expected: raise ValueError('Missing evaluation rows')
    for row in rows:
        if row['control']=='shuffled':
            donor=ids[(ids.index(row['example_id'])+1)%len(ids)]
            if donor==row['example_id'] or row['donor_example_id']!=donor or row['donor_received_sha256']!=frames[donor][1]: raise ValueError('Wrong shuffled donor')
        elif row['donor_example_id'] is not None or row['donor_received_sha256'] is not None: raise ValueError('Unexpected donor')
    return rows


def history(root,c,phase,planned,expected_ids):
    hist=jsonlines(root/'history.jsonl'); audit=read(root/'learning_audit.json'); done=read(root/'TRAIN_COMPLETE.json'); trainability=read(root/'trainability.json')
    if len(hist)!=planned['additional_updates'] or trainability['parent_optimizer_restored'] or trainability['loss']!='pure cross entropy' or trainability['batch']!=2 or trainability['updates']!=len(hist) or trainability['optimizer']!='NEW AdamW' or trainability['parent_global_update']!=planned['parent_updates'] or trainability['smoke'] or not trainability['frozen_BART'] or trainability['parameters']!=25618432: raise ValueError('Changed loss/optimizer/budget')
    seen=set()
    for i,row in enumerate(hist):
        g=planned['parent_updates']+i+1; ix=choose_indices(len(expected_ids),2,g-1,0,c['seed'],1); ids=[expected_ids[j] for j in ix]; seen.update(ids)
        if row['local_update']!=i+1 or row['global_update']!=g or row['batch_example_ids']!=ids or row['unique_examples_seen']!=len(seen) or row['cumulative_presentations']!=g*2 or row['phase_presentations']!=(i+1)*2:
            raise ValueError('Global data cursor/order/presentations differ')
        if not math.isfinite(row['loss']) or not close(row['lr'],phase_lr(phase,i,len(hist))) or row['smoke']: raise ValueError('Invalid actual loss/LR/history')
    if len(seen)!=2048 or audit['updates']!=len(hist) or audit['unique_examples_seen']!=len(seen) or not audit['passed'] or set(audit['groups'])!={'compressor','tx_normalization','receiver','memory'} or any(not v['nonzero_gradient'] or not v['changed'] or v['parameters']<=0 for v in audit['groups'].values()): raise ValueError('Required groups/sample diversity did not learn')
    if done['updates']!=len(hist) or done['global_update']!=planned['target_updates'] or done['loss']!='pure CE' or not done['weights_only_warm_start']: raise ValueError('Training completion differs')
    return hist


def diagnostic(root,declared,receipts,matched):
    rows=jsonlines(root/'rows.jsonl'); done=read(root/'DIAGNOSTIC_COMPLETE.json')
    if [r['example_id'] for r in rows]!=declared['example_ids'] or len(rows)!=done['rows'] or digest(root/'rows.jsonl')!=done['rows_sha256']: raise ValueError('Prefix diagnosis matrix differs')
    for row in rows:
        eid=row['example_id']; source=receipts[eid]
        if row['source_text_sha256']!=source['text_sha256'] or row['source_tokens']!=source['source_tokens'] or row['actual_wire_sha256']!=matched[eid]['tx_wire_sha256'] or not row['diagnostic_only'] or not row['gold_prefix_is_privileged_NOT_actual_RX'] or not row['native_source_memory_NOT_transmitted'] or row['slot_count']!=9 or len(row['slots'])!=9: raise ValueError('Diagnosis treated as real recovery/source mismatch')
        for j,s in enumerate(row['slots']):
            n=min(62,max(0,row['source_tokens']-2-62*j))
            seq=matched[eid]['chunk_generated_token_ids'][j]
            end=next((k for k,t in enumerate(seq[1:],1) if t==2),len(seq)-1)
            common=min(n,max(0,min(64,end)-1))
            if s['slot']!=j or s['gold_content_BPE']!=n or s['common_content_positions']!=common or not 0<=s['TF_argmax_early_EOS']<=n: raise ValueError('Diagnosis content/EOS counts differ')
            for key in ('TF_pEOS_before_gold_end','TF_pEOS_at_gold_end','TF_content_accuracy'):
                v=s[key]
                if v is not None and (not math.isfinite(v) or not 0<=v<=1): raise ValueError('Invalid diagnostic probability')
            for key,expected_count in (('TF_content_CE',n),('native_TF_content_CE',n),('TF_CE_common',common),('free_prefix_CE_common',common)):
                v=s[key]
                if (not expected_count and v is not None) or (expected_count and (v is None or not math.isfinite(v) or v<0)): raise ValueError('Invalid/empty diagnostic CE')
        for key,weight in (('TF_content_CE','gold_content_BPE'),('native_TF_content_CE','gold_content_BPE'),('TF_content_accuracy','gold_content_BPE'),('TF_CE_common','common_content_positions'),('free_prefix_CE_common','common_content_positions')):
            count=sum(s[weight] for s in row['slots']); average=sum(s[key]*s[weight] for s in row['slots'] if s[weight])/max(1,count)
            if count and (not math.isfinite(row[key]) or not close(row[key],average)): raise ValueError('Diagnostic weighted aggregation differs '+key)
        if row['common_content_positions']!=sum(s['common_content_positions'] for s in row['slots']) or not math.isfinite(row['TF_total_CE']) or row['TF_total_CE']<0: raise ValueError('Diagnostic total differs')
    return rows


def verify(root):
    root=Path(root); identity=read(root/'run_identity.json'); c=read(root/'config.json'); phase=load(root/'phase_config.json'); planned=schedule(phase)
    if (root/'FAILED.json').exists(): raise ValueError('Failed experiment cannot be sealed')
    approved=parent(root/'parent_reference',False)
    manifest=read(root/'source_snapshot/MANIFEST.json')
    if manifest!=read(FOLDER/'MANIFEST.json'): raise ValueError('Unknown verifier/source revision')
    for n,sha in manifest['files'].items():
        if digest(file(root,'source_snapshot/'+n))!=sha: raise ValueError('Source snapshot changed')
    if identity['schema']!='docci-text-continue-v11' or identity['status']!='TRAIN_AND_EVAL_FINISHED' or identity['schedule']!=planned or identity['phase']!=phase or identity['config']!=c or c!=approved['config']: raise ValueError('Run/config/schedule differs')
    if identity['gpu_physical']!=1 or identity['optimizer_restored'] or identity['smoke_weights_reused'] or identity['source_at_actual_RX'] or identity['FSO_training'] or identity['CLIP_Common_image_diffusion_used'] or identity['loss']!='pure CE' or not identity['gold_prefix_diagnostics_NOT_actual_recovery']: raise ValueError('Training/RX isolation differs')
    if identity['geometry']!=geometry(c,'T1_chunks') or identity['initial_state_hashes']!=approved['identity']['final_state_hashes'] or identity['parent_checkpoint_sha256']!=approved['checkpoint_sha256'] or identity['backbone_signature_before']!=identity['backbone_signature_after'] or identity['backbone_signature_before']!=approved['identity']['backbone_signature_after'] or identity['active_parameters']!=25618432 or identity['checkpoint_policy']!='fixed final, no validation selection': raise ValueError('Loaded parent/backbone/resources differ')
    finished=read(root/'RUN_FINISHED.json')
    if finished['run_identity_sha256']!=digest(root/'run_identity.json') or not finished['independent_CPU_verification_PENDING']: raise ValueError('GPU finish marker binding differs')
    data=read(root/'data_identity.json'); old=approved['data']
    if data['source']!=old['source'] or data['train']['records']!=old['train']['records'] or data['validation']['records']!=old['validation']['records'] or data['diagnostic_ids']!=old['diagnostic_ids']: raise ValueError('Approved data identity changed')
    tr=[r['example_id'] for r in data['train']['records']]; va=[r['example_id'] for r in data['validation']['records']]
    receipts={r['example_id']:r for r in data['train']['records']+data['validation']['records']}
    hist=history(root/'train_ce',c,phase,planned,tr)
    expected_probes=probes(planned['target_updates'],10000,phase['probe_every'])
    if identity['probes']!=expected_probes: raise ValueError('Missing fixed probes/final checkpoint')
    expected_evals=[]; expected_diag=[]; evaluated={}
    for g in expected_probes:
        stage='g'+str(g); final=g==planned['target_updates']
        for cohort,ids,tag in (('diagnostic_train',tr[:32] if final else tr[:phase['probe_samples']],'train'),('validation',va if final else va[:phase['probe_samples']],'validation')):
            d=dict(folder=stage+'_'+tag,stage=stage,cohort=cohort,example_ids=ids,global_update=g); expected_evals.append(d)
            print('VERIFY '+d['folder']+' metrics/frames',flush=True); evaluated[d['folder']]=evaluation(root/d['folder'],c,d,receipts)
        expected_diag.append(dict(folder=stage+'_prefix_diagnostic',global_update=g,example_ids=va[:phase['probe_samples']]))
        checkpoint=root/('checkpoint_'+str(g))
        if g!=10000:
            ck=read(checkpoint/'CHECKPOINT.json')
            if ck['global_update']!=g or not ck['actual_disk_roundtrip'] or not ck['optimizer_and_RNG_saved']: raise ValueError('Checkpoint/optimizer roundtrip audit differs')
            if (checkpoint/'last.pth').exists() and digest(checkpoint/'last.pth')!=ck['checkpoint_sha256']: raise ValueError('Saved checkpoint changed')
            if final and ck['state_hashes']!=identity['final_state_hashes']: raise ValueError('Final checkpoint identity differs')
    if identity['evaluations']!=expected_evals or identity['diagnostics']!=expected_diag: raise ValueError('Unexpected/foreign/missing stage')
    diag={}
    for d in expected_diag:
        stage='g'+str(d['global_update']); matched={r['example_id']:r for r in evaluated[stage+'_validation'] if r['control']=='matched'}
        diag[d['folder']]=diagnostic(root/d['folder'],d,receipts,matched)
    smoke=read(root/'smoke/SMOKE_COMPLETE.json'); audit=read(root/'smoke/train/learning_audit.json')
    if not smoke['actual_parent_loaded'] or not smoke['actual_backward_and_disk_reload'] or not smoke['saved_smoke_weights_NEVER_reused'] or smoke['parent_restored_state_hashes']!=identity['initial_state_hashes'] or smoke['rows']!=6 or not audit['passed'] or audit['updates']!=2: raise ValueError('Authentic smoke failed or weights reused')
    return dict(identity=identity,evaluations=evaluated,diagnostics=diag,history=hist,parent=approved)


def finish(root,result):
    root=Path(root); ident=result['identity']; final='g'+str(ident['schedule']['target_updates']); c=ident['config']
    train=result['evaluations'][final+'_train']; val=result['evaluations'][final+'_validation']
    train_gate=gate(train,'text',c,True); val_gate=gate(val,'text',c,False)
    original=[r for r in jsonlines(root/'parent_reference/clean_validation/rows.jsonl') if r['control']=='matched']
    received=[r for r in val if r['control']=='matched']; index={r['example_id']:r for r in received}
    delta=np.asarray([index[r['example_id']]['normalized_wer']-r['normalized_wer'] for r in original]); rng=np.random.default_rng(c['seed'])
    bootstrap=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(1)
    outcome=dict(schema='docci-text-continue-v11-summary',engineering_verified=True,
        legacy_quality_status='CLEAN_READY' if train_gate['passed'] and val_gate['passed'] else 'QUALITY_STOP',
        legacy_rule_unchanged=True,train_gate=train_gate,validation_gate=val_gate,
        final_validation=summary(received),baseline_validation=summary(original),
        paired_WER_delta=dict(mean=float(delta.mean()),CI95=np.quantile(bootstrap,[.025,.975]).tolist(),
            improved=int((delta<0).sum()),tied=int((delta==0).sum()),worse=int((delta>0).sum()),
            scope='conditional description bootstrap, one training replicate; not test/training-seed uncertainty'),
        trajectory={str(g):summary([r for r in result['evaluations']['g'+str(g)+'_validation'] if r['control']=='matched' and r['example_id'] in ident['diagnostics'][0]['example_ids']]) for g in ident['probes']},
        prefix_diagnostics=result['diagnostics'],gold_prefix_is_NOT_actual_recovery=True,
        warning='Pure CE weights-only continuation; fixed last; no FSO/joint/digital/hardware claim')
    if (root/'summary.json').exists():
        if read(root/'summary.json')!=outcome: raise FileExistsError('Existing partial summary differs; preserved')
    else: atomic_json(root/'summary.json',outcome)
    e=lambda v:html.escape(str(v)); parts=['<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;padding:24px}table{border-collapse:collapse;table-layout:fixed;width:100%}td,th{padding:10px;border:1px solid #ccd7df;vertical-align:top;overflow-wrap:anywhere;white-space:pre-wrap}</style><h1>DOCCI纯CE权重续训：完整验证恢复</h1>',
        '<p>固定末次权重，旧质量门槛未改。真实前缀诊断不等于传输恢复；源长度只用于评价。所有100篇保留。</p><p>'+e(outcome['legacy_quality_status'])+'</p><table><tr><th>原文（仅评价）</th><th>原10000步权重</th><th>累计'+str(ident['schedule']['target_updates'])+'步权重</th></tr>']
    for row in original:
        new=index[row['example_id']]; parts+=['<tr><td>'+e(row['example_id'])+'\n'+e(row['source_text'])+'</td><td>'+e(row['received_caption'])+'\nWER {:.2%}</td><td>'.format(row['normalized_wer'])+e(new['received_caption'])+'\nWER {:.2%}；到上限 {}</td></tr>'.format(new['normalized_wer'],new['stopped_by_capacity'])]
    parts+=['</table>']
    rendered='\n'.join(parts).encode('utf-8'); page=root/'actual_text_recovery.html'
    if page.exists():
        if page.read_bytes()!=rendered: raise FileExistsError('Existing partial HTML differs; preserved')
    else:
        with page.open('xb') as f: f.write(rendered)
    atomic_json(root/'COMPLETE.json',dict(schema='docci-text-continue-v11',engineering_completion_NOT_quality=True,
        run_identity_sha256=digest(root/'run_identity.json'),summary_sha256=digest(root/'summary.json'),
        legacy_quality_status=outcome['legacy_quality_status']))
    return outcome


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root'); a=p.parse_args(); result=verify(a.root)
    print('Verified existing records; no training or artifact mutation')
