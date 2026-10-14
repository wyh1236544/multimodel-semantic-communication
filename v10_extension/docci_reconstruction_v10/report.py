"""Independent matrix/metrics/resources verifier; completion is not quality."""
import argparse
import hashlib
import html
import math
from pathlib import Path
import sys
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
import numpy as np
from docci_modal_v9.report import local,close,image_row,text_row as native_text_row
from docci_reconstruction_v10.core import (read,digest,jsonlines,load,plan,geometry,gate,metrics,
    chunks,joined_ids,atomic_json,TEXT_ARMS,IMAGE_ARMS,choose_indices,objective)
from docci_reconstruction_v10.channel import resource


def text_row(row,c,arm):
    if row['input_truncated'] is not False or not 2<=row['source_tokens']<=c['max_text_tokens']: raise ValueError('Full original text truncated')
    if hashlib.sha256(row['source_text'].encode()).hexdigest()!=row['source_text_sha256']: raise ValueError('Source caption hash')
    expected=metrics(row['source_text'],row['source_text'],row['received_caption'])
    for k,v in expected.items():
        if not close(row[k],v) if isinstance(v,(int,float)) else row[k]!=v: raise ValueError('Full-source text metrics differ')
    sequences=row['chunk_generated_token_ids']; cap=c['chunk_tokens'] if arm=='T1_chunks' else c['max_text_tokens']
    if row['receiver_failure']:
        if sequences or row['generated_token_ids'] or row['received_caption'] or row['stopped_by_capacity']: raise ValueError('Failed receiver generated text')
        return
    if len(sequences)!=(chunks(c) if arm=='T1_chunks' else 1): raise ValueError('RX secretly used source chunk count')
    for seq in sequences:
        if not seq or len(seq)>cap+1 or seq[0]!=2 or any(type(v) is not int or not 0<=v<50265 for v in seq): raise ValueError('Invalid unconstrained decoder tokens')
    ids,stopped=joined_ids(sequences)
    if ids!=row['generated_token_ids'] or stopped!=row['stopped_by_capacity']: raise ValueError('Whole BPE/EOS reassembly differs')


def evaluation(root,expected_ids,c,arm,stage,cohort):
    matrix=read(root/'matrix.json'); done=read(root/'EVAL_COMPLETE.json'); rows=jsonlines(root/'rows.jsonl')
    branch='text' if arm in TEXT_ARMS else 'image'; g=geometry(c,arm)
    if done['rows_sha256']!=digest(root/'rows.jsonl') or done['matrix_sha256']!=digest(root/'matrix.json') or done['rows']!=len(rows): raise ValueError('Evaluation completion differs')
    if matrix['example_ids']!=expected_ids or len(set(expected_ids))!=len(expected_ids) or matrix['arm']!=arm or matrix['branch']!=branch or matrix['stage']!=stage or matrix['cohort']!=cohort: raise ValueError('Cohort/model/stage differs')
    if matrix['controls']!=['matched','zero','shuffled'] or matrix['source_count_or_mask_at_RX'] or matrix['Common_used'] or not matrix['failures_retained']: raise ValueError('Wrong controls/RX scope')
    if matrix['resources']!=resource(c,arm): raise ValueError('All-symbol waveform accounting differs')
    conditions=matrix['conditions']
    if conditions[0]!='clean' or len(set(conditions))!=len(conditions) or not set(conditions)<={'clean','weak_p06','moderate_p18','moderate_p21','strong_p18'}: raise ValueError('Condition matrix')
    expected={(cid,eid,ctrl) for cid in conditions for eid in expected_ids for ctrl in matrix['controls']}; seen=set(); frames={}
    for row in rows:
        key=row['condition'],row['example_id'],row['control']
        if key not in expected or key in seen or row['arm']!=arm or row['branch']!=branch or row['stage']!=stage or row['cohort']!=cohort: raise ValueError('Duplicate/foreign evaluation row')
        seen.add(key)
        if row['official_split']!=('train' if cohort=='diagnostic_train' else 'qual_dev'): raise ValueError('Official test leakage')
        file=local(root,row['frame_file'])
        if digest(file)!=row['frame_sha256']: raise ValueError('Public received file hash differs')
        with np.load(file,allow_pickle=False) as z:
            if set(z.files)!={'observed','available','pilot_snr_db','reliability'}: raise ValueError('Non-public RX data')
            y=z['observed'].copy(); ok=bool(z['available']); snr=float(z['pilot_snr_db']); rel=float(z['reliability'])
        if y.shape!=(g['total_reals'],) or not np.isfinite(y).all() or hashlib.sha256(y.tobytes()).hexdigest()!=row['received_sha256']: raise ValueError('RX symbol budget/hash')
        stats=row['stats']
        if stats['resources']!=matrix['resources'] or stats['receiver_available']!=ok or row['receiver_failure']==ok or not close(snr,stats['pilot_snr_db']) or not close(rel,stats['analog_reliability']): raise ValueError('Receiver/pilot/resource mismatch')
        if stats['payload_complex_symbols']!=g['payload_complex_symbols'] or stats['analog_repeats']!=1 or stats['raw_utf8_wire_bytes']!=0 or stats['pre_fec_ber'] is not None or stats['post_fec_ber'] is not None: raise ValueError('Hidden repetitions/raw-text/binary BER')
        if row['condition']=='clean' and (snr!=40 or rel!=1 or not ok or not stats['clean_sentinel_not_measured_snr']): raise ValueError('Clean40 sentinel mismatch')
        if row['condition']!='clean':
            if stats['used_samples']>stats['resources']['slot_samples'] or not stats['resource_feasible'] or not math.isfinite(stats['analog_feature_mse']): raise ValueError('Actual physical frame resource differs')
        framekey=row['condition'],row['example_id']; value=(row['frame_sha256'],row['received_sha256'],row['tx_wire_sha256'],row['noise_seed'])
        if framekey in frames and frames[framekey]!=value: raise ValueError('Controls did not share received frame')
        frames[framekey]=value
        if branch=='text': text_row(row,c,arm)
        else: image_row(root,row,c)
    if seen!=expected: raise ValueError('Incomplete row matrix; no dropping failures')
    for row in rows:
        if row['control']=='shuffled':
            donor=expected_ids[(expected_ids.index(row['example_id'])+1)%len(expected_ids)]
            if donor==row['example_id'] or row['donor_example_id']!=donor or row['donor_received_sha256']!=frames[(row['condition'],donor)][1]: raise ValueError('Invalid shuffled donor')
        elif row['donor_example_id'] is not None or row['donor_received_sha256'] is not None: raise ValueError('Unexpected donor')
    return rows


def verify(root):
    root=Path(root); identity=read(root/'run_identity.json'); c=load(root/'config.json'); arm=identity['arm']; branch=identity['branch']
    if identity['schema']!='docci-reconstruction-v10' or c!=identity['config'] or branch!=('text' if arm in TEXT_ARMS else 'image') or arm not in TEXT_ARMS+IMAGE_ARMS: raise ValueError('Run arm/schema differs')
    schedule=plan(c,identity['mode'],branch)
    if identity['schedule']!=schedule or identity['geometry']!=geometry(c,arm) or identity['gpu_physical']!=1: raise ValueError('Declared schedule/resource/GPU mismatch')
    if any(identity[k] for k in ('Common_used','diffusion_used','opposite_modality_loaded','raw_UTF8_transmitted','source_mask_or_length_at_RX')): raise ValueError('Wrong isolation boundary')
    if identity['backbone_signature_before']!=identity['backbone_signature_after']: raise ValueError('Frozen backbone changed')
    from docci_modal_v9.backbones import historical_backbone
    historical_backbone(branch,identity['backbone']['files'])
    init=identity['initialization']
    if init['checkpoint']:
        from docci_modal_v9.core import TEXT_SHA,IMAGE_SHA
        if not init['loaded'] or init['sha256']!=(TEXT_SHA if branch=='text' else IMAGE_SHA) or not init['old_checkpoint_unchanged']: raise ValueError('Historical initialization pin differs')
    if arm=='I2_shifted' and init['checkpoint'] is not None: raise ValueError('New architecture incorrectly claims historical tensor initialization')
    manifest=read(local(root,'source_snapshot/MANIFEST.json')); published=Path(__file__).with_name('MANIFEST.json')
    if manifest['schema']!='docci-reconstruction-v10-source' or manifest['files']!=identity['source_snapshot_sha256']: raise ValueError('Wrong source snapshot')
    if published.exists() and digest(published)!=digest(root/'source_snapshot/MANIFEST.json'): raise ValueError('Untrusted source revision')
    for name,sha in manifest['files'].items():
        if digest(local(root,'source_snapshot/'+name))!=sha: raise ValueError('Source snapshot changed')
    data=read(root/'data_identity.json'); tr=data['train']['records']; va=data['validation']['records']; ti=[r['example_id'] for r in tr]; vi=[r['example_id'] for r in va]; di=data['diagnostic_ids']
    if len(ti)!=schedule['train_samples'] or len(vi)!=schedule['val_samples'] or len(set(ti))!=len(ti) or len(set(vi))!=len(vi) or set(ti)&set(vi) or di!=ti[:schedule['gate_train_samples']]: raise ValueError('Fixed full source view differs')
    if data['source']['official_counts']!={'train':9647,'test':5000,'qual_dev':100,'qual_test':100}: raise ValueError('Not official DOCCI counts')
    for split,items in (('train',tr),('qual_dev',va)):
        if any(r['official_split']!=split for r in items): raise ValueError('Test leakage')
        if branch=='text' and any(r['fixed_chunks']!=chunks(c) or not r['no_truncation'] for r in items): raise ValueError('Full original BPE cache not preserved')
    receipts={r['example_id']:r for r in tr+va}
    def bind(rows):
        for row in rows:
            source=receipts[row['example_id']]
            if branch=='text':
                if row['source_text_sha256']!=source['text_sha256'] or row['source_tokens']!=source['source_tokens']: raise ValueError('Scored full source receipt differs')
            elif row['source_image_sha256']!=source['image_sha256'] or row['native_image_size']!=source['native_size'] or row['evaluation_rectangle']!=source['rectangle']: raise ValueError('Scored image receipt differs')
    for name,ids in (('native_train',di),('native_validation',vi)):
        folder=root/name; rows=jsonlines(folder/'rows.jsonl'); done=read(folder/'NATIVE_COMPLETE.json')
        if [r['example_id'] for r in rows]!=ids or len(rows)!=done['rows'] or digest(folder/'rows.jsonl')!=done['rows_sha256'] or not done['no_transmission']: raise ValueError('Native reference matrix')
        bind(rows)
        for row in rows:
            if 'ORACLE' not in row['scope']: raise ValueError('Native reference treated as transmitted')
            if branch=='text': native_text_row(row,c)
            else: image_row(folder,row,c)
    if arm=='T1_chunks':
        folder=root/'native_chunk_validation'; rows=jsonlines(folder/'rows.jsonl'); done=read(folder/'NATIVE_COMPLETE.json')
        if [r['example_id'] for r in rows]!=vi or done['rows']!=len(rows) or digest(folder/'rows.jsonl')!=done['rows_sha256'] or not done['no_transmission']: raise ValueError('Native chunk oracle matrix differs')
        bind(rows)
        for row in rows:
            if 'ORACLE' not in row['scope']: raise ValueError('Native chunk source-mask oracle treated as RX')
            text_row(row,c,arm)
    evaluations={}
    required={'initial_validation':('initial','validation',vi),'clean_train':('clean','diagnostic_train',di),'clean_validation':('clean','validation',vi)}
    if identity['status']=='PIPELINE_COMPLETE':
        for stage in ('clean','clean_continue','fso_continue'): required['physical_'+stage]=(stage,'validation',vi)
    if set(identity['evaluations'])!=set(required) or len(identity['evaluations'])!=len(required): raise ValueError('Missing/unexpected stage evaluation')
    for name,(stage,cohort,ids) in required.items():
        evaluations[name]=evaluation(root/name,ids,c,arm,stage,cohort); bind(evaluations[name])
    train_gate=gate(evaluations['clean_train'],branch,c,True); validation_gate=gate(evaluations['clean_validation'],branch,c,False)
    if train_gate!=identity['train_gate'] or validation_gate!=identity['validation_gate']: raise ValueError('Recovery quality gate mismatch')
    quality=train_gate['passed'] and validation_gate['passed']
    status='SMOKE_COMPLETE' if identity['mode']=='smoke' else 'QUALITY_STOP' if not quality else 'PIPELINE_COMPLETE' if identity['mode']=='pipeline' else 'CLEAN_READY'
    if status!=identity['status']: raise ValueError('Execution confused with quality success')
    stages=['train_clean']+(['train_clean_continue','train_fso_continue'] if status=='PIPELINE_COMPLETE' else [])
    if identity['stages']!=stages: raise ValueError('Training stage set differs')
    histories={}
    for name in stages:
        folder=root/name; label=name[6:]; updates=schedule['updates'] if label=='clean' else c[branch+'_continue_updates']
        history=jsonlines(folder/'history.jsonl'); audit=read(folder/'learning_audit.json'); done=read(folder/'TRAIN_COMPLETE.json'); trainability=read(folder/'trainability.json')
        size=min(c['text_batch_size'],len(ti)) if branch=='text' else 1; accum=1 if branch=='text' else c['image_grad_accum']
        if [r['update'] for r in history]!=list(range(1,updates+1)) or done['optimizer_updates']!=updates or done['arm']!=arm or done['stage']!=label or not done['exact_disk_roundtrip']: raise ValueError('Checkpoint/optimizer budget differs')
        if trainability['training_examples']!=ti or trainability['microbatch_size']!=size or trainability['gradient_accumulation']!=accum or trainability['example_presentations']!=updates*size*accum: raise ValueError('Actual diverse TRAIN presentation budget differs')
        if not audit['passed'] or audit['actual_optimizer_updates']!=updates or any(not v['nonzero_gradient'] or not v['changed'] or v['parameters']<=0 for v in audit['groups'].values()): raise ValueError('Required codec did not learn')
        seen=set()
        for r in history:
            expected_ids=[]
            for micro in range(accum): expected_ids += [ti[i] for i in choose_indices(len(ti),size,r['update']-1,micro,c['seed'],accum)]
            seen.update(expected_ids)
            if r['batch_example_ids']!=expected_ids or r['unique_examples_seen']!=len(seen) or r['cumulative_example_presentations']!=r['update']*size*accum or not math.isfinite(r['loss']): raise ValueError('Actual order/count/loss differs')
            if label=='clean':
                expected_loss=r['ce_or_latent_mse'] if branch=='text' else objective(c,arm,r['ce_or_latent_mse'],r['pixel_mse'])
                if not close(r['loss'],expected_loss): raise ValueError('Declared clean loss weights differ from measured components')
        if audit['unique_examples_seen']!=len(seen) or (updates*size*accum>=len(ti) and seen!=set(ti)): raise ValueError('TRAIN samples not genuinely used')
        if not read(folder/'forward_consistency.json')['passed']: raise ValueError('Grad/no-grad mismatch')
        if branch=='image':
            probes=history[0]['loss_gradient_probes']
            if not probes or any(not math.isfinite(v[k]) or v[k]<=0 for v in probes for k in ('latent_weighted_gradient_norm','pixel_weighted_gradient_norm')): raise ValueError('Missing/nonfinite/zero actual weighted pixel/latent gradient audit')
        if (folder/'last.pth').exists() and digest(folder/'last.pth')!=done['checkpoint_sha256']: raise ValueError('Saved checkpoint changed')
        histories[name]=history
    if status=='PIPELINE_COMPLETE':
        for name in ('physical_clean','physical_clean_continue','physical_fso_continue'):
            if {r['condition'] for r in evaluations[name]}!={'clean','weak_p06','moderate_p18','moderate_p21','strong_p18'}: raise ValueError('Incomplete full DSP evaluation')
        a=histories['train_clean_continue']; b=histories['train_fso_continue']
        if [(r['batch_example_ids'],r['lr']) for r in a]!=[(r['batch_example_ids'],r['lr']) for r in b]: raise ValueError('Noise/clean continuation not matched')
    return dict(identity=identity,evaluations=evaluations,histories=histories,verified=True)


def finish(root):
    root=Path(root); result=verify(root); identity=result['identity']; branch=identity['branch']; c=identity['config']
    summaries={name:{cid:gate([r for r in rows if r['condition']==cid],branch,c,name=='clean_train') for cid in sorted({r['condition'] for r in rows})} for name,rows in result['evaluations'].items()}
    atomic_json(root/'summary.json',dict(arm=identity['arm'],status=identity['status'],resources=resource(c,identity['arm']),results=summaries,
        warning='Exploratory validation / one training replicate. Chunking charges ALL9 segments; no joint/compression/digital superiority claim.'))
    e=lambda value:html.escape(str(value)); output=['<!doctype html><meta charset="utf-8"><style>body{font:16px sans-serif;padding:24px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px;vertical-align:top}img{max-width:512px;width:100%}</style>',
        '<h1>'+e(identity['arm'])+'</h1><p>'+e(identity['status'])+'；固定末步，原文只用于发送和评价，原生参考未经传输。</p>',
        '<p>'+e(resource(c,identity['arm'])['payload_complex_symbols'])+' 个连续复符号；R=1；分段空段也计费。PSNR/WER ≠ 压缩效率或联合收益。</p>']
    for name,rows in result['evaluations'].items():
        output+=['<h2>'+e(name)+'</h2>']
        for cid in sorted({r['condition'] for r in rows}):
            selected=[r for r in rows if r['condition']==cid]; ids=list(dict.fromkeys(r['example_id'] for r in selected))
            output+=['<h3>'+e(cid)+'</h3><table><tr><th>源输入（仅评价）</th><th>正常接收</th><th>清零</th><th>打乱</th></tr>']
            for eid in ids:
                controls={r['control']:r for r in selected if r['example_id']==eid}; first=controls['matched']
                source=e(first['source_text']) if branch=='text' else '<img src="'+e(name+'/'+first['source_view_file'])+'">' if first.get('source_view_file') else e(eid)
                output+=['<tr><td>'+source+'</td>']
                for ctrl in ('matched','zero','shuffled'):
                    row=controls[ctrl]
                    value=e(row['received_caption'])+'<p>全文WER {:.1%}；容量停止 {}</p>'.format(row['normalized_wer'],e(row['stopped_by_capacity'])) if branch=='text' else ('<img src="'+e(name+'/'+row['restored_image_file'])+'">' if row['restored_image_file'] else '预览未保存')+'<p>PSNR {:.2f}dB；SSIM {:.3f}</p>'.format(row['psnr'],row['ssim'])
                    output+=['<td>'+value+'</td>']
                output+=['</tr>']
            output+=['</table>']
    for name in ('native_validation',)+(('native_chunk_validation',) if identity['arm']=='T1_chunks' else ()):
        output+=['<h2>'+e(name)+' — 原生参考，未压缩/未传输</h2>']
        for row in jsonlines(root/name/'rows.jsonl'):
            value=e(row['received_caption'])+'<p>全文WER {:.1%}</p>'.format(row['normalized_wer']) if branch=='text' else '<img src="'+e(name+'/'+row['restored_image_file'])+'"><p>PSNR {:.2f}dB</p>'.format(row['psnr'])
            output+=['<p>'+e(row['example_id'])+'</p>'+value]
    with (root/'actual_recovery.html').open('x',encoding='utf-8') as f: f.write('\n'.join(output))
    atomic_json(root/(identity['status']+'.json'),dict(verified=True,run_identity_sha256=digest(root/'run_identity.json'),summary_sha256=digest(root/'summary.json'),engineering_completion_not_quality=True))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root'); a=p.parse_args(); result=verify(a.root)
    print('Verified {} status={}; no training'.format(result['identity']['arm'],result['identity']['status']))
