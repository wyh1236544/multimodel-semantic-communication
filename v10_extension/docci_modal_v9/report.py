"""Independent records verifier and clearly labelled private-branch displays."""
import argparse
import html
import json
import math
from pathlib import Path
import sys
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
import numpy as np
from PIL import Image
from docci_modal_v9.core import atomic_json,digest,read,jsonlines,geometry,gate,metrics,schedule,load,TEXT_SHA,IMAGE_SHA


def local(root,name):
    path=Path(name)
    if path.is_absolute() or any(x in ('.','..') for x in path.parts): raise ValueError('Unsafe artifact path')
    at=Path(root)
    for part in path.parts:
        at=at/part
        if at.is_symlink(): raise ValueError('Symlink artifact rejected')
    if not at.is_file(): raise FileNotFoundError(str(at))
    return at


def close(a,b): return math.isclose(float(a),float(b),rel_tol=1e-5,abs_tol=1e-5)


def text_row(row,c):
    if row['input_truncated'] is not False or not 2<=row['source_tokens']<=c['max_text_tokens']: raise ValueError('Truncated/invalid full source')
    import hashlib
    if hashlib.sha256(row['source_text'].encode()).hexdigest()!=row['source_text_sha256']: raise ValueError('Source caption hash differs')
    recalculated=metrics(row['source_text'],row['source_text'],row['received_caption'])
    if any((not close(row[k],v) if isinstance(v,(int,float)) else row[k]!=v) for k,v in recalculated.items()): raise ValueError('Full-source text scores differ')
    ids=row['generated_token_ids']
    if len(ids)>c['max_text_tokens']+1 or any(type(x) is not int or not 0<=x<50265 for x in ids): raise ValueError('Invalid public decoder tokens')
    if row['receiver_failure']:
        if row['received_caption'] or ids or row['stopped_by_capacity']: raise ValueError('Failed receiver hallucinated text')
    elif not ids or bool(row['stopped_by_capacity'])==(2 in ids[1:]): raise ValueError('Natural EOS/capacity declaration differs')


def image_row(root,row,c):
    if row['image_size']!=c['image_size'] or not all(math.isfinite(row[k]) for k in ('mse','psnr','ssim')): raise ValueError('Image size/finite scores differ')
    if not 0<=row['mse']<=1 or not -.01<=row['ssim']<=1.01 or not close(row['psnr'],-10*math.log10(max(1e-12,row['mse']))): raise ValueError('Image PSNR/MSE differs')
    l,t,r,b=row['evaluation_rectangle']; size=c['image_size']
    if not 0<=l<r<=size or not 0<=t<b<=size: raise ValueError('Invalid valid full-view rectangle')
    if row['restored_image_file']:
        file=local(root,row['restored_image_file']); source=local(root,row['source_view_file'])
        if digest(file)!=row['restored_image_sha256'] or digest(source)!=row['source_view_sha256']: raise ValueError('Actual image artifact hash differs')
        with Image.open(file) as im: x=np.asarray(im.convert('RGB')).copy()
        with Image.open(source) as im: y=np.asarray(im.convert('RGB')).copy()
        if x.shape!=(size,size,3) or y.shape!=x.shape: raise ValueError('Actual restored size differs')
        import hashlib
        if hashlib.sha256(x.tobytes()).hexdigest()!=row['quantized_image_sha256']: raise ValueError('PNG pixel hash differs')
        mse=float(np.mean(((x[t:b,l:r].astype(float)-y[t:b,l:r])/255)**2))
        # Float-domain score vs nearest8-bit PNG: use the exact quantization bound.
        if abs(math.sqrt(mse)-math.sqrt(row['mse']))>1/510+1e-6: raise ValueError('Displayed image does not support scored MSE')
        if row['receiver_failure'] and x.any(): raise ValueError('Failed receiver not retained as black')


def evaluation(root,expected_ids,branch,c,expected_stage=None,expected_cohort=None):
    matrix=read(root/'matrix.json'); done=read(root/'EVAL_COMPLETE.json'); rows=jsonlines(root/'rows.jsonl')
    if done['rows_sha256']!=digest(root/'rows.jsonl') or done['matrix_sha256']!=digest(root/'matrix.json') or done['rows']!=len(rows): raise ValueError('Evaluation completion identity differs')
    if matrix['example_ids']!=expected_ids or len(set(expected_ids))!=len(expected_ids) or matrix['branch']!=branch or matrix['controls']!=['matched','zero','shuffled']: raise ValueError('Expected private controls/cohort differs')
    if expected_stage and matrix['stage']!=expected_stage or expected_cohort and matrix['cohort']!=expected_cohort: raise ValueError('Stage/cohort label differs')
    if matrix['resources']['total_reals']!=geometry(c,branch)['total_reals'] or matrix['resources']['analog_repeats']!=1: raise ValueError('Private resource differs')
    conditions=matrix['conditions']
    if not conditions or conditions[0]!='clean' or len(set(conditions))!=len(conditions) or not set(conditions)<= {'clean','weak_p06','moderate_p18','moderate_p21','strong_p18'}: raise ValueError('Condition matrix differs')
    if matrix['Common_or_other_modality_used'] or not matrix['failures_retained'] or not matrix['raw_text_never_RX']: raise ValueError('Wrong isolation/privacy scope')
    expected={(cid,eid,ctrl) for cid in conditions for eid in expected_ids for ctrl in matrix['controls']}; seen=set(); frames={}
    for row in rows:
        key=row['condition'],row['example_id'],row['control']
        if key not in expected or key in seen or row['branch']!=branch or row['stage']!=matrix['stage'] or row['cohort']!=matrix['cohort']: raise ValueError('Invalid/missing/duplicate row matrix')
        seen.add(key)
        if row['official_split']!=('train' if matrix['cohort']=='diagnostic_train' else 'qual_dev'): raise ValueError('Official test or wrong source split')
        frame_file=local(root,row['frame_file'])
        if digest(frame_file)!=row['frame_sha256']: raise ValueError('Saved receiver frame differs')
        with np.load(frame_file,allow_pickle=False) as z:
            if set(z.files)!={'observed','available','pilot_snr_db','reliability'}: raise ValueError('Non-public source data at RX')
            observed=z['observed'].copy(); available=bool(z['available']); snr=float(z['pilot_snr_db']); rel=float(z['reliability'])
        import hashlib
        if observed.shape!=(geometry(c,branch)['total_reals'],) or not np.isfinite(observed).all() or hashlib.sha256(observed.tobytes()).hexdigest()!=row['received_sha256']: raise ValueError('Received payload hash/size differs')
        stats=row['stats']
        if stats['receiver_available']!=available or not close(stats['pilot_snr_db'],snr) or not close(stats['analog_reliability'],rel) or row['receiver_failure']==available: raise ValueError('Public receiver state differs')
        if stats['resources']!=matrix['resources'] or stats['payload_complex_symbols']!=geometry(c,branch)['payload_complex_symbols'] or stats['analog_repeats']!=1 or stats['raw_utf8_wire_bytes']!=0: raise ValueError('Actual private resource/hidden text differs')
        if row['condition']=='clean' and (snr!=40 or rel!=1 or not available or not stats['clean_sentinel_not_measured_snr']): raise ValueError('Clean sentinel misrepresented')
        if stats['pre_fec_ber'] is not None or stats['post_fec_ber'] is not None: raise ValueError('BER cannot be defined for continuous features')
        frame_key=row['condition'],row['example_id']; identity=(row['frame_sha256'],row['received_sha256'],row['tx_wire_sha256'],row['noise_seed'])
        if frame_key in frames and frames[frame_key]!=identity: raise ValueError('Controls not same received frame')
        frames[frame_key]=identity
        if branch=='text': text_row(row,c)
        else: image_row(root,row,c)
    if seen!=expected: raise ValueError('Incomplete controls matrix; retain failures')
    for row in rows:
        if row['control']=='shuffled':
            donor=expected_ids[(expected_ids.index(row['example_id'])+1)%len(expected_ids)]
            if row['donor_example_id']!=donor or row['donor_received_sha256']!=frames[(row['condition'],donor)][1]: raise ValueError('Shuffled donor is not held-out received frame')
        elif row['donor_example_id'] is not None or row['donor_received_sha256'] is not None: raise ValueError('Unexpected donor')
    return rows


def verify(root):
    root=Path(root); identity=read(root/'run_identity.json'); c=load(root/'config.json'); data=read(root/'data_identity.json'); branch=identity['branch']
    if identity['schema']!='docci-modal-v9' or c!=identity['config'] or identity['schedule']!=schedule(c,identity['mode'],branch): raise ValueError('Run identity/schedule differs')
    if identity['Common_used'] or identity['opposite_modality_model_loaded'] or identity['raw_UTF8_transmitted'] or identity['diffusion_used']: raise ValueError('Not an independent private branch')
    if identity['backbone_signature_before']!=identity['backbone_signature_after']: raise ValueError('Frozen backbone changed')
    init=identity['initialization']
    if init['checkpoint']:
        from docci_modal_v9.backbones import historical_backbone
        historical_backbone(branch,identity['backbone']['files'])
        if not init['loaded'] or not init['old_checkpoint_unchanged'] or (init.get('preset') and init['sha256']!=(TEXT_SHA if branch=='text' else IMAGE_SHA)): raise ValueError('Historical initialization identity differs')
    saved_manifest=read(local(root,'source_snapshot/MANIFEST.json'))
    if saved_manifest['schema']!='docci-modal-v9-source' or saved_manifest['files']!=identity['source_snapshot_sha256']: raise ValueError('Incomplete published source snapshot')
    trusted=Path(__file__).with_name('MANIFEST.json')
    if trusted.exists() and digest(trusted)!=digest(root/'source_snapshot/MANIFEST.json'): raise ValueError('Source manifest not the trusted V9 revision')
    for name,sha in identity['source_snapshot_sha256'].items():
        if digest(local(root,'source_snapshot/'+name))!=sha: raise ValueError('Source snapshot differs')
    train=data['train']['records']; val=data['validation']['records']; diag=data['diagnostic_ids']; ti=[r['example_id'] for r in train]; vi=[r['example_id'] for r in val]
    plan=identity['schedule']
    if len(ti)!=plan['train_samples'] or len(vi)!=plan['val_samples'] or diag!=ti[:plan['diagnostic_samples']] or set(ti)&set(vi): raise ValueError('DOCCI independent fixed view differs')
    if data['source']['official_counts']!={'train':9647,'test':5000,'qual_dev':100,'qual_test':100}: raise ValueError('Dataset count differs')
    for split,records in (('train',train),('qual_dev',val)):
        if any(r['official_split']!=split for r in records): raise ValueError('Leakage from official test or overlap')
    receipts={r['example_id']:r for r in train+val}
    def bind_source(rows):
        for row in rows:
            r=receipts[row['example_id']]
            if branch=='text':
                if row['source_text_sha256']!=r['text_sha256'] or row['source_tokens']!=r['source_tokens']: raise ValueError('Scored original caption not bound to source receipt')
            elif row['source_image_sha256']!=r['image_sha256'] or row['native_image_size']!=r['native_size'] or row['evaluation_rectangle']!=r['rectangle']: raise ValueError('Scored original image not bound to source receipt')
    for name in ('native_train','native_validation'):
        folder=root/name; rows=jsonlines(folder/'rows.jsonl'); done=read(folder/'NATIVE_COMPLETE.json'); ids=diag if name=='native_train' else vi
        if [r['example_id'] for r in rows]!=ids or done['rows']!=len(ids) or done['rows_sha256']!=digest(folder/'rows.jsonl') or not done['no_transmission']: raise ValueError('Native oracle matrix differs')
        bind_source(rows)
        for row in rows:
            if 'ORACLE' not in row['scope']: raise ValueError('Native source features incorrectly treated as transmission')
            if branch=='text': text_row(row,c)
            else: image_row(folder,row,c)
    evaluations={}
    for name in identity['evaluations']:
        folder=root/name; m=read(folder/'matrix.json'); ids=diag if m['cohort']=='diagnostic_train' else vi
        evaluations[name]=evaluation(folder,ids,branch,c); bind_source(evaluations[name])
    if len(identity['evaluations'])!=len(evaluations): raise ValueError('Duplicate evaluation folder')
    for required in ('initial_validation','overfit_train','overfit_validation'):
        if required not in evaluations: raise ValueError('Mandatory baseline/diagnostic output absent')
    for name,stage,cohort in (('initial_validation','initial','validation'),('overfit_train','overfit','diagnostic_train'),('overfit_validation','overfit_DIAGNOSTIC_WEIGHTS','validation')):
        m=read(root/name/'matrix.json')
        if m['stage']!=stage or m['cohort']!=cohort: raise ValueError('Baseline/diagnostic inference weights mislabelled')
    if gate(evaluations['overfit_train'],branch,c,True)!=identity['overfit_gate']: raise ValueError('Overfit quality gate differs')
    if identity['mode']=='smoke': expected_status='SMOKE_COMPLETE'
    elif identity['mode']=='diagnostic' or not identity['overfit_gate']['passed']:
        expected_status='DIAGNOSTIC_COMPLETE' if identity['mode']=='diagnostic' and identity['overfit_gate']['passed'] else 'QUALITY_STOP'
    else:
        if not identity['clean_reset_from_history_not_overfit'] or 'clean_validation' not in evaluations: raise ValueError('Clean stage not reset or not evaluated')
        if gate(evaluations['clean_validation'],branch,c,False)!=identity['clean_gate']: raise ValueError('Clean validation gate differs')
        expected_status='QUALITY_STOP' if not identity['clean_gate']['passed'] else ('SCREEN_COMPLETE' if identity['mode']=='screen' else 'COMPLETE')
    if identity['status']!=expected_status: raise ValueError('Engineering completion confused with recovery success')
    expected_stages=['train_overfit']+(['train_clean'] if 'clean_gate' in identity else [])+(['train_clean_continue','train_fso_continue'] if expected_status=='COMPLETE' else [])
    if identity['stages']!=expected_stages: raise ValueError('Unexpected/omitted learning stages')
    for stage in expected_stages:
        folder=root/stage; label=stage.replace('train_',''); expected_updates=plan['diagnostic_updates'] if label=='overfit' else plan['clean_updates'] if label=='clean' else plan['continue_updates']
        history=jsonlines(folder/'history.jsonl'); audit=read(folder/'learning_audit.json'); done=read(folder/'TRAIN_COMPLETE.json'); trainability=read(folder/'trainability.json')
        if [r['update'] for r in history]!=list(range(1,expected_updates+1)) or audit['actual_optimizer_updates']!=expected_updates or done['optimizer_updates']!=expected_updates or not done['exact_disk_roundtrip']: raise ValueError('Actual optimizer-update/checkpoint budget differs')
        size=min(c['text_batch_size'],len(diag) if label=='overfit' else len(ti)) if branch=='text' else 1; accum=1 if branch=='text' else c['image_grad_accum']
        if trainability['microbatch_size']!=size or trainability['gradient_accumulation']!=accum or trainability['example_presentations']!=expected_updates*size*accum: raise ValueError('Microsteps mistaken for optimizer updates')
        if not audit['passed'] or any(not v['parameters'] or not v['changed'] or not v['nonzero_gradient'] for v in audit['groups'].values()): raise ValueError('Required source/TX/RX group did not learn')
        expected_ids=set(diag if label=='overfit' else ti)
        for r in history:
            if len(r['batch_example_ids'])!=size*accum or not set(r['batch_example_ids'])<=expected_ids or r['cumulative_example_presentations']!=r['update']*size*accum or not math.isfinite(r['loss']): raise ValueError('Learning history count/leakage differs')
        if not read(folder/'forward_consistency.json')['passed']: raise ValueError('Grad/no-grad interface differs')
    if expected_status=='COMPLETE':
        for stage in ('clean','clean_continue','fso_continue'):
            rows=evaluations.get('physical_'+stage)
            if rows is None or {r['condition'] for r in rows}!={'clean','weak_p06','moderate_p18','moderate_p21','strong_p18'}: raise ValueError('Missing actual full-waveform matrix')
        a=jsonlines(root/'train_clean_continue/history.jsonl'); b=jsonlines(root/'train_fso_continue/history.jsonl')
        if [r['batch_example_ids'] for r in a]!=[r['batch_example_ids'] for r in b] or [r['lr'] for r in a]!=[r['lr'] for r in b]: raise ValueError('Continuation training budgets/orders not matched')
    return dict(identity=identity,evaluations=evaluations,verified=True)


def finish(root):
    root=Path(root); result=verify(root); identity=result['identity']; branch=identity['branch']; c=identity['config']
    summaries={}
    for name,rows in result['evaluations'].items():
        summaries[name]={cid:gate([r for r in rows if r['condition']==cid],branch,c,name=='overfit_train' or name.startswith('snapshot_')) for cid in sorted({r['condition'] for r in rows})}
    atomic_json(root/'summary.json',dict(branch=branch,status=identity['status'],overfit_gate=identity['overfit_gate'],clean_gate=identity.get('clean_gate'),
        results=summaries,warning='TRAIN4 overfit is diagnostic, VAL8 is exploratory; no Common/joint gain, no independent test or digital comparison claimed'))
    e=lambda s:html.escape(str(s)); chunks=['<!doctype html><meta charset="utf-8"><title>DOCCI private branch</title><style>body{font:16px sans-serif;padding:24px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:10px;vertical-align:top}img{width:100%;max-width:512px}td{word-break:break-word}</style>',
        '<h1>DOCCI 独立'+('文本' if branch=='text' else '图像')+'分支</h1><p>'+e(identity['status'])+'；诊断记忆与验证集表现分开。无 Common、跨模态融合或扩散；原生参考不经过传输。</p>']
    for name,rows in result['evaluations'].items():
        if name.startswith('snapshot_'): continue
        chunks+=['<h2>'+e(name)+'</h2>']
        for cid in sorted({r['condition'] for r in rows}):
            selected=[r for r in rows if r['condition']==cid]; ids=list(dict.fromkeys(r['example_id'] for r in selected)); chunks+=['<h3>'+e(cid)+'</h3><table><tr><th>原始输入（仅评价）</th><th>正常接收</th><th>清零</th><th>打乱</th></tr>']
            for eid in ids:
                controls={r['control']:r for r in selected if r['example_id']==eid}; first=controls['matched']
                source=e(first['source_text']) if branch=='text' else ('<img src="'+e(name+'/'+first['source_view_file'])+'">' if first.get('source_view_file') else e(eid))
                chunks+=['<tr><td>'+source+'</td>']
                for key in ('matched','zero','shuffled'):
                    row=controls[key]
                    content=e(row['received_caption'])+'<p>WER {:.1%}；容量停止 {}</p>'.format(row['normalized_wer'],e(row['stopped_by_capacity'])) if branch=='text' else ('<img src="'+e(name+'/'+row['restored_image_file'])+'">' if row['restored_image_file'] else '未保存本张预览')+'<p>PSNR {:.2f}dB；SSIM {:.3f}</p>'.format(row['psnr'],row['ssim'])
                    chunks+=['<td>'+content+'</td>']
                chunks+=['</tr>']
            chunks+=['</table>']
    for name in ('native_train','native_validation'):
        chunks+=['<h2>'+e(name)+' — 原生参考 ORACLE，未压缩/未经过通信</h2>']
        for row in jsonlines(root/name/'rows.jsonl'):
            chunks+=['<p>'+e(row['example_id'])+'</p>']
            chunks+=['<p>'+e(row['received_caption'])+'</p>'] if branch=='text' else ['<img src="'+e(name+'/'+row['restored_image_file'])+'">']
    with (root/'actual_recovery.html').open('x',encoding='utf-8') as f: f.write('\n'.join(chunks))
    atomic_json(root/(identity['status']+'.json'),dict(verified=True,status=identity['status'],branch=branch,
        run_identity_sha256=digest(root/'run_identity.json'),summary_sha256=digest(root/'summary.json'),quality_success_not_implied_by_verified=True))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root'); a=p.parse_args(); result=verify(a.root)
    print('Verified independent {} records; status={}. No new training.'.format(result['identity']['branch'],result['identity']['status']))
