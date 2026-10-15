"""Independent CPU record verification. Never imports Torch or loads pickle."""
import argparse
import hashlib
import html
import math
from pathlib import Path
import sys
import numpy as np
from PIL import Image
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
from docci_semantic_probe_v12.core import (FOLDER,SCHEMA,read,jsonlines,digest,atomic_json,indices,lr,
    compare_pairs,damage_numpy,seed_for)
from docci_text_continue_v11.metrics import metrics,summary
from docci_reconstruction_v10.core import gate,joined_ids


def ah(x): return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def file_at(root, name):
    p=Path(name)
    if p.is_absolute() or not p.parts or any(x in ('.','..') for x in p.parts): raise ValueError('Unsafe record path')
    at=Path(root)
    for part in p.parts:
        at=at/part
        if at.is_symlink(): raise ValueError('Symlink record rejected')
    return at


def matrix(rows, ids, conditions, controls, key='control'):
    expected={(i,c,k) for i in ids for c in conditions for k in controls}; actual=set()
    for row in rows:
        item=(row['example_id'],row['condition'],row[key])
        if item not in expected or item in actual: raise ValueError('Missing/duplicate/unexpected evaluation cell')
        actual.add(item)
    if actual!=expected: raise ValueError('Incomplete evaluation matrix')


def validate_history(folder, c, task, arm, ids, smoke):
    t=read(folder/'trainability.json'); h=jsonlines(folder/'history.jsonl'); audit=read(folder/'learning_audit.json')
    updates=2 if smoke else c[task+'_updates']; size=c['text_batch_size'] if task=='text' else c['image_grad_accum']
    if t['arm']!=arm or t['updates']!=updates or t['training_example_ids']!=ids or len(h)!=updates or not t['backbone_frozen']:
        raise ValueError('Training declaration differs')
    if not audit['passed'] or audit['updates']!=updates or not audit['groups'] or not all(v['parameters'] and v['changed'] and v['nonzero_gradient'] for v in audit['groups'].values()):
        raise ValueError('Learning audit failed')
    seen=set()
    for u,row in enumerate(h):
        selected=[ids[i] for i in indices(len(ids),size,u,c['seed'])]; seen.update(selected)
        if row['update']!=u+1 or row['batch_example_ids']!=selected or row['unique_examples_seen']!=len(seen): raise ValueError('Example order/count differs')
        if not math.isclose(row['lr'],lr(c,task,u,updates),rel_tol=1e-12,abs_tol=1e-15): raise ValueError('LR schedule differs')
        if not all(math.isfinite(row[k]) and row[k]>=0 for k in ('loss','grad_norm','elapsed_seconds')): raise ValueError('Nonfinite training metric')
        if task=='text':
            weight=c['kd_weight'] if arm=='B1_ce_kd' else 0.
            if row['teacher_executed']!=(weight>0) or not math.isclose(row['loss'],row['CE']+weight*row['KD'],rel_tol=1e-6,abs_tol=1e-6): raise ValueError('KD loss/execution differs')
            if arm=='B0_ce' and row['KD']!=0: raise ValueError('Teacher leaked into CE control')
        elif row['condition']!=c['image_train_conditions'][u%2] or row['teacher_executed']:
            raise ValueError('Image condition/loss declaration differs')
    cp=read(folder/'CHECKPOINT.json')
    if cp['updates']!=updates or not cp['exact_disk_roundtrip'] or cp['parent_optimizer_restored'] or read(folder/'TRAIN_COMPLETE.json')['updates']!=updates:
        raise ValueError('Checkpoint/training completion differs')
    weight=folder/'last.pth'
    if weight.exists() and digest(weight)!=cp['checkpoint_sha256']: raise ValueError('Own saved checkpoint differs')
    return t,cp


def text_evaluation(folder, receipts, cohort):
    m=read(folder/'matrix.json'); rows=jsonlines(folder/'rows.jsonl'); done=read(folder/'EVAL_COMPLETE.json')
    ids=list(receipts)
    if m['example_ids']!=ids or m['cohort']!=cohort or m['conditions']!=['clean'] or m['controls']!=['matched','zero','shuffled']:
        raise ValueError('Text matrix identity differs')
    if m['source_count_or_mask_at_RX'] or m['Common_used'] or not m['clean_snr40_is_sentinel']: raise ValueError('Text side channel declaration differs')
    matrix(rows,ids,['clean'],m['controls'])
    if done['rows']!=len(rows) or digest(folder/'rows.jsonl')!=done['rows_sha256'] or digest(folder/'matrix.json')!=done['matrix_sha256']:
        raise ValueError('Text completion hashes differ')
    wires={}
    for eid in ids:
        f=folder/'frames'/('clean__'+eid+'.npz')
        with np.load(f,allow_pickle=False) as z:
            if set(z.files)!={'observed','available','pilot_snr_db','reliability'}: raise ValueError('Text frame fields differ')
            y=z['observed'].copy()
            if not bool(z['available']) or float(z['pilot_snr_db'])!=40. or float(z['reliability'])!=1.: raise ValueError('Clean sentinel differs')
        if y.shape!=(18432,) or not np.isfinite(y).all() or abs(float(np.mean(y**2))-1)>.0001: raise ValueError('Text budget/power differs')
        wires[eid]=(ah(y),digest(f))
    for row in rows:
        eid=row['example_id']; source=receipts[eid]; donor=ids[(ids.index(eid)+1)%len(ids)]
        if row['official_split']!=source['official_split'] or row['source_text_sha256']!=source['text_sha256'] or hashlib.sha256(row['source_text'].encode()).hexdigest()!=source['text_sha256']:
            raise ValueError('Text source identity differs')
        if row['source_tokens']!=source['source_tokens'] or row['input_truncated'] or row['receiver_failure']: raise ValueError('Text source/framing/failure differs')
        if (row['received_sha256'],row['frame_sha256'])!=wires[eid] or row['tx_wire_sha256']!=wires[eid][0]: raise ValueError('Text actual wire differs')
        if row['control']=='shuffled':
            if row['donor_example_id']!=donor or row['donor_received_sha256']!=wires[donor][0]: raise ValueError('Shuffled text donor differs')
        elif row['donor_example_id'] is not None or row['donor_received_sha256'] is not None: raise ValueError('Unexpected donor')
        sequences=row['chunk_generated_token_ids']
        if len(sequences)!=9 or any(not s or s[0]!=2 or len(s)>65 or any(type(t) is not int or not 0<=t<50265 for t in s) for s in sequences): raise ValueError('Nonpublic generated text layout')
        joined,capacity=joined_ids(sequences)
        if joined!=row['generated_token_ids'] or capacity!=row['stopped_by_capacity']: raise ValueError('Actual public chunk joining differs')
        for k,v in metrics(row['source_text'],row['received_caption']).items():
            if isinstance(v,bool) or isinstance(v,str):
                if row[k]!=v: raise ValueError('Text metric differs '+k)
            elif not math.isclose(row[k],v,rel_tol=1e-11,abs_tol=1e-12): raise ValueError('Recomputed text metric differs '+k)
        if row['stats']['payload_complex_symbols']!=9216 or row['stats']['analog_repeats']!=1 or not row['stats']['clean_sentinel_not_measured_snr']:
            raise ValueError('Text frame resources differ')
    return rows


def verify_text(root,c,smoke):
    from docci_text_collect_fix_v11.engine import diagnostic
    identity=read(root/'identity.json'); data=read(root/'data_identity.json')
    if identity['source_signature_before']!=identity['source_signature_after'] or identity['actual_RX_source_text_memory_mask_or_length'] or identity['FSO'] or not identity['teacher_TRAIN_loss_only']:
        raise ValueError('Frozen text/RX contract differs')
    train=data['train']['records']; val=data['validation']['records']
    expected_train,expected_val=(8,2) if smoke else (c['train_samples'],c['val_samples'])
    if len(train)!=expected_train or len(val)!=expected_val: raise ValueError('Text sample sizes differ')
    if any(x['official_split']!='train' for x in train) or any(x['official_split']!='qual_dev' for x in val): raise ValueError('Official split isolation differs')
    if len({r['example_id'] for r in train+val})!=len(train)+len(val): raise ValueError('Text duplicate/overlap')
    arms={}; starts=[]; sizes=[]
    for arm in ('B0_ce','B1_ce_kd'):
        a=root/arm; t,cp=validate_history(a/'train',c,'text',arm,[r['example_id'] for r in train],smoke)
        starts.append(t['initial_state_hashes']); sizes.append(t['active_parameters'])
        if cp['state_hashes']!=identity['arms'][arm]['final_state_hashes']: raise ValueError('Text final tensor receipt differs')
        selected=train[:2 if smoke else c['train_eval_samples']]
        train_rows=text_evaluation(a/'train_evaluation',{r['example_id']:r for r in selected},'diagnostic_train')
        val_rows=text_evaluation(a/'validation',{r['example_id']:r for r in val},'validation')
        matched={r['example_id']:r for r in val_rows if r['control']=='matched'}; selected_diag=val[:min(c['diagnostic_samples'],len(val))]; audit=[]
        diag=diagnostic(a/'prefix_diagnostic',dict(example_ids=[r['example_id'] for r in selected_diag]),
            {r['example_id']:r for r in val},matched,audit)
        train_gate=gate(train_rows,'text',identity['config'],True); val_gate=gate(val_rows,'text',identity['config'],False)
        arms[arm]=dict(validation={control:summary([r for r in val_rows if r['control']==control]) for control in ('matched','zero','shuffled')},
            matched_rows=matched,train_gate=train_gate,validation_gate=val_gate,
            quality_status='SMOKE_ONLY' if smoke else 'CLEAN_READY' if train_gate['passed'] and val_gate['passed'] else 'QUALITY_STOP',
            prefix_content_CE=float(np.mean([r['TF_content_CE'] for r in diag])),
            prefix_content_accuracy=float(np.mean([r['TF_content_accuracy'] for r in diag])),
            privileged_prefix_diagnostic_NOT_actual_recovery=True,FP32_aggregation_checks=len(audit))
    if starts[0]!=starts[1] or sizes[0]!=sizes[1]: raise ValueError('Text controls do not share initialization/parameter budget')
    return dict(arms=arms,WER_B1_minus_B0=compare_pairs(arms['B1_ce_kd']['matched_rows'],arms['B0_ce']['matched_rows'],'normalized_wer'),
        unchanged_legacy_quality_gates=True,teacher_TRAIN_only=True,FSO=False)


def verify_image(root,c,smoke):
    identity=read(root/'identity.json'); data=read(root/'data_identity.json'); train=data['train']['records']; val=data['validation']['records']
    ntrain,nval=(8,2) if smoke else (c['train_samples'],c['val_samples'])
    if len(train)!=ntrain or len(val)!=nval or any(r['official_split']!='train' for r in train) or any(r['official_split']!='qual_dev' for r in val): raise ValueError('Image data/splits differ')
    if identity['FSO'] or identity['Common'] or identity['diffusion'] or not identity['ORACLE_text_is_privileged_NOT_actual_communication']:
        raise ValueError('Image probe scope differs')
    ids=[r['example_id'] for r in val]
    if len({r['example_id'] for r in train+val})!=ntrain+nval: raise ValueError('Image duplicate/overlap')
    sem={r['example_id']:r for r in data['validation_semantics']}; wire={r['example_id']:r for r in data['validation_wires']}
    if list(sem)!=ids or list(wire)!=ids or any(not sem[e]['ORACLE_untransmitted'] or sem[e]['input_truncated'] for e in ids): raise ValueError('Semantic/wire pairing differs')
    if identity['frozen_image_codec_before']!=identity['frozen_image_codec_after'] or identity['frozen_VAE_before']!=identity['frozen_VAE_after'] or data['longclip_signature_before']!=data['longclip_signature_after']:
        raise ValueError('Frozen image backbone changed')
    frames={}
    for condition in c['image_eval_conditions']:
        for eid in ids:
            f=root/'frames'/(condition+'__'+eid+'.npz')
            with np.load(f,allow_pickle=False) as z:
                if set(z.files)!={'observed','transmitted'}: raise ValueError('Image frame fields differ')
                x=z['transmitted'].copy(); y=z['observed'].copy()
            seed=seed_for(c['seed'],'EVAL',condition,eid)
            if x.shape!=(8192,) or ah(x)!=wire[eid]['wire_sha256'] or not np.isfinite(x).all() or abs(float(np.mean(x**2))-1)>.0001:
                raise ValueError('Image payload/power differs')
            if not np.array_equal(damage_numpy(x,condition,seed),y): raise ValueError('Shared source-independent damage differs')
            frames[(condition,eid)]=(digest(f),ah(x),ah(y),seed)
    starts=[]; sizes=[]; groups={}; png_checks=0
    for tag in ('P0_parent','A0_image_only_evaluation','A1_oracle_text_evaluation'):
        if tag!='P0_parent':
            arm=tag[:-len('_evaluation')]; t,cp=validate_history(root/arm/'train',c,'image',arm,[r['example_id'] for r in train],smoke)
            starts.append(t['initial_state_hashes']); sizes.append(t['active_parameters'])
            if cp['state_hashes']!=identity['arms'][arm]['final_state_hashes']: raise ValueError('Image final tensor receipt differs')
        folder=root/tag; m=read(folder/'matrix.json'); rows=jsonlines(folder/'rows.jsonl'); done=read(folder/'EVAL_COMPLETE.json')
        oracle=tag=='A1_oracle_text_evaluation'; controls=['matched','shuffled','zero'] if oracle else ['none']
        if m['example_ids']!=ids or m['controls']!=controls or m['conditions']!=c['image_eval_conditions'] or m['ORACLE_text_untransmitted']!=oracle:
            raise ValueError('Image matrix declaration differs')
        if m['FSO'] or m['image_complex_symbols']!=4096 or m['LPIPS'] is not None: raise ValueError('Unsupported image claim')
        matrix(rows,ids,c['image_eval_conditions'],controls)
        if done['rows']!=len(rows) or done['rows_sha256']!=digest(folder/'rows.jsonl') or done['matrix_sha256']!=digest(folder/'matrix.json'): raise ValueError('Image completion differs')
        for row in rows:
            eid=row['example_id']; condition=row['condition']; donor=ids[(ids.index(eid)+1)%len(ids)]; ctrl=row['control']
            if (row['frame_sha256'],row['transmitted_sha256'],row['observed_sha256'],row['damage_seed'])!=frames[(condition,eid)]: raise ValueError('Image comparison did not share identical frame')
            receipt=val[ids.index(eid)]
            if row['source_image_sha256']!=receipt['image_sha256'] or row['rectangle']!=receipt['rectangle'] or row['official_split']!='qual_dev': raise ValueError('Image source differs')
            if row['source_text_sha256']!=sem[eid]['text_sha256'] or hashlib.sha256(row['source_text'].encode()).hexdigest()!=sem[eid]['text_sha256']: raise ValueError('Image text source differs')
            expected_sem=(sem[donor]['semantic_sha256'] if ctrl=='shuffled' else ah(np.zeros(512,np.float32)) if ctrl=='zero' else sem[eid]['semantic_sha256']) if oracle else None
            if row['semantic_sha256']!=expected_sem or row['semantic_donor_example_id']!=(donor if ctrl=='shuffled' else None) or row['oracle_text_not_transmitted']!=oracle: raise ValueError('Oracle intervention differs')
            if row['payload_complex_symbols']!=4096 or row['analog_repeats']!=1 or row['physical_snr_db'] is not None or row['BER'] is not None or row['LPIPS'] is not None: raise ValueError('Fabricated physical/perceptual metric')
            if not all(math.isfinite(row[k]) for k in ('psnr','mse','ssim')) or row['mse']<0 or not -1<=row['ssim']<=1: raise ValueError('Invalid image metric')
            if not math.isclose(row['psnr'],-10*math.log10(max(row['mse'],1e-12)),rel_tol=2e-6,abs_tol=2e-5): raise ValueError('MSE/PSNR arithmetic differs')
            if ('image_file' in row)!=(ids.index(eid)<c['save_images']): raise ValueError('Predeclared visual subset missing/changed')
            if 'image_file' in row:
                output=file_at(root,row['image_file']); source=file_at(root,row['source_view_file'])
                if digest(output)!=row['image_file_sha256'] or digest(source)!=row['source_view_sha256']: raise ValueError('PNG identity differs')
                with Image.open(output) as im: a=np.asarray(im.convert('RGB')).copy()
                with Image.open(source) as im: b=np.asarray(im.convert('RGB')).copy()
                if a.shape!=(512,512,3) or b.shape!=a.shape or ah(a)!=row['quantized_image_sha256']: raise ValueError('PNG raster differs')
                l,t,r,bottom=row['rectangle']; q=float(np.mean(((a[t:bottom,l:r].astype(np.float64)-b[t:bottom,l:r])/255)**2))
                allowance=2*(.5/255)*math.sqrt(row['mse'])+(.5/255)**2+2e-7
                if abs(q-row['mse'])>allowance: raise ValueError('Saved PNG and reported floating reconstruction differ')
                png_checks+=1
        groups[tag]={(r['condition'],r['control'],r['example_id']):r for r in rows}
    if starts[0]!=starts[1] or sizes[0]!=sizes[1]: raise ValueError('Image controls differ in init/parameter budget')
    result={}
    for condition in c['image_eval_conditions']:
        a={e:groups['A1_oracle_text_evaluation'][(condition,'matched',e)] for e in ids}
        b={e:groups['A0_image_only_evaluation'][(condition,'none',e)] for e in ids}
        shuffled={e:groups['A1_oracle_text_evaluation'][(condition,'shuffled',e)] for e in ids}
        result[condition]=dict(PSNR_A1_minus_A0=compare_pairs(a,b,'psnr'),PSNR_matched_minus_shuffled=compare_pairs(a,shuffled,'psnr'),
            SSIM_A1_minus_A0=compare_pairs(a,b,'ssim'),means={})
        for tag,g in groups.items():
            for ctrl in (['matched','shuffled','zero'] if tag.startswith('A1') else ['none']):
                values=[g[(condition,ctrl,e)] for e in ids]
                result[condition]['means'][tag+'/'+ctrl]={k:float(np.mean([r[k] for r in values])) for k in ('psnr','ssim')}
    # New predeclared diagnostic decision, NOT replacement/relaxation of old quality gates.
    primary=result[c['image_primary_condition']]
    promising=primary['PSNR_A1_minus_A0']['CI95'][0]>0 and primary['PSNR_matched_minus_shuffled']['CI95'][0]>0 and result['clean']['PSNR_A1_minus_A0']['mean']>=0
    return dict(conditions=result,decision='SMOKE_ONLY' if smoke else 'PROMISING_DIAGNOSTIC_ONLY' if promising else 'NEEDS_REVIEW',
        decision_is_NOT_joint_communication_success=True,ORACLE_text_uncharged=True,LPIPS='NOT_MEASURED',
        PNG_subset_MSE_rechecks=png_checks,all_saved_metric_checks='identity, complete pairing and PSNR/MSE arithmetic; no new neural inference or all-sample SSIM recomputation')


def verify(root, smoke_only=False):
    root=Path(root).resolve(strict=True)
    if 'torch' in sys.modules: raise RuntimeError('Independent collector must not import Torch')
    launch=read(root/'launch_identity.json'); tasks=launch['tasks']; c=launch['config']
    if launch['schema']!=SCHEMA or tasks not in (['text'],['image'],['text','image']): raise ValueError('Unknown experiment')
    if (root/'FAILED.json').exists(): raise ValueError('Failed launcher preserved; not a finished experiment')
    if not smoke_only:
        done=read(root/'LAUNCH_FINISHED.json')
        if done['launch_identity_sha256']!=digest(root/'launch_identity.json'): raise ValueError('Launch identity differs')
    result=dict(schema=SCHEMA,task_results={},source_only=False,independent_GPU_reinference=False)
    for mode in (['smoke'] if smoke_only or launch['action']=='smoke' else ['smoke','main']):
        folder=root/mode; identity=read(folder/'run_identity.json'); completed=read(folder/'RUN_FINISHED.json')
        if (folder/'FAILED.json').exists() or completed['run_identity_sha256']!=digest(folder/'run_identity.json') or identity['tasks']!=tasks or identity['config']!=c:
            raise ValueError('GPU run incomplete or identity differs')
        if identity['mode']!=('smoke' if mode=='smoke' else 'train') or identity['smoke_weights_reused'] or identity['FSO']: raise ValueError('Execution scope differs')
        manifest=read(FOLDER/'MANIFEST.json')
        for name,sha in manifest['files'].items():
            if digest(folder/'source_snapshot'/name)!=sha: raise ValueError('Actual source snapshot differs: '+name)
        if digest(folder/'source_snapshot/MANIFEST.json')!=digest(FOLDER/'MANIFEST.json'): raise ValueError('Source manifest differs')
        result['task_results'][mode]={}
        for task in tasks:
            spec=read(FOLDER/'parents.json')[task]
            for name,sha in spec['files'].items():
                if digest(file_at(folder/task/'parent_reference',name))!=sha: raise ValueError('Approved parent evidence differs')
            task_data=read(folder/task/'data_identity.json')
            approved_data=read(folder/task/'parent_reference/data_identity.json')
            if task_data['source']!=approved_data['source']: raise ValueError('Dataset identity differs from approved source')
            result['task_results'][mode][task]=(verify_text if task=='text' else verify_image)(folder/task,c,mode=='smoke')
    return result


def render(root,result):
    root=Path(root); parts=['<!doctype html><meta charset="utf-8"><title>V12 诊断结果</title>',
        '<style>body{font:16px system-ui;max-width:1250px;margin:35px auto;padding:0 20px}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #ccc;padding:8px;vertical-align:top}img{max-width:240px}pre{white-space:pre-wrap}</style>',
        '<h1>V12：文本蒸馏与理想文本辅助图像</h1><p>固定末次权重；qual_dev 已用于研究诊断，不是独立测试。图片组文本未经过传输、未计费，不能宣称同资源图文联合收益。未使用 FSO 或扩散；LPIPS 未测。</p>']
    mode='main' if 'main' in result['task_results'] else 'smoke'; data=result['task_results'][mode]
    if 'text' in data:
        parts.append('<h2>文本：CE 与 CE+KD</h2><pre>'+html.escape(json.dumps({k:v for k,v in data['text'].items() if k!='arms'},ensure_ascii=False,indent=2))+'</pre>')
        a=jsonlines(root/mode/'text/B0_ce/validation/rows.jsonl'); b=jsonlines(root/mode/'text/B1_ce_kd/validation/rows.jsonl')
        aa={r['example_id']:r for r in a if r['control']=='matched'}; bb={r['example_id']:r for r in b if r['control']=='matched'}
        parts.append('<table><tr><th>ID／原文（仅供评价）</th><th>CE 实际恢复</th><th>CE+KD 实际恢复</th></tr>')
        for eid in aa:
            parts.append('<tr><td>'+html.escape(eid+'\n'+aa[eid]['source_text'])+'</td><td>'+html.escape(aa[eid]['received_caption'])+'</td><td>'+html.escape(bb[eid]['received_caption'])+'</td></tr>')
        parts.append('</table>')
    if 'image' in data:
        parts.append('<h2>图像：全部数值与预先固定前几例</h2><pre>'+html.escape(json.dumps(data['image'],ensure_ascii=False,indent=2))+'</pre>')
        records=[]
        for tag in ('P0_parent','A0_image_only_evaluation','A1_oracle_text_evaluation'):
            records+=jsonlines(root/mode/'image'/tag/'rows.jsonl')
        ids=list(dict.fromkeys(r['example_id'] for r in records if 'image_file' in r))
        for eid in ids:
            parts.append('<h3>'+html.escape(eid)+'</h3>')
            first=next(r for r in records if r['example_id']==eid); parts.append('<p>原始描述（理想辅助／评价）：'+html.escape(first['source_text'])+'</p>')
            source=mode+'/image/'+first['source_view_file']; parts.append('<img alt="源图，仅供评价" src="'+html.escape(source,quote=True)+'">')
            for condition in ('clean','mask25'):
                parts.append('<h4>'+condition+'（mask25 不是 FSO）</h4><table><tr>')
                for r in records:
                    if r['example_id']==eid and r['condition']==condition and 'image_file' in r:
                        path=mode+'/image/'+r['image_file']
                        parts.append('<td>'+html.escape(r['tag']+'/'+r['control'])+'<br><img alt="重建图" src="'+html.escape(path,quote=True)+'"><br>PSNR {:.3f} dB；SSIM {:.4f}</td>'.format(r['psnr'],r['ssim']))
                parts.append('</tr></table>')
    page='\n'.join(parts)
    return page


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root'); a=p.parse_args(); result=verify(a.root)
    print(json.dumps({k:v for k,v in result.items() if k!='task_results'},ensure_ascii=False,indent=2))
    print('Independent CPU verification passed. No training, no new inference, no quality-gate changes.')
