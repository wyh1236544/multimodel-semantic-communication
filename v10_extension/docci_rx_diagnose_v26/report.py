"""Independent diagnostic aggregation, arithmetic/identity checks and PRIVATE review forms."""
import argparse
import csv
import html
import json
import math
from pathlib import Path
import statistics
import sys
import numpy as np
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_rx_diagnose_v26.core import (read,rows,sha,atomic,array_sha,config,selected,
    matrix,key,reference_counts,close,verify_files,REGIMES,CHANNELS,CONTROLS,PINS,seed,source_check)

def losses(row,reference):
    expected=reference_counts(reference['gold_ids'])
    if len(row['slots'])!=9:raise ValueError('Missing slot diagnostics')
    for j,(r,want) in enumerate(zip(row['slots'],expected)):
        if r['slot']!=j or r['counts']!=want or not 0<=r['content_correct']<=want['content']:
            raise ValueError('Wrong TF gold counts/accuracy')
        if any(not math.isfinite(v) or v<0 for v in r['nll'].values()):raise ValueError('Invalid NLL')
        if not close(r['nll']['all'],math.fsum(r['nll'][k] for k in ('content','eos','bos'))):
            raise ValueError('TF component loss differs')
    for k in ('all','content','eos','bos'):
        n=sum(s['counts'][k] for s in row['slots']);nll=math.fsum(s['nll'][k] for s in row['slots'])
        if row['counts'][k]!=n or not close(row['nll'][k],nll):raise ValueError('Weighted slot aggregation differs')
    if row['counts']['all']!=reference['source_tokens']+16 or row['counts']['content']!=reference['source_tokens']-2:
        raise ValueError('Source text content was lost')
    if not close(row['TF_all_CE'],row['nll']['all']/row['counts']['all']):raise ValueError('All CE differs')
    nc=row['counts']['content']
    want=row['nll']['content']/nc if nc else None
    if (want is None and row['TF_content_CE'] is not None) or (want is not None and not close(row['TF_content_CE'],want)):
        raise ValueError('Content CE differs')
    want=sum(s['content_correct'] for s in row['slots'])/nc if nc else None
    if (want is None and row['TF_content_accuracy'] is not None) or (want is not None and not close(row['TF_content_accuracy'],want)):
        raise ValueError('Content accuracy differs')
    if not row['teacher_forced_gold_prefix'] or not row['gold_masks_used_for_measurement_only'] or row['source_mask_at_receiver'] or row['training_updates']:
        raise ValueError('TF diagnostic vs free RX boundary differs')

def trace(row,reference,checkpoint):
    enabled=row['arm']!='base' and row['control']!='disabled';v=row['trace']
    if v['enabled']!=enabled or len(v['layers'])!=(2 if enabled else 0):raise ValueError('Trace activation differs')
    tt=CHANNELS.index(REGIMES[row['regime']][1]);gate=reference['received_gates'][tt]
    for i,layer in enumerate(v['layers']):
        saved=checkpoint['adapters'][row['arm']]['layers'][i]
        if layer['layer']!=i or not close(layer['alpha'],saved['alpha']) or not close(layer['tanh_alpha'],saved['tanh_alpha']):
            raise ValueError('Actual checkpoint alpha differs')
        if not close(layer['tanh_alpha'],math.tanh(layer['alpha'])):raise ValueError('Invalid gate coefficient')
        if layer['received_content_gates']!=gate:raise ValueError('Trace used wrong receiver-only soft gate')
        for scope in ('all','content'):
            if len(layer['scopes'][scope])!=9:raise ValueError('Missing residual slots')
            for j,s in enumerate(layer['scopes'][scope]):
                if s['positions']!=row['slots'][j]['counts'][scope]:raise ValueError('Residual mask differs')
                h,d=s['hidden_l2_sq'],s['delta_l2_sq']
                if not math.isfinite(h) or not math.isfinite(d) or min(h,d)<0:raise ValueError('Invalid actual residual')
                expected=math.sqrt(d/h) if s['positions'] and h else None
                if expected is None:
                    if s['effective_ratio'] is not None:raise ValueError('Empty residual scope must be null')
                elif not close(s['effective_ratio'],expected):raise ValueError('Residual ratio differs')

def summarize(tf,gen,c):
    from docci_blip_feature_rx_v25.report import bootstrap
    groups={}
    for r in tf:groups.setdefault((r['cohort'],r['regime'],r['arm'],r['control']),[]).append(r)
    out=[]
    for (cohort,regime,arm,ctrl),rs in groups.items():
        summed=math.fsum(r['nll']['content'] for r in rs);n=sum(r['counts']['content'] for r in rs)
        per=[r['TF_content_CE'] for r in rs if r['TF_content_CE'] is not None]
        matching=[r for r in gen if (r['cohort'],r['regime'],r['arm'],r['control'])==(cohort,regime,arm,ctrl)]
        ratios=[s['effective_ratio'] for r in rs for layer in r['trace']['layers']
                for s in layer['scopes']['content'] if s['effective_ratio'] is not None]
        out.append(dict(cohort=cohort,regime=regime,arm=arm,control=ctrl,n=len(rs),
            token_weighted_content_CE=summed/n if n else None,mean_example_content_CE=statistics.mean(per) if per else None,
            mean_greedy_WER=statistics.mean(r['normalized_wer'] for r in matching),
            mean_content_residual_ratio=statistics.mean(ratios) if ratios else None,
            mean_repeat_fraction=statistics.mean(r['output_audit']['adjacent_repeat_fraction'] for r in matching)))
    by={key(r):r for r in tf};gains=[]
    for cohort in ('train','dev'):
        ids=sorted({r['example_id'] for r in tf if r['cohort']==cohort})
        for regime in c['regimes']:
            for arm,ctrl,label in (('self','matched','self'),('visual','shuffled','wrong_image'),('base','matched','frozen_text')):
                delta=[by[e,regime,arm,ctrl]['TF_content_CE']-by[e,regime,'visual','matched']['TF_content_CE'] for e in ids
                    if by[e,regime,arm,ctrl]['TF_content_CE'] is not None]
                gains.append(dict(cohort=cohort,regime=regime,comparison=label,n=len(delta),
                    mean_content_CE_reduction=statistics.mean(delta) if delta else None,paired95CI=bootstrap(delta) if delta else None))
    return dict(groups=out,TF_content_comparisons=gains,training_updates=0,
        interpretation='diagnostic evidence, NOT an automatic unique-cause classifier or semantic success',
        confidence='single fixed checkpoint; reused cohorts; pointwise example-bootstrap; uncorrected exploratory intervals',
        facts='PENDING',semantic_success=False,changed_old_V25_screen=False)

def verify(root,inputs):
    from docci_rx_diagnose_v26.parents import lexical
    from docci_text_continue_v11.metrics import body
    root=Path(root);inputs=Path(inputs);verify_files(root);verify_files(inputs)
    index=read(inputs/'PARENT_INDEX.json')
    if sha(inputs/'PARENT_INDEX.json')!=PINS['cache/index.json']:raise ValueError('Pinned reference index differs')
    if sha(inputs/'V25_EVALUATION_ROWS.jsonl')!=PINS['evaluation/rows.jsonl']:raise ValueError('Pinned old generation rows differ')
    if sha(inputs/'V25_CACHE_COMPLETE.json')!=PINS['cache/COMPLETE.json']:raise ValueError('Pinned old cache receipt differs')
    cache_files=read(inputs/'V25_CACHE_COMPLETE.json')['files']
    for n,old_name in (('CAPTION_ONLY.jsonl','caption_only.jsonl'),('PARENT_GRAY.json','gray.json')):
        if sha(inputs/n)!=cache_files[old_name]['sha256']:raise ValueError('Copied native caption/gray source differs')
    previous={key(r):r for r in rows(inputs/'V25_EVALUATION_ROWS.jsonl')}
    cfg=read(root/'config.json')
    if cfg!=config(cfg['profile']):raise ValueError('Fixed diagnostic profile differs')
    selected_entries,donors=selected(index,cfg['profile']);orig={r['example_id']:r for r in index['records']}
    wanted=[r for v in selected_entries.values() for r in v]
    ref=read(root/'REFERENCES.json');refs={r['example_id']:r for r in ref['records']}
    if ref['gray_visual_slice_sha256']!=read(inputs/'PARENT_GRAY.json')['visual_slice_sha256']:
        raise ValueError('Actual gray feature reference differs')
    if list(refs)!=[r['example_id'] for r in wanted] or ref['donors']!=donors or ref['parent_index_sha256']!=PINS['cache/index.json']:
        raise ValueError('Fixed sample/donor selection differs')
    for r in refs.values():
        for k,v in orig[r['example_id']].items():
            if r[k]!=v:raise ValueError('Copied parent reference differs '+k)
        reference_counts(r['gold_ids'])
        if len(r['received_gates'])!=3 or any(len(s)!=9 or any(not 0<=x<=1 for x in s) for s in r['received_gates']):
            raise ValueError('Cached soft gate geometry differs')
    tf=rows(root/'tf_rows.jsonl');gen=rows(root/'generations.jsonl');expected=matrix(wanted,cfg)
    tf_by={key(r):r for r in tf};gen_by={key(r):r for r in gen}
    if len(tf_by)!=len(tf) or len(gen_by)!=len(gen) or set(tf_by)!=expected or set(gen_by)!=expected:
        raise ValueError('Missing/duplicate diagnostic matrix')
    checkpoints=read(root/'CHECKPOINTS.json');notes=[]
    if checkpoints['parent_pins']!=PINS or checkpoints['trainable_parameters'] or checkpoints['BLIP_loaded'] or checkpoints['VAE_loaded'] or checkpoints['source_codec_loaded']:
        raise ValueError('Read-only model load contract differs')
    if set(checkpoints['adapters'])!={'self','visual'}:raise ValueError('Actual two checkpoints required')
    for arm in ('self','visual'):
        if checkpoints['adapters'][arm]['parameters']!=1516290 or len(checkpoints['adapters'][arm]['layers'])!=2:
            raise ValueError('Actual saved adapter geometry differs')
    for r in tf:
        o=refs[r['example_id']];it,tt=[CHANNELS.index(s) for s in REGIMES[r['regime']]]
        if (r['cohort']!=o['cohort'] or r['official_split']!='train' or r['source_text']!=o['text'] or
                r['source_tokens']!=o['source_tokens'] or r['source_text_sha256']!=o['text_sha256'] or
                (r['image_snr_db'],r['text_snr_db'])!=REGIMES[r['regime']] or
                r['memory_sha256']!=o['memory_slice_sha256'][tt] or
                r['gate_sha256']!=array_sha(np.asarray(o['received_gates'][tt:tt+1],dtype='float32'))):
            raise ValueError('Own frozen received data/reference differs')
        donor=donors[r['example_id']] if r['control']=='shuffled' else None
        if r['donor_id']!=donor:raise ValueError('Diagnostic donor differs')
        if r['arm']=='visual':
            expected_aux=(orig[donor]['visual_slice_sha256'][it] if donor else
                ref['gray_visual_slice_sha256'] if r['control']=='gray' else
                array_sha(np.zeros((1,32,768),dtype='float32')) if r['control']=='zero' else o['visual_slice_sha256'][it])
            if r['auxiliary_sha256']!=expected_aux:raise ValueError('Visual perturbation differs')
        if r['arm']=='base' and r['auxiliary_sha256'] is not None:raise ValueError('Base must not use auxiliary')
        losses(r,o);trace(r,o,checkpoints)
        base=tf_by[r['example_id'],r['regime'],'base','matched']
        if r['control']=='disabled' and any(not close(r['nll'][k],base['nll'][k]) for k in ('all','content','eos','bos')):
            raise ValueError('Disabled TF is not base identity')
        g=gen_by[key(r)]
        for k in ('cohort','source_text','source_tokens','source_text_sha256','memory_sha256','auxiliary_sha256','gate_sha256','donor_id'):
            if g[k]!=r[k]:raise ValueError('Greedy/TF inputs differ')
        if g['source_text_at_RX'] or g['source_mask_at_RX'] or g['source_image_at_RX'] or g['training_updates']:
            raise ValueError('Free-generation source leak/update')
        if g['generation_provenance']!=('V25_VERIFIED_DEV_REUSE' if o['cohort']=='dev' else 'V26_NEW_TRAIN_GREEDY'):
            raise ValueError('Generation provenance differs')
        if o['cohort']=='dev' and any(g[k]!=previous[key(g)][k] for k in ('received_caption','generated_token_ids','chunk_generated_token_ids','stopped_by_capacity')):
            raise ValueError('Reused V25 actual generation differs')
        notes.extend(lexical(g))
        if len(g['chunk_generated_token_ids'])!=9:raise ValueError('Greedy chunk protocol differs')
        for seq in g['chunk_generated_token_ids']:body(seq)
        if r['control']=='disabled':
            b=gen_by[r['example_id'],r['regime'],'base','matched']
            if any(g[k]!=b[k] for k in ('received_caption','generated_token_ids','chunk_generated_token_ids','stopped_by_capacity')):
                raise ValueError('Disabled greedy is not base')
    anchors=read(root/'ANCHORS.json')
    expected_anchors={(r['example_id'],arm,ctrl) for r in selected_entries['dev'][:2]
        for arm,ctrl in (('base','matched'),('self','matched'),('visual','matched'),('visual','shuffled'))}
    if (not anchors['passed'] or len(anchors['actual_anchors'])!=8 or not all(r['tokens_equal'] for r in anchors['actual_anchors']) or
            {(r['example_id'],r['arm'],r['control']) for r in anchors['actual_anchors']}!=expected_anchors):
        raise ValueError('Actual saved-parent anchors failed')
    result=read(root/'RESULT.json')
    nt=cfg['train']*len(cfg['regimes'])*len(CONTROLS);nd=cfg['dev']*len(cfg['regimes'])*len(CONTROLS)
    if (result['TF_calls']!=nt+nd or result['new_TRAIN_greedy']!=nt or result['reused_DEV_greedy']!=nd or
        result['training_updates'] or result['optimizer_created'] or result['BART_before']!=result['BART_after'] or not result['adapter_tensors_unchanged']):
        raise ValueError('Actual read-only execution incomplete')
    if result['BART_before']!=checkpoints['BART_before']:raise ValueError('BART tensor receipt differs')
    expected_signatures={arm:checkpoints['adapters'][arm]['tensor_sha256'] for arm in ('self','visual')}
    if result['adapter_before']!=expected_signatures or result['adapter_after']!=expected_signatures:
        raise ValueError('Saved adapter tensor digest changed')
    return dict(profile=cfg['profile'],rows=len(tf),new_greedy=nt,reused_greedy=nd,BLEU_roundoff_only=notes,
        results=summarize(tf,gen,cfg),engineering='VERIFIED',facts='PENDING',semantic_success=False,training_updates=0)

def pipeline(root):
    root=Path(root);state=read(root/'PIPELINE.json')
    from docci_rx_diagnose_v26.core import SCHEMA,FOLDER
    if state['schema']!=SCHEMA or state['status']!='COMPLETE' or state['source_manifest_sha256']!=sha(FOLDER/'MANIFEST.json'):
        raise ValueError('Pipeline incomplete/failed')
    inputs=root/'inputs';verify_files(inputs)
    proof=read(inputs/'PARENTS.json')
    if (proof['pins']!=PINS or not proof['heavy_arrays_and_weights_verified'] or not proof['launch_checked'] or
            not proof['historical_WER_and_counts_exact'] or proof['old_code_and_results_modified']):
        raise ValueError('Actual full parent preflight failed')
    if read(inputs/'PARENT_LAUNCH.json')['status']!='VERIFIED':raise ValueError('Parent launch not finished')
    out={stage:verify(root/stage,inputs) for stage in ('smoke','diagnose')}
    if out['smoke']['profile']!='smoke' or out['diagnose']['profile']!='diagnose':raise ValueError('Wrong stage profile')
    return dict(schema=SCHEMA,stages=out,training_updates=0,semantic_success=False,facts='PENDING')

def facts(root,inputs,destination):
    """Blind PRIMARY review preparation, no automatic facts or scores."""
    from docci_rx_diagnose_v26.core import seed
    root=Path(root);destination=Path(destination);cfg=read(root/'config.json')
    if cfg['profile']!='diagnose':return
    refs=[r for r in read(root/'REFERENCES.json')['records'] if r['cohort']=='dev']
    source=rows(root/'generations.jsonl')
    chosen=[r for r in source if r['cohort']=='dev' and r['regime']=='both0' and
        (r['arm'],r['control']) in (('base','matched'),('self','matched'),('visual','matched'),('visual','shuffled'))]
    # Native caption is an additional image-only baseline, not a transmitted long-text recovery.
    for r in rows(Path(inputs)/'CAPTION_ONLY.jsonl'):
        if r['image_snr_db']==0:
            chosen.append(dict(example_id=r['example_id'],arm='caption',control='matched',received_caption=r['output_text']))
    chosen.sort(key=lambda r:seed(42,'V26-blind-review',r['example_id'],r['arm'],r['control']))
    key_rows=[]
    with (destination/'FACT_CANDIDATES.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['case_id','reference_id','candidate_text','retained_fact_ids','contradicted_fact_ids','unsupported_added_facts','reviewer','status'])
        for i,r in enumerate(chosen):
            case='case_{:04d}'.format(i+1)
            writer.writerow([case,r['example_id'],r['received_caption'],'','','','','PENDING'])
            key_rows.append(dict(case_id=case,example_id=r['example_id'],arm=r['arm'],control=r['control']))
    atomic(destination/'FACT_REFERENCES.json',dict(status='PENDING',automatic_fact_extraction=False,
        items=[dict(reference_id=r['example_id'],source_text=r['text'],facts=[],
            required_categories=['objects','attributes','colors','counts','relations'],reviewer='') for r in refs]))
    atomic(destination/'FACT_KEY_PRIVATE.json',dict(status='PENDING',items=key_rows,
        note='Keep this method key hidden from candidate raters; random order does not guarantee complete blinding.'))

def render(result):
    e=html.escape
    text=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>V26 接收诊断</title>',
        '<style>body{font:16px/1.65 system-ui;max-width:1400px;margin:25px auto;padding:0 20px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:8px}pre{white-space:pre-wrap}</style>',
        '<h1>V26：只诊断，不训练</h1><p>TF loss使用真实前缀，只供诊断，不是实际自由恢复。DEV自由生成复用V25；TRAIN为新测。实际视觉残差在TF有效位置上被动测量，不是整个贪心路径的残差。无自动语义成功/唯一原因判定。</p>',
        '<table><tr><th>cohort</th><th>条件</th><th>方法</th><th>内容TF CE（token加权）</th><th>自由生成WER</th><th>内容位置残差比均值</th></tr>']
    for r in result['results']['groups']:
        def number(x):return '未测/不适用' if x is None else '{:.6f}'.format(x)
        text.append('<tr>'+''.join('<td>'+e(str(v))+'</td>' for v in (r['cohort'],r['regime'],r['arm']+'/'+r['control'],number(r['token_weighted_content_CE']),number(r['mean_greedy_WER']),number(r['mean_content_residual_ratio'])))+'</tr>')
    text.append('</table><details><summary>配对区间与完整边界</summary><pre>'+e(json.dumps(result,ensure_ascii=False,indent=2))+'</pre></details></html>')
    return '\n'.join(text)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--inputs');p.add_argument('--pipeline',action='store_true');a=p.parse_args()
    source_check()
    if not a.pipeline and not a.inputs:raise ValueError('--inputs required for stage verification')
    print(json.dumps(pipeline(a.root) if a.pipeline else verify(a.root,a.inputs),ensure_ascii=False,indent=2))
