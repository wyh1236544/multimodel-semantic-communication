"""Immutable V25 full-artifact checks; WER exact, known BLEU roundoff logged."""
import math
from pathlib import Path
import numpy as np
from docci_rx_diagnose_v26.core import read,rows,sha,PINS,atomic

def lexical(row):
    from docci_text_continue_v11.metrics import metrics
    from docci_caption_calibration_v24.core import text_audit
    notes=[]
    for k,value in metrics(row['source_text'],row['received_caption']).items():
        if row[k]!=value:
            if (k=='bleu2' and isinstance(value,float) and math.isfinite(row[k]) and
                    math.isclose(row[k],value,rel_tol=1e-14,abs_tol=1e-16)):
                notes.append(dict(key=k,recorded=row[k],recomputed=value))
            else:raise ValueError('Historical lexical metric differs '+k)
    if row['output_audit']!=text_audit(row['received_caption']):raise ValueError('Historical output audit differs')
    return notes

def evaluate(root,arrays):
    from docci_blip_feature_rx_v25 import report as v
    from docci_text_continue_v11.metrics import body
    cache=root/'cache';folder=root/'evaluation';v.verify_files(folder,'evaluation',arrays)
    entries=[r for r in read(cache/'index.json')['records'] if r['cohort']=='dev'];origin={r['example_id']:r for r in entries}
    data=rows(folder/'rows.jsonl');by={v.key(r):r for r in data};cfg=read(cache/'config.json')
    if len(data)!=len(by) or set(by)!=v.expected_keys(cfg,entries):raise ValueError('Historical evaluation matrix differs')
    donors=v.donor_map([r['example_id'] for r in entries]);notes=[]
    for r in data:
        o=origin[r['example_id']]
        if (r['source_text']!=o['text'] or r['source_text_sha256']!=o['text_sha256'] or r['source_tokens']!=o['source_tokens']):
            raise ValueError('Historical source reference differs')
        if r['source_text_at_RX'] or r['source_image_at_RX'] or r['source_mask_at_RX'] or r['training_updates']!=0 or not r['same_text_observation_all_controls']:
            raise ValueError('Historical receiver contract differs')
        if (r['image_snr_db'],r['text_snr_db'])!=v.REGIMES[r['regime']]:raise ValueError('Historical channel differs')
        used=4128 if r['arm']=='caption' else 9248 if r['arm'] in ('base','self') or r['control'] in ('gray','zero','disabled') else 13376
        if r['shared_trial_allocated_symbols']!=13376 or r['minimum_input_symbols']!=used:raise ValueError('Historical resource differs')
        donor=donors[r['example_id']] if r['control']=='shuffled' else None
        if r['donor_id']!=donor:raise ValueError('Historical donor differs')
        it,tt=[v.CHANNELS.index(s) for s in v.REGIMES[r['regime']]]
        if r['memory_sha256']!=o['memory_slice_sha256'][tt] or r['image_stats']!=o['image_stats'][it] or r['text_stats']!=o['text_stats'][tt]:
            raise ValueError('Historical received observation differs')
        if r['arm']=='visual':
            expected=(origin[donor]['visual_slice_sha256'][it] if donor else
                read(cache/'gray.json')['visual_slice_sha256'] if r['control']=='gray' else
                v.array_sha(np.zeros((1,32,768),np.float32)) if r['control']=='zero' else o['visual_slice_sha256'][it])
            if r['auxiliary_sha256']!=expected:raise ValueError('Historical visual intervention differs')
        notes.extend(dict(example_id=r['example_id'],regime=r['regime'],arm=r['arm'],control=r['control'],**n) for n in lexical(r))
        base=by[r['example_id'],r['regime'],'base','matched']
        if r['memory_sha256']!=base['memory_sha256']:raise ValueError('Historical own text changed')
        if r['arm']!='caption':
            if len(r['chunk_generated_token_ids'])!=9:raise ValueError('Historical public geometry differs')
            for seq in r['chunk_generated_token_ids']:body(seq)
        if r['control']=='disabled' and any(r[k]!=base[k] for k in ('received_caption','chunk_generated_token_ids','generated_token_ids','stopped_by_capacity')):
            raise ValueError('Historical disabled output differs')
    stats=read(folder/'EVAL_RESULT.json')
    if stats['rows']!=len(data) or stats['BART_before']!=stats['BART_after'] or not stats['frozen_unchanged']:
        raise ValueError('Historical evaluation not frozen/complete')
    return notes

def verify(root,arrays=True,require_launch=True):
    from docci_blip_feature_rx_v25 import report as v
    from docci_blip_feature_rx_v25.core import source_check
    root=Path(root);source_check()
    for n,h in PINS.items():
        if not arrays and n.endswith('.pth'):continue
        if sha(root/n)!=h:raise ValueError('Pinned completed V25 pilot differs '+n)
    p=read(root/'PIPELINE.json')
    if p['schema']!='docci-blip-feature-rx-v25' or p['status']!='COMPLETE' or p['profile']!='pilot':
        raise ValueError('Completed V25 pilot required')
    if require_launch and read(root/'LAUNCH_FINISHED.json')['status']!='VERIFIED':raise ValueError('V25 launch not finalized')
    v.cache(root/'cache',arrays)
    for arm in ('self','visual'):v.training(root/('train_'+arm),root/'cache',arrays)
    a,b=[read(root/('train_'+arm)/'identity.json') for arm in ('self','visual')]
    if a['initial_parameters']!=b['initial_parameters'] or a['trainable_parameters']!=b['trainable_parameters']:
        raise ValueError('Historical matched-size initialization differs')
    notes=evaluate(root,arrays)
    return dict(schema='v26-parent-verification',profile='pilot',pins=PINS,
        heavy_arrays_and_weights_verified=arrays,launch_checked=require_launch,
        historical_WER_and_counts_exact=True,BLEU_rounding_only=notes,
        old_code_and_results_modified=False,training_updates=0)

def after(root):
    for n,h in PINS.items():
        if sha(Path(root)/n)!=h:raise ValueError('Read-only parent changed '+n)
    root=Path(root)
    for entry in read(root/'cache/index.json')['records']:
        if sha(root/'cache'/entry['file'])!=entry['sha256']:raise ValueError('Read-only parent array changed')
