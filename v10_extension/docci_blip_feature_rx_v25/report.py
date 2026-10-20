"""Independent CPU verification; light records never imply rechecked heavy arrays."""
import argparse
import html
import json
from pathlib import Path
import statistics
import sys
import numpy as np
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT),str(EXT.parent)]
from docci_blip_feature_rx_v25.core import (read,rows,sha,array_sha,digest,config,stream,REGIMES,
    CHANNELS,CONTROLS,regimes,expected_keys,key,verify_files,source_check,donor_map,local)

def parameter_count(dim=768,width=256,layers=2):
    projection = 2*dim+dim*width+width
    return projection+layers*(projection+4*width*width+4*width+width*dim+dim+1)

def cache(root,arrays=True):
    from docci_caption_calibration_v24.core import token_audit,text_audit
    root = Path(root); verify_files(root,'cache',arrays)
    c = read(root/'config.json')
    if c!=config(c['profile']): raise ValueError('Cache config differs')
    entries = read(root/'index.json')['records']
    if len(entries)!=c['train']+c['dev'] or len({r['example_id'] for r in entries})!=len(entries):
        raise ValueError('Cache cohort matrix differs')
    for group in ('train','dev'):
        rs = [r for r in entries if r['cohort']==group]
        if len(rs)!=c[group]: raise ValueError('Cohort size differs')
        if any(r['official_split']!='train' or r['input_truncated'] or r['source_mask_at_RX'] for r in rs):
            raise ValueError('TRAIN/dev identity or leakage flags differ')
    for field in ('example_id','text_sha256','source_image_sha256'):
        if {r[field] for r in entries if r['cohort']=='train'} & {r[field] for r in entries if r['cohort']=='dev'}:
            raise ValueError('TRAIN/dev overlap')
    for r in entries:
        import hashlib
        if hashlib.sha256(r['text'].encode()).hexdigest()!=r['text_sha256']: raise ValueError('Text hash differs')
        if not 2<=r['source_tokens']<=512: raise ValueError('Full source capacity differs')
        for trace in r['visual_traces']:
            if (trace['shape']!=[1,32,768] or trace['input_has_text'] or trace['source_image_at_RX'] or
                    not trace['finite'] or trace['processed_pixels_shape']!=[1,3,224,224]):
                raise ValueError('Actual native Q-Former trace differs')
        if r['cohort']=='dev':
            for value in r['image_files'].values():
                if sha(local(root,value['file']))!=value['sha256']: raise ValueError('Actual RX PNG differs')
        if arrays:
            path = local(root,r['file'])
            if sha(path)!=r['sha256']: raise ValueError('Cache array bytes differ')
            with np.load(path,allow_pickle=False) as z:
                shapes = dict(ids=(9,64),memory=(3,9,32,768),gate=(3,9),visual=(3,32,768),received_rgb=(3,512,512,3))
                if set(z.files)!=set(shapes): raise ValueError('Array schema differs')
                for k,shape in shapes.items():
                    x = z[k]
                    if x.shape!=shape or not np.isfinite(x).all() or array_sha(x)!=r['array_hashes'][k]:
                        raise ValueError('Invalid array '+k)
                if z['ids'].dtype!=np.int64 or z['received_rgb'].dtype!=np.uint8: raise ValueError('Array dtype differs')
                if any(z[k].dtype!=np.float32 for k in ('memory','gate','visual')): raise ValueError('Expected lossless FP32 cache')
                for j in range(3):
                    if (array_sha(z['memory'][j:j+1])!=r['memory_slice_sha256'][j] or
                            array_sha(z['visual'][j:j+1])!=r['visual_slice_sha256'][j]): raise ValueError('Per-condition slice hash differs')
                if ((z['gate']<0)|(z['gate']>1)).any(): raise ValueError('Invalid content gate')
                count = 0
                for slot in z['ids']:
                    if slot[0]!=0 or list(slot).count(2)!=1: raise ValueError('Invalid public chunk source labels')
                    end = list(slot).index(2)
                    if (slot[1:end]<=2).any() or (slot[end+1:]!=1).any(): raise ValueError('Invalid full source framing')
                    count += end-1
                if count+2!=r['source_tokens']: raise ValueError('Source text was truncated')
    cap = rows(root/'caption_only.jsonl'); dev = [r for r in entries if r['cohort']=='dev']
    if len(cap)!=3*len(dev) or {(r['example_id'],r['image_snr_db']) for r in cap}!={(r['example_id'],s) for r in dev for s in CHANNELS}:
        raise ValueError('Caption-only control matrix differs')
    for r in cap:
        origin = next(e for e in dev if e['example_id']==r['example_id'])
        ix = CHANNELS.index(r['image_snr_db']); t = r['telemetry']
        audit = token_audit(t['returned_token_ids'],t['new_token_count'],t['eos_token_ids'],64)
        if any(t[k]!=v for k,v in audit.items()) or r['text_audit']!=text_audit(r['output_text']):
            raise ValueError('Actual caption token/EOS record differs')
        if (r['image_sha256']!=origin['image_files'][str(r['image_snr_db'])]['sha256'] or
                t['processed_pixels']['float32_sha256']!=origin['visual_traces'][ix]['processed_pixels_float32_sha256']):
            raise ValueError('Caption and feature extraction did not use identical received pixels')
    frozen = read(root/'frozen_audit.json')
    if frozen['BLIP_before']!=frozen['BLIP_after'] or frozen['pretrained_parameters_trained']!=0:
        raise ValueError('Frozen pretrained audit differs')
    if arrays:
        with np.load(root/'gray.npz',allow_pickle=False) as z:
            if z['visual'].shape!=(32,768) or not np.isfinite(z['visual']).all(): raise ValueError('Invalid actual gray features')
            if array_sha(z['visual'][None])!=read(root/'gray.json')['visual_slice_sha256']:
                raise ValueError('Gray feature hash differs')
    return dict(stage='cache',profile=c['profile'],examples=len(entries),arrays_verified=arrays)

def training(root,cache_root,arrays=True):
    root = Path(root); verify_files(root,'train',arrays)
    c = read(Path(cache_root)/'config.json'); identity = read(root/'identity.json'); history = rows(root/'history.jsonl')
    if identity['config']!=c or identity['cache_index_sha256']!=sha(Path(cache_root)/'index.json'):
        raise ValueError('Training configuration/cache differs')
    entries = [r for r in read(Path(cache_root)/'index.json')['records'] if r['cohort']=='train']
    if identity['train_ids']!=[r['example_id'] for r in entries] or identity['trainable_parameters']!=parameter_count():
        raise ValueError('Train data/parameter contract differs')
    if (not all(identity[k] for k in ('BART_frozen','BLIP_frozen','TX_RX_frozen')) or
            identity['source_masks_at_RX'] or identity['matching_gate']): raise ValueError('Frozen/receiver contract differs')
    plan = list(stream(c))
    if len(history)!=len(plan): raise ValueError('Training updates incomplete')
    seen = set()
    for r,(step,epoch,ix,rs) in zip(history,plan):
        expected = [entries[i]['example_id'] for i in ix]; seen.update(expected)
        if (r['update']!=step+1 or r['epoch']!=epoch+1 or r['ids']!=expected or r['regimes']!=rs or
                r['valid_target_tokens']!=sum(entries[i]['source_tokens']+16 for i in ix) or
                not np.isfinite([r['CE'],r['gradient_norm'],r['lr']]).all()):
            raise ValueError('Actual optimizer stream differs')
    learned = read(root/'learning_audit.json')
    if not learned['passed'] or not learned['frozen_unchanged'] or learned['BART_before']!=learned['BART_after']:
        raise ValueError('Learning/frozen audit failed')
    for group in learned['groups'].values():
        if not group['parameters'] or not group['changed'] or not group['nonzero_gradient']:
            raise ValueError('Real trainable group did not learn')
    result = read(root/'TRAIN_RESULT.json')
    if (result['updates']!=c['updates'] or result['unique_training_ids']!=len(seen) or
            not result['actual_disk_roundtrip'] or not result['fixed_final']): raise ValueError('Final checkpoint policy differs')
    if arrays and sha(root/'last.pth')!=result['checkpoint_sha256']: raise ValueError('Saved weight bytes differ')
    return dict(stage='train',arm=identity['arm'],updates=len(history),arrays_verified=arrays)

def evaluate(root,cache_root,arrays=True):
    from docci_text_continue_v11.metrics import metrics,body
    from docci_caption_calibration_v24.core import text_audit
    root = Path(root); verify_files(root,'evaluation',arrays)
    c = read(Path(cache_root)/'config.json'); dev = [r for r in read(Path(cache_root)/'index.json')['records'] if r['cohort']=='dev']
    source = {r['example_id']:r for r in dev}; data = rows(root/'rows.jsonl'); seen = {key(r):r for r in data}
    if len(seen)!=len(data) or set(seen)!=expected_keys(c,dev): raise ValueError('Evaluation controls missing/duplicate')
    ids = [r['example_id'] for r in dev]; donors = donor_map(ids)
    for r in data:
        origin = source[r['example_id']]
        if r['source_text']!=origin['text'] or r['source_text_sha256']!=origin['text_sha256'] or r['source_tokens']!=origin['source_tokens']:
            raise ValueError('Evaluation source reference differs')
        if (r['source_text_at_RX'] or r['source_mask_at_RX'] or r['source_image_at_RX'] or
                not r['same_text_observation_all_controls'] or r['training_updates']!=0):
            raise ValueError('RX leakage/update flags differ')
        if (r['image_snr_db'],r['text_snr_db'])!=REGIMES[r['regime']]: raise ValueError('Channel regime differs')
        minimum = 4128 if r['arm']=='caption' else 9248 if r['arm'] in ('base','self') or r['control'] in ('gray','zero','disabled') else 13376
        if r['shared_trial_allocated_symbols']!=13376 or r['minimum_input_symbols']!=minimum:
            raise ValueError('Charged/used communication budget differs')
        expected_donor = donors[r['example_id']] if r['control']=='shuffled' else None
        if r['donor_id']!=expected_donor: raise ValueError('Within-cohort donor differs')
        it,tt = [CHANNELS.index(x) for x in REGIMES[r['regime']]]
        if (r['image_stats']!=origin['image_stats'][it] or r['text_stats']!=origin['text_stats'][tt] or
                r['memory_sha256']!=origin['memory_slice_sha256'][tt]):
            raise ValueError('Own observation/physical condition differs')
        if r['arm']=='visual':
            expected_aux = (source[expected_donor]['visual_slice_sha256'][it] if r['control']=='shuffled' else
                read(Path(cache_root)/'gray.json')['visual_slice_sha256'] if r['control']=='gray' else
                array_sha(np.zeros((1,32,768),np.float32)) if r['control']=='zero' else origin['visual_slice_sha256'][it])
            if r['auxiliary_sha256']!=expected_aux: raise ValueError('Auxiliary intervention content differs')
        score = metrics(origin['text'],r['received_caption'])
        if any(r[k]!=v for k,v in score.items()) or r['output_audit']!=text_audit(r['received_caption']):
            raise ValueError('Independent lexical metrics differ')
        base = seen[(r['example_id'],r['regime'],'base','matched')]
        if r['memory_sha256']!=base['memory_sha256']: raise ValueError('Own text memory changed across controls')
        if r['arm']!='caption':
            if len(r['chunk_generated_token_ids'])!=9: raise ValueError('Public chunk geometry differs')
            for seq in r['chunk_generated_token_ids']: body(seq)
        if r['control']=='disabled' and any(r[k]!=base[k] for k in ('received_caption','chunk_generated_token_ids','generated_token_ids','stopped_by_capacity')):
            raise ValueError('Disabled branch is not original receiver')
    stats = read(root/'EVAL_RESULT.json')
    if stats['rows']!=len(data) or stats['BART_before']!=stats['BART_after'] or not stats['frozen_unchanged']:
        raise ValueError('Evaluation completion differs')
    return dict(stage='evaluation',rows=len(data),profile=c['profile'],arrays_verified=arrays)

def bootstrap(delta,seed=250042):
    x = np.asarray(delta,float)
    if len(x)<3: return None
    rng = np.random.Generator(np.random.PCG64(seed))
    means = x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return [float(v) for v in np.quantile(means,[.025,.975])]

def summarize(data,c):
    from docci_text_continue_v11.metrics import body
    def empty_activation(r):
        if r['arm']=='caption': return None
        n = r['source_tokens']-2
        return sum(bool(body(seq)[0]) for j,seq in enumerate(r['chunk_generated_token_ids']) if j*62>=n)
    groups = {}
    for r in data: groups.setdefault((r['regime'],r['arm'],r['control']),[]).append(r)
    summary = []
    for (regime,arm,control),rs in groups.items():
        v = [r['normalized_wer'] for r in rs]
        summary.append(dict(regime=regime,arm=arm,control=control,n=len(rs),
            mean_WER=statistics.mean(v),SD_WER=statistics.stdev(v) if len(v)>1 else None,
            corpus_WER=sum(r['normalized_word_errors'] for r in rs)/max(1,sum(r['normalized_source_words'] for r in rs)),
            mean_adjacent_repeat=statistics.mean(r['output_audit']['adjacent_repeat_fraction'] for r in rs),
            empty_outputs=sum(r['output_audit']['empty'] for r in rs),
            empty_slot_activations=None if arm=='caption' else sum(empty_activation(r) for r in rs)))
    by = {key(r):r for r in data}; ids = sorted({r['example_id'] for r in data})
    comparisons = {}
    for arm,control,label in [('self','matched','vs_self'),('visual','shuffled','vs_shuffled'),('base','matched','vs_frozen')]:
        delta = [by[e,'both0',arm,control]['normalized_wer']-by[e,'both0','visual','matched']['normalized_wer'] for e in ids]
        comparisons[label] = dict(mean_WER_reduction=statistics.mean(delta),paired95CI=bootstrap(delta))
    numeric = c['profile']=='pilot'
    for name,threshold in [('vs_self',.02),('vs_shuffled',.01),('vs_frozen',0.)]:
        v = comparisons[name]; numeric = numeric and v['mean_WER_reduction']>=threshold and v['paired95CI'] is not None and v['paired95CI'][0]>0
    matched = groups['both0','visual','matched']; self_rows = groups['both0','self','matched']
    numeric = numeric and statistics.mean(r['output_audit']['adjacent_repeat_fraction'] for r in matched)<=statistics.mean(r['output_audit']['adjacent_repeat_fraction'] for r in self_rows)
    numeric = numeric and sum(empty_activation(r) for r in matched)<=sum(empty_activation(r) for r in self_rows)
    return dict(groups=summary,primary='both0',comparisons=comparisons,numeric_screen_passed=bool(numeric),
        smoke_is_NOT_performance=c['profile']=='smoke',facts='PENDING',semantic_success=False,
        caption_vs_joint_factual_gain='NOT_MEASURED',independent_final_test=False,
        CI_scope='single trained seed, reused dev, pointwise image-paired; not training variability or corrected significance')

def verify_pipeline(root,arrays=True):
    root = Path(root); identity = read(root/'PIPELINE.json')
    if (identity['schema']!='docci-blip-feature-rx-v25' or identity['status']!='COMPLETE' or
            identity['source_manifest_sha256']!=sha(Path(__file__).with_name('MANIFEST.json'))):
        raise ValueError('Pipeline incomplete/failed')
    output = dict(cache=cache(root/'cache',arrays))
    for arm in ('self','visual'): output['train_'+arm] = training(root/('train_'+arm),root/'cache',arrays)
    a,b = [read(root/('train_'+arm)/'identity.json') for arm in ('self','visual')]
    if a['initial_parameters']!=b['initial_parameters'] or a['trainable_parameters']!=b['trainable_parameters']:
        raise ValueError('Equal-size/common initialization control differs')
    if rows(root/'train_self/history.jsonl') and [r['ids'] for r in rows(root/'train_self/history.jsonl')]!=[r['ids'] for r in rows(root/'train_visual/history.jsonl')]:
        raise ValueError('Training pair order differs')
    output['evaluation'] = evaluate(root/'evaluation',root/'cache',arrays)
    output['results'] = summarize(rows(root/'evaluation/rows.jsonl'),read(root/'cache/config.json'))
    output['engineering'] = 'VERIFIED'; output['heavy_arrays_and_weights_reverified_here'] = arrays
    return output

def render(root,result,prefix='data/'):
    root = Path(root); e = html.escape
    c = read(root/'cache/config.json'); dev = [r for r in read(root/'cache/index.json')['records'] if r['cohort']=='dev']
    data = {key(r):r for r in rows(root/'evaluation/rows.jsonl')}
    text = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>V25 BLIP特征辅助文本</title>',
        '<style>body{font:16px/1.6 system-ui;max-width:1400px;margin:24px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:10px;vertical-align:top}img{max-width:260px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.warn{color:#a33}</style>',
        '<h1>V25：BLIP-2视觉特征辅助文字恢复</h1><p class="warn">PRIVATE · 不代表双向互促成功。主比较 both0 为图文各0dB AWGN；text0为图像clean、文字0dB的异质条件诊断。clean不是原图或20dB；−5为冻结codec/适配器训练外。</p>',
        '<p>只有小适配器训练；BART/BLIP2/图文codec冻结。图文完整预算13376复符号。Caption-only是描述，不是原文恢复；不同长度WER不能单独裁决语义好坏。</p>',
        '<details><summary>完整聚合与证据边界</summary><pre>'+e(json.dumps(result,ensure_ascii=False,indent=2))+'</pre></details>']
    for item in dev[:4]:
        text.append('<h2>'+e(item['example_id'])+'</h2><p>原文仅评价参照：</p><pre>'+e(item['text'])+'</pre>')
        for regime in regimes(c):
            source = data[item['example_id'],regime,'visual','matched']; snr = REGIMES[regime][0]
            image_file = item['image_files'][str(snr)]['file']
            text.append('<h3>'+e(regime)+'</h3><img src="'+e(prefix+'cache/'+image_file,quote=True)+'"><table><tr><th>方法/对照</th><th>实际输出全文</th><th>WER，仅词面指标</th></tr>')
            for arm,control in [('base','matched'),('caption','matched'),('self','matched'),('visual','matched'),('visual','shuffled'),('visual','gray'),('visual','zero')]:
                r = data[item['example_id'],regime,arm,control]
                text.append('<tr><td>'+e(arm+'/'+control)+'</td><td>'+e(r['received_caption'])+'</td><td>{:.2%}</td></tr>'.format(r['normalized_wer']))
            text.append('</table>')
    return '\n'.join(text+['</html>'])

def main():
    p = argparse.ArgumentParser(); p.add_argument('stage',choices=('cache','train','evaluation','pipeline'))
    p.add_argument('root'); p.add_argument('--cache'); p.add_argument('--light',action='store_true'); a = p.parse_args()
    source_check()
    if a.stage=='cache': result = cache(a.root,not a.light)
    elif a.stage=='train': result = training(a.root,a.cache,not a.light)
    elif a.stage=='evaluation': result = evaluate(a.root,a.cache,not a.light)
    else: result = verify_pipeline(a.root,not a.light)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
