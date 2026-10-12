"""Independent matrix, source, actual-text, learning and archive verifier."""
import argparse
from collections import defaultdict
import html
import json
import math
from pathlib import Path
import sys
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from bart_coco_modal_v2 import VERSION,MODES
from bart_coco_modal_v2.core import digest,file_under,geometry,metrics,summarize,quality_gate

MARKERS=dict(complete='COMPLETE.json',partial_complete='PARTIAL_COMPLETE.json',smoke_complete='SMOKE_COMPLETE.json')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    with Path(path).open(encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]


def text_check(row):
    calculated=metrics(row['source_text'],row['covered_text'],row['received_caption'])
    for k,v in calculated.items():
        if isinstance(v,(float,int)) and not isinstance(v,bool):
            if not math.isfinite(row[k]) or abs(row[k]-v)>1e-9: raise ValueError('Actual received text/metric differs: '+k)
        elif row[k]!=v: raise ValueError('Actual text flag differs: '+k)


def controls(path):
    groups=defaultdict(list); seen=set()
    for row in lines(path):
        text_check(row); key=(row['control'],row['image_id'])
        if key in seen: raise ValueError('Duplicate controls')
        seen.add(key); groups[row['control']].append(row)
    if set(groups)!={'matched','zero','shuffled'}: raise ValueError('Missing controls')
    ids=[{r['image_id'] for r in groups[k]} for k in ('matched','zero','shuffled')]
    if len(ids[0])<2 or not ids[0]==ids[1]==ids[2]: raise ValueError('Unpaired controls')
    return {k:summarize(v) for k,v in groups.items()}


def check_gate(root,label,cfg,diagnostic):
    calculated=quality_gate(controls(root/(label+'_controls.jsonl')),cfg,diagnostic)
    saved=read(root/(label+'_gate.json'))
    for k in ('passed','matched_wer','margins','quality_limit','memory_margin_limit','groups'):
        if saved[k]!=calculated[k]: raise ValueError('Stored gate differs from actual text: '+label)
    return calculated


def matrix_check(rows,design,statuses):
    cfg=design['config']; expected=set()
    for mode,state in statuses.items():
        view=design['caption_views'][mode]; train=view['train']; val=view['validation']
        if set(train['image_ids'])&set(val['image_ids']): raise ValueError('Source split overlap')
        for part,receipt in view.items():
            if receipt['original_split_not_rewritten'] is not True or receipt['mode']!=mode: raise ValueError('Source view changed')
            if receipt['text_encoder_executed']!=(mode!='image_only') or receipt['image_file_opened']!=(mode!='text_only'): raise ValueError('Opposite source entered encoder')
            if len(receipt['image_ids'])!=(cfg['train_samples'] if part=='train' else cfg['val_samples']): raise ValueError('Source subset silently reduced')
        if state['evaluated']:
            expected.update((mode,cid,i) for cid in ['clean']+[v['id'] for v in design['channel_config']['conditions']] for i in val['image_ids'])
    actual={(r['mode'],r['condition'],r['image_id']) for r in rows}
    if actual!=expected or len(rows)!=len(expected): raise ValueError('Missing/duplicate/unplanned evaluation')
    for r in rows:
        mode=r['mode']; view=design['caption_views'][mode]['validation']; iid=str(r['image_id'])
        if r['source_split']!='validation' or r['source_sha256']!=view['source_sha256'][iid] or r['annotation_id']!=view['annotation_ids'][iid]: raise ValueError('Wrong source/annotation identity')
        if r['geometry']!=geometry(mode): raise ValueError('Wrong actual compact layout')
        if r['text_used_for_source']!=(mode!='image_only') or r['image_used_for_source']!=(mode!='text_only'): raise ValueError('Opposite source leakage')
        if mode=='image_only':
            if r['received_caption'] is not None or 'covered_wer' in r or r['source_utf8_bytes'] is not None: raise ValueError('Image-only claims a text transmission')
        else:
            text_check(r)
            if not r['stats']['receiver_available'] and r['received_caption']!='': raise ValueError('Erased frame text not empty')
        if mode=='text_only':
            if r['psnr'] is not None or r['ssim'] is not None or 'image_file' in r: raise ValueError('Text-only claims image reconstruction')
        elif not all(math.isfinite(r[k]) for k in ('psnr','ssim')): raise ValueError('Nonfinite image task metric')
        stats=r['stats']
        if stats.get('pre_fec_ber') is not None or stats.get('post_fec_ber') is not None: raise ValueError('Fabricated analog bit error rate')
        if r['condition']=='clean':
            if not stats['clean_control_no_physical_transmission'] or stats['scheduled_seconds'] is not None: raise ValueError('Clean is not a physical frame')
        else:
            res=design['resources'][mode]
            if stats['clean_control_no_physical_transmission'] or stats['analog_repeats']!=1 or stats['payload_complex_symbols']!=geometry(mode)['complex_symbols']: raise ValueError('Wrong physical mode/repeat')
            if stats['used_samples']!=res['scheduled_samples'] or abs(stats['scheduled_seconds']-res['scheduled_seconds'])>1e-15: raise ValueError('Physical resource accounting differs')
        for key in ('image_file','source_image_file','constellation_file'):
            if key in r and not file_under(design['_root'],r[key]).is_file(): raise ValueError('Missing visual evidence')


def inspect(root):
    root=Path(root); identity=read(root/'run_identity.json'); design=read(root/'design.json'); states=read(root/'mode_status.json')
    status=identity['status']; cfg=design['config']
    if identity['version']!=VERSION or design['version']!=VERSION or status not in MARKERS: raise ValueError('Incomplete/wrong run')
    split_sha=identity.get('split_sha256','')
    if not isinstance(split_sha,str) or len(split_sha)!=64 or any(v not in '0123456789abcdef' for v in split_sha): raise ValueError('Missing exact image split hash')
    if set(states)!=set(design['modes']) or not set(states)<=set(MODES): raise ValueError('Missing requested mode')
    for mode,state in states.items():
        allowed=('evaluated','smoke_evaluated','overfit_stop','clean_validation_stop')
        if state['status'] not in allowed or state['evaluated']!=(state['status'] in ('evaluated','smoke_evaluated')): raise ValueError('Mode state contradicts execution')
        if mode=='image_only' and not state['evaluated']: raise ValueError('Image execution cannot be a text quality stop')
        if (state['status']=='smoke_evaluated')!=bool(design['smoke'] and state['evaluated']): raise ValueError('Mode smoke provenance mismatch')
        if state['evaluated'] and state.get('rows')!=cfg['val_samples']*(len(design['channel_config']['conditions'])+1): raise ValueError('Mode row count differs')
    if not design['clean_training_only'] or not design['no_test_source_used']: raise ValueError('Unexpected training/data scope')
    if status=='smoke_complete' and not design['smoke']: raise ValueError('False smoke status')
    if status!='smoke_complete' and design['smoke']: raise ValueError('Smoke claims main result')
    if status=='complete' and not all(v['evaluated'] for v in states.values()): raise ValueError('False complete status')
    if status=='partial_complete' and all(v['evaluated'] for v in states.values()): raise ValueError('False partial status')
    if not all(read(root/'frozen_verification.json').get(k) is True for k in ('passed','BART_reference_VAE_CLIP_unchanged','no_test_source_used')): raise ValueError('Missing frozen verification')
    for mode,state in states.items():
        folder=root/mode; stages=[]
        for view in design['caption_views'][mode].values():
            if view.get('base_image_split_sha256')!=identity.get('split_sha256'): raise ValueError('Caption view changed original image split')
        if mode!='image_only':
            native=lines(folder/'pretrained_uncompressed.jsonl')
            if len(native)!=cfg['diagnostic_samples'] or any(r['source_split']!='train' for r in native): raise ValueError('Wrong native diagnostic')
            for r in native: text_check(r)
            check_gate(folder,'private_T32',cfg,True)
            gate=check_gate(folder,'mode_overfit',cfg,True)
            stages+=['private_T32_diagnostic','mode_diagnostic']
            if state['status']=='overfit_stop':
                if gate['passed'] or state['evaluated'] or (folder/'clean').exists(): raise ValueError('Wrong overfit stop')
            else:
                if not gate['passed'] and not design['smoke']: raise ValueError('Overfit quality bypass')
                val=check_gate(folder,'clean_validation',cfg,False)
                if state['status']=='clean_validation_stop':
                    if val['passed'] or state['evaluated']: raise ValueError('Wrong validation stop')
                elif not val['passed'] and not design['smoke']: raise ValueError('Validation quality bypass')
        if state['status']!='overfit_stop': stages+=['clean']
        for name in stages:
            stage=folder/name; audit=read(stage/'learning_audit.json'); forward=read(stage/'forward_consistency.json'); end=read(stage/'TRAIN_COMPLETE.json')
            if not audit['passed'] or not audit['frozen_unchanged'] or not forward['passed']: raise ValueError('Learning/frozen/forward gate failed')
            if not audit['groups'] or any(not all(v.get(k) for k in ('parameters','nonzero_gradient','changed')) for v in audit['groups'].values()): raise ValueError('Missing actual group learning')
            if any(not math.isfinite(v) or v>cfg['forward_tolerance'] for v in forward['errors'].values()): raise ValueError('Forward audit inconsistent')
            history=lines(stage/'history.jsonl'); count=cfg['train_samples']*cfg['clean_epochs'] if name=='clean' else cfg['diagnostic_updates']
            if len(history)!=count or [r['step'] for r in history]!=list(range(1,count+1)) or end['steps']!=count or end['optimizer_updates']!=math.ceil(count/cfg['grad_accum']): raise ValueError('Fixed step/update budget differs')
    rows=lines(root/'samples.jsonl'); design['_root']=str(root); matrix_check(rows,design,states); del design['_root']
    controls_rows=lines(root/'common_controls.jsonl')
    for r in controls_rows:
        if r['mode'] not in states or not states[r['mode']]['evaluated'] or r['diagnostic_fixed_received_tensor_not_new_physical_transmission'] is not True: raise ValueError('Mislabelled receive diagnostic')
        if r['received_caption'] is not None: text_check(r)
    return identity,design,states,rows


def report(root):
    from v10.io import atomic_json
    root=Path(root); identity,design,states,rows=inspect(root); groups=defaultdict(list)
    for row in rows: groups[(row['mode'],row['condition'])].append(row)
    summaries={}
    for (mode,cid),rr in groups.items():
        value=dict(entries=len(rr),receiver_available_fraction=sum(r['stats']['receiver_available'] for r in rr)/len(rr))
        if mode!='image_only': value.update(summarize(rr))
        if mode!='text_only': value.update(psnr=sum(r['psnr'] for r in rr)/len(rr),ssim=sum(r['ssim'] for r in rr)/len(rr))
        for k in ('physical_symbol_snr_db','pilot_snr_db','scheduled_seconds','launch_signal_energy_j'):
            numbers=[r['stats'].get(k) for r in rr if r['stats'].get(k) is not None]
            value[k]=sum(numbers)/len(numbers) if numbers else None
        summaries[mode+'__'+cid]=value
    summary=dict(version=VERSION,status=identity['status'],smoke=design['smoke'],mode_status=states,rows=len(rows),
        groups=summaries,resources=design['resources'],limits=design['limitations'],
        analog_ber='undefined, not fabricated',text_metrics_independently_recomputed=True,
        image_metrics='actual Torch tensor PSNR/SSIM, finite checked; pixels saved as quantized PNG')
    atomic_json(root/'summary.json',summary); esc=lambda x:html.escape(str(x),quote=True)
    out=['<!doctype html><meta charset="utf-8"><title>COCO single modality</title><style>body{font:15px system-ui;margin:28px;color:#172338}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd7e6;padding:8px;vertical-align:top}img{width:144px}pre{white-space:pre-wrap}</style>',
        '<h1>原始 COCO 短描述：图片／文字单独传输</h1><p>状态：'+esc(identity['status'])+'</p>',
        '<p>图片模式不编码文字；文字模式不读取图片像素。公共 C 在单模态模式中仅来自该模态，不是图文联合语义。非传输模态指标不适用，不填零。单次连续传输，物理波形计费。训练仅为干净训练，不能据此声称新增带噪训练增益。</p>',
        '<pre>'+esc(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>']
    for mode in design['modes']:
        folder=root/mode
        if mode=='image_only': continue
        for label,name in [('未经压缩的预训练 BART（TRAIN诊断）','pretrained_uncompressed.jsonl'),('T32短文本压缩（TRAIN诊断）','private_T32_controls.jsonl'),('当前 C/T 模式（TRAIN诊断）','mode_overfit_controls.jsonl'),('未见训练图片的短文本（探索性验证）','clean_validation_controls.jsonl')]:
            if not (folder/name).exists(): continue
            out+=['<h2>'+esc(mode+' / '+label)+'</h2><table><tr><th>组</th><th>原文，仅供评价</th><th>实际生成</th><th>覆盖 WER</th></tr>']
            for r in lines(folder/name): out+=['<tr><td>'+esc(r.get('control','native'))+'</td><td>'+esc(r['source_text'])+'</td><td>'+esc(r['received_caption'])+'</td><td>'+esc(round(r['covered_wer'],4))+'</td></tr>']
            out+=['</table>']
    out+=['<h2>实际干净／FSO恢复</h2><table><tr><th>模式／链路</th><th>原图</th><th>恢复图</th><th>评价原文／恢复文字</th><th>任务指标／链路</th></tr>']
    for r in rows:
        if r['mode']=='text_only' and r['image_id'] not in design['caption_views'][r['mode']]['validation']['image_ids'][:design['config']['save_images']]: continue
        if r['mode']!='text_only' and 'image_file' not in r: continue
        pics=[('<img src="'+esc(r[k])+'">') if k in r else '不适用' for k in ('source_image_file','image_file')]
        text='仅供评价的图片描述：'+r['source_text'] if r['mode']=='image_only' else '原文：'+r['source_text']+'\n实际恢复：'+r['received_caption']
        details={k:r.get(k) for k in ('psnr','ssim','covered_wer','text_cer','bleu2')}; details['link']=r['stats']
        out+=['<tr><td>'+esc(r['mode']+' / '+r['condition'])+'</td><td>'+pics[0]+'</td><td>'+pics[1]+'</td><td><pre>'+esc(text)+'</pre></td><td><pre>'+esc(json.dumps(details,ensure_ascii=False,indent=2))+'</pre></td></tr>']
    out+=['</table><h2>公共符号置零／错配诊断</h2><p>固定接收张量上的敏感性检查，不是独立训练的最优私有符号基线，也不是额外物理传输。</p><pre>'+esc(json.dumps(lines(root/'common_controls.jsonl'),ensure_ascii=False,indent=2))+'</pre>']
    with (root/'actual_recovery.html').open('x',encoding='utf-8') as f: f.write('\n'.join(out))
    return summary


def seal(root):
    root=Path(root); identity,_,_,_=inspect(root); marker=MARKERS[identity['status']]
    files=[f for f in root.rglob('*') if f.is_file() and f.name not in MARKERS.values()]
    value=dict(version=VERSION,status=identity['status'],wall_seconds=identity['wall_seconds'],
        result_sha256={str(f.relative_to(root)):digest(f) for f in files if f.suffix!='.pth'},
        checkpoint_sha256={str(f.relative_to(root)):digest(f) for f in files if f.suffix=='.pth'})
    with (root/marker).open('x',encoding='utf-8') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def verify(root,weights=True):
    root=Path(root); identity,_,_,_=inspect(root); marker=MARKERS[identity['status']]; end=read(root/marker)
    if end['version']!=VERSION or end['status']!=identity['status']: raise ValueError('Wrong completion marker')
    files=[f for f in root.rglob('*') if f.is_file() and f.name not in MARKERS.values() and f.suffix!='.pth']
    if {str(f.relative_to(root)) for f in files}!=set(end['result_sha256']): raise ValueError('Unsealed/missing records')
    for name,sha in end['result_sha256'].items():
        if digest(file_under(root,name))!=sha: raise ValueError('Records changed: '+name)
    if weights:
        for name,sha in end['checkpoint_sha256'].items():
            if digest(file_under(root,name))!=sha: raise ValueError('Checkpoint changed: '+name)
    return identity['status']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('run'); p.add_argument('--records-only',action='store_true'); a=p.parse_args()
    print('Verified '+verify(a.run,weights=not a.records_only))
