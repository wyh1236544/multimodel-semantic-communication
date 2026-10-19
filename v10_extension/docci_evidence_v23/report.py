"""Independent matrix/identity/PNG/receiver verification; facts never invented."""
import argparse
import csv
import html
import json
import math
from pathlib import Path
import shutil
import sys
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
import numpy as np
from PIL import Image
from docci_evidence_v23.core import (CONTROLS,CAPTION_SNR,prepared,verify_files,
    read,rows,sha,digest,local,case_key,metric,prompts,digital_observation,atomic,safe,residual_stats,Budget)

def matrix(data,wanted):
    by={(case_key(r),r['control']):r for r in data}
    if len(by)!=len(data) or set(by)!=wanted: raise ValueError('Missing/duplicate/unexpected result matrix')
    return by

def verify(root,prepared_root=None,target_root=None,arrays=True,budget=lambda:None):
    root=Path(root);receipt=verify_files(root,arrays=arrays);stage=receipt['stage']
    if stage in ('inventory','assets'): return dict(stage=stage,training=False,semantic_success=False)
    if stage=='prepare':
        _,rs,_=prepared(root)
        if len(rs)!=receipt['metadata']['rows']: raise ValueError('Prepared row count differs')
        return dict(stage=stage,rows=len(rs),training=False,semantic_success=False)
    data=rows(root/'rows.jsonl');identity=read(root/'identity.json')
    if len(data)!=receipt['metadata']['rows']: raise ValueError('Completion count differs')
    if stage=='calibrate':
        wanted={(r['example_id'],s,0) for r in data for s in CAPTION_SNR}
        if (identity['count'] not in (8,32) or len({r['example_id'] for r in data})!=identity['count'] or len(data)!=len(wanted)
                or {case_key(r) for r in data}!=wanted or identity['timestep_mapping'] is not None):
            raise ValueError('Calibration matrix differs')
        for r in data:
            budget()
            if r['cohort']!='train' or not r['source_latent_used_only_after_receive_for_diagnosis']:
                raise ValueError('Calibration must be TRAIN-only diagnostics')
            with Image.open(local(root,r['source_image'])) as im:truth=np.asarray(im.convert('RGB'))
            if sha(local(root,r['source_image']))!=r['source_image_sha256']:raise ValueError('Calibration truth differs')
            for label in ('direct','roundtrip'):
                image=local(root,r[label+'_image'])
                if sha(image)!=r[label+'_image_sha256']:raise ValueError('Calibration output differs')
                with Image.open(image) as im:output=np.asarray(im.convert('RGB'))
                expected=metric(truth,output,r['rectangle'])
                if expected!=r[label]:raise ValueError('Calibration PNG metrics differ')
            if arrays:
                path=local(root,r['array_file'])
                if sha(path)!=r['array_sha256']: raise ValueError('Latent array bytes differ')
                with np.load(path,allow_pickle=False) as z:
                    if set(z.files)!={'source_latent','received_latent','clean_received_latent','reencoded_latent'}:
                        raise ValueError('Wrong latent matrix')
                    for key,a,b in [('source_error','received_latent','source_latent'),
                        ('incremental_channel_error','received_latent','clean_received_latent'),
                        ('roundtrip_error','reencoded_latent','received_latent')]:
                        expected=residual_stats(z[a]-z[b])
                        for name,value in expected.items():
                            actual=r[key][name]
                            if isinstance(value,float):
                                if not math.isclose(actual,value,rel_tol=2e-6,abs_tol=1e-8): raise ValueError('Latent statistic differs '+name)
                            elif value!=actual: raise ValueError('Latent statistic differs '+name)
        return dict(stage=stage,rows=len(data),arrays_verified=arrays,semantic_success=False,timestep_mapping=None)
    if not prepared_root: raise ValueError('Explicit preparation required')
    index,base,examples=prepared(prepared_root);by={case_key(r):r for r in base}
    if identity['prepared_sha256']!=sha(Path(prepared_root)/'COMPLETE.json'): raise ValueError('Preparation lineage differs')
    if stage=='digital':
        from docci_pretrained_checks_v22.digital import PROFILES,encode,receive,reconstruct,llrs
        matrix(data,{(k,'jpeg_'+c) for k in by for c in PROFILES})
        for i,r in enumerate(data):
            budget();eid,snr,rep=case_key(r);profile=r['control'][5:]
            payload=(root/'tx'/(eid+'__'+profile+'.jpg')).read_bytes()
            x,alloc,private=encode(payload,0,profile);y=digital_observation(x,index,r);rx=receive(y,profile)
            decoded=rx.pop('decoded_bits');value=rx.pop('payload')
            if (alloc!=r['allocation'] or r['source_coding']['codec']!='jpeg' or len(payload)!=r['allocation']['payload_bytes']
                    or r['allocation']['total_symbols']!=4128 or r['source_image_at_RX'] or r['source_text_at_RX']
                    or r['post_noise_renormalization'] or abs(float(np.mean(abs(x)**2))-1)>2e-6):
                raise ValueError('Digital resource/receiver contract differs')
            for key,v in rx.items():
                actual=r['rx'][key]
                if isinstance(v,float):
                    if not math.isclose(actual,v,rel_tol=1e-12,abs_tol=1e-12): raise ValueError('Digital RX telemetry differs')
                elif actual!=v: raise ValueError('Digital receiver decision differs '+key)
            errors=None if decoded is None else int(np.count_nonzero(decoded!=private['packet']))
            if r['post_fec_errors']!=errors or r['post_fec_bits']!=(0 if errors is None else len(private['packet'])):
                raise ValueError('PostFEC BER differs')
            hard=np.concatenate(((llrs(y[:680],1.,2)<0).astype(np.uint8),
                (llrs(y[680:],1.,alloc['bps'],alloc['body_gain'])<0).astype(np.uint8)))
            if r['pre_fec_bits']!=len(hard) or r['pre_fec_errors']!=int(np.count_nonzero(hard!=private['bits'])):
                raise ValueError('PreFEC BER differs')
            expected=reconstruct(value,0) if rx['packet_crc'] else Image.new('RGB',(512,512),(128,128,128))
            with Image.open(local(root,r['image_file'])) as im:
                if im.mode!='RGB' or im.size!=(512,512) or not np.array_equal(np.asarray(im),np.asarray(expected)):
                    raise ValueError('Digital reconstructed pixels differ')
            ex=examples[eid]
            with Image.open(local(prepared_root,ex['source_image'])) as im:truth=np.asarray(im.convert('RGB'))
            pixel=metric(truth,np.asarray(expected),ex['rectangle'])
            if abs(pixel['MSE']-r['MSE'])>1e-14 or pixel['identical']!=r['identical']:
                raise ValueError('Digital pixel metric differs')
            if pixel['psnr'] is None:
                if r['psnr'] is not None:raise ValueError('Identical PSNR must be null')
            elif r['psnr'] is None or abs(pixel['psnr']-r['psnr'])>1e-8:raise ValueError('Digital PSNR differs')
            if sha(local(root,r['image_file']))!=r['image_sha256']: raise ValueError('Digital PNG hash differs')
            if snr is None and not rx['packet_crc']: raise ValueError('Clean digital packet failed')
            if i%100==0: print('VERIFY DIGITAL {}/{}'.format(i+1,len(data)),flush=True)
    elif stage=='caption':
        wanted={(k,c) for k in by if k[1] in CAPTION_SNR for c in CONTROLS};matrix(data,wanted)
        for r in data:
            budget();src=by[case_key(r)];donor=by[src['donor_id'],r['snr_db'],r['repeat']]
            cap,joint=prompts(src['received_caption']);control=r['control']
            prompt=None if control=='received_text' else cap if control=='caption_only' else (
                prompts(donor['received_caption'])[1] if control=='joint_wrong_text' else joint)
            image_sha=None
            if control=='joint_blank_image': image_sha=digest(np.full((512,512,3),128,np.uint8).tobytes())
            elif control!='received_text':
                origin=donor if control=='joint_wrong_image' else src
                with Image.open(local(prepared_root,origin['received_image'])) as im: image_sha=digest(np.asarray(im.convert('RGB')).tobytes())
            if (r['source_text_at_RX'] or r['total_symbols']!=13376 or r['output_sha256']!=digest(r['output_text'].encode())
                    or r['received_caption_sha256']!=src['received_caption_sha256']
                    or r['prompt_sha256']!=(None if prompt is None else digest(prompt.encode()))
                    or r['conditioning_image_sha256']!=image_sha
                    or (control=='received_text' and r['output_text']!=src['received_caption'])):
                raise ValueError('Caption input/receiver contract differs')
            if control!='received_text' and (r['telemetry']['do_sample'] or r['telemetry']['maximum_new_tokens']!=512
                    or r['telemetry']['capacity_stop']==r['telemetry']['natural_EOS']):
                raise ValueError('Caption termination telemetry differs')
    elif stage=='score':
        if target_root is None: raise ValueError('Portable explicit target required')
        target=Path(target_root);tr=verify_files(target)
        if sha(target/'COMPLETE.json')!=identity['target_sha256']: raise ValueError('Score target differs')
        target_by={(case_key(r),r.get('control','continuous')):r for r in rows(target/'rows.jsonl')}
        matrix(data,set(target_by))
        for r in data:
            budget();origin=target_by[case_key(r),r['control']];name=origin.get('image_file',origin.get('received_image'))
            with Image.open(local(target,name)) as im:output=np.asarray(im.convert('RGB'))
            ex=examples[r['example_id']]
            with Image.open(local(prepared_root,ex['source_image'])) as im:truth=np.asarray(im.convert('RGB'))
            expected=metric(truth,output,ex['rectangle'])
            if r['image_sha256']!=sha(local(target,name)) or not math.isfinite(r['LPIPS']) or r['LPIPS']<0:
                raise ValueError('Score image/LPIPS differs')
            if abs(r['MSE']-expected['MSE'])>1e-14 or r['identical']!=expected['identical']:
                raise ValueError('Independent PNG MSE differs')
            if expected['psnr'] is None:
                if r['psnr'] is not None: raise ValueError('Infinite PSNR must be null')
            elif r['psnr'] is None or abs(r['psnr']-expected['psnr'])>1e-8: raise ValueError('Independent PNG PSNR differs')
    else: raise ValueError('Unknown stage')
    return dict(stage=stage,rows=len(data),training=False,semantic_success=False,facts='PENDING')

def curve_svg(groups,metric_name,path):
    """Descriptive means only. Clean is separate, never placed at fictitious SNR."""
    curves={}
    for key,group in groups.items():
        label,snr=key.rsplit(' / ',1);value=group['mean'].get(metric_name)
        if snr=='None' or value is None:continue
        if label.startswith('digital:') and 'score:'+label[len('digital:'):]+' / '+snr in groups:continue
        if label.startswith('pixel:') and 'score:'+label[len('pixel:'):]+' / '+snr in groups:continue
        if label.startswith(('score:','digital:','pixel:')):
            curves.setdefault(label,[]).append((float(snr),value))
    if not curves:return False
    xs=[x for pts in curves.values() for x,y in pts];ys=[y for pts in curves.values() for x,y in pts]
    lo,hi=min(ys),max(ys);pad=max((hi-lo)*.1,.01 if metric_name=='LPIPS' else .1);lo-=pad;hi+=pad
    xmin,xmax=min(xs),max(xs)
    def x(v):return 70+650*(v-xmin)/(xmax-xmin or 1)
    def y(v):return 340-280*(v-lo)/(hi-lo)
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#333333']
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="410" viewBox="0 0 1100 410">',
        '<rect width="1100" height="410" fill="white"/>',
        '<text x="70" y="25" font-family="sans-serif" font-size="17">'+metric_name+' — descriptive image means, no inference</text>',
        '<path d="M70 55V340H730" fill="none" stroke="black"/>']
    for i in range(6):
        value=lo+(hi-lo)*i/5
        parts.append('<text x="64" y="{:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{:.3f}</text>'.format(y(value)+4,value))
    for v in sorted(set(xs)):
        parts.append('<text x="{:.2f}" y="365" text-anchor="middle" font-family="sans-serif" font-size="12">{:g}</text>'.format(x(v),v))
    for i,(label,pts) in enumerate(sorted(curves.items())):
        color=colors[i%len(colors)];pts=sorted(pts)
        points=' '.join('{:.2f},{:.2f}'.format(x(a),y(b)) for a,b in pts)
        parts.append('<polyline points="'+points+'" fill="none" stroke="'+color+'" stroke-width="2"/>')
        for a,b in pts:parts.append('<circle cx="{:.2f}" cy="{:.2f}" r="3" fill="{}"/>'.format(x(a),y(b),color))
        parts.append('<text x="750" y="{}" font-family="sans-serif" font-size="12" fill="{}">{}</text>'.format(65+i*24,color,html.escape(label)))
    parts.append('<text x="350" y="395" font-family="sans-serif" font-size="14">Nominal AWGN channel SNR (dB); clean shown in table</text></svg>')
    path.write_text('\n'.join(parts),encoding='utf-8');return True

def render(prepared_root,stages,output,budget=lambda:None):
    output=safe(output)
    if output.exists(): raise FileExistsError('Existing report preserved')
    index,base,examples=prepared(prepared_root)
    roots={sha(Path(prepared_root)/'COMPLETE.json'):Path(prepared_root)}
    roots.update({sha(Path(p)/'COMPLETE.json'):Path(p) for p in stages})
    summary=dict(schema='v23-report',facts='PENDING',semantic_success=False,training=False,
        reused_dev=True,independent_test=False,groups={},stages=[],
        interpretation='Descriptive smoke n=2; not significance. JPEG is not strongest digital. Caption-only is not mutual recovery.')
    output.mkdir();(output/'images').mkdir()
    captions=[];image_rows=[]
    for r in base:
        budget();ex=examples[r['example_id']]
        with Image.open(local(prepared_root,ex['source_image'])) as im:truth=np.asarray(im.convert('RGB'))
        with Image.open(local(prepared_root,r['received_image'])) as im:output_image=np.asarray(im.convert('RGB'))
        measures=metric(truth,output_image,ex['rectangle']);measures.pop('identical')
        key='pixel:'+Path(prepared_root).name+'/continuous / '+str(r['snr_db'])
        summary['groups'].setdefault(key,{}).setdefault(r['example_id'],[]).append(measures)
    for path in map(Path,stages):
        receipt=read(path/'COMPLETE.json');stage=receipt['stage'];target=None
        if stage=='score': target=roots.get(read(path/'identity.json')['target_sha256'])
        result=verify(path,prepared_root,target,budget=budget);summary['stages'].append(result)
        data=rows(path/'rows.jsonl') if (path/'rows.jsonl').exists() else []
        for r in data:
            if stage=='score':
                label='score:'+roots[read(path/'identity.json')['target_sha256']].name+'/'+r['control']
                measures={k:r[k] for k in ('psnr','MSE','LPIPS')}
            elif stage=='caption':
                from docci_text_continue_v11.metrics import metrics
                label='caption:'+path.name+'/'+r['control'];m=metrics(examples[r['example_id']]['source_text'],r['output_text'])
                measures={'normalized_WER':m['normalized_wer'],'capacity_stop':float(r.get('telemetry',{}).get('capacity_stop',False))}
                captions.append((path,r))
            elif stage=='digital':
                label='digital:'+path.name+'/'+r['control'];measures={'packet_success':float(r['rx']['packet_crc']),'psnr':r['psnr'],'MSE':r['MSE']}
                image_rows.append((path,r))
            else: continue
            key=label+' / '+str(r['snr_db']);summary['groups'].setdefault(key,{}).setdefault(r['example_id'],[]).append(measures)
    for key,images in list(summary['groups'].items()):
        values=[]
        for eid,rr in images.items():
            values.append({k:None if any(x[k] is None for x in rr) else float(np.mean([x[k] for x in rr])) for k in rr[0]})
        summary['groups'][key]=dict(independent_images=len(values),repeats_averaged_within_image=True,
            mean={k:None if any(x[k] is None for x in values) else float(np.mean([x[k] for x in values])) for k in values[0]},
            per_image=[dict(example_id=eid,**v) for eid,v in zip(images,values)])
    atomic(output/'summary.json',summary)
    plot_names=[]
    for name in ('psnr','LPIPS'):
        if curve_svg(summary['groups'],name,output/(name+'_curve.svg')):plot_names.append(name+'_curve.svg')
    def portable(path):
        name=sha(path)+'.png';dest=output/'images'/name
        if not dest.exists():shutil.copyfile(path,dest)
        return 'images/'+name
    page=['<!doctype html><meta charset="utf-8"><title>V23 PRIVATE evidence</title>',
        '<style>body{font:16px/1.6 system-ui;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px;vertical-align:top}img{width:240px}pre{white-space:pre-wrap}details{margin:12px 0}</style>',
        '<h1>V23：免训练图像对照与图像辅助文本检查</h1>',
        '<p>PRIVATE。AWGN，clean无噪声；负SNR为分布外推理。复用dev，不是新test。事实审核PENDING，未宣布互促或连续优于最强数字。无FSO、无扩散生成、无训练。</p>',
        '<p>折线仅连接已测SNR的图片均值，不包含clean，不表示两图预检有统计显著性。PSNR高为好、LPIPS低为好。</p>'+''.join('<p><img style="width:1100px;max-width:100%" src="'+name+'"></p>' for name in plot_names),
        '<h2>聚合（同图噪声重复先平均）</h2><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>']
    with (output/'fact_review.csv').open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['blind_id','source_text','output_text','source_atomic_fact','fact_type','preserved_0_1','unsupported_added_fact','reviewer'])
        keys=[]
        for path,r in sorted(captions,key=lambda x:seed('V23-human',x[0].name,x[1]['control'],*case_key(x[1]))):
            if r['repeat']!=0:continue
            bid=str(seed('V23-blind',path.name,r['control'],*case_key(r)))
            writer.writerow([bid,examples[r['example_id']]['source_text'],r['output_text'],'','','','',''])
            keys.append(dict(blind_id=bid,control=r['control'],case=list(case_key(r))))
    atomic(output/'fact_review_PRIVATE_key.json',keys)
    for r in base:
        if r['example_id'] not in list(examples)[:2] or r['repeat']!=0:continue
        ex=examples[r['example_id']];page.append('<h2>'+html.escape(str(case_key(r)))+'</h2><table><tr>')
        for label,p in [('原图（仅评价）',local(prepared_root,ex['source_image'])),('连续接收',local(prepared_root,r['received_image']))]:
            page.append('<td>'+label+'<br><img src="'+portable(p)+'"></td>')
        for path,d in image_rows:
            if case_key(d)==case_key(r):page.append('<td>'+html.escape(d['control'])+'<br><img src="'+portable(local(path,d['image_file']))+'"></td>')
        page.append('</tr></table><details><summary>原文、实际接收及各文本对照</summary><p>原文（评价）：'+html.escape(ex['source_text'])+'</p>')
        if r['received_caption'] is not None:page.append('<p>接收文字：'+html.escape(r['received_caption'])+'</p>')
        for _,d in captions:
            if case_key(d)==case_key(r):page.append('<p>'+html.escape(d['control']+': '+d['output_text'])+'</p>')
        page.append('</details>')
    page.append('<p>图像单路4128复符号；加入传输文字共13376。BLIP-2可能幻觉，WER不是语义准确率。CAPTURE/BERTScore/事实得分未运行。</p>')
    (output/'index.html').write_text('\n'.join(page),encoding='utf-8')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('action',choices=('verify','render'));p.add_argument('root',nargs='?')
    p.add_argument('--prepared');p.add_argument('--target');p.add_argument('--stages',nargs='*',default=[]);p.add_argument('--output');p.add_argument('--max-hours',type=float,default=2.)
    a=p.parse_args();b=Budget(a.max_hours)
    result=verify(a.root,a.prepared,a.target,budget=b) if a.action=='verify' else render(a.prepared,a.stages,a.output,b)
    print(json.dumps(result,ensure_ascii=False,indent=2))
