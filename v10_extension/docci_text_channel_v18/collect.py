"""Independent PRIVATE collection; no inference or automatic retraining."""
import argparse
import html
import io
import json
import math
from pathlib import Path
import shutil
import sys
import tarfile
import time
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_text_channel_v18.core import safe,atomic,read,lines,sha
from docci_text_channel_v18.report import pipeline

def render(stage,result):
    esc=html.escape
    parts=['<!doctype html><meta charset="utf-8"><title>V18 PRIVATE recovery</title><style>body{font:16px/1.65 system-ui;max-width:1200px;margin:30px auto;padding:0 20px}table{width:100%;border-collapse:collapse}td,th{border:1px solid #bbb;padding:10px;vertical-align:top}th{background:#eef3f6}p{overflow-wrap:anywhere}summary{cursor:pointer}.note{background:#fff4dc;padding:16px}</style>',
        '<h1>V18：文本 codec 带噪适应</h1><p class="note">状态 '+esc(result['status'])+'。无图像辅助、Common、FSO或数字对照；BART冻结，原文本TX/RX codec可训练。语义事实审核PENDING，工程完成不等于效果通过。</p>']
    comparison=result.get('comparison')
    if comparison:
        parts.append('<h2>WER曲线数据（越低越好）</h2><table><tr><th>名义SNR</th><th>原V14冻结</th><th>clean继续训练</th><th>带噪＋25%clean回放</th></tr>')
        for channel,v in comparison['scores'].items():
            parts.append('<tr><th>'+channel+'</th>'+''.join('<td>{:.2f}%</td>'.format(v[a]['mean_WER']*100) for a in ('frozen','clean_continue','noise_replay'))+'</tr>')
        parts.append('</table><p>clean无噪声/SNR不适用。每个噪声SNR每篇3次实现，先篇内平均再做描述级区间；不能当300篇独立样本。</p><h2>配对差值与固定筛查</h2><pre>'+esc(json.dumps(comparison['effects'],ensure_ascii=False,indent=2))+'</pre>')
        parts.append('<p>数字筛查：'+str(comparison['numerical_screen_pass'])+'；'+esc(comparison['predeclared_screen'])+'</p>')
        # Simple vector curve: clean displayed separately because it is not 0dB.
        colors={'frozen':'#777777','clean_continue':'#E69F00','noise_replay':'#0072B2'}
        svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 490"><rect width="900" height="490" fill="white"/><g font-family="Arial,sans-serif">']
        ymax=max(100,20*math.ceil(max(v[a]['mean95CI_WER'][1]*100 for v in comparison['scores'].values() for a in colors)/20))
        scale=310/ymax
        for tick in range(0,int(ymax)+1,20):
            y=375-tick*scale;svg.append('<line x1="75" x2="850" y1="'+str(y)+'" y2="'+str(y)+'" stroke="#ddd"/><text x="58" y="'+str(y+5)+'" text-anchor="end">'+str(tick)+'</text>')
        for arm,col in colors.items():
            points=[]
            for i,snr in enumerate([0,5,10,15,20]):
                score=comparison['scores'][str(snr)][arm]
                x=120+i*170;y=375-score['mean_WER']*100*scale
                lo,hi=score['mean95CI_WER']
                svg.append('<line x1="'+str(x)+'" x2="'+str(x)+'" y1="'+str(375-hi*100*scale)+'" y2="'+str(375-lo*100*scale)+'" stroke="'+col+'" stroke-width="2"/>')
                points.append(str(x)+','+str(y));svg.append('<circle cx="'+str(x)+'" cy="'+str(y)+'" r="5" fill="'+col+'"/>')
            svg.append('<polyline points="'+' '.join(points)+'" fill="none" stroke="'+col+'" stroke-width="2"/>')
        for i,snr in enumerate([0,5,10,15,20]):svg.append('<text x="'+str(120+i*170)+'" y="405" text-anchor="middle">'+str(snr)+'</text>')
        svg.append('<text x="450" y="435" text-anchor="middle">Nominal text-symbol AWGN SNR (dB)</text><text x="22" y="240" transform="rotate(-90 22 240)">Mean WER (%)</text>')
        for i,(arm,col) in enumerate(colors.items()):svg.append('<text x="'+str(70+i*280)+'" y="465" fill="'+col+'">'+arm+'</text>')
        svg.append('</g></svg>')
        (stage/'curve.svg').write_text(''.join(svg),encoding='utf-8')
        parts.append('<p><img src="curve.svg" style="width:100%"></p><p>误差条为描述级bootstrap的边际95%区间；方法差异须看上方配对区间。均为点态未校正、单训练种子，不是多训练重复。WER可超过100%，纵轴按结果自动扩展。</p>')
    parts.append('<h2>全部真实样例（固定ID顺序，未选优）</h2>')
    completed={}
    for arm in ('frozen','clean_continue','noise_replay'):
        p=stage/'pipeline'/('eval_'+arm)/'rows.jsonl'
        if 'eval_'+arm in result['verified_stages']:completed[arm]=lines(p)
    if completed:
        first=next(iter(completed.values()))
        for eid in sorted({r['example_id'] for r in first}):
            source=next(r for r in first if r['example_id']==eid)['source_text']
            parts.append('<details><summary>'+eid+'</summary><p>原文（仅评价）：'+esc(source)+'</p>')
            for snr in (None,20,15,10,5,0):
                parts.append('<h3>'+('clean' if snr is None else str(snr)+'dB')+'，噪声实现0</h3><table>')
                for arm,rows in completed.items():
                    r=next(r for r in rows if r['example_id']==eid and r['snr_db']==snr and r['repeat']==0 and r['control']=='matched')
                    parts.append('<tr><th>'+arm+'<br>WER {:.2f}%</th><td>'.format(r['normalized_wer']*100)+esc(r['received_caption'])+'</td></tr>')
                parts.append('</table>')
            parts.append('</details>')
    else:parts.append('<p>暂无完整主评价；smoke或部分生成不作为主结果展示。</p>')
    return ''.join(parts)

def collect(root,destination):
    root=Path(root).resolve(strict=True);out=safe(destination);stage=out.with_name(out.name+'.contents')
    if root==out or root in out.parents or out.exists() or stage.exists() or out.with_name(out.name+'.sha256').exists():raise FileExistsError('NEW archive outside original run required')
    result=pipeline(root,full=True,max_minutes=60)
    stage.mkdir(parents=True,exist_ok=False);target=stage/'pipeline';target.mkdir()
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('Symlink in results')
        if not p.is_file() or p.suffix not in ('.py','.json','.jsonl','.md','.txt','.log') or p==root/'collection.log':continue
        q=target/p.relative_to(root);q.parent.mkdir(parents=True,exist_ok=True)
        h=sha(p);shutil.copyfile(p,q)
        if sha(p)!=h or sha(q)!=h:raise ValueError('Result changed during collection')
    atomic(stage/'summary.json',result)
    (stage/'index.html').write_text(render(stage,result),encoding='utf-8')
    atomic(stage/'COLLECTION.json',dict(PRIVATE=True,status=result['status'],complete=result['status']=='COMPLETE',no_inference=True,
        old_caches_and_weights_excluded=True,originals_read_only=True,large_arrays_and_weights_rechecked_on_server=True))
    hashes={p.relative_to(stage).as_posix():sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}
    atomic(stage/'ARCHIVE_MANIFEST.json',hashes)
    with out.open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz',format=tarfile.USTAR_FORMAT) as tf:
            for n in sorted(hashes)+['ARCHIVE_MANIFEST.json']:
                value=(stage/n).read_bytes();info=tarfile.TarInfo('v18_records/'+n);info.size=len(value);info.mode=0o644;info.mtime=0;tf.addfile(info,io.BytesIO(value))
    with out.with_name(out.name+'.sha256').open('x',encoding='utf-8') as f:f.write(sha(out)+'  '+out.name+'\n')
    receipt=root/'COLLECTED.json'
    if receipt.exists():receipt=root/('COLLECTED_'+sha(out)[:12]+'.json')
    atomic(receipt,dict(archive=str(out),complete=result['status']=='COMPLETE',status=result['status']))
    print('PRIVATE records collected: '+str(out),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('destination');a=p.parse_args();collect(a.root,a.destination)
