"""Plain PRIVATE HTML/SVG, fixed examples, no generative media or selected bests."""
import csv
import html
import json
import io
import math
from pathlib import Path
from docci_digital_compare_v19.common import read,lines

COLORS={'frozen':'#777777','clean_continue':'#E69F00','noise_replay':'#0072B2',
        'qpsk_ldpc12':'#009E73','qpsk_ldpc34':'#CC79A7'}
STYLE='<style>body{font:16px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:0 20px}table{border-collapse:collapse;width:100%;margin:12px 0}td,th{border:1px solid #bbb;padding:8px;text-align:left;vertical-align:top}th{background:#edf2f6}td{overflow-wrap:anywhere}pre{white-space:pre-wrap}.note{background:#fff1d6;padding:15px}summary{cursor:pointer;margin:8px 0}img{max-width:100%}</style>'

def write_derived(path,text):
    path=Path(path)
    if path.exists():
        if path.read_text(encoding='utf-8')!=text:raise FileExistsError('Existing derived display differs; preserve: '+str(path))
    else:
        with path.open('x',encoding='utf-8') as f:f.write(text)

def curve(comparison):
    scores=comparison['scores'];snrs=sorted(int(s) for s in scores if s!='clean')
    ymax=max(100,20*math.ceil(max(x['WER']['CI95'][1]*100 for s in scores.values() for x in s.values())/20))
    def x(s):return 80+(s+10)/30*940
    def y(v):return 370-v*100/ymax*310
    p=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 520"><rect width="1100" height="520" fill="white"/><g font-family="Arial,sans-serif" font-size="15">']
    for i in range(6):
        v=ymax*i/5;yy=y(v/100)
        p.append('<line x1="80" x2="1020" y1="{0}" y2="{0}" stroke="#ddd"/><text x="65" y="{1}" text-anchor="end">{2:g}</text>'.format(yy,yy+5,v))
    for s in snrs:p.append('<text x="{}" y="400" text-anchor="middle">{}</text>'.format(x(s),s))
    for j,(arm,color) in enumerate(COLORS.items()):
        pts=[]
        for s in snrs:
            if arm not in scores[str(s)]:continue
            v=scores[str(s)][arm]['WER'];xx=x(s);yy=y(v['mean']);lo,hi=v['CI95'];pts.append('{},{}'.format(xx,yy))
            p.append('<line x1="{0}" x2="{0}" y1="{1}" y2="{2}" stroke="{3}"/><circle cx="{0}" cy="{4}" r="4" fill="{3}"/>'.format(xx,y(lo),y(hi),color,yy))
        if pts:p.append('<polyline points="{}" fill="none" stroke="{}" stroke-width="2" {}/>'.format(' '.join(pts),color,'stroke-dasharray="7 4"' if arm.startswith('qpsk') else ''))
        p.append('<text x="{}" y="{}" fill="{}">{}</text>'.format(80+(j%3)*330,465+(j//3)*25,color,arm))
    p+=['<text x="550" y="430" text-anchor="middle">AWGN Es/N0 (dB); clean is separate</text><text transform="translate(23 280) rotate(-90)">Mean normalized WER (%)</text></g></svg>']
    return ''.join(p)

def constellation(points):
    chosen=[r for s in (None,10,0,-2,-5,-10) for r in points if r['arm']=='qpsk_ldpc12' and r['snr_db']==s]
    lim=max(2,math.ceil(max((max(abs(a),abs(b)) for r in chosen for a,b in r['received_IQ']),default=2)))
    p=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1020 660"><rect width="1020" height="660" fill="white"/><g font-family="Arial,sans-serif">']
    for j,r in enumerate(chosen):
        left=35+(j%3)*340;top=40+(j//3)*320
        p.append('<text x="{}" y="{}">{}; first256 observed body symbols</text>'.format(left,top-10,'clean' if r['snr_db'] is None else str(r['snr_db'])+' dB'))
        p.append('<rect x="{}" y="{}" width="280" height="260" fill="none" stroke="#aaa"/>'.format(left,top))
        for a,b in r['received_IQ']:p.append('<circle cx="{}" cy="{}" r="1.8" fill="#0072B2" opacity=".6"/>'.format(left+140+a/lim*130,top+130-b/lim*120))
        p.append('<text x="{}" y="{}">I,Q each ±{}; no fitted rescaling</text>'.format(left,top+280,lim))
    return ''.join(p)+'</g></svg>'

def render(root,result):
    root=Path(root);esc=html.escape;data=result['comparison'];c=read(root/'config.json')
    parts=['<!doctype html><meta charset="utf-8"><title>V19 PRIVATE matched comparison</title>'+STYLE,
           '<h1>V19：同资源文本数字/连续对照</h1><p class="note">'+esc(result['status'])+'。没有新训练；每篇9216复符号、单位符号能量、同AWGN。负SNR为V18范围外推理。不是FSO/图文联合/实际光线速实验，关键事实审核PENDING。</p>']
    if data:
        write_derived(root/'curve.svg',curve(data))
        parts+=['<img src="curve.svg"><p>描述级bootstrap点态95%区间；3次噪声先篇内平均。单训练种子、复用开发集；不能把低WER称语义事实准确率。</p>',
                '<h2>WER：越低越好；clean不是0 dB</h2><table><tr><th>SNR</th>'+''.join('<th>'+a+'</th>' for a in COLORS)+'</tr>']
        for s in ['clean']+[str(s) for s in c['snr_db']]:
            if s not in data['scores']:continue
            v=data['scores'][s];parts.append('<tr><th>'+s+'</th>'+''.join('<td>{:.2f}%</td>'.format(v[a]['WER']['mean']*100) if a in v else '<td>NOT_RUN</td>' for a in COLORS)+'</tr>')
        parts+=['</table><h2>实际数字包与误码统计</h2><p>头失败时正文BER未知，不填0；失败文本输出为空并计入WER。零次观察错误不代表理论零误码。</p><table><tr><th>SNR</th><th>配置</th><th>包失败率</th><th>头失败率</th><th>硬判BER</th><th>正文FEC后BER（头成功条件下）</th></tr>']
        for s,v in data['scores'].items():
            for a in c['profiles']:
                if a not in v:continue
                r=v[a];post=r['post_FEC_BER_conditional_on_header']
                parts.append('<tr><td>{}</td><td>{}</td><td>{:.2%}</td><td>{:.2%}</td><td>{:.5g}</td><td>{}</td></tr>'.format(s,a,r['packet_fail_fraction'],r['header_fail_fraction'],r['pre_FEC_BER'],'UNKNOWN' if post is None else '{:.5g}'.format(post)))
        parts+=['</table><h2>连续noise_replay相对数字的配对增益（pp，正值才是连续更好）</h2><pre>'+esc(json.dumps(data['continuous_gain_pp_vs_digital'],ensure_ascii=False,indent=2))+'</pre>',
                '<h2>限制</h2><ul>'+''.join('<li>'+esc(s)+'</li>' for s in data['limits'])+'</ul>']
    if (root/'digital/capacity.json').exists():
        cap=read(root/'digital/capacity.json');ls=cap['lengths']
        parts+=['<h2>理论参考，不是实测成功率</h2><p>zlib平均{:.1f}bit，最大{}bit。容量表中的length_fit_fraction只是压缩长度是否低于渐近预算；未扣开销，不替代实际编码结果。</p>'.format(sum(r['zlib'] for r in ls)/len(ls),max(r['zlib'] for r in ls)),
                '<p>所有预算/压缩器/2dB假设敏感性见 digital/capacity.json；缩小预算没有在本版训练或评价连续模型。</p>']
    if (root/'digital/constellations.json').exists():
        write_derived(root/'constellation.svg',constellation(read(root/'digital/constellations.json')))
        parts+=['<h2>实际接收星座点（固定首篇、repeat0）</h2><img src="constellation.svg">']
    parts+=['<p><a href="actual_text.html">全部100篇实际恢复文本</a>；<a href="fact_review.csv">固定前30篇源事实审核表</a>；<a href="REPORT.json">完整统计</a></p>']
    write_derived(root/'index.html',''.join(parts))
    sources=read(root/'prepared/sources.json');rows=lines(root/'prepared/old_rows.jsonl')
    for label in ('digital','negative'):
        if label in result['stages']:rows+=lines(root/label/'rows.jsonl')
    ix={(r['arm'],r['example_id'],r['snr_db'],r['repeat']):r for r in rows}
    p=['<!doctype html><meta charset="utf-8"><title>V19 PRIVATE actual text</title>'+STYLE,
       '<h1>真实全文恢复</h1><p>所有ID固定顺序，不选优；展示repeat0，统计包含全部重复。原文仅评价，未送入接收端。空输出是数字包失败，不是遗漏样本。</p>']
    for source in sorted(sources,key=lambda r:r['example_id']):
        eid=source['example_id'];p+=['<details><summary>'+esc(eid)+'</summary><p>原文：'+esc(source['text'])+'</p>']
        for s in [None]+c['snr_db']:
            p+=['<h3>'+('clean' if s is None else str(s)+' dB')+'</h3><table>']
            for a in COLORS:
                r=ix.get((a,eid,s,0))
                if r:p.append('<tr><th>{}<br>WER {:.2f}%</th><td>{}</td></tr>'.format(a,r['normalized_wer']*100,esc(r['received_caption']) or '〔接收为空〕'))
            p.append('</table>')
        p.append('</details>')
    write_derived(root/'actual_text.html',''.join(p))
    # A human may have filled this file after the first collection. Never erase annotations.
    if not (root/'fact_review.csv').exists():
        with (root/'fact_review.csv').open('x',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(['example_id','source_text','source_atomic_facts_first','reviewer','model','snr_db','received_facts_retained','unsupported_new_facts','notes'])
            for s in sorted(sources,key=lambda r:r['example_id'])[:30]:writer.writerow([s['example_id'],s['text'],'','','','','','',''])
