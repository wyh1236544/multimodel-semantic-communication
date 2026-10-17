"""PRIVATE local report; source text/images are never public repository content."""
import html
import json
from pathlib import Path
from docci_text_guided_image_v20.core import read,rows

def render(root,result):
    root = Path(root)
    esc = html.escape
    parts = ['<!doctype html><meta charset="utf-8"><title>V20 图文恢复</title>',
        '<style>body{font:16px sans-serif;max-width:1400px;margin:30px auto}img{width:360px;max-width:100%}'
        'article{border-top:1px solid #bbb;padding:16px 0}pre{white-space:pre-wrap}figure{display:inline-block;'
        'max-width:400px;vertical-align:top;margin:8px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}</style>',
        '<h1>V20：文字辅助图像编解码</h1><p>真实接收图像与文字。文字分支冻结，尚未验证双向互促。'
        'oracle 原文仅诊断；AWGN 不是 FSO。PRIVATE：此页包含私人 DOCCI 数据，不上传公共仓库。</p>',
        '<h2>独立验收与配对结果</h2><pre>'+esc(json.dumps(result,ensure_ascii=False,indent=2))+'</pre>']
    for prefix in ('','smoke_'):
        for arm in ('A_self','B_tx_text','C_tx_rx_text'):
            folder = root/(prefix+'eval_'+arm)
            if not (folder/'EVAL_COMPLETE.json').exists():
                continue
            parts.append('<h2>'+esc(prefix+arm)+'</h2>')
            image_rows = rows(folder/'rows.jsonl')
            displayed = {x['example_id'] for x in image_rows if x['image_file']}
            sources_shown = set()
            for r in image_rows:
                if r['image_file'] is None:
                    continue
                base = folder.name+'/'
                if r['example_id'] not in sources_shown:
                    parts.append('<figure><img src="'+esc(base+r['source_image_file'])+
                        '"><figcaption>'+esc(r['example_id'])+' 原图（评价参考）</figcaption></figure>')
                    sources_shown.add(r['example_id'])
                parts.append('<figure><img src="'+esc(base+r['image_file'])+'"><figcaption>'+
                    esc('{} {} SNR={} {} | PSNR {:.2f} / SSIM {:.3f} / LPIPS {:.4f}'.format(
                        r['cohort'],r['example_id'],'clean' if r['snr_db'] is None else r['snr_db'],
                        r['control'],r['psnr'],r['ssim'],r['LPIPS']))+'</figcaption></figure>')
            for r in rows(folder/'text_rows.jsonl'):
                if r['example_id'] not in displayed:
                    continue
                parts.append('<article><h3>'+esc('{} {} SNR={}'.format(r['cohort'],r['example_id'],r['snr_db']))+
                    '</h3><p>原文（评价参考）：'+esc(r['source_text'])+'</p><p>实际接收文字（未受图像改善）：'+
                    esc(r['received_caption'])+'</p></article>')
    target = root/'index.html'
    content = ''.join(parts)
    if target.exists() and target.read_text(encoding='utf-8')!=content:
        raise ValueError('Existing differing PRIVATE report preserved')
    if not target.exists():
        with target.open('x',encoding='utf-8') as f:
            f.write(content)
