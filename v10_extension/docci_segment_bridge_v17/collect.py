"""PRIVATE CPU collection, including explicit partial/NOT_RUN states, never retrains."""
import argparse
import hashlib
import html
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_segment_bridge_v17.core import safe, sha, read, atomic, lines, ARMS, CONDITIONS
from docci_segment_bridge_v17.report import pipeline


def render(root, result):
    esc = html.escape
    parts = ['<!doctype html><meta charset="utf-8"><title>V17 PRIVATE actual recovery</title>',
             '<style>body{font:16px/1.6 system-ui;max-width:1350px;margin:30px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:10px;vertical-align:top}th{background:#eef3f6}pre{white-space:pre-wrap}img{width:250px}.row{display:flex;gap:20px}.warn{background:#fff4dc;padding:16px}</style>',
             '<h1>V17 按段视觉修复：真实结果</h1>',
             '<p class="warn">流水状态：'+esc(result['pipeline_status'])+'。oracle使用原图，不是可部署的等资源通信结果；仅文本符号加AWGN，图像理想。无FSO，无新Common，无新图像分支训练。事实审核PENDING。</p>',
             '<h2>恢复WER（越低越好）</h2><table><tr><th>组别</th><th>clean</th><th>10dB</th><th>0dB</th></tr>']
    for arm in ARMS:
        data = result['arm_results'].get(arm)
        parts.append('<tr><th>'+esc(arm)+'</th>')
        for channel in CONDITIONS:
            value = '{:.2f}%'.format(data['scores'][channel]['matched']['mean_WER']*100) if data else 'NOT_RUN / 未完成'
            parts.append('<td>'+value+'</td>')
        parts.append('</tr>')
    parts.append('</table><h2>配对内容相关筛查</h2><pre>'+esc(json.dumps(result['comparisons'], ensure_ascii=False, indent=2))+'</pre>')
    inputs = root/'inputs'
    if 'inputs' in result['arm_results'] and inputs.is_dir() and (inputs/'baseline.jsonl').is_file():
        baselines = lines(inputs/'baseline.jsonl')
        ids = list(dict.fromkeys(r['example_id'] for r in baselines))[:4]
        completed = [arm for arm in ARMS if arm in result['arm_results']]
        output = {arm: lines(root/arm/'rows.jsonl') for arm in completed}
        parts.append('<h2>固定前4条真实样例（非选优）</h2>')
        for eid in ids:
            base = next(r for r in baselines if r['example_id'] == eid)
            parts.append('<h3>'+esc(eid)+'</h3><div class="row">')
            for suffix, caption in [('source', '源图，仅oracle诊断'), ('received', '旧V14接收重建，本轮未训练图像分支')]:
                name = eid+'__'+suffix+'.png'
                if (inputs/'images'/name).is_file():
                    parts.append('<figure><img src="inputs/images/'+esc(name, quote=True)+'"><figcaption>'+caption+'</figcaption></figure>')
            parts.append('</div><p>原文（仅评价）：'+esc(base['source_text'])+'</p>')
            for channel in CONDITIONS:
                parts.append('<h4>'+esc(channel)+'</h4><table><tr>')
                for arm in completed:
                    r = next(x for x in output[arm] if x['example_id'] == eid and x['condition'] == channel and x['control'] == 'matched')
                    parts.append('<td><b>'+esc(arm)+'</b><p>WER {:.2f}%</p><p>'.format(r['normalized_wer']*100)+esc(r['received_caption'])+'</p></td>')
                parts.append('</tr></table>')
    parts.append('<p>配对区间只覆盖重复使用的100条qual_dev，不覆盖训练随机性。完整逐样本、所有donor与对照保存在各组rows.jsonl。数字筛查通过也不等于事实正确或双向互促已成立。</p>')
    return ''.join(parts)


def collect(root, destination, max_minutes=60):
    root, out = Path(root).resolve(strict=True), safe(destination)
    stage = out.with_name(out.name+'.contents')
    if out == root or root in out.parents or out.exists() or stage.exists() or out.with_name(out.name+'.sha256').exists():
        raise FileExistsError('Archive/staging must be NEW and outside original run')
    if not 0 < max_minutes <= 180:
        raise ValueError('Bounded CPU collection required')
    start = time.monotonic()
    def budget():
        if time.monotonic()-start > max_minutes*60:
            raise TimeoutError('CPU collection time exceeded; no retraining needed')
    result = pipeline(root, arrays=True, max_minutes=max_minutes)
    stage.mkdir(parents=True, exist_ok=False)
    target = stage/'pipeline'; target.mkdir()
    allowed = {'.py', '.json', '.jsonl', '.md', '.txt', '.log', '.png', '.html'}
    for p in sorted(root.rglob('*')):
        budget()
        if p.is_symlink():
            raise ValueError('Symlink result collection rejected')
        if not p.is_file() or p.suffix not in allowed or '__pycache__' in p.parts or p == root/'collection.log':
            continue
        q = target/p.relative_to(root); q.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(p); shutil.copyfile(p, q)
        if sha(p) != digest or sha(q) != digest:
            raise ValueError('Result changed during copy')
    atomic(stage/'summary.json', result)
    with (target/'index.html').open('x', encoding='utf-8') as f:
        f.write(render(target, result))
    atomic(stage/'COLLECTION.json', dict(PRIVATE=True, original_files_read_only=True,
           partial_is_not_complete=result['pipeline_status'] != 'COMPLETE',
           neural_inference_replayed=False, checkpoints_and_large_arrays_excluded=True,
           arrays_and_final_checkpoints_hashed_for_verified_stages=result['completed_stages'],
           unverified_partial_files_are_diagnostic_only=True,
           original_cache_verified_at_pipeline_preflight=(root/'cache_verified.json').is_file(),
           elapsed_seconds=time.monotonic()-start))
    hashes = {p.relative_to(stage).as_posix(): sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}
    atomic(stage/'ARCHIVE_MANIFEST.json', hashes)
    with out.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz', format=tarfile.USTAR_FORMAT) as tar:
            for name in sorted(hashes)+['ARCHIVE_MANIFEST.json']:
                budget(); data = (stage/name).read_bytes()
                item = tarfile.TarInfo('v17_records/'+name); item.mode, item.mtime, item.size = 0o644, 0, len(data)
                tar.addfile(item, io.BytesIO(data))
    with out.with_name(out.name+'.sha256').open('x', encoding='utf-8') as f:
        f.write(sha(out)+'  '+out.name+'\n')
    print('PRIVATE V17 records collected: '+str(out), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('destination')
    p.add_argument('--max-minutes', type=float, default=60); a = p.parse_args()
    collect(a.root, a.destination, a.max_minutes)
