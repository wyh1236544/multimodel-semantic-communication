"""Independent CPU matrix/telemetry verifier and PRIVATE, fully escaped gallery."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import sys
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_caption_calibration_v24.core import (SCHEMA, FOLDER, PROTOCOL, QUESTIONS,
    read, rows, sha, digest, local, make_plan, token_audit, text_audit, review_keys,
    generation_kwargs, source_check)

def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite JSON scalar')
    if isinstance(value, dict):
        for x in value.values(): finite(x)
    if isinstance(value, list):
        for x in value: finite(x)

def verify(root):
    from PIL import Image
    from docci_evidence_v23.core import prepared
    root = Path(root).resolve(strict=True); c = read(root/'COMPLETE.json')
    if (c['schema'] != SCHEMA or c['source_manifest_sha256'] != sha(FOLDER/'MANIFEST.json') or
            c['training_updates'] != 0 or c['semantic_success'] is not False or c['facts'] != 'PENDING'):
        raise ValueError('Completion identity differs')
    found = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and
        p.relative_to(root).as_posix() not in ('COMPLETE.json', 'LAUNCH.json')}
    if found != set(c['files']):
        raise ValueError('Missing/unexpected records')
    for n, h in c['files'].items():
        if sha(local(root, n)) != h:
            raise ValueError('Artifact hash differs '+n)
    if (root/'FAILED.json').exists():
        raise ValueError('Failed inference is not complete')
    run = read(root/'RUN.json'); inference = read(root/'INFERENCE.json'); model = read(root/'MODEL.json')
    if (run['schema'] != SCHEMA or run['stage'] not in ('caption', 'qa') or run['protocol'] != PROTOCOL or
            run['training_updates'] != 0 or run['automatic_next_stage'] is not False or
            run['source_manifest_sha256'] != sha(FOLDER/'MANIFEST.json')):
        raise ValueError('Run protocol differs')
    index, original, _ = prepared(root/'prepared')
    if sha(root/'prepared/COMPLETE.json') != run['prepared_complete_sha256'] or sha(root/'ASSET_LOCK.json') != run['asset_lock_sha256']:
        raise ValueError('Input identity differs')
    plan = make_plan(index, original, run['stage'])
    with Image.open(root/'gray.png') as im:
        if im.mode != 'RGB' or im.size != (512, 512) or im.getextrema() != ((128, 128),)*3:
            raise ValueError('Blank control is not constant RGB gray')
    for p in plan:
        name = p.pop('source_file')
        p['image_file'] = 'gray.png' if name is None else 'prepared/'+name
        if name is None: p['image_sha256'] = sha(root/'gray.png')
    if read(root/'PLAN.json') != plan:
        raise ValueError('Control requests differ from sealed original/donor/gray matrix')
    actual = rows(root/'rows.jsonl')
    if len(actual) != len(plan) or {r['case_id'] for r in actual} != {r['case_id'] for r in plan}:
        raise ValueError('Duplicate/missing output matrix')
    by = {r['case_id']: r for r in actual}; seen = {}; unique = []
    for request in plan:
        r = by[request['case_id']]; finite(r)
        if any(r.get(k) != v for k, v in request.items()):
            raise ValueError('Actual model request differs')
        if sha(local(root, r['image_file'])) != r['image_sha256']:
            raise ValueError('Conditioning image differs')
        if hashlib.sha256(r['output_text'].encode()).hexdigest() != r['output_sha256'] or r['text_audit'] != text_audit(r['output_text']):
            raise ValueError('Output/lexical diagnostics differ')
        t = r['telemetry']; cap = 64 if run['stage'] == 'caption' else 32
        audit = token_audit(t['returned_token_ids'], t['new_token_count'], t['eos_token_ids'], cap)
        if any(t.get(k) != v for k, v in audit.items()) or len(t['score_trace']) != t['new_token_count']:
            raise ValueError('Token/EOS/score trace differs')
        if t['implicit_BOS'] != (r['prompt'] is None) or (r['prompt'] is None and t['input_token_ids'] is not None):
            raise ValueError('Native caption must omit text entirely')
        if not isinstance(t['raw_decoded_text'] if 'raw_decoded_text' in t else r['raw_decoded_text'], str):
            raise ValueError('Raw token decode missing')
        expected_shape = [1, 3, model['config']['vision_config']['image_size'], model['config']['vision_config']['image_size']]
        if t['processed_pixels']['shape'] != expected_shape or t['processed_pixels']['finite'] is not True:
            raise ValueError('Actual processor geometry differs')
        if set(t['forward_probes']) != {'vision_model', 'qformer', 'language_projection', 'language_model'} or any(x['finite'] is not True for x in t['forward_probes'].values()):
            raise ValueError('Missing actual finite forward probes')
        for j, s in enumerate(t['score_trace']):
            if (s['step'] != j or s['selected_token_id'] != t['new_token_ids'][j] or
                    len(s['top5_ids']) != 5 or len(s['top5_probabilities']) != 5 or
                    not 0 <= s['eos_probability'] <= 1.000001 or
                    any(not 0 <= p <= 1.000001 for p in s['top5_probabilities']) or
                    s['masked_negative_infinity_count'] < 0):
                raise ValueError('Invalid generation probability trace')
        call = digest([r['image_sha256'], r['prompt'], run['stage'], run['dtype']])
        if r['inference_id'] != call or r['reused_inference'] != (call in seen):
            raise ValueError('Unique inference/reuse accounting differs')
        content = {k: r[k] for k in ('output_text', 'raw_decoded_text', 'text_audit', 'telemetry')}
        if call in seen and content != seen[call]:
            raise ValueError('Same model inputs claimed reused but outputs differ')
        if call not in seen: unique.append(call)
        seen[call] = content
    if inference['unique_inference_ids'] != unique or inference['unique_inferences'] != len(unique) or inference['rows'] != len(plan):
        raise ValueError('Inference count differs')
    if (model['trainable_parameters'] != 0 or inference['training_updates'] != 0 or
            inference['semantic_success'] is not False or inference['facts'] != 'PENDING' or
            inference['state_before_sha256'] != inference['state_after_sha256'] or
            model['state_before_sha256'] != inference['state_before_sha256'] or
            inference['source_text_at_model'] is not False or inference['received_text_at_model'] is not False or
            inference['source_images_privileged_diagnosis_only'] is not True or inference['GPU'] != 'physical1'):
        raise ValueError('Frozen/input boundary differs')
    if model['explicit_generation_kwargs'] != generation_kwargs(run['stage']):
        raise ValueError('Generation protocol differs')
    if any(model['loading_info'].get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Unaccepted loading mismatch')
    assets = read(root/'ASSET_AUDIT.json')
    if assets['before'] != assets['after'] or assets['full_hash_before_load'] is not True or assets['full_hash_after_load'] is not False:
        raise ValueError('Asset mutation/audit scope differs')
    finite(inference); finite(model)
    for k in ('peak_allocated_gib', 'peak_reserved_gib', 'startup_free_gib'):
        if not isinstance(inference[k], (int, float)) or inference[k] <= 0:
            raise ValueError('GPU measurement missing')
    manifest = read(FOLDER/'MANIFEST.json')
    for n, h in dict(manifest['files'], **{'MANIFEST.json': sha(FOLDER/'MANIFEST.json')}).items():
        if sha(local(root/'source_snapshot', n)) != h:
            raise ValueError('Runtime source snapshot differs '+n)
    basic = review_keys(plan) if run['stage'] == 'caption' else []
    if run['stage'] == 'qa':
        authorization = read(root/'QA_AUTHORIZATION.json')
        if authorization['review']['decision'] != 'APPROVE_QA_ONLY':
            raise ValueError('Explicit QA authorization absent')
    return dict(schema=SCHEMA, stage=run['stage'], engineering='VERIFIED', rows=len(actual),
        unique_inferences=len(unique), empty_outputs=sum(r['text_audit']['empty'] for r in actual),
        capacity_stops=sum(r['telemetry']['capacity_stop'] for r in actual),
        empty_basic_outputs=sum(by[k]['text_audit']['empty'] for k in basic),
        factual_usability='HUMAN_REVIEW_PENDING', facts='PENDING', semantic_success=False,
        inference=inference, inference_complete_sha256=sha(root/'COMPLETE.json'))

def review_template(root):
    plan = read(Path(root)/'PLAN.json')
    return dict(caption_complete_sha256=sha(Path(root)/'COMPLETE.json'), decision='PENDING', reviewer='',
        instructions='Review both ORIGINAL privileged images and CLEAN RECEIVED images. Do not approve empty/off-topic/looping outputs. This authorizes QA only, NOT training or expansion.',
        items=[dict(case_id=k, on_topic=None, main_object_supported=None, no_long_loop=None, notes='') for k in review_keys(plan)])

def render(root, result, prefix='data/'):
    root = Path(root); e = html.escape
    run = read(root/'RUN.json'); index = read(root/'prepared/index.json')
    sources = {r['example_id']: r['source_text'] for r in index['examples']}
    original = {(r['example_id'], r['snr_db']): r for r in rows(root/'prepared/rows.jsonl')}
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>V24 BLIP-2 基础诊断</title>',
        '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 18px}article{border:1px solid #ccc;padding:14px;margin:16px 0}img{max-width:350px;max-height:300px}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}.warn{color:#a30}</style>',
        '<h1>V24：BLIP-2 基础看图校准</h1><p class="warn">PRIVATE · 两张复用开发图 · 无训练 · 不是互促成功结果。原图仅为特权诊断，真实接收组没有原图/原文输入。</p>',
        '<p>clean = 没有额外信道噪声，不等于原图，也不是 20 dB。0/−5 是名义 AWGN SNR；负 SNR 为原父模型分布外推理。</p>',
        '<p>只生成短描述/短回答；错图和灰图相同输入复用，不能当独立试验。输出全文、空文本与循环均保留。</p>',
        '<pre>'+e(json.dumps(result, ensure_ascii=False, indent=2))+'</pre>']
    for r in rows(root/'rows.jsonl'):
        snr = 'clean' if r['snr_db'] is None else str(r['snr_db'])+' dB'
        label = '原图：特权诊断' if r['privileged'] else r['control']
        parts.append('<article><h2>'+e(r['example_id']+' / '+snr+' / '+label+' / '+r['task'])+'</h2>')
        parts.append('<img src="'+e(prefix+r['image_file'], quote=True)+'" alt="实际模型输入">')
        parts.append('<p>提示：'+e(str(r['prompt']))+'；实际生成复用：'+str(r['reused_inference'])+'</p>')
        parts.append('<h3>BLIP-2 实际输出</h3><pre>'+e(r['output_text'] or '[空输出]')+'</pre>')
        parts.append('<details><summary>评价参照：原文与旧接收文字（没有喂给本次模型）</summary><pre>'+e(sources[r['example_id']])+'</pre>')
        if not r['privileged']:
            parts.append('<pre>'+e(original[r['example_id'], r['snr_db']]['received_caption'])+'</pre>')
        parts.append('</details><details><summary>完整 token / EOS / 数值与耗时</summary><pre>'+e(json.dumps(r['telemetry'], ensure_ascii=False, indent=2))+'</pre></details></article>')
    parts.append('</html>')
    return '\n'.join(parts)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root'); a = p.parse_args()
    source_check(); print(json.dumps(verify(a.root), ensure_ascii=False, indent=2))
