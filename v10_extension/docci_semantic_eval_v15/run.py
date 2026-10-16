"""Prepare/verify semantic evaluation without models, GPU, installs or training."""
import argparse
import html
from pathlib import Path
import random
import statistics

from common import (FOLDER, fresh, inside, legacy, matrix, read, rows, seal, sha,
                    stats, text_sha, verify_seal, verify_source, write, write_rows)
from challenge import cases as challenge_cases


def v13_cases(root):
    root = Path(root).resolve(strict=True)
    manifest = read(root / 'PREPARED.json')
    if manifest.get('schema') != 'v13-prepared' or manifest.get('n_examples') != 100 or manifest.get('n_cases') != 1100:
        raise ValueError('Need full original V13 preparation, not smoke or altered cohort')
    for name in manifest['files']:
        inside(root, name)
    for name in manifest['source_code']:
        inside(FOLDER.parent / 'docci_semantic_eval_v13', name)
    _, cases = legacy().verify_prepared(root)
    groups, refs = matrix(cases, 100)
    expected = {a + '/' + c for a in ('V11_g20480', 'V12_CE', 'V12_CE_KD') for c in ('matched', 'shuffled', 'zero')}
    expected |= {'native_BART/matched', 'unrelated_description/deranged'}
    if set(groups) != expected:
        raise ValueError('Wrong V13 group matrix')
    for case in cases:
        if case['arm'].startswith(('V11_', 'V12_')):
            r = case['communication_resources']
            if r['payload_complex_symbols'] != 9216 or r['analog_repeats'] != 1 or r['raw_utf8_wire_bytes'] != 0:
                raise ValueError('Changed historical transmission resources')
    chosen = read(root / 'audit_selection.json')['example_ids']
    if len(chosen) != 30 or len(set(chosen)) != 30 or not set(chosen) <= set(refs):
        raise ValueError('Wrong original human audit selection')
    return cases, chosen, dict(kind='V13_prepared', manifest_sha256=sha(root / 'PREPARED.json'),
                              cases_sha256=sha(root / 'cases.jsonl'), scope='Pinned V13 preparation and lexical matrix verified')


def v14_cases(root, refs):
    """Read a finished, extracted V14 main/calibration root; NEVER import its code."""
    root = Path(root).resolve(strict=True)
    manifest = read(root / 'LIGHT_MANIFEST.json')
    done = read(root / 'RUN_FINISHED.json')
    if sha(root / 'LIGHT_MANIFEST.json') != done['manifest_sha256']:
        raise ValueError('Unfinished/changed V14 manifest')
    for name, expected in manifest.items():
        if sha(inside(root, name)) != expected:
            raise ValueError('V14 light file changed: ' + name)
    design = read(root / 'design.json')
    if design['task'] not in ('text', 'both') or design['mode'] not in ('train', 'calibrate'):
        raise ValueError('V14 completed text main/calibration required')
    arms = ['historical_reference'] if design['mode'] == 'calibrate' else design['text_arms']
    folders = ['text/native_validation'] + ['text/' + arm + '/validation' for arm in arms]
    result = []
    lex = legacy().lexical
    for folder in folders:
        p = inside(root, folder)
        m, e = read(p / 'matrix.json'), read(p / 'EVAL_COMPLETE.json')
        rr = rows(p / 'rows.jsonl')
        native = folder == 'text/native_validation'
        controls = ['native'] if native else ['matched', 'shuffled', 'zero']
        if (m['example_ids'] != sorted(refs) or m['controls'] != controls
                or m['strategies'] != ['greedy', 'deployment'] or m['evaluation_split'] != 'qual_dev'
                or m['task'] != 'text' or m['cohort'] != 'validation'
                or m['native_privileged_NOT_transmission'] != native):
            raise ValueError('V14 text matrix/cohort/strategies differ')
        if e['rows_sha256'] != sha(p / 'rows.jsonl') or e['matrix_sha256'] != sha(p / 'matrix.json') or e['rows'] != len(rr):
            raise ValueError('V14 evaluation checksum/count differs')
        if not native and (m['geometry']['payload_complex_symbols'] != 9216 or m['geometry']['analog_repeats'] != 1):
            raise ValueError('V14 changed text payload')
        expected = {(i, c, s) for i in refs for c in controls for s in ('greedy', 'deployment')}
        for r in rr:
            key = (r['example_id'], r['control'], r['strategy'])
            if key not in expected:
                raise ValueError('Duplicate/unknown V14 case')
            expected.remove(key)
            if (r['condition'] != 'ideal_clean' or r['snr_db'] is not None or r['official_split'] != 'qual_dev'
                    or r['cohort'] != 'validation' or r['task'] != 'text' or r['native'] != native
                    or r['input_truncated'] or r['source_text'] != refs[r['example_id']]
                    or r['source_text_sha256'] != text_sha(r['source_text'])):
                raise ValueError('V14 source/channel/split differs')
            metric = lex(r['source_text'], r['received_caption'])
            if abs(metric['normalized_WER'] - r['normalized_wer']) > 1e-12:
                raise ValueError('V14 string WER differs')
            arm = 'V14_' + ('native' if native else folder.split('/')[1]) + '_' + r['strategy']
            control = 'matched' if native else r['control']
            result.append(dict(case_id=arm + '/' + control + '/' + r['example_id'], arm=arm,
                control=control, example_id=r['example_id'], reference=r['source_text'], candidate=r['received_caption'],
                reference_sha256=r['source_text_sha256'], candidate_sha256=text_sha(r['received_caption']),
                lexical=metric, scope='privileged no-compression diagnostic' if native else 'saved clean continuous receiver',
                communication_resources=None if native else dict(payload_complex_symbols=9216, analog_repeats=1, raw_utf8_wire_bytes=0)))
        if expected:
            raise ValueError('Incomplete V14 matrix')
    return result, dict(kind='V14_extracted_light_records', manifest_sha256=sha(root / 'LIGHT_MANIFEST.json'),
         scope='All listed light-file hashes, complete text row matrices and WER checked; NOT neural inference/frame-array/history reexecution')


def split_audit(chosen, refs):
    ranked = sorted(chosen, key=lambda i: (len(legacy().words(refs[i])), i))
    dev, confirm = [], []
    rng = random.Random(15042)
    for j in range(3):
        group = ranked[j * 10:(j + 1) * 10]
        rng.shuffle(group)
        dev += group[:5]
        confirm += group[5:]
    return dict(development=sorted(dev), confirmation=sorted(confirm))


def render(cases, summary):
    e = html.escape
    parts = ['<!doctype html><meta charset="utf-8"><title>V15 文本语义核查</title>',
        '<style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:9px;text-align:left;vertical-align:top}td{white-space:pre-wrap}pre{white-space:pre-wrap}summary{cursor:pointer}h2{margin-top:30px}</style>',
        '<h1>V15：语义相似不等于事实正确</h1><p>只读旧输出；无噪声，SNR不适用；没有新训练。人工事实审核尚未完成，旧QUALITY_STOP不变；数字基线NOT_RUN。原生BART不是等资源上界。</p>',
        '<p><a href="source_review.html">先做源事实清单</a> · <a href="source_facts_template.json">源事实JSON模板</a> · <a href="metric_cases.jsonl">可选评价器完整输入（含合成挑战）</a></p>',
        '<h2>逐字指标（不是语义正确率）</h2><table><tr><th>组别</th><th>n</th><th>WER</th><th>全文一致</th><th>相邻重复率</th></tr>']
    for name, data in summary.items():
        parts.append('<tr><td>{}</td><td>{}</td><td>{:.2%}</td><td>{}</td><td>{:.2%}</td></tr>'.format(
            e(name), data['n'], data['WER']['mean'], data['exact'], data['adjacent_repeat']))
    parts.append('</table><p>不要把1100条输出当作1100个独立样本；同一原文的多组输出相互配对。</p>')
    by_id = {}
    for c in cases:
        by_id.setdefault(c['example_id'], []).append(c)
    for eid, rr in sorted(by_id.items()):
        parts.append('<details><summary>' + e(eid) + '：完整原文与恢复</summary><p>' + e(rr[0]['reference']) + '</p><table>')
        for c in rr:
            parts.append('<tr><td>' + e(c['arm'] + '/' + c['control']) + '</td><td>' + e(c['candidate']) + '</td></tr>')
        parts.append('</table></details>')
    return '\n'.join(parts)


def source_page(template):
    # No candidate text or model key in this source-first page.
    e = html.escape
    parts = ['<!doctype html><meta charset="utf-8"><title>仅看原文：事实清单</title>',
             '<style>body{font:17px system-ui;max-width:1050px;margin:35px auto}pre{white-space:pre-wrap}article{border-bottom:1px solid #aaa;padding:15px}</style>',
             '<h1>第一阶段：只看原文定义事实</h1><p>请先填写同目录source_facts_template.json，另存为source_facts_completed.json。不要先看恢复输出。每条事实应是一个能单独判断的断言，附原文精确引句；颜色、数量、关系和否定均需覆盖。不能自动生成“已审核”标签。</p>']
    for r in template['examples']:
        parts.append('<article><h2>' + e(r['example_id']) + '</h2><p>' + e(r['reference']) + '</p></article>')
    return '\n'.join(parts)


def prepare(prepared, output, v14_root=None):
    verify_source()
    cases, chosen, receipt = v13_cases(prepared)
    refs = {c['example_id']: c['reference'] for c in cases}
    sources = [receipt]
    if v14_root:
        more, receipt = v14_cases(v14_root, refs)
        cases += more
        sources.append(receipt)
    cases = sorted(cases, key=lambda r: r['case_id'])
    groups, refs = matrix(cases, 100)
    selection = dict(example_ids=chosen, **split_audit(chosen, refs))
    summary = {g: dict(n=len(rr), WER=stats([c['lexical']['normalized_WER'] for c in rr]),
        exact=sum(c['lexical']['exact_text'] for c in rr),
        adjacent_repeat=statistics.mean(c['lexical']['adjacent_repeat'] for c in rr)) for g, rr in sorted(groups.items())}
    root = fresh(output)
    write(root / 'protocol.json', read(FOLDER / 'protocol.json'))
    write_rows(root / 'cases.jsonl', cases)
    write(root / 'audit_selection.json', selection)
    template = dict(schema='v15-source-facts', inventory_reviewer='', examples=[dict(
        example_id=i, reference=refs[i], reference_sha256=text_sha(refs[i]), inventory_complete=False, facts=[]) for i in chosen])
    write(root / 'source_facts_template.json', template)
    write(root / 'lexical_summary.json', summary)
    challenge = challenge_cases()
    write_rows(root / 'challenge.jsonl', challenge)
    write_rows(root / 'metric_cases.jsonl', cases + challenge)
    (root / 'index.html').write_text(render(cases, summary), encoding='utf-8')
    (root / 'source_review.html').write_text(source_page(template), encoding='utf-8')
    seal(root, 'PREPARED.json', dict(schema='v15-prepared', n_descriptions=100, n_cases=len(cases),
        n_challenge=len(challenge), sources=sources, original_v13_manifest_sha256=sources[0]['manifest_sha256'],
        original_v13_protocol_sha256=read(Path(prepared) / 'PREPARED.json')['protocol_sha256'],
        human='PENDING', CAPTURE='NOT_RUN', AlignScore='NOT_RUN', historical_quality='UNCHANGED',
        digital='NOT_RUN', training='NONE', expected_human_outputs=30 * len(groups)))
    print('V15_PREPARED {} descriptions / {} outputs + {} synthetic challenge; no training'.format(100, len(cases), len(challenge)), flush=True)
    return root


def verify_prepared(root):
    m = verify_seal(root, 'PREPARED.json')
    cases = rows(Path(root) / 'cases.jsonl')
    groups, refs = matrix(cases, 100)
    selection = read(Path(root) / 'audit_selection.json')
    if (m['n_cases'] != len(cases) or selection != dict(example_ids=selection['example_ids'], **split_audit(selection['example_ids'], refs))
            or read(Path(root) / 'protocol.json') != read(FOLDER / 'protocol.json')):
        raise ValueError('Frozen design/matrix changed')
    if rows(Path(root) / 'metric_cases.jsonl') != cases + challenge_cases() or rows(Path(root) / 'challenge.jsonl') != challenge_cases():
        raise ValueError('Challenge/input matrix differs')
    return m, cases, groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'verify', 'check'])
    p.add_argument('--prepared', help='original V13 preparation for prepare; V15 output for verify')
    p.add_argument('--v14-root', help='optional extracted finished V14 main/calibration root')
    p.add_argument('--output')
    a = p.parse_args()
    if a.action == 'check':
        verify_source()
        print('V15 source/dependencies verified. No model, GPU, downloads or training.')
    elif a.action == 'prepare':
        if not a.prepared or not a.output:
            p.error('prepare requires --prepared and --output')
        prepare(a.prepared, a.output, a.v14_root)
    else:
        m, _, _ = verify_prepared(a.prepared)
        print('Verified V15 {} cases; human={}; digital={}'.format(m['n_cases'], m['human'], m['digital']))


if __name__ == '__main__':
    main()
