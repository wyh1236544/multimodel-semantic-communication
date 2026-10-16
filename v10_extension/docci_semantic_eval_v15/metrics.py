"""Strict historical BERTScore reuse and optional frozen external score imports."""
import argparse
import html
import math
from pathlib import Path
import random
import statistics

from common import (FOLDER, anchor_position, finite, fresh, read, rows, seal, sha,
                    stats, verify_seal, write, write_rows)
from run import verify_prepared


def ratio_interval(a, b, c, repeats=2000):
    rng = random.Random(42)
    valid, undefined = [], 0
    for _ in range(repeats):
        ix = [rng.randrange(len(a)) for _ in a]
        value = anchor_position([a[i] for i in ix], [b[i] for i in ix], [c[i] for i in ix])['value']
        if value is None:
            undefined += 1
        else:
            valid.append(value)
    # Do not silently omit unstable resamples and still report a CI.
    if undefined:
        return dict(CI95=None, undefined_resamples=undefined, replicates=repeats)
    valid.sort()
    def q(p):
        x = (len(valid) - 1) * p
        lo, hi = math.floor(x), math.ceil(x)
        return valid[lo] + (valid[hi] - valid[lo]) * (x - lo)
    return dict(CI95=[q(.025), q(.975)], undefined_resamples=0, replicates=repeats)


def import_bert(prepared, scores, summary_path, output):
    m, cases, groups = verify_prepared(prepared)
    settings = read(FOLDER / 'protocol.json')['bertscore']
    summary, rr = read(summary_path), rows(scores)
    if summary['status'] != 'AUXILIARY_SIMILARITY_COMPLETE' or summary['n'] != 1100:
        raise ValueError('Need full 1100-row V13 scores, not smoke')
    if (summary['prepared_sha256'] != m['original_v13_manifest_sha256']
            or summary['protocol_sha256'] != m['original_v13_protocol_sha256']):
        raise ValueError('Scores belong to different original prepared inputs')
    for key in ('model_id', 'resolved_revision', 'num_layers', 'idf', 'rescaled'):
        if summary[key] != settings[key]:
            raise ValueError('Frozen evaluator setting changed: ' + key)
    if summary['package_versions']['bert-score'] != settings['package_version']:
        raise ValueError('Wrong BERTScore package version')
    expected = {c['case_id']: c for c in cases if not c['arm'].startswith('V14_')}
    if len(rr) != len(expected) or len(expected) != 1100:
        raise ValueError('Score/input count differs')
    values, by_group = {}, {}
    for r in rr:
        cid = r['case_id']
        if cid not in expected or cid in values:
            raise ValueError('Duplicate/unknown BERTScore case')
        c = expected[cid]
        if r['example_id'] != c['example_id'] or r['group'] != c['arm'] + '/' + c['control']:
            raise ValueError('BERTScore identity differs')
        if not all(finite(r[k]) and -1 <= r[k] <= 1.000001 for k in ('P', 'R', 'F1')):
            raise ValueError('Nonfinite/out-of-range raw score')
        values[cid] = r
        by_group.setdefault(r['group'], []).append(r)
    if set(summary['groups']) != set(by_group):
        raise ValueError('Score summary group matrix differs')
    aggregates = {}
    for group, items in sorted(by_group.items()):
        s = summary['groups'][group]
        if s['n'] != len(items) or any(abs(s[k] - statistics.mean(r[k] for r in items)) > 1e-12 for k in ('P', 'R', 'F1')):
            raise ValueError('Original group means do not match rows')
        aggregates[group] = {k: stats([r[k] for r in items]) for k in ('P', 'R', 'F1')}
    comparisons, anchors = {}, {}
    for arm in ('V11_g20480', 'V12_CE', 'V12_CE_KD'):
        ids = sorted(r['example_id'] for r in by_group[arm + '/matched'])
        a = [values[arm + '/matched/' + i]['F1'] for i in ids]
        b = [values[arm + '/shuffled/' + i]['F1'] for i in ids]
        n = [values['native_BART/matched/' + i]['F1'] for i in ids]
        comparisons[arm + ': matched minus shuffled'] = stats([x - y for x, y in zip(a, b)])
        anchors[arm] = dict(anchor_position(a, b, n), **ratio_interval(a, b, n))
    result = dict(status='HISTORICAL_BERT_REANALYSIS_COMPLETE_NOT_MODEL_QUALITY_PASS', groups=aggregates,
        matched_minus_shuffled_F1=comparisons, relative_anchor_positions=anchors,
        source_summary_sha256=sha(summary_path), source_rows_sha256=sha(scores),
        V15_prepared_sha256=sha(Path(prepared) / 'PREPARED.json'),
        V14_BERTScore='NOT_MEASURED' if any(c['arm'].startswith('V14_') for c in cases) else 'NOT_INCLUDED',
        model_inference='NOT_RERUN; scalar reanalysis only', rescaled=False,
        human_facts='PENDING', digital='NOT_RUN', historical_quality='UNCHANGED',
        interpretation='Relative anchor position is NOT accuracy or percent semantic recovery; no clipping. Native is privileged.')
    root = fresh(output)
    write(root / 'summary.json', result)
    write_rows(root / 'imported_bertscore_rows.jsonl', rr)
    write(root / 'original_summary.json', summary)
    e = html.escape
    lines = ['<!doctype html><meta charset="utf-8"><title>V15 已有分数重分析</title>',
        '<style>body{font:18px system-ui;max-width:1250px;margin:35px auto}table{border-collapse:collapse;width:100%}td,th{padding:12px;border:1px solid #aaa;text-align:left}</style>',
        '<h1>文本确实带回了相关信息，但事实质量仍待审核</h1>',
        '<p>同100篇DOCCI qual_dev，理想无噪声，SNR不适用。复用已计算1100条分数；未重新训练或运行评价模型。</p>',
        '<table><tr><th>模型/控制</th><th>原始F1</th><th>95%描述级区间</th><th>WER</th></tr>']
    for g, v in aggregates.items():
        lo, hi = v['F1']['description_bootstrap_CI95']
        wer = statistics.mean(c['lexical']['normalized_WER'] for c in groups[g])
        lines.append('<tr><td>{}</td><td>{:.4f}</td><td>[{:.4f}, {:.4f}]</td><td>{:.2%}</td></tr>'.format(e(g), v['F1']['mean'], lo, hi, wer))
    lines.append('</table><h2>相对参照位置：不是准确率</h2><p>按“正常−错配”除以“原生−错配”计算。数值不截到0—1；参照间距过小则不定义。</p>')
    for arm, v in anchors.items():
        lines.append('<p>{}: {}；95%配对区间 {}。</p>'.format(e(arm), e(str(v['value'])), e(str(v['CI95']))))
    lines.append('<p>人工事实PENDING；CAPTURE/AlignScore未运行；数字基线NOT_RUN；历史QUALITY_STOP不变。CI只反映固定模型、当前描述的抽样变化，不含训练随机性。</p>')
    (root / 'index.html').write_text('\n'.join(lines), encoding='utf-8')
    seal(root, 'COMPLETE.json', dict(status=result['status'], prepared_sha256=result['V15_prepared_sha256']))
    print('Historical BERTScore reanalysis complete; human facts still PENDING', flush=True)
    return result


def freeze_metric(prepared, spec_path, output):
    verify_prepared(prepared)
    spec = read(spec_path)
    if spec['metric'] not in ('CAPTURE', 'AlignScore', 'BERTScore_challenge'):
        raise ValueError('Unknown optional metric')
    if not spec.get('evaluator_revision', '').strip() or not isinstance(spec.get('settings'), dict) or not spec.get('assets'):
        raise ValueError('Explicit pinned evaluator revision/settings/assets required')
    # Freeze actual local bytes; this does not certify provenance or execute them.
    assets = []
    for asset in spec['assets']:
        p = Path(asset['path']).resolve(strict=True)
        value = sha(p)
        if value != asset['sha256']:
            raise ValueError('Local evaluator asset hash differs')
        assets.append(dict(name=p.name, sha256=value, bytes=p.stat().st_size))
    source = Path(prepared) / 'metric_cases.jsonl'
    cases = rows(source)
    root = fresh(output)
    write_rows(root / 'requests.jsonl', cases)
    write(root / 'plan.json', dict(metric=spec['metric'], evaluator_revision=spec['evaluator_revision'],
        settings=spec['settings'], assets=assets, n=len(cases), prepared_sha256=sha(Path(prepared) / 'PREPARED.json'),
        requests_sha256=sha(root / 'requests.jsonl'), scope='COMPLETE_FIXED_INPUTS_NO_MODEL_EXECUTION',
        warning='Asset hashes/revision are declared by operator, not independent verification of metric execution or official equivalence'))
    seal(root, 'PLAN_FROZEN.json', dict(status='FROZEN_BEFORE_OPTIONAL_SCORING_NOT_SCORED'))
    print('Metric plan frozen; NO metric runtime installed or executed', flush=True)


def import_metric(plan_root, score_rows, output):
    verify_seal(plan_root, 'PLAN_FROZEN.json')
    p = Path(plan_root)
    plan = read(p / 'plan.json')
    cases = {c['case_id']: c for c in rows(p / 'requests.jsonl')}
    expected_plan = sha(p / 'plan.json')
    rr = rows(score_rows)
    scored = {}
    for r in rr:
        cid = r['case_id']
        if cid not in cases or cid in scored:
            raise ValueError('Duplicate/unknown external metric output')
        c = cases[cid]
        if r['plan_sha256'] != expected_plan or r['reference_sha256'] != c['reference_sha256'] or r['candidate_sha256'] != c['candidate_sha256']:
            raise ValueError('External score plan/text identity changed')
        if not finite(r['score']) or not isinstance(r.get('details'), dict):
            raise ValueError('Nonfinite/missing score/details; no skipped parse failures')
        scored[cid] = r['score']
    if set(scored) != set(cases):
        raise ValueError('Incomplete external score matrix, including failures/empty candidates/challenge')
    groups, challenges = {}, {}
    for cid, c in cases.items():
        if c['arm'] == 'challenge':
            challenges.setdefault(c['example_id'], {})[c['control']] = scored[cid]
        else:
            groups.setdefault(c['arm'] + '/' + c['control'], []).append(scored[cid])
    diagnostic = {}
    for kind in ('count', 'color', 'binding', 'spatial', 'negation', 'action', 'omission', 'unrelated'):
        differences = [d['paraphrase'] - d[kind] for d in challenges.values()]
        diagnostic[kind] = dict(scenes=len(differences), paraphrase_higher=sum(x > 0 for x in differences),
            ties=sum(x == 0 for x in differences), paraphrase_lower=sum(x < 0 for x in differences),
            per_scene_score_gap=differences, interpretation='Direction diagnostic only; four synthetic scenes, not factual accuracy')
    root = fresh(output)
    write_rows(root / 'metric_rows.jsonl', rr)
    write(root / 'summary.json', dict(metric=plan['metric'], groups={g: stats(v) for g, v in groups.items()},
        challenge_diagnostics=diagnostic, plan_sha256=expected_plan, score_file_sha256=sha(score_rows),
        status='EXTERNAL_SCALARS_VALIDATED_NOT_INDEPENDENT_RUNTIME_VERIFIED',
        factual_accuracy='NOT_ESTABLISHED', human='PENDING', historical_quality='UNCHANGED'))
    seal(root, 'COMPLETE.json', dict(status='EXTERNAL_SCORE_IMPORT_COMPLETE_NOT_QUALITY_PASS'))
    print('Optional scalar scores imported; no calibrated fact/contradiction rate inferred', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    b = sub.add_parser('import-bert')
    for key in ('prepared', 'scores', 'summary', 'output'):
        b.add_argument('--' + key, required=True)
    f = sub.add_parser('freeze')
    for key in ('prepared', 'spec', 'output'):
        f.add_argument('--' + key, required=True)
    i = sub.add_parser('import')
    for key in ('plan', 'scores', 'output'):
        i.add_argument('--' + key, required=True)
    a = p.parse_args()
    if a.action == 'import-bert':
        import_bert(a.prepared, a.scores, a.summary, a.output)
    elif a.action == 'freeze':
        freeze_metric(a.prepared, a.spec, a.output)
    else:
        import_metric(a.plan, a.scores, a.output)


if __name__ == '__main__':
    main()
