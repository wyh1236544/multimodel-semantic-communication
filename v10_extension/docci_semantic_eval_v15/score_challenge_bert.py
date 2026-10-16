"""Optional CPU-only scoring of 40 synthetic perturbations using existing V13 assets."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import time

from common import FOLDER, finite, fresh, inside, read, rows, seal, sha, write, write_rows
from run import verify_prepared


def check_lengths(cases, tokenizer):
    lengths = []
    for c in cases:
        for key in ('reference', 'candidate'):
            n = len(tokenizer.encode(c[key].strip(), add_special_tokens=True,
                                     add_prefix_space=True, truncation=False))
            if n > 4096:
                raise ValueError('Full text overflow; no truncation or skipping')
            lengths.append(n)
    return max(lengths)


def score(prepared, model_path, output, max_minutes=20):
    if not finite(max_minutes) or max_minutes <= 0:
        raise ValueError('Positive finite time budget required')
    verify_prepared(prepared)
    cases = rows(Path(prepared) / 'challenge.jsonl')
    protocol = read(FOLDER / 'protocol.json')['bertscore']
    path = Path(model_path).resolve(strict=True)
    receipt = read(path / 'DOWNLOAD_RECEIPT.json')
    if receipt['model_id'] != protocol['model_id'] or receipt['resolved_revision'] != protocol['resolved_revision']:
        raise ValueError('Use exact approved V13 Longformer assets')
    for name, expected in receipt['files'].items():
        if sha(inside(path, name)) != expected:
            raise ValueError('Evaluator asset changed')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if importlib.metadata.version('bert-score') != protocol['package_version']:
        raise ValueError('Use the existing isolated V13 CPU environment')
    from bert_score import BERTScorer
    evaluator = BERTScorer(model_type=str(path), num_layers=7, lang='en', idf=False,
        rescale_with_baseline=False, use_fast_tokenizer=False, device='cpu', batch_size=1, nthreads=4)
    maximum = check_lengths(cases, evaluator._tokenizer)
    evaluator._tokenizer.model_max_length = 4096
    root = fresh(output)
    started, result = time.monotonic(), []
    for i, c in enumerate(cases):
        if time.monotonic() - started > max_minutes * 60:
            raise TimeoutError('Challenge scoring budget exceeded; preserve failure output, no auto retry')
        p, r, f = evaluator.score([c['candidate']], [c['reference']], batch_size=1)
        v = [float(x[0]) for x in (p, r, f)]
        if not all(finite(x) for x in v):
            raise ValueError('Nonfinite synthetic score')
        result.append(dict(case_id=c['case_id'], example_id=c['example_id'], kind=c['control'],
             reference_sha256=c['reference_sha256'], candidate_sha256=c['candidate_sha256'],
             P=v[0], R=v[1], F1=v[2]))
        print('BERT challenge {}/40 elapsed={:.1f}s'.format(i + 1, time.monotonic() - started), flush=True)
    lookup = {(r['example_id'], r['kind']): r for r in result}
    ids = sorted({r['example_id'] for r in result})
    diagnostic = {}
    for kind in ('count', 'color', 'binding', 'spatial', 'negation', 'action', 'omission', 'unrelated'):
        d = [lookup[(i, 'paraphrase')]['F1'] - lookup[(i, kind)]['F1'] for i in ids]
        diagnostic[kind] = dict(scenes=4, paraphrase_higher=sum(x > 0 for x in d), ties=sum(x == 0 for x in d),
             paraphrase_lower=sum(x < 0 for x in d), per_scene_F1_gap=d)
    write_rows(root / 'challenge_bertscore_rows.jsonl', result)
    write(root / 'summary.json', dict(status='SYNTHETIC_EVALUATOR_DIAGNOSTIC_ONLY', n=40,
        model_id=protocol['model_id'], resolved_revision=protocol['resolved_revision'], num_layers=7,
        idf=False, rescaled=False, max_input_tokens=maximum, diagnostic=diagnostic,
        elapsed_seconds=time.monotonic() - started, device='cpu', threads=4,
        asset_receipt_sha256=sha(path / 'DOWNLOAD_RECEIPT.json'),
        versions={k: importlib.metadata.version(k) for k in ('bert-score', 'torch', 'transformers')},
        interpretation='Four hand-authored scenes, NOT recovered texts, population validation or factual accuracy'))
    seal(root, 'COMPLETE.json', dict(status='SYNTHETIC_CHALLENGE_COMPLETE', prepared_sha256=sha(Path(prepared) / 'PREPARED.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('prepared', 'model-path', 'output'):
        p.add_argument('--' + k, required=True)
    p.add_argument('--max-minutes', type=float, default=20)
    a = p.parse_args()
    score(a.prepared, a.model_path, a.output, a.max_minutes)
