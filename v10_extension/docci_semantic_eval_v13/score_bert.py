"""Optional, offline-only independent Longformer BERTScore. No server installs."""
import argparse
import importlib.metadata
import math
import os
from pathlib import Path
import statistics
import time

from core import FOLDER, file_hash, fresh, interval, jsonlines, load, save, verify_prepared


def check_lengths(cases, tokenizer, maximum):
    """Use the same slow RoBERTa-family encoding convention as BERTScore 0.3.13."""
    lengths = {}
    for case in cases:
        for key in ('reference', 'candidate'):
            text = case[key].strip()
            if text not in lengths:
                lengths[text] = len(tokenizer.encode(text, add_special_tokens=True,
                     add_prefix_space=True, truncation=False))
            if lengths[text] > maximum:
                raise ValueError('FULL-TEXT overflow {} {}: {}>{}; no truncation, no score'.format(
                    case['case_id'], key, lengths[text], maximum))
    return dict(unique_texts=len(lengths), max_encoded_tokens=max(lengths.values()),
                limit=maximum, policy='complete texts; zero omitted/truncated samples')


def score(prepared, model_path, output, smoke=False, batch_size=1, max_minutes=120):
    if batch_size < 1 or max_minutes <= 0:
        raise ValueError('Positive batch size and time budget required')
    manifest, all_cases = verify_prepared(prepared)
    protocol = load(FOLDER / 'protocol.json')['bertscore']
    model_path = Path(model_path).resolve(strict=True)
    receipt = load(model_path / 'DOWNLOAD_RECEIPT.json')
    if receipt['model_id'] != protocol['model_id'] or not receipt['resolved_revision']:
        raise ValueError('Need the approved independent evaluator download receipt')
    for name, expected in receipt['files'].items():
        if file_hash(model_path / name) != expected:
            raise ValueError('Evaluator bytes changed: ' + name)
    config = load(model_path / 'config.json')
    if config['model_type'] != 'longformer' or config['max_position_embeddings'] < 4098:
        raise ValueError('Wrong independent long-context evaluator')
    # Enforce offline before importing/loading HF. No dependency installation here.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if importlib.metadata.version('bert-score') != protocol['package_version']:
        raise ValueError('Use isolated local bert-score==0.3.13 environment')
    from bert_score import BERTScorer
    scorer = BERTScorer(model_type=str(model_path), num_layers=protocol['num_layers'],
                       lang='en', idf=False, rescale_with_baseline=False,
                       use_fast_tokenizer=False, device='cpu', batch_size=batch_size, nthreads=4)
    audit = check_lengths(all_cases, scorer._tokenizer, protocol['max_tokens_with_special'])
    scorer._tokenizer.model_max_length = protocol['max_tokens_with_special']
    cases = all_cases[:10] if smoke else all_cases
    root = fresh(output)
    started = time.monotonic()
    scores = []
    for offset in range(0, len(cases), batch_size):
        if time.monotonic() - started > max_minutes * 60:
            raise TimeoutError('CPU scoring budget exhausted; preserve partial output, use new attempt')
        batch = cases[offset:offset + batch_size]
        p, r, f = scorer.score([c['candidate'] for c in batch], [c['reference'] for c in batch], batch_size=batch_size)
        for case, precision, recall, f1 in zip(batch, p.tolist(), r.tolist(), f.tolist()):
            if not all(math.isfinite(v) for v in (precision, recall, f1)):
                raise ValueError('Nonfinite similarity; no silent sample skip')
            scores.append(dict(case_id=case['case_id'], example_id=case['example_id'],
                 group=case['arm'] + '/' + case['control'], P=precision, R=recall, F1=f1))
        if offset == 0 or (offset + len(batch)) % 25 == 0:
            print('BERTScore {}/{} elapsed={:.1f}s'.format(len(scores), len(cases), time.monotonic() - started), flush=True)
    groups = {}
    for row in scores:
        groups.setdefault(row['group'], []).append(row)
    summary = {}
    for group, rows in groups.items():
        summary[group] = dict(n=len(rows), **{key: statistics.mean(r[key] for r in rows) for key in ('P', 'R', 'F1')})
        summary[group]['F1_description_bootstrap_CI95'] = interval([r['F1'] for r in rows])
    # Paired deltas use complete identical IDs, not independent-sample uncertainty.
    comparisons = {}
    for left, right in [('V12_CE_KD/matched', 'V12_CE/matched'),
                         ('V12_CE/matched', 'V11_g20480/matched'),
                         ('V12_CE_KD/matched', 'V12_CE_KD/shuffled'),
                         ('V12_CE_KD/matched', 'unrelated_description/deranged')]:
        if left in groups and right in groups:
            a = {r['example_id']: r['F1'] for r in groups[left]}
            b = {r['example_id']: r['F1'] for r in groups[right]}
            if set(a) != set(b):
                raise ValueError('Unpaired similarity cohort')
            delta = [a[i] - b[i] for i in sorted(a)]
            comparisons[left + ' minus ' + right] = dict(mean=statistics.mean(delta), CI95=interval(delta))
    jsonlines(root / 'bertscore_rows.jsonl', scores)
    save(root / 'summary.json', dict(status='SMOKE_ONLY_NOT_FULL_EVALUATION' if smoke else 'AUXILIARY_SIMILARITY_COMPLETE',
         n=len(scores), groups=summary, paired_deltas=comparisons, length_audit=audit,
         model_id=protocol['model_id'], resolved_revision=receipt['resolved_revision'],
         download_receipt_sha256=file_hash(model_path / 'DOWNLOAD_RECEIPT.json'),
         prepared_sha256=file_hash(Path(prepared) / 'PREPARED.json'), protocol_sha256=manifest['protocol_sha256'],
         num_layers=protocol['num_layers'], idf=False, rescaled=False,
         package_versions={p: importlib.metadata.version(p) for p in ('bert-score', 'transformers', 'torch', 'tokenizers')},
         scorer_setting_hash=scorer.hash, device='cpu', elapsed_seconds=time.monotonic() - started,
         facts='PENDING_SEPARATE_HUMAN_AUDIT', digital='NOT_RUN', historical_quality='UNCHANGED',
         interpretation='NOT accuracy percentage; long-context raw score, not comparable numerically to RoBERTa rescaled scores'))
    print('CPU auxiliary scoring finished: ' + str(root), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-minutes', type=float, default=120)
    args = parser.parse_args()
    score(args.prepared, args.model_path, args.output, args.smoke, args.batch_size, args.max_minutes)


if __name__ == '__main__':
    main()
