"""Prepare immutable offline inputs; no Torch, training, or server dependencies."""
import argparse
from pathlib import Path
import time

from core import (FOLDER, Source, cases_from_sources, digest, file_hash, fresh,
                  html_table, jsonlines, load, save, select_audit, summarize)


def prepare(v11_path, v12_path, output):
    started = time.monotonic()
    protocol = load(FOLDER / 'protocol.json')
    a, b = Source(v11_path), Source(v12_path)
    try:
        cases, references = cases_from_sources(a, b, protocol['expected_examples'])
    finally:
        a.close()
        b.close()
    selected = select_audit(references, protocol['audit_examples'], protocol['seed'])
    root = fresh(output)
    save(root / 'protocol.json', protocol)
    jsonlines(root / 'cases.jsonl', cases)
    summary = summarize(cases)
    save(root / 'lexical_summary.json', summary)
    # Source-only inventories: reviewers define facts BEFORE seeing blinded candidates.
    template = dict(schema='v13-source-facts', inventory_reviewer='', examples=[
        dict(example_id=i, reference_sha256=digest(references[i].encode('utf-8')),
             reference=references[i], inventory_complete=False, facts=[])
        for i in selected])
    save(root / 'source_facts_template.json', template)
    save(root / 'audit_selection.json', dict(example_ids=selected, n=len(selected),
         selection='three equal rank-length strata; random sample within strata; seed42',
         expected_candidate_judgements=len(selected) * (len(cases) // len(references))))
    with (root / 'actual_text.html').open('x', encoding='utf-8') as handle:
        handle.write(html_table(cases, summary))
    files = {p.name: file_hash(p) for p in sorted(root.iterdir()) if p.is_file()}
    manifest = dict(schema='v13-prepared', status='PREPARED_LEXICAL_ONLY',
                    semantic_metrics='PENDING; no fact or BERTScore scores yet',
                    digital='NOT_RUN', historical_quality='UNCHANGED',
                    n_examples=len(references), n_cases=len(cases),
                    protocol_sha256=file_hash(FOLDER / 'protocol.json'),
                    sources=[dict(identity=s.identity, consumed_members=s.receipts) for s in (a, b)],
                    source_code={p.name: file_hash(p) for p in sorted(FOLDER.glob('*.py'))},
                    files=files, elapsed_seconds=time.monotonic() - started)
    save(root / 'PREPARED.json', manifest)
    print('PREPARED {} examples / {} cases; {:.1f}s; no semantic scores, no training'.format(
        len(references), len(cases), manifest['elapsed_seconds']), flush=True)
    print(root, flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'verify'])
    parser.add_argument('--v11-records')
    parser.add_argument('--v12-records')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        if not args.v11_records or not args.v12_records:
            parser.error('Both explicit V11 and V12 records roots or tar archives are required')
        if Path(args.output).exists():
            raise FileExistsError('Existing output preserved; use attempt2')
        prepare(args.v11_records, args.v12_records, args.output)
    else:
        from core import verify_prepared
        manifest, cases = verify_prepared(args.output)
        print('Verified frozen inputs: {} cases; status={}'.format(len(cases), manifest['status']))


if __name__ == '__main__':
    main()
