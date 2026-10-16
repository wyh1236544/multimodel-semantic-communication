"""Source-first atomic facts; opaque model identity; strict complete human audit."""
import argparse
from pathlib import Path
import secrets
import statistics

from common import (FOLDER, fresh, read, rows, seal, sha, stats, text_sha,
                    verify_seal, write, write_rows)
from run import verify_prepared


def inventories(prepared, facts_path, cases):
    doc = read(facts_path)
    selected = set(read(Path(prepared) / 'audit_selection.json')['example_ids'])
    refs = {r['example_id']: r['reference'] for r in cases}
    if doc.get('schema') not in ('v15-source-facts', 'v13-source-facts') or not doc.get('inventory_reviewer', '').strip():
        raise ValueError('Completed source-first inventory and reviewer required')
    cats = read(FOLDER / 'protocol.json')['fact_categories']
    out = {}
    for r in doc['examples']:
        eid = r['example_id']
        if eid not in selected or eid in out or r['inventory_complete'] is not True:
            raise ValueError('Duplicate/unknown/unreviewed source inventory')
        if r['reference'] != refs[eid] or r['reference_sha256'] != text_sha(refs[eid]) or not r['facts']:
            raise ValueError('Changed source/empty facts')
        ids = set()
        for f in r['facts']:
            if not isinstance(f['fact_id'], str) or not f['fact_id'].strip() or f['fact_id'] in ids:
                raise ValueError('Invalid/duplicate fact ID')
            ids.add(f['fact_id'])
            if f['category'] not in cats or not f['proposition'].strip():
                raise ValueError('Invalid fact category/proposition')
            if not f['reference_quote'] or f['reference_quote'] not in refs[eid]:
                raise ValueError('Fact requires exact source evidence')
            if 'visually_verifiable' in f and type(f['visually_verifiable']) is not bool:
                raise ValueError('Visual metadata must be a boolean; private text facts must not be dropped')
        out[eid] = r
    if set(out) != selected:
        raise ValueError('Source facts must cover all original 30 descriptions')
    return doc, out


def blind(prepared, facts, output):
    _, cases, _ = verify_prepared(prepared)
    doc, sources = inventories(prepared, facts, cases)
    selection = read(Path(prepared) / 'audit_selection.json')
    key, review = [], []
    seen = set()
    for c in cases:
        if c['example_id'] not in sources:
            continue
        bid = secrets.token_hex(12)
        if bid in seen:
            raise RuntimeError('Opaque ID collision; stop before writing')
        seen.add(bid)
        key.append(dict(blind_id=bid, case_id=c['case_id']))
        review.append(dict(blind_id=bid, example_id=c['example_id'], reference=c['reference'],
             candidate=c['candidate'], candidate_sha256=c['candidate_sha256'], reviewer='',
             reference_facts=[dict(f, state='UNREVIEWED', candidate_quote='', note='') for f in sources[c['example_id']]['facts']],
             prediction_inventory_complete=False, prediction_facts=[]))
    root = fresh(output)
    write(root / 'frozen_source_facts.json', doc)
    write_rows(root / 'candidate_review_template.jsonl', sorted(review, key=lambda r: r['blind_id']))
    write_rows(root / 'PRIVATE_KEY_DO_NOT_SHARE.jsonl', key)
    write(root / 'audit_selection.json', selection)
    seal(root, 'AUDIT_READY.json', dict(status='HUMAN_REVIEW_PENDING', n_outputs=len(review),
         prepared_sha256=sha(Path(prepared) / 'PREPARED.json'),
         warning='Give reviewer ONLY candidate template plus instructions, never key, source result pages or prepared model group mappings'))
    print('Blind audit ready: {} candidate outputs; manual judgments PENDING'.format(len(review)), flush=True)


def judge(row, case, source, protocol):
    for k in ('reference', 'candidate', 'candidate_sha256', 'example_id'):
        if row[k] != case[k]:
            raise ValueError('Reviewer changed original text/identity')
    if not row.get('reviewer', '').strip() or row.get('prediction_inventory_complete') is not True:
        raise ValueError('Reviewer and complete prediction inventory required')
    gold = {f['fact_id']: f for f in source['facts']}
    decisions = row['reference_facts']
    if len(decisions) != len(gold) or {f['fact_id'] for f in decisions} != set(gold):
        raise ValueError('Missing/duplicate source facts in judgment')
    for f in decisions:
        if any(f.get(k) != v for k, v in gold[f['fact_id']].items()):
            raise ValueError('Frozen source fact changed')
        if f['state'] not in protocol['reference_fact_states']:
            raise ValueError('Unreviewed/unknown reference state')
        if f['state'] in ('entailed', 'contradicted') and (not f['candidate_quote'] or f['candidate_quote'] not in case['candidate']):
            raise ValueError('Entailment/contradiction needs exact candidate evidence')
    predictions, ids = row['prediction_facts'], set()
    if case['candidate'].strip() and not predictions:
        raise ValueError('Nonempty output requires a real prediction-fact denominator')
    if not case['candidate'].strip() and predictions:
        raise ValueError('Empty output cannot contain prediction assertions')
    for f in predictions:
        if not isinstance(f['fact_id'], str) or not f['fact_id'].strip() or f['fact_id'] in ids:
            raise ValueError('Invalid prediction fact identity')
        ids.add(f['fact_id'])
        if f['state'] not in protocol['prediction_fact_states'] or not f['proposition'].strip():
            raise ValueError('Unreviewed prediction fact')
        if not f['candidate_quote'] or f['candidate_quote'] not in case['candidate']:
            raise ValueError('Prediction assertion needs exact candidate evidence')
        if f['state'] in ('supported', 'contradicted') and (not f.get('reference_quote') or f['reference_quote'] not in case['reference']):
            raise ValueError('Supported/contradicted assertion needs source evidence')
    nc = {s: sum(f['state'] == s for f in decisions) for s in protocol['reference_fact_states']}
    pc = {s: sum(f['state'] == s for f in predictions) for s in protocol['prediction_fact_states']}
    cat = {}
    for category in protocol['fact_categories']:
        rr = [f for f in decisions if f['category'] == category]
        cat[category] = dict(n=len(rr), entailed=sum(f['state'] == 'entailed' for f in rr),
                            contradicted=sum(f['state'] == 'contradicted' for f in rr),
                            not_mentioned=sum(f['state'] == 'not_mentioned' for f in rr),
                            unclear=sum(f['state'] == 'unclear' for f in rr))
    return dict(case_id=case['case_id'], group=case['arm'] + '/' + case['control'],
         example_id=case['example_id'], reference_counts=nc, prediction_counts=pc,
         source_facts=len(decisions), predicted_facts=len(predictions), category_counts=cat,
         fact_retention=nc['entailed'] / len(decisions),
         unsupported_prediction_rate=(pc['unsupported'] + pc['contradicted']) / len(predictions) if predictions else None)


def aggregate(rr):
    unknown = [r for r in rr if r['unsupported_prediction_rate'] is None]
    known = [r['unsupported_prediction_rate'] for r in rr if r['unsupported_prediction_rate'] is not None]
    n, m = sum(r['source_facts'] for r in rr), sum(r['predicted_facts'] for r in rr)
    cats = {}
    for cat in read(FOLDER / 'protocol.json')['fact_categories']:
        counts = {k: sum(r['category_counts'][cat][k] for r in rr) for k in ('n', 'entailed', 'contradicted', 'not_mentioned', 'unclear')}
        cats[cat] = dict(counts, micro_retention=counts['entailed'] / counts['n'] if counts['n'] else None)
    return dict(fact_retention=stats([r['fact_retention'] for r in rr]),
        unsupported_prediction_rate=stats(known) if known else None,
        unsupported_defined_descriptions=len(known), empty_output_descriptions=len(unknown),
        micro_retention=sum(r['reference_counts']['entailed'] for r in rr) / n,
        micro_unsupported_rate=sum(r['prediction_counts']['unsupported'] + r['prediction_counts']['contradicted'] for r in rr) / m if m else None,
        reference_unclear=sum(r['reference_counts']['unclear'] for r in rr),
        prediction_unclear=sum(r['prediction_counts']['unclear'] for r in rr), categories=cats)


def score(prepared, audit_root, judgements, output):
    _, cases, _ = verify_prepared(prepared)
    ar = Path(audit_root)
    ready = verify_seal(ar, 'AUDIT_READY.json')
    if ready['prepared_sha256'] != sha(Path(prepared) / 'PREPARED.json'):
        raise ValueError('Audit belongs to another preparation')
    _, sources = inventories(prepared, ar / 'frozen_source_facts.json', cases)
    lookup = {c['case_id']: c for c in cases if c['example_id'] in sources}
    key_rows = rows(ar / 'PRIVATE_KEY_DO_NOT_SHARE.jsonl')
    key = {r['blind_id']: r['case_id'] for r in key_rows}
    if len(key) != len(key_rows) or len(key) != len(lookup) or set(key.values()) != set(lookup):
        raise ValueError('Invalid blind key matrix')
    protocol = read(FOLDER / 'protocol.json')
    results, seen = [], set()
    for row in rows(judgements):
        bid = row['blind_id']
        if bid not in key or bid in seen:
            raise ValueError('Duplicate/unknown blind review')
        seen.add(bid)
        case = lookup[key[bid]]
        results.append(judge(row, case, sources[case['example_id']], protocol))
    if seen != set(key):
        raise ValueError('Incomplete human matrix; no selectively reported scores')
    selection = read(ar / 'audit_selection.json')
    reports = {}
    for split in ('development', 'confirmation'):
        grouped = {}
        for r in results:
            if r['example_id'] in selection[split]:
                grouped.setdefault(r['group'], []).append(r)
        reports[split] = {g: aggregate(rr) for g, rr in grouped.items()}
    root = fresh(output)
    write_rows(root / 'fact_scores.jsonl', results)
    write(root / 'summary.json', dict(status='COMPLETE_HUMAN_AUDIT_NOT_MODEL_QUALITY_PASS',
        splits=reports, n_outputs=len(results), judgments_sha256=sha(judgements),
        prepared_sha256=sha(Path(prepared) / 'PREPARED.json'), audit_ready_sha256=sha(ar / 'AUDIT_READY.json'),
        historical_quality='UNCHANGED', digital='NOT_RUN', automatic_metric_selection='NONE',
        interpretation='Source-unsupported does not mean visually false. 15+15 reused descriptions, not independent model test; single-rater labels not machine-certified.'))
    seal(root, 'COMPLETE.json', dict(status='HUMAN_FACT_AUDIT_COMPLETE_NOT_QUALITY_PASS'))
    print('Human fact audit complete; historical quality unchanged', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    b = sub.add_parser('blind')
    for k in ('prepared', 'facts', 'output'):
        b.add_argument('--' + k, required=True)
    s = sub.add_parser('score')
    for k in ('prepared', 'audit', 'judgements', 'output'):
        s.add_argument('--' + k, required=True)
    a = p.parse_args()
    if a.action == 'blind':
        blind(a.prepared, a.facts, a.output)
    else:
        score(a.prepared, a.audit, a.judgements, a.output)


if __name__ == '__main__':
    main()
