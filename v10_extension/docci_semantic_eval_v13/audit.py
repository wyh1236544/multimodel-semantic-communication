"""Two-phase, model-identity-masked human fact audit. Never invent automatic facts."""
import argparse
from pathlib import Path
import statistics

from core import (FOLDER, digest, encode, file_hash, fresh, interval, jsonlines, load,
                  readlines, save, verify_prepared)


def inventories(prepared, path, cases):
    document = load(path)
    if document.get('schema') != 'v13-source-facts' or not document.get('inventory_reviewer', '').strip():
        raise ValueError('Need named inventory reviewer and source-facts schema')
    selected = set(load(Path(prepared) / 'audit_selection.json')['example_ids'])
    reference = {r['example_id']: r['reference'] for r in cases}
    by_id = {}
    categories = set(load(FOLDER / 'protocol.json')['fact_categories'])
    for row in document['examples']:
        example = row['example_id']
        if example in by_id or example not in selected or row.get('inventory_complete') is not True:
            raise ValueError('Missing, duplicate, incomplete or unselected source inventory')
        if (row['reference_sha256'] != digest(reference[example].encode('utf-8'))
                or row['reference'] != reference[example] or not row['facts']):
            raise ValueError('Reference mismatch or empty fact inventory')
        seen = set()
        for fact in row['facts']:
            key = fact['fact_id']
            if not isinstance(key, str) or not key.strip() or key in seen:
                raise ValueError('Duplicate/empty fact identifier')
            seen.add(key)
            if fact['category'] not in categories or not fact['proposition'].strip():
                raise ValueError('Invalid fact category/proposition')
            if not fact['reference_quote'] or fact['reference_quote'] not in reference[example]:
                raise ValueError('Every reference fact needs an exact supporting source quote')
        by_id[example] = row
    if set(by_id) != selected:
        raise ValueError('Source inventories must cover the full preselected cohort')
    return document, by_id


def blind_id(case):
    return 'B-' + digest(('v13-identity-mask/' + case['case_id']).encode('utf-8'))[:20]


def make_audit(prepared, facts, output):
    _, cases = verify_prepared(prepared)
    document, by_id = inventories(prepared, facts, cases)
    rows, key = [], []
    for case in cases:
        if case['example_id'] not in by_id:
            continue
        source = by_id[case['example_id']]
        bid = blind_id(case)
        rows.append(dict(blind_id=bid, example_id=case['example_id'],
                         candidate=case['candidate'], candidate_sha256=case['candidate_sha256'],
                         reference=case['reference'], reviewer='',
                         reference_facts=[dict(**fact, state='UNREVIEWED', candidate_quote='', note='') for fact in source['facts']],
                         prediction_inventory_complete=False, prediction_facts=[]))
        key.append(dict(blind_id=bid, case_id=case['case_id']))
    root = fresh(output)
    # Do not hand the key/prepared directory to candidate reviewers.
    jsonlines(root / 'candidate_judgements_template.jsonl', sorted(rows, key=lambda r: r['blind_id']))
    jsonlines(root / 'PRIVATE_model_key.jsonl', key)
    save(root / 'frozen_source_facts.json', document)
    save(root / 'AUDIT_READY.json', dict(schema='v13-audit-ready', n_candidates=len(rows),
         facts_sha256=file_hash(root / 'frozen_source_facts.json'),
         prepared_manifest_sha256=file_hash(Path(prepared) / 'PREPARED.json'),
         warning='Identity masking is procedural, not cryptographic blinding; human semantic judgments are not machine-verified'))
    print('AUDIT_READY: {} outputs from {} descriptions; human review PENDING'.format(len(rows), len(by_id)))


def validate_judgement(row, case, inventory, protocol):
    if row['candidate'] != case['candidate'] or row['reference'] != case['reference'] or row['example_id'] != case['example_id']:
        raise ValueError('Judgement text/example changed')
    if row['candidate_sha256'] != case['candidate_sha256'] or not row.get('reviewer', '').strip():
        raise ValueError('Candidate digest or reviewer missing')
    gold = {f['fact_id']: f for f in inventory['facts']}
    decisions = row['reference_facts']
    if len(decisions) != len(gold) or {f['fact_id'] for f in decisions} != set(gold):
        raise ValueError('Reference fact coverage differs')
    for fact in decisions:
        for k, value in gold[fact['fact_id']].items():
            if fact.get(k) != value:
                raise ValueError('Frozen reference fact edited')
        if fact['state'] not in protocol['reference_fact_states']:
            raise ValueError('Unreviewed/invalid reference fact state')
        if fact['state'] in ('entailed', 'contradicted'):
            if not fact['candidate_quote'] or fact['candidate_quote'] not in case['candidate']:
                raise ValueError('Positive/contradictory judgments need candidate evidence')
    if row.get('prediction_inventory_complete') is not True:
        raise ValueError('Prediction fact inventory not reviewed')
    predictions = row['prediction_facts']
    if case['candidate'].strip() and not predictions:
        raise ValueError('Nonempty prediction requires atomic assertion audit, not an empty perfect score')
    seen = set()
    for fact in predictions:
        if fact['fact_id'] in seen or not str(fact['fact_id']).strip():
            raise ValueError('Duplicate/empty predicted fact')
        seen.add(fact['fact_id'])
        if fact['state'] not in protocol['prediction_fact_states'] or not fact['proposition'].strip():
            raise ValueError('Unreviewed/invalid prediction fact')
        if not fact['candidate_quote'] or fact['candidate_quote'] not in case['candidate']:
            raise ValueError('Prediction fact needs exact candidate evidence')
        if fact['state'] in ('supported', 'contradicted'):
            if not fact.get('reference_quote') or fact['reference_quote'] not in case['reference']:
                raise ValueError('Supported/contradicted fact needs source evidence')
    n = len(decisions)
    m = len(predictions)
    counts = {s: sum(f['state'] == s for f in decisions) for s in protocol['reference_fact_states']}
    pc = {s: sum(f['state'] == s for f in predictions) for s in protocol['prediction_fact_states']}
    return dict(case_id=case['case_id'], example_id=case['example_id'], group=case['arm'] + '/' + case['control'],
                n_reference_facts=n, n_prediction_facts=m, reference_counts=counts, prediction_counts=pc,
                fact_retention=counts['entailed'] / n,
                omission_rate=counts['not_mentioned'] / n,
                reference_contradiction_rate=counts['contradicted'] / n,
                source_unsupported_prediction_rate=(pc['contradicted'] + pc['unsupported']) / m if m else None,
                reference_unclear_rate=counts['unclear'] / n,
                prediction_unclear_rate=pc['unclear'] / m if m else None)


def score_facts(prepared, audit_root, judgements, output):
    manifest, cases = verify_prepared(prepared)
    ar = Path(audit_root)
    ready = load(ar / 'AUDIT_READY.json')
    if ready['prepared_manifest_sha256'] != file_hash(Path(prepared) / 'PREPARED.json'):
        raise ValueError('Audit is for another preparation')
    if file_hash(ar / 'frozen_source_facts.json') != ready['facts_sha256']:
        raise ValueError('Frozen source facts edited')
    _, inv = inventories(prepared, ar / 'frozen_source_facts.json', cases)
    expected = {blind_id(c): c for c in cases if c['example_id'] in inv}
    result, seen = [], set()
    protocol = load(FOLDER / 'protocol.json')
    for row in readlines(judgements):
        bid = row['blind_id']
        if bid not in expected or bid in seen:
            raise ValueError('Duplicate/unknown blinded judgement')
        seen.add(bid)
        result.append(validate_judgement(row, expected[bid], inv[row['example_id']], protocol))
    if seen != set(expected):
        raise ValueError('Incomplete candidate matrix; do not report selective manual scores')
    groups = {}
    for row in result:
        groups.setdefault(row['group'], []).append(row)
    summary = {}
    for group, rows in groups.items():
        retention = [r['fact_retention'] for r in rows]
        unsupported = [r['source_unsupported_prediction_rate'] for r in rows if r['source_unsupported_prediction_rate'] is not None]
        pred_n = sum(r['n_prediction_facts'] for r in rows)
        summary[group] = dict(n_descriptions=len(rows), macro_fact_retention=statistics.mean(retention),
             retention_description_bootstrap_CI95=interval(retention),
             macro_unsupported_rate=statistics.mean(unsupported) if unsupported else None,
             unsupported_description_bootstrap_CI95=interval(unsupported) if unsupported else None,
             unsupported_defined_n=len(unsupported), empty_prediction_n=len(rows) - len(unsupported),
             micro_fact_retention=sum(r['reference_counts']['entailed'] for r in rows) / sum(r['n_reference_facts'] for r in rows),
             micro_unsupported_rate=sum(r['prediction_counts']['unsupported'] + r['prediction_counts']['contradicted'] for r in rows) / pred_n if pred_n else None,
             reference_unclear_facts=sum(r['reference_counts']['unclear'] for r in rows),
             prediction_unclear_facts=sum(r['prediction_counts']['unclear'] for r in rows))
    paired = {}
    for left, right in [('V12_CE_KD/matched', 'V12_CE/matched'),
                         ('V12_CE/matched', 'V11_g20480/matched'),
                         ('V12_CE_KD/matched', 'V12_CE_KD/shuffled'),
                         ('V12_CE_KD/matched', 'unrelated_description/deranged')]:
        if left in groups and right in groups:
            a = {r['example_id']: r['fact_retention'] for r in groups[left]}
            b = {r['example_id']: r['fact_retention'] for r in groups[right]}
            if set(a) != set(b):
                raise ValueError('Human audit cohorts not paired')
            delta = [a[i] - b[i] for i in sorted(a)]
            paired[left + ' minus ' + right] = dict(n=len(delta), mean=statistics.mean(delta), CI95=interval(delta))
    root = fresh(output)
    jsonlines(root / 'fact_scores.jsonl', result)
    save(root / 'summary.json', dict(status='HUMAN_FACT_AUDIT_COMPLETE_NOT_MODEL_QUALITY_PASS',
         protocol_sha256=manifest['protocol_sha256'], prepared_sha256=file_hash(Path(prepared) / 'PREPARED.json'),
         facts_sha256=ready['facts_sha256'], judgements_sha256=file_hash(judgements),
         groups=summary, paired_retention_deltas=paired,
         scope='30 fixed descriptions, exploratory human audit; source unsupported does NOT mean false in the image',
         automatic_SPICE_or_LLM_scores='NOT_MEASURED', historical_quality='UNCHANGED'))
    print('Human audit complete; historical quality unchanged: ' + str(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    make = sub.add_parser('make')
    make.add_argument('--prepared', required=True)
    make.add_argument('--facts', required=True)
    make.add_argument('--output', required=True)
    score = sub.add_parser('score')
    score.add_argument('--prepared', required=True)
    score.add_argument('--audit', required=True)
    score.add_argument('--judgements', required=True)
    score.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.action == 'make':
        make_audit(args.prepared, args.facts, args.output)
    else:
        score_facts(args.prepared, args.audit, args.judgements, args.output)


if __name__ == '__main__':
    main()
