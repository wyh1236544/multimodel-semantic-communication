"""Source-first, identity-blind human fact audit; no fabricated automatic facts."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import statistics
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_baseline_v14.core import safe, atomic, sha, rows, read
from docci_baseline_v14.report import verify


def write_rows(path, data):
    with path.open('x', encoding='utf-8') as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def prepare(root, output):
    result = verify(root)
    candidates = [(name, r) for name, v in result['evaluations'].items()
                  if name.startswith('text/') and name.endswith('validation')
                  for r in v['rows'] if r['strategy']=='greedy']
    if not candidates:
        raise ValueError('Text validation required')
    source = {r['example_id']: r for _, r in candidates}
    ids = sorted(source, key=lambda eid: (source[eid]['source_tokens'], eid))
    # Length stratification uses source only, never model quality.
    strata = [ids[i*len(ids)//3:(i+1)*len(ids)//3] for i in range(3)]
    rng = random.Random(42)
    chosen = []
    for group in strata:
        chosen += sorted(rng.sample(group, min(10, len(group))))
    out = safe(output)
    out.mkdir(parents=True, exist_ok=False)
    write_rows(out/'source_facts.jsonl', [dict(example_id=eid, source_text=source[eid]['source_text'],
        source_text_sha256=source[eid]['source_text_sha256'], reviewed=False, facts=[]) for eid in chosen])
    atomic(out/'AUDIT_SOURCE.json', dict(run_manifest_sha256=sha(Path(root)/'LIGHT_MANIFEST.json'),
           selected_ids=chosen, instruction='Fill facts BEFORE opening candidate outputs. Each fact: id,text,visually_verifiable(bool). Set reviewed=true. Do not change original descriptions.',
           source_selection='source-length-stratified; no quality selection', pending=True))


def blind(root, facts, output):
    result = verify(root)
    original = {r['example_id']:r for name,v in result['evaluations'].items()
                if name.startswith('text/') and name.endswith('validation') for r in v['rows']}
    inventory = rows(facts)
    if not inventory or len({r['example_id'] for r in inventory}) != len(inventory):
        raise ValueError('Empty/duplicate fact source inventory')
    # Audit selection must match the frozen source-first plan, not selected gains.
    plan = read(Path(facts).parent/'AUDIT_SOURCE.json')
    if plan['run_manifest_sha256'] != sha(Path(root)/'LIGHT_MANIFEST.json') or plan['selected_ids'] != [r['example_id'] for r in inventory]:
        raise ValueError('Fact source selection/experiment changed')
    for r in inventory:
        ref = original[r['example_id']]
        if r['reviewed'] is not True or not r['facts'] or r['source_text'] != ref['source_text'] or r['source_text_sha256'] != ref['source_text_sha256']:
            raise ValueError('Source facts must be completed/frozen before blind review')
        if len({f['id'] for f in r['facts']}) != len(r['facts']) or any(not f['text'].strip() or type(f['visually_verifiable']) is not bool for f in r['facts']):
            raise ValueError('Invalid fact schema')
    out = safe(output)
    out.mkdir(parents=True, exist_ok=False)
    shutil_facts = out/'frozen_source_facts.jsonl'
    with shutil_facts.open('xb') as f:
        f.write(Path(facts).read_bytes())
    source = {r['example_id']:r for r in inventory}
    candidates = [(name,r) for name,v in result['evaluations'].items()
                  if name.startswith('text/') and name.endswith('validation') for r in v['rows']
                  if r['strategy']=='greedy' and r['example_id'] in source]
    random.Random(43).shuffle(candidates)
    key, review = [], []
    for i,(name,r) in enumerate(candidates):
        cid = 'C{:04d}'.format(i+1)
        key.append(dict(candidate_id=cid, experiment=name, control=r['control'], example_id=r['example_id'],
                        recovered_sha256=hashlib.sha256(r['received_caption'].encode()).hexdigest()))
        review.append(dict(candidate_id=cid, source_text=r['source_text'], recovered=r['received_caption'],
            reviewed=False, facts=[dict(fact_id=f['id'], fact=f['text'], judgment='PENDING') for f in source[r['example_id']]['facts']],
            source_unsupported_claims=None, generated_claim_count=None))
    write_rows(out/'blind_review.jsonl', review)
    atomic(out/'PRIVATE_KEY_DO_NOT_GIVE_REVIEWER.json', key)
    atomic(out/'AUDIT.json', dict(frozen_facts_sha256=sha(shutil_facts), source_inventory_sha256=sha(facts),
           private_key_sha256=sha(out/'PRIVATE_KEY_DO_NOT_GIVE_REVIEWER.json'),
           rows=len(review), judgments=['retained','contradicted','omitted','unclear'],
           unsupported_is_NOT_necessarily_visually_false=True, pending=True))


def score(root, output):
    root = Path(root).resolve(strict=True)
    if safe(output).exists():
        raise FileExistsError('Human scoring output exists; use a new version')
    design = read(root/'AUDIT.json')
    if sha(root/'frozen_source_facts.jsonl') != design['frozen_facts_sha256']:
        raise ValueError('Frozen source facts changed after blind preparation')
    original = {r['example_id']:r for r in rows(root/'frozen_source_facts.jsonl')}
    key = read(root/'PRIVATE_KEY_DO_NOT_GIVE_REVIEWER.json')
    if sha(root/'PRIVATE_KEY_DO_NOT_GIVE_REVIEWER.json') != design['private_key_sha256']:
        raise ValueError('Private candidate identity changed')
    mapping = {r['candidate_id']:r for r in key}
    reviewed = rows(root/'blind_review.jsonl')
    if len(reviewed) != design['rows'] or len(mapping) != len(reviewed):
        raise ValueError('Incomplete human review matrix')
    seen, grouped = set(), {}
    for r in reviewed:
        cid = r['candidate_id']
        if cid not in mapping or cid in seen or r['reviewed'] is not True:
            raise ValueError('Duplicate/unreviewed human candidate')
        seen.add(cid)
        k = mapping[cid]
        source = original[k['example_id']]
        if r['source_text'] != source['source_text'] or [f['fact_id'] for f in r['facts']] != [f['id'] for f in source['facts']] or [f['fact'] for f in r['facts']] != [f['text'] for f in source['facts']]:
            raise ValueError('Reviewed facts differ from frozen source')
        if hashlib.sha256(r['recovered'].encode()).hexdigest() != k['recovered_sha256']:
            raise ValueError('Reviewer changed the recovered model output')
        states = [f['judgment'] for f in r['facts']]
        if any(x not in ('retained','contradicted','omitted','unclear') for x in states):
            raise ValueError('Unfinished fact labels')
        claims = r['source_unsupported_claims']
        n = r['generated_claim_count']
        if not isinstance(claims,list) or any(not isinstance(x,str) or not x.strip() for x in claims) or type(n) is not int or not 0 <= len(claims) <= n:
            raise ValueError('Human unsupported-claim list AND explicit denominator required')
        group = k['experiment']+'/'+k['control']
        total = len(states)
        values = dict(example_id=k['example_id'],facts=total,retained=states.count('retained'),
                      contradicted=states.count('contradicted'),unclear=states.count('unclear'),
                      fact_retention=states.count('retained')/total,unsupported_claims=len(claims),generated_claims=n)
        grouped.setdefault(group,[]).append(values)
    aggregates = {}
    for name, rr in grouped.items():
        aggregates[name] = dict(n_descriptions=len(rr),mean_fact_retention=statistics.mean(r['fact_retention'] for r in rr),
            source_facts=sum(r['facts'] for r in rr),retained=sum(r['retained'] for r in rr),
            contradicted=sum(r['contradicted'] for r in rr),unclear=sum(r['unclear'] for r in rr),
            unsupported_claims=sum(r['unsupported_claims'] for r in rr),generated_claims=sum(r['generated_claims'] for r in rr),
            unsupported_claim_fraction=sum(r['unsupported_claims'] for r in rr)/max(1,sum(r['generated_claims'] for r in rr)))
    atomic(safe(output),dict(status='HUMAN_REVIEW_COMPLETE',groups=aggregates,per_description=grouped,
           uncertainty='No training-replicate or inter-rater uncertainty measured; no success gate applied',
           source_unsupported_is_NOT_necessarily_visually_false=True,
           unclear_source_facts_remain_in_denominator=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('prepare','blind','score'))
    p.add_argument('root')
    p.add_argument('output')
    p.add_argument('--facts')
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(a.root, a.output)
    elif a.action == 'blind':
        if not a.facts:
            p.error('blind needs --facts completed source_facts.jsonl')
        blind(a.root, a.facts, a.output)
    else:
        score(a.root,a.output)
