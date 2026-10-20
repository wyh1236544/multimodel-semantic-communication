"""Validate explicit human submissions; never infer/approve factual truth automatically."""
from core import (SCHEMA, CATEGORIES, VISIBILITY, LABELS, require_keys, short,
                  text_sha, config)


def validate_references(doc, material, prepared_sha):
    require_keys(doc, ('schema', 'kind', 'prepared_sha256', 'reviewer', 'human_attested', 'prior_candidate_exposure', 'items'))
    if doc['schema'] != SCHEMA or doc['kind'] != 'references' or doc['prepared_sha256'] != prepared_sha:
        raise ValueError('Reference submission is bound to another prepared run')
    reviewer = short(doc['reviewer'], 'reference reviewer', 100)
    if doc['human_attested'] is not True:
        raise ValueError('Explicit human review attestation required')
    if doc['prior_candidate_exposure'] not in ('yes', 'no', 'unknown'):
        raise ValueError('Record whether reference selector saw candidate outputs before this review')
    refs = {r['reference_id']: r for r in material['references']}
    if not isinstance(doc['items'], list) or len(doc['items']) != len(refs):
        raise ValueError('All 32 references required; no selected-subset scoring')
    done = {}; c = config()
    for item in doc['items']:
        require_keys(item, ('reference_id', 'source_text_sha256', 'facts'))
        rid = item['reference_id']
        if rid not in refs or rid in done or item['source_text_sha256'] != refs[rid]['source_text_sha256']:
            raise ValueError('Duplicate/wrong source reference')
        fs = item['facts']
        if not isinstance(fs, list) or not c['minimum_reference_facts'] <= len(fs) <= c['maximum_reference_facts']:
            raise ValueError('Each description needs 3..30 independently chosen atomic facts')
        facts = []; seen = set(); statements = set()
        for f in fs:
            require_keys(f, ('fact_id', 'category', 'visibility', 'statement', 'source_quote'))
            fid = short(f['fact_id'], 'fact ID', 20)
            statement = short(f['statement'], 'atomic statement', 240)
            quote = short(f['source_quote'], 'source evidence', 1500)
            if fid in seen or statement.casefold() in statements:
                raise ValueError('Duplicate fact ID/statement')
            if f['category'] not in CATEGORIES or f['visibility'] not in VISIBILITY:
                raise ValueError('Unknown fact category/visibility')
            if quote not in refs[rid]['source_text']:
                raise ValueError('Source quote is not a literal span in this description')
            seen.add(fid); statements.add(statement.casefold())
            facts.append(dict(fact_id=fid, category=f['category'], visibility=f['visibility'],
                              statement=statement, source_quote=quote))
        done[rid] = dict(reference_id=rid, source_text_sha256=item['source_text_sha256'], facts=facts)
    return dict(schema=SCHEMA, kind='references', prepared_sha256=prepared_sha,
                reviewer=reviewer, human_attested=True,
                prior_candidate_exposure=doc['prior_candidate_exposure'],
                items=[done[r['reference_id']] for r in material['references']])


def validate_ratings(doc, material, references, review_sha):
    require_keys(doc, ('schema', 'kind', 'review_sha256', 'reviewer', 'human_attested', 'items'))
    if doc['schema'] != SCHEMA or doc['kind'] != 'ratings' or doc['review_sha256'] != review_sha:
        raise ValueError('Ratings bound to another frozen review')
    reviewer = short(doc['reviewer'], 'candidate reviewer', 100)
    if doc['human_attested'] is not True:
        raise ValueError('Explicit candidate review attestation required')
    candidates = {r['case_id']: r for r in material['candidates']}
    refs = {r['reference_id']: r['facts'] for r in references['items']}
    if not isinstance(doc['items'], list) or len(doc['items']) != len(candidates):
        raise ValueError('All 160 candidate cases required; missing cases cannot silently score zero')
    done = {}
    for item in doc['items']:
        require_keys(item, ('case_id', 'candidate_sha256', 'judgments', 'unsupported'))
        cid = item['case_id']
        if cid not in candidates or cid in done:
            raise ValueError('Duplicate/unknown candidate case')
        r = candidates[cid]
        if item['candidate_sha256'] != text_sha(r['text']):
            raise ValueError('Candidate text hash differs')
        wanted = {f['fact_id'] for f in refs[r['reference_id']]}
        judgments = {}
        if not isinstance(item['judgments'], list):
            raise ValueError('Judgments must be a list')
        for j in item['judgments']:
            require_keys(j, ('fact_id', 'label', 'candidate_quote', 'note'))
            fid, label = j['fact_id'], j['label']
            if fid not in wanted or fid in judgments or label not in LABELS:
                raise ValueError('Unknown/duplicate fact judgment or unfilled label')
            quote, note = j['candidate_quote'], j['note']
            if not isinstance(quote, str) or not isinstance(note, str) or max(len(quote), len(note)) > 2000:
                raise ValueError('Invalid judgment evidence/note')
            if quote and quote not in r['text']:
                raise ValueError('Candidate quote is not a literal span of this case')
            if label in ('retained', 'contradicted') and not quote.strip():
                raise ValueError('Retained/contradicted decisions require candidate evidence')
            if label == 'missing' and quote:
                raise ValueError('Missing fact must not have invented evidence')
            if label == 'uncertain' and not note.strip():
                raise ValueError('Uncertain decision requires a note')
            judgments[fid] = dict(fact_id=fid, label=label, candidate_quote=quote, note=note)
        if set(judgments) != wanted:
            raise ValueError('Every frozen reference fact must be judged')
        unsupported = item['unsupported']; seen = set()
        if not isinstance(unsupported, list) or len(unsupported) > 100:
            raise ValueError('Unsupported additions must be an explicit list, not a missing value')
        added = []
        for u in unsupported:
            require_keys(u, ('statement', 'candidate_quote', 'note'))
            statement = short(u['statement'], 'unsupported statement', 400)
            quote = short(u['candidate_quote'], 'unsupported statement evidence', 1500)
            note = short(u['note'], 'why source does not support this addition', 2000)
            if statement.casefold() in seen or quote not in r['text']:
                raise ValueError('Duplicate addition or incorrect evidence')
            seen.add(statement.casefold())
            added.append(dict(statement=statement, candidate_quote=quote, note=note))
        done[cid] = dict(case_id=cid, candidate_sha256=item['candidate_sha256'],
                        judgments=[judgments[f['fact_id']] for f in refs[r['reference_id']]], unsupported=added)
    return dict(schema=SCHEMA, kind='ratings', review_sha256=review_sha, reviewer=reviewer,
                human_attested=True, items=[done[r['case_id']] for r in material['candidates']])


def reference_template(material, prepared_sha):
    return dict(schema=SCHEMA, kind='references', prepared_sha256=prepared_sha, reviewer='',
                human_attested=False, prior_candidate_exposure='unknown', items=[dict(reference_id=r['reference_id'],
                source_text_sha256=r['source_text_sha256'], facts=[]) for r in material['references']])


def rating_template(material, references, review_sha):
    refs = {r['reference_id']: r['facts'] for r in references['items']}
    return dict(schema=SCHEMA, kind='ratings', review_sha256=review_sha, reviewer='', human_attested=False,
                items=[dict(case_id=r['case_id'], candidate_sha256=r['candidate_sha256'],
                judgments=[dict(fact_id=f['fact_id'], label='', candidate_quote='', note='')
                           for f in refs[r['reference_id']]], unsupported=[]) for r in material['candidates']])
