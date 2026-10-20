"""Exploratory paired, source-referenced factual measures, not image-ground-truth scoring."""
import math
import random
import re
import statistics
from core import METHODS, CATEGORIES, config


def quantile(values, q):
    values = sorted(values)
    p = (len(values) - 1) * q; lo = int(p); hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - p) + values[hi] * (p - lo) if hi != lo else values[lo]


def bootstrap(delta):
    c = config(); rng = random.Random(c['bootstrap_seed']); n = len(delta)
    if n < 3:
        return None
    values = [math.fsum(delta[rng.randrange(n)] for _ in range(n)) / n
              for _ in range(c['bootstrap_draws'])]
    return [quantile(values, .025), quantile(values, .975)]


def summarize(material, references, ratings):
    fact_by = {r['reference_id']: r['facts'] for r in references['items']}
    rated = {r['case_id']: r for r in ratings['items']}
    by = {}; details = []
    for candidate in material['candidates']:
        r = rated[candidate['case_id']]; fs = fact_by[candidate['reference_id']]
        labels = {j['fact_id']: j['label'] for j in r['judgments']}
        n = len(fs); counts = {label: sum(v == label for v in labels.values())
                              for label in ('retained', 'contradicted', 'missing', 'uncertain')}
        visual = [f['fact_id'] for f in fs if f['visibility'] == 'potentially_visual']
        detail = dict(case_id=candidate['case_id'], reference_id=candidate['reference_id'],
            method=candidate['method'], facts=n, counts=counts,
            retention_lower=counts['retained'] / n,
            retention_upper=(counts['retained'] + counts['uncertain']) / n,
            contradiction_fraction=counts['contradicted'] / n,
            unsupported_added_facts=len(r['unsupported']),
            potentially_visual_facts=len(visual),
            potentially_visual_retained=sum(labels[f] == 'retained' for f in visual))
        details.append(detail); by[candidate['reference_id'], candidate['method']] = (detail, labels)
    groups = []
    for method in METHODS:
        rs = [r for r in details if r['method'] == method]
        totals = {k: sum(r['counts'][k] for r in rs) for k in ('retained', 'contradicted', 'missing', 'uncertain')}
        n = sum(r['facts'] for r in rs)
        categories = {}
        for category in CATEGORIES:
            labels = [j['label'] for candidate in material['candidates'] if candidate['method'] == method
                for f, j in zip(fact_by[candidate['reference_id']], rated[candidate['case_id']]['judgments'])
                if f['category'] == category]
            counts = {k: labels.count(k) for k in ('retained', 'contradicted', 'missing', 'uncertain')}
            categories[category] = dict(facts=len(labels), counts=counts,
                retention_lower=counts['retained']/len(labels) if labels else None,
                retention_upper=(counts['retained']+counts['uncertain'])/len(labels) if labels else None)
        groups.append(dict(method=method, descriptions=len(rs), total_reference_facts=n, counts=totals, by_category=categories,
            macro_retention_lower=statistics.mean(r['retention_lower'] for r in rs),
            macro_retention_upper=statistics.mean(r['retention_upper'] for r in rs),
            micro_retention_lower=totals['retained'] / n,
            micro_retention_upper=(totals['retained'] + totals['uncertain']) / n,
            macro_contradiction_fraction=statistics.mean(r['contradiction_fraction'] for r in rs),
            mean_unsupported_added_facts=statistics.mean(r['unsupported_added_facts'] for r in rs)))
    ids = [r['reference_id'] for r in material['references']]
    comparisons = []
    for control in ('self', 'caption', 'visual_shuffled'):
        changes = [by[e, 'visual_matched'][0]['retention_lower'] - by[e, control][0]['retention_lower'] for e in ids]
        comparisons.append(dict(comparison='visual_matched_minus_' + control,
            macro_retention_gain=statistics.mean(changes), pointwise_paired95CI=bootstrap(changes)))
    opportunities = []
    for rid in ids:
        caption = by[rid, 'caption'][1]; text = by[rid, 'self'][1]
        eligible = [f['fact_id'] for f in fact_by[rid] if f['visibility'] == 'potentially_visual']
        available = [f for f in eligible if caption[f] == 'retained' and text[f] in ('missing', 'contradicted')]
        opportunities.append(dict(reference_id=rid, supported_missing_fact_ids=available, count=len(available)))
    unknown = sum(r['counts']['uncertain'] for r in details)
    threshold = config()['opportunity_screen']
    count = sum(r['count'] for r in opportunities); descriptions = sum(r['count'] > 0 for r in opportunities)
    decision = ('PENDING_ADJUDICATION' if unknown else
                'WORTH_PROPOSING_A_SMALL_CORRECTION_PILOT' if
                count >= threshold['minimum_supported_missing_facts'] and
                descriptions >= threshold['minimum_helpable_descriptions'] else
                'INSUFFICIENT_OPPORTUNITY_IN_THIS_FIXED_REVIEW')
    return dict(groups=groups, historical_comparisons=comparisons, per_case=details,
        opportunity=dict(decision=decision, supported_missing_facts=count, helpable_descriptions=descriptions,
                         uncertain_judgments=unknown, per_description=opportunities, fixed_thresholds=threshold),
        stage='historical_evidence_review_not_new_model_evaluation',
        reference_selector_prior_candidate_exposure=references['prior_candidate_exposure'],
        reference_and_candidate_reviewer_differ=references['reviewer'] != ratings['reviewer'],
        independent_test=False, intervals='10000 description-level paired bootstrap; pointwise uncorrected exploratory',
        limitations=['Source-referenced judgments, not verified against actual image pixels.',
                     'Human attestation is self-reported, not proof of reviewer identity or independence.',
                     'Unknown judgments produce bounds and block the opportunity decision.',
                     'Unsupported additions mean unsubstantiated by source, not necessarily false in the image.',
                     'Reference coverage does not provide candidate precision: no factual F1 is fabricated.',
                     'All 32 reused DEV descriptions are included; this is not a new independent test.'],
        semantic_success=False, automatic_training_approval=False, training_updates=0)


def lexical_hints(material):
    """Analyst-only literal overlap hints: NOT facts, grounding, recall or a training gate."""
    vocabulary = {
        'colors': ('red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'gray', 'grey', 'orange', 'pink', 'purple'),
        'counts': ('one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'),
        'positions': ('left', 'right', 'behind', 'above', 'below', 'beside', 'between', 'under')}
    tokens = lambda s: set(re.findall(r"[a-z]+", s.lower()))
    by = {(r['reference_id'], r['method']): r for r in material['candidates']}
    out = []
    for r in material['references']:
        source = tokens(r['source_text']); cap = tokens(by[r['reference_id'], 'caption']['text'])
        received = tokens(by[r['reference_id'], 'self']['text'])
        hints = {k: sorted(set(v) & source & cap - received) for k, v in vocabulary.items()}
        out.append(dict(reference_id=r['reference_id'], literal_overlap_hints=hints,
                        warning='No entity binding/negation/semantic verification; do not show to fact selectors before freezing references.'))
    return out
