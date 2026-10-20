"""CPU-only immutable three-stage factual review. Deliberately no training command."""
import argparse
from pathlib import Path
import shutil
import time
from core import (SCHEMA, FOLDER, read, sha, config, atomic, write_text, new_output,
                  snapshot, seal, source_check, Budget)
from parent import check_parent, copy_evidence, materialize
from facts import validate_references, validate_ratings, reference_template, rating_template
from metrics import summarize, lexical_hints
from ui import reference_form, rating_form, index
from report import verify, expected_prepared


def start(root, stage):
    snapshot(root)
    atomic(root / 'RUN.json', dict(schema=SCHEMA, stage=stage, training_updates=0,
        models_loaded=False, GPU_allocated=False, downloads=False, automatic_training_approval=False))


def prepare(parent, records, output, budget):
    print('V27 verify pinned completed V26; CPU only, no weights or new inference', flush=True)
    source, proof = check_parent(parent, records, budget)
    root = new_output(output, (source, parent))
    start(root, 'prepared'); copy_evidence(source, root / 'evidence'); budget()
    material = materialize(root / 'evidence')
    atomic(root / 'IMPORT.json', proof); atomic(root / 'config.json', config())
    atomic(root / 'MATERIAL.json', material)
    atomic(root / 'PREPARED.json', expected_prepared(root, material))
    template = reference_template(material, sha(root / 'PREPARED.json'))
    atomic(root / 'references_template.json', template)
    atomic(root / 'literal_overlap_hints.json', lexical_hints(material))
    write_text(root / 'references.html', reference_form(material, template))
    write_text(root / 'index.html', index('prepared'))
    seal(root, 'prepared', 'PENDING_REFERENCE_REVIEW')
    result = verify(root, budget)
    print('V27 PREPARED 32 references / 160 existing outputs; facts PENDING. No training.', flush=True)
    return root, result


def seal_references(prepared, submission, output, budget):
    prepared = Path(prepared).resolve(strict=True)
    if verify(prepared, budget)['stage'] != 'prepared':
        raise ValueError('Expected prepared stage')
    material = read(prepared / 'MATERIAL.json')
    raw = read(submission)
    frozen = validate_references(raw, material, sha(prepared / 'PREPARED.json'))
    root = new_output(output, (prepared, submission)); start(root, 'review')
    shutil.copytree(prepared, root / 'prepared'); budget()
    atomic(root / 'references_submission.json', raw); atomic(root / 'REFERENCES.json', frozen)
    atomic(root / 'REVIEW.json', dict(schema=SCHEMA, prepared_sha256=sha(root / 'prepared/PREPARED.json'),
        prepared_complete_sha256=sha(root / 'prepared/COMPLETE.json'), references_sha256=sha(root / 'REFERENCES.json'),
        facts_frozen_before_candidate_review=True))
    template = rating_template(material, frozen, sha(root / 'REVIEW.json'))
    atomic(root / 'ratings_template.json', template)
    write_text(root / 'rate_candidates.html', rating_form(material, frozen, template))
    write_text(root / 'index.html', index('review'))
    seal(root, 'review', 'PENDING_CANDIDATE_REVIEW')
    result = verify(root, budget)
    print('V27 facts FROZEN; 160 candidate cases await human labels. No training.', flush=True)
    return root, result


def score(review, submission, output, budget):
    review = Path(review).resolve(strict=True)
    if verify(review, budget)['stage'] != 'review':
        raise ValueError('Expected frozen reference/candidate-review stage')
    material = read(review / 'prepared/MATERIAL.json'); refs = read(review / 'REFERENCES.json')
    raw = read(submission)
    rated = validate_ratings(raw, material, refs, sha(review / 'REVIEW.json'))
    result = summarize(material, refs, rated); budget()
    root = new_output(output, (review, submission)); start(root, 'scored')
    shutil.copytree(review, root / 'review')
    atomic(root / 'ratings_submission.json', raw); atomic(root / 'RATINGS.json', rated)
    atomic(root / 'RESULT.json', result); write_text(root / 'index.html', index('scored', result))
    status = 'REVIEW_COMPLETE_WITH_UNCERTAINTY' if result['opportunity']['uncertain_judgments'] else 'REVIEW_COMPLETE'
    seal(root, 'scored', status)
    checked = verify(root, budget)
    print('V27 '+status+'; opportunity='+result['opportunity']['decision']+'; training approval=false', flush=True)
    return root, checked


def main():
    p = argparse.ArgumentParser(description=__doc__); sp = p.add_subparsers(dest='action', required=True)
    q = sp.add_parser('prepare'); q.add_argument('output')
    g = q.add_mutually_exclusive_group(); g.add_argument('--v26-root', default='experiment_results/v26_rx_diagnose_01'); g.add_argument('--v26-records')
    q = sp.add_parser('seal-facts'); q.add_argument('prepared'); q.add_argument('submission'); q.add_argument('output')
    q = sp.add_parser('score'); q.add_argument('review'); q.add_argument('submission'); q.add_argument('output')
    q = sp.add_parser('verify'); q.add_argument('root')
    for q in sp.choices.values():
        q.add_argument('--max-minutes', type=float, default=10)
    a = p.parse_args(); source_check(); budget = Budget(a.max_minutes)
    if a.action == 'prepare':
        prepare(a.v26_records or a.v26_root, bool(a.v26_records), a.output, budget)
    elif a.action == 'seal-facts':
        seal_references(a.prepared, a.submission, a.output, budget)
    elif a.action == 'score':
        score(a.review, a.submission, a.output, budget)
    else:
        import json
        print(json.dumps(verify(a.root, budget), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
