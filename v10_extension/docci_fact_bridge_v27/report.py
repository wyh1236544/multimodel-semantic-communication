"""Independent re-indexing, human-submission validation and score recomputation."""
import math
from pathlib import Path
from core import read, sha, config, verify_files, digest, SCHEMA
from parent import materialize
from facts import validate_references, validate_ratings, reference_template, rating_template
from metrics import summarize, lexical_hints
from ui import reference_form, rating_form, index


def equal(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or a is None or isinstance(a, (str, int)):
        return type(a) is type(b) and a == b
    return isinstance(b, (float, int)) and math.isfinite(a) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def expected_prepared(root, material):
    return dict(schema=SCHEMA, material_sha256=sha(Path(root) / 'MATERIAL.json'),
                source_manifest_sha256=sha(Path(__file__).parent / 'MANIFEST.json'),
                config_sha256=sha(Path(root) / 'config.json'), examples=32, candidate_cases=160,
                primary='both0', facts='PENDING', training_updates=0)


def verify(root, budget=lambda: None):
    root = Path(root); budget(); receipt = verify_files(root); stage = receipt['stage']
    if stage == 'prepared':
        material = materialize(root / 'evidence')
        if read(root / 'MATERIAL.json') != material or read(root / 'config.json') != config():
            raise ValueError('Frozen prepared inputs/config differ')
        if read(root / 'PREPARED.json') != expected_prepared(root, material):
            raise ValueError('Prepared input binding differs')
        template = reference_template(material, sha(root / 'PREPARED.json'))
        if read(root / 'references_template.json') != template:
            raise ValueError('Reference template differs')
        if read(root / 'literal_overlap_hints.json') != lexical_hints(material):
            raise ValueError('Literal hints changed')
        if (root / 'references.html').read_text(encoding='utf-8') != reference_form(material, template):
            raise ValueError('Source-only reference page differs')
        expected_status = 'PENDING_REFERENCE_REVIEW'
        if read(root / 'IMPORT.json')['pinned_files'] != __import__('parent').PINS:
            raise ValueError('Imported parent identity differs')
        output = dict(stage=stage, status=expected_status, examples=32, candidate_cases=160,
                      facts='PENDING', fact_scores=None)
    elif stage == 'review':
        verify(root / 'prepared', budget)
        material = read(root / 'prepared/MATERIAL.json')
        frozen = validate_references(read(root / 'references_submission.json'), material,
                                     sha(root / 'prepared/PREPARED.json'))
        if read(root / 'REFERENCES.json') != frozen:
            raise ValueError('Frozen references differ from human submission')
        want = dict(schema=SCHEMA, prepared_sha256=sha(root / 'prepared/PREPARED.json'),
                    prepared_complete_sha256=sha(root / 'prepared/COMPLETE.json'),
                    references_sha256=sha(root / 'REFERENCES.json'), facts_frozen_before_candidate_review=True)
        if read(root / 'REVIEW.json') != want:
            raise ValueError('Candidate-review binding differs')
        template = rating_template(material, frozen, sha(root / 'REVIEW.json'))
        if read(root / 'ratings_template.json') != template:
            raise ValueError('Rating template differs')
        if (root / 'rate_candidates.html').read_text(encoding='utf-8') != rating_form(material, frozen, template):
            raise ValueError('Blinded candidate page differs')
        expected_status = 'PENDING_CANDIDATE_REVIEW'
        output = dict(stage=stage, status=expected_status, examples=32, candidate_cases=160,
                      frozen_reference_facts=sum(len(r['facts']) for r in frozen['items']),
                      facts='PENDING', fact_scores=None)
    elif stage == 'scored':
        verify(root / 'review', budget)
        material = read(root / 'review/prepared/MATERIAL.json')
        refs = read(root / 'review/REFERENCES.json')
        ratings = validate_ratings(read(root / 'ratings_submission.json'), material, refs,
                                   sha(root / 'review/REVIEW.json'))
        if read(root / 'RATINGS.json') != ratings:
            raise ValueError('Validated candidate labels differ')
        budget(); result = summarize(material, refs, ratings); budget()
        if not equal(result, read(root / 'RESULT.json')):
            raise ValueError('Independent factual aggregation differs')
        expected_status = 'REVIEW_COMPLETE_WITH_UNCERTAINTY' if result['opportunity']['uncertain_judgments'] else 'REVIEW_COMPLETE'
        if (root / 'index.html').read_text(encoding='utf-8') != index('scored', read(root / 'RESULT.json')):
            raise ValueError('Scored review page differs')
        output = dict(stage=stage, status=expected_status, examples=32, candidate_cases=160,
                      facts='HUMAN_ATTESTED_SOURCE_REFERENCED', result=result)
    else:
        raise ValueError('Unknown V27 review stage')
    if receipt['status'] != expected_status:
        raise ValueError('Stage status inconsistent with actual review')
    if stage != 'scored' and (root / 'index.html').read_text(encoding='utf-8') != index(stage):
        raise ValueError('Stage index page differs')
    return dict(output, training_updates=0, semantic_success=False, automatic_training_approval=False,
                images_verified=False, private=True)
