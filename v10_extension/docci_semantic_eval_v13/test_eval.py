"""Synthetic CPU contracts. No pretrained performance assertions."""
import copy
import io
import json
from pathlib import Path
import random
import tarfile
import tempfile
import unittest
from unittest import mock
import types

import core
from audit import blind_id, inventories, make_audit, score_facts, validate_judgement
from score_bert import check_lengths
import score_bert


def historical(example, text, control='matched'):
    return dict(example_id=example, official_split='qual_dev', cohort='validation',
                condition='clean', input_truncated=False, source_text=text,
                source_text_sha256=core.digest(text.encode()), received_caption=text,
                control=control, receiver_failure=False, normalized_wer=0.,
                stats=dict(clean_sentinel_not_measured_snr=True,
                           resources=dict(payload_complex_symbols=9216, analog_repeats=1, raw_utf8_wire_bytes=0)))


class MemorySource:
    def __init__(self, mapping):
        self.mapping = mapping

    def rows(self, name):
        return self.mapping[name]


def fixture():
    matrix = [historical('q' + str(i), 'There are {} red dogs.'.format(i + 1), c)
              for i in range(3) for c in ('matched', 'shuffled', 'zero')]
    native = [historical('q' + str(i), 'There are {} red dogs.'.format(i + 1)) for i in range(3)]
    a = MemorySource({'g20480_validation/rows.jsonl': copy.deepcopy(matrix),
                      'parent_reference/native_validation/rows.jsonl': native})
    b = MemorySource({'main/text/B0_ce/validation/rows.jsonl': copy.deepcopy(matrix),
                      'main/text/B1_ce_kd/validation/rows.jsonl': copy.deepcopy(matrix)})
    return a, b


def prepared(root):
    a, b = fixture()
    cases, refs = core.cases_from_sources(a, b, 3)
    root.mkdir()
    core.jsonlines(root / 'cases.jsonl', cases)
    core.save(root / 'audit_selection.json', dict(example_ids=sorted(refs)))
    files = {p.name: core.file_hash(p) for p in root.iterdir()}
    core.save(root / 'PREPARED.json', dict(files=files, protocol_sha256=core.file_hash(core.FOLDER / 'protocol.json')))
    doc = dict(schema='v13-source-facts', inventory_reviewer='reviewer A', examples=[dict(
        example_id=i, reference=refs[i], reference_sha256=core.digest(refs[i].encode()),
        inventory_complete=True, facts=[dict(fact_id='F1', category='object', proposition='Dogs are present',
            reference_quote='dogs')]) for i in sorted(refs)])
    return cases, doc


class Contracts(unittest.TestCase):
    def test_exact_distance_against_independent_dynamic_programming(self):
        rng = random.Random(4)
        for _ in range(150):
            a = [rng.randrange(4) for _ in range(rng.randrange(15))]
            b = [rng.randrange(4) for _ in range(rng.randrange(15))]
            dp = list(range(len(b) + 1))
            for i, x in enumerate(a, 1):
                row = [i]
                for j, y in enumerate(b, 1):
                    row.append(min(dp[j] + 1, row[-1] + 1, dp[j - 1] + (x != y)))
                dp = row
            self.assertEqual(core.distance(a, b), dp[-1])

    def test_normalization_empty_and_repetition_not_semantic_score(self):
        self.assertEqual(core.lexical('RED dog!', 'red dog')['normalized_WER'], 0.)
        self.assertEqual(core.lexical('red dog', '')['normalized_WER'], 1.)
        self.assertGreater(core.lexical('red dog', 'dog dog dog dog')['normalized_WER'], 1.)
        self.assertEqual(core.lexical('red dog', 'dog dog dog')['adjacent_repeat'], 1.)

    def test_all_arms_controls_and_reference_derangement(self):
        a, b = fixture()
        cases, refs = core.cases_from_sources(a, b, 3)
        self.assertEqual(len(cases), 33)
        self.assertEqual(len({r['case_id'] for r in cases}), 33)
        self.assertTrue(all(r['donor_example_id'] != r['example_id'] for r in cases if r['arm'] == 'unrelated_description'))

    def test_bad_wer_source_split_and_missing_control_rejected(self):
        for failure in ('wer', 'hash', 'split', 'missing', 'duplicate', 'truncated', 'reference', 'resources'):
            a, b = fixture()
            rows = b.mapping['main/text/B0_ce/validation/rows.jsonl']
            if failure == 'wer': rows[0]['normalized_wer'] = .5
            if failure == 'hash': rows[0]['source_text_sha256'] = 'wrong'
            if failure == 'split': rows[0]['official_split'] = 'test'
            if failure == 'missing': rows.pop()
            if failure == 'duplicate': rows.append(copy.deepcopy(rows[0]))
            if failure == 'truncated': rows[0]['input_truncated'] = True
            if failure == 'resources': rows[0]['stats']['resources']['payload_complex_symbols'] = 100
            if failure == 'reference':
                rows[0]['source_text'] = 'Different'
                rows[0]['source_text_sha256'] = core.digest(b'Different')
            with self.assertRaises(ValueError, msg=failure):
                core.cases_from_sources(a, b, 3)

    def test_archive_read_without_extraction_and_ambiguous_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.tar.gz'
            with tarfile.open(path, 'w:gz') as tar:
                data = b'{"x": 3}\n'
                info = tarfile.TarInfo('records/a/rows.jsonl')
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            src = core.Source(path)
            self.assertEqual(src.rows('a/rows.jsonl'), [{'x': 3}])
            src.close()
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ['rows.tar.gz'])

    def test_archive_links_and_traversal_and_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ('link', 'traversal', 'duplicate'):
                path = Path(tmp) / (kind + '.tar')
                with tarfile.open(path, 'w') as tar:
                    info = tarfile.TarInfo('../a' if kind == 'traversal' else 'a')
                    if kind == 'link':
                        info.type, info.linkname = tarfile.SYMTYPE, '/tmp'
                    tar.addfile(info)
                    if kind == 'duplicate': tar.addfile(info)
                with self.assertRaises(ValueError): core.Source(path)

    def test_fresh_output_preserves_existing_and_refuses_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            core.fresh(root / 'output')
            with self.assertRaises(FileExistsError): core.fresh(root / 'output')
            (root / 'link').symlink_to(root / 'output', target_is_directory=True)
            with self.assertRaises(ValueError): core.fresh(root / 'link' / 'child')

    def test_full_text_length_audit_never_truncates(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                self.assertion = kwargs
                return [0] * (len(text.split()) + 2)
        tok = Tokenizer()
        cases = [dict(case_id='x', reference='one two three', candidate='one')]
        self.assertEqual(check_lengths(cases, tok, 5)['max_encoded_tokens'], 5)
        self.assertFalse(tok.assertion['truncation'])
        with self.assertRaises(ValueError): check_lengths(cases, tok, 4)

    def test_sample_is_deterministic_length_stratified_and_not_result_selected(self):
        refs = {'q' + str(i): ('word ' * (i + 1)) for i in range(100)}
        selected = core.select_audit(refs, 30, 42)
        self.assertEqual(selected, core.select_audit(refs, 30, 42))
        ranks = sorted(refs, key=lambda i: (len(core.words(refs[i])), i))
        self.assertEqual([len(set(selected) & set(ranks[i * 100 // 3:(i + 1) * 100 // 3])) for i in range(3)], [10, 10, 10])

    def test_html_escapes_text(self):
        row = dict(example_id='<img>', reference='<script>x</script>', candidate='<script>y</script>',
                   arm='a', control='c', lexical=dict(normalized_WER=0.))
        result = core.html_table([row], {})
        self.assertNotIn('<script>', result)
        self.assertIn('&lt;script&gt;', result)

    def test_empty_source_facts_and_changed_preparation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve() / 'prepared'
            cases, doc = prepared(root)
            path = Path(tmp) / 'facts.json'
            doc['examples'][0]['facts'] = []
            core.save(path, doc)
            with self.assertRaises(ValueError): inventories(root, path, cases)
            with (root / 'cases.jsonl').open('a') as handle: handle.write('\n')
            with self.assertRaises(ValueError): core.verify_prepared(root)

    def test_complete_blinded_manual_workflow_and_refuse_unreviewed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = str(Path(tmp).resolve())
            root = Path(tmp) / 'prepared'
            cases, doc = prepared(root)
            facts = Path(tmp) / 'facts.json'
            core.save(facts, doc)
            audit = Path(tmp) / 'audit'
            make_audit(root, facts, audit)
            template = core.readlines(audit / 'candidate_judgements_template.jsonl')
            self.assertEqual(len(template), 33)
            self.assertTrue(all('arm' not in r and 'control' not in r and 'case_id' not in r for r in template))
            with self.assertRaises(ValueError): score_facts(root, audit, audit / 'candidate_judgements_template.jsonl', Path(tmp) / 'bad')
            for row in template:
                row['reviewer'] = 'Reviewer B'
                for f in row['reference_facts']:
                    f['state'], f['candidate_quote'] = 'entailed', 'dogs'
                row['prediction_inventory_complete'] = True
                row['prediction_facts'] = [dict(fact_id='P1', proposition='Dogs present', state='supported',
                    candidate_quote='dogs', reference_quote='dogs')]
            labels = Path(tmp) / 'labels.jsonl'
            core.jsonlines(labels, template)
            score_facts(root, audit, labels, Path(tmp) / 'scored')
            summary = core.load(Path(tmp) / 'scored' / 'summary.json')
            self.assertTrue(all(v['macro_fact_retention'] == 1. for v in summary['groups'].values()))
            self.assertEqual(summary['historical_quality'], 'UNCHANGED')

    def test_judgement_missing_prediction_evidence_and_unclear_not_dropped(self):
        a, b = fixture()
        cases, _ = core.cases_from_sources(a, b, 3)
        case = cases[0]
        inv = dict(facts=[dict(fact_id='f', category='object', proposition='dogs', reference_quote='dogs')])
        row = dict(example_id=case['example_id'], candidate=case['candidate'], reference=case['reference'],
            candidate_sha256=case['candidate_sha256'], reviewer='R',
            reference_facts=[dict(**inv['facts'][0], state='unclear', candidate_quote='')],
            prediction_inventory_complete=True, prediction_facts=[])
        protocol = core.load(core.FOLDER / 'protocol.json')
        with self.assertRaises(ValueError): validate_judgement(row, case, inv, protocol)
        row['prediction_facts'] = [dict(fact_id='p', proposition='dogs', state='unclear', candidate_quote='dogs')]
        result = validate_judgement(row, case, inv, protocol)
        self.assertEqual(result['fact_retention'], 0.)
        self.assertEqual(result['reference_unclear_rate'], 1.)

    def test_optional_scorer_real_entry_with_synthetic_backend_and_frozen_receipt(self):
        class Tensor:
            def __init__(self, values): self.values = values
            def tolist(self): return self.values
        class Tokenizer:
            def encode(self, text, **kwargs): return list(range(len(text.split()) + 2))
        class Scorer:
            def __init__(self, **kwargs):
                self._tokenizer = Tokenizer()
                self.hash = 'SYNTHETIC_TEST_ONLY'
                if kwargs['device'] != 'cpu' or kwargs['rescale_with_baseline']:
                    raise AssertionError('Wrong fixed profile')
            def score(self, candidates, refs, batch_size):
                return tuple(Tensor([.8] * len(candidates)) for _ in range(3))
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            root = base / 'prepared'
            prepared(root)
            model = base / 'model'
            model.mkdir()
            core.save(model / 'config.json', dict(model_type='longformer', max_position_embeddings=4098))
            core.save(model / 'DOWNLOAD_RECEIPT.json', dict(model_id='allenai/longformer-base-4096',
                resolved_revision='synthetic-test-only', files={'config.json': core.file_hash(model / 'config.json')}))
            with mock.patch.dict('sys.modules', {'bert_score': types.SimpleNamespace(BERTScorer=Scorer)}), \
                 mock.patch.object(score_bert.importlib.metadata, 'version', return_value='0.3.13'), \
                 mock.patch.dict('os.environ', {}, clear=False):
                score_bert.score(root, model, base / 'scores', batch_size=3)
            result = core.load(base / 'scores/summary.json')
            self.assertEqual(result['n'], 33)
            self.assertEqual(result['digital'], 'NOT_RUN')
            self.assertEqual(result['facts'], 'PENDING_SEPARATE_HUMAN_AUDIT')
            self.assertEqual(len(result['paired_deltas']), 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
