"""Synthetic structural contracts. Labels/scores here are fixtures, NOT research results."""
import copy
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest

import audit
import common as c
from challenge import cases as challenge_cases
import collect
import metrics
import run
from score_challenge_bert import check_lengths


def fake_v13(folder):
    old = c.legacy()
    rr = []
    for i in range(100):
        text = 'This synthetic description number {} has two dogs near a bench.'.format(i)
        for control in ('matched', 'shuffled', 'zero'):
            rr.append(dict(example_id='qual_dev_{:05d}'.format(i), official_split='qual_dev', cohort='validation',
                condition='clean', input_truncated=False, source_text=text, source_text_sha256=c.text_sha(text),
                received_caption=text, control=control, receiver_failure=False, normalized_wer=0.,
                stats=dict(clean_sentinel_not_measured_snr=True, resources=dict(payload_complex_symbols=9216,
                           analog_repeats=1, raw_utf8_wire_bytes=0))))
    class Source:
        def rows(self, name):
            return [copy.deepcopy(r) for r in rr if r['control'] == 'matched'] if 'native' in name else copy.deepcopy(rr)
    cases, refs = old.cases_from_sources(Source(), Source(), 100)
    folder.mkdir()
    c.write_rows(folder / 'cases.jsonl', cases)
    c.write(folder / 'audit_selection.json', dict(example_ids=old.select_audit(refs, 30, 42)))
    files = {p.name: c.sha(p) for p in folder.iterdir()}
    c.write(folder / 'PREPARED.json', dict(schema='v13-prepared', n_examples=100, n_cases=1100,
        files=files, source_code={}, protocol_sha256=c.sha(old.FOLDER / 'protocol.json')))
    return cases


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(dir=str(c.PROJECT))
        cls.root = Path(cls.tmp.name)
        cls.original = cls.root / 'v13'
        cls.cases = fake_v13(cls.original)
        cls.prepared = cls.root / 'v15'
        run.prepare(cls.original, cls.prepared)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def temp(self):
        return Path(tempfile.mkdtemp(dir=str(self.root)))

    def test_source_pins(self):
        self.assertGreater(len(c.verify_source()['dependencies']), 1)

    def test_complete_matrix_and_lexical_recompute(self):
        m, cases, groups = run.verify_prepared(self.prepared)
        self.assertEqual(m['n_cases'], 1100)
        self.assertEqual(len(groups), 11)
        self.assertEqual(len(cases), 1100)

    def test_duplicate_missing_and_reference_mismatch_rejected(self):
        for mode in ('duplicate', 'missing', 'source', 'lexical'):
            cc = copy.deepcopy(self.cases)
            if mode == 'duplicate': cc.append(cc[0])
            if mode == 'missing': cc.pop()
            if mode == 'source':
                cc[0]['reference'] += '!'
                cc[0]['reference_sha256'] = c.text_sha(cc[0]['reference'])
            if mode == 'lexical': cc[0]['lexical']['normalized_WER'] = .123
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                c.matrix(cc, 100)

    def test_frozen_file_mutation_rejected(self):
        clone = self.temp() / 'copy'
        shutil.copytree(self.prepared, clone)
        (clone / 'cases.jsonl').write_text('[]', encoding='utf-8')
        with self.assertRaises(ValueError): run.verify_prepared(clone)

    def test_no_overwrite(self):
        with self.assertRaises(FileExistsError): c.fresh(self.prepared)

    def test_project_and_path_traversal_guards(self):
        for bad in ('../x', '/tmp/x', 'a/../b', 'a\\b', './x'):
            with self.subTest(path=bad), self.assertRaises(ValueError): c.inside(self.root, bad)
        with self.assertRaises(ValueError): c.fresh('/tmp/not-a-v15-project-output')

    def test_source_first_page_has_no_candidates_or_keys(self):
        page = (self.prepared / 'source_review.html').read_text()
        self.assertNotIn('PRIVATE_KEY', page)
        self.assertNotIn('V12_CE_KD', page)
        self.assertIn('只看原文', page)

    def test_disjoint_audit_split_fixed_and_source_only(self):
        s = c.read(self.prepared / 'audit_selection.json')
        self.assertEqual(len(s['development']), 15)
        self.assertEqual(len(s['confirmation']), 15)
        self.assertFalse(set(s['development']) & set(s['confirmation']))
        self.assertEqual(set(s['example_ids']), set(s['development']) | set(s['confirmation']))

    def test_challenge_is_synthetic_complete_and_hash_bound(self):
        cc = challenge_cases()
        self.assertEqual(len(cc), 40)
        self.assertEqual(len({r['example_id'] for r in cc}), 4)
        self.assertEqual(len({r['case_id'] for r in cc}), 40)
        for r in cc:
            self.assertEqual(r['candidate_sha256'], c.text_sha(r['candidate']))
            self.assertIn('SYNTHETIC', r['scope'])

    def test_full_text_length_overflow_no_truncation(self):
        class Tok:
            def encode(self, text, **kw):
                assert kw['truncation'] is False
                return list(range(len(text)))
        self.assertGreater(check_lengths(challenge_cases(), Tok()), 0)
        with self.assertRaises(ValueError): check_lengths([dict(reference='x' * 4097, candidate='x')], Tok())

    def test_anchor_not_clipped_not_accuracy(self):
        self.assertEqual(c.anchor_position([3.], [0.], [2.])['value'], 1.5)
        self.assertEqual(c.anchor_position([-.5], [0.], [1.])['value'], -.5)
        self.assertIsNone(c.anchor_position([.8], [.9], [.9])['value'])

    def test_paired_ratio_uncertainty_and_undefined_resamples(self):
        r = metrics.ratio_interval([.9, .95], [.8, .8], [1., 1.])
        self.assertEqual(r['undefined_resamples'], 0)
        self.assertEqual(metrics.ratio_interval([.8], [.9], [.9])['undefined_resamples'], 2000)

    def bert_fixture(self, root):
        rr, groups = [], {}
        for x in self.cases:
            group = x['arm'] + '/' + x['control']
            value = .998 if x['arm'] == 'native_BART' else (.93 if x['control'] == 'matched' else .845)
            r = dict(case_id=x['case_id'], example_id=x['example_id'], group=group, P=value, R=value, F1=value)
            rr.append(r)
            groups[group] = dict(n=100, P=value, R=value, F1=value)
        c.write_rows(root / 'rows.jsonl', rr)
        s = dict(c.read(c.FOLDER / 'protocol.json')['bertscore'])
        s.update(status='AUXILIARY_SIMILARITY_COMPLETE', n=1100, groups=groups,
            prepared_sha256=c.sha(self.original / 'PREPARED.json'),
            protocol_sha256=c.read(self.original / 'PREPARED.json')['protocol_sha256'], package_versions={'bert-score': '0.3.13'})
        c.write(root / 'summary.json', s)
        return rr, s

    def test_real_schema_bert_import_and_anchor(self):
        t = self.temp()
        self.bert_fixture(t)
        result = metrics.import_bert(self.prepared, t / 'rows.jsonl', t / 'summary.json', t / 'scored')
        self.assertAlmostEqual(result['relative_anchor_positions']['V12_CE_KD']['value'], (.93-.845)/(.998-.845))
        self.assertEqual(result['human_facts'], 'PENDING')

    def test_smoke_wrong_settings_summary_and_duplicate_bert_rejected(self):
        for mode in ('smoke', 'layer', 'identity', 'mean', 'duplicate', 'nan'):
            t = self.temp()
            rr, s = self.bert_fixture(t)
            if mode == 'smoke': s['status'] = 'SMOKE_ONLY_NOT_FULL_EVALUATION'
            if mode == 'layer': s['num_layers'] = 12
            if mode == 'identity': s['prepared_sha256'] = 'wrong'
            if mode == 'mean': s['groups'][rr[0]['group']]['F1'] = .2
            if mode == 'duplicate': rr[-1] = rr[0]
            if mode == 'nan': rr[0]['F1'] = 'nan'
            (t / 'rows.jsonl').unlink()
            (t / 'summary.json').unlink()
            c.write_rows(t / 'rows.jsonl', rr)
            c.write(t / 'summary.json', s)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                metrics.import_bert(self.prepared, t / 'rows.jsonl', t / 'summary.json', t / 'bad')

    def facts_fixture(self):
        doc = c.read(self.prepared / 'source_facts_template.json')
        doc['inventory_reviewer'] = 'SYNTHETIC_TEST_REVIEWER'
        for r in doc['examples']:
            r['inventory_complete'] = True
            r['facts'] = [dict(fact_id='F1', category='object', proposition='Dogs are present', reference_quote='dogs')]
        return doc

    def test_unreviewed_source_inventory_refused(self):
        t = self.temp()
        with self.assertRaises(ValueError): audit.blind(self.prepared, self.prepared / 'source_facts_template.json', t / 'bad')

    def test_full_blind_audit_schema_and_score(self):
        t = self.temp()
        c.write(t / 'facts.json', self.facts_fixture())
        audit.blind(self.prepared, t / 'facts.json', t / 'blind')
        rr = c.rows(t / 'blind/candidate_review_template.jsonl')
        self.assertEqual(len(rr), 330)
        self.assertNotIn('V12_', json.dumps(rr))
        for r in rr:
            r['reviewer'] = 'SYNTHETIC_TEST_REVIEWER'
            for f in r['reference_facts']:
                f.update(state='entailed', candidate_quote='dogs')
            r['prediction_inventory_complete'] = True
            r['prediction_facts'] = [dict(fact_id='P1', category='object', proposition='Dogs present',
                state='supported', candidate_quote='dogs', reference_quote='dogs')]
        c.write_rows(t / 'judged.jsonl', rr)
        audit.score(self.prepared, t / 'blind', t / 'judged.jsonl', t / 'scores')
        s = c.read(t / 'scores/summary.json')
        self.assertEqual(s['n_outputs'], 330)
        self.assertEqual(len(s['splits']['confirmation']), 11)
        c.write_rows(t / 'missing.jsonl', rr[:-1])
        with self.assertRaises(ValueError): audit.score(self.prepared, t / 'blind', t / 'missing.jsonl', t / 'missing')

    def test_candidate_evidence_immutable_and_unclear_denominator(self):
        case = self.cases[0]
        source = dict(facts=[dict(fact_id='F1', category='count', proposition='There are two dogs', reference_quote='two dogs')])
        row = dict(reference=case['reference'], candidate=case['candidate'], candidate_sha256=case['candidate_sha256'],
            example_id=case['example_id'], reviewer='SYNTHETIC', prediction_inventory_complete=True,
            reference_facts=[dict(source['facts'][0], state='unclear', candidate_quote='')],
            prediction_facts=[dict(fact_id='P1', state='unclear', proposition='Dogs', candidate_quote='dogs')])
        p = c.read(c.FOLDER / 'protocol.json')
        v = audit.judge(row, case, source, p)
        self.assertEqual(v['fact_retention'], 0.)
        self.assertEqual(v['source_facts'], 1)
        row['candidate'] += ' edited'
        with self.assertRaises(ValueError): audit.judge(row, case, source, p)

    def test_empty_output_not_perfect_and_nonempty_requires_predictions(self):
        case = copy.deepcopy(self.cases[0])
        case.update(candidate='', candidate_sha256=c.text_sha(''))
        fact = dict(fact_id='F1', category='object', proposition='Dogs', reference_quote='dogs')
        row = dict(reference=case['reference'], candidate='', candidate_sha256=c.text_sha(''), example_id=case['example_id'],
            reviewer='SYNTHETIC', prediction_inventory_complete=True,
            reference_facts=[dict(fact, state='not_mentioned', candidate_quote='')], prediction_facts=[])
        p = c.read(c.FOLDER / 'protocol.json')
        self.assertIsNone(audit.judge(row, case, dict(facts=[fact]), p)['unsupported_prediction_rate'])
        row['candidate'] = case['candidate'] = 'dogs'
        row['candidate_sha256'] = case['candidate_sha256'] = c.text_sha('dogs')
        with self.assertRaises(ValueError): audit.judge(row, case, dict(facts=[fact]), p)

    def test_optional_metric_frozen_plan_complete_import_and_rejections(self):
        t = self.temp()
        (t / 'fixture.asset').write_text('SYNTHETIC_NOT_MODEL')
        c.write(t / 'spec.json', dict(metric='CAPTURE', evaluator_revision='SYNTHETIC_TEST_ONLY', settings={},
            assets=[dict(path=str(t / 'fixture.asset'), sha256=c.sha(t / 'fixture.asset'))]))
        metrics.freeze_metric(self.prepared, t / 'spec.json', t / 'plan')
        plan_sha = c.sha(t / 'plan/plan.json')
        rr = [dict(case_id=r['case_id'], reference_sha256=r['reference_sha256'], candidate_sha256=r['candidate_sha256'],
                   plan_sha256=plan_sha, score=.5, details={'fixture': True}) for r in c.rows(t / 'plan/requests.jsonl')]
        c.write_rows(t / 'scores.jsonl', rr)
        metrics.import_metric(t / 'plan', t / 'scores.jsonl', t / 'import')
        s = c.read(t / 'import/summary.json')
        self.assertEqual(s['challenge_diagnostics']['binding']['ties'], 4)
        c.write_rows(t / 'missing.jsonl', rr[:-1])
        with self.assertRaises(ValueError): metrics.import_metric(t / 'plan', t / 'missing.jsonl', t / 'bad')
        rr[0]['candidate_sha256'] = 'wrong'
        c.write_rows(t / 'changed.jsonl', rr)
        with self.assertRaises(ValueError): metrics.import_metric(t / 'plan', t / 'changed.jsonl', t / 'bad2')

    def test_collection_roundtrip_source_only_snapshot_and_no_overwrite(self):
        t = self.temp()
        output = t / 'private.tar.gz'
        collect.collect(self.prepared, output)
        with tarfile.open(output) as tf:
            self.assertTrue(all(m.isfile() for m in tf.getmembers()))
            self.assertFalse(any(m.name.endswith(('.pth', '.pt', '.bin')) for m in tf.getmembers()))
        self.assertTrue(output.with_name(output.name + '.sha256').is_file())
        with self.assertRaises(ValueError): collect.collect(self.prepared, output)


if __name__ == '__main__':
    unittest.main(verbosity=2)
