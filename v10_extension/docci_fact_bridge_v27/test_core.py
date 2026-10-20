"""Synthetic TEST_ONLY contracts. These never constitute human factual approval."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from core import (SCHEMA, FOLDER, PROJECT, atomic, read, sha, text_sha, local, new_output,
                  digest, source_check, Budget)
from parent import assemble, MODES
from facts import (reference_template, rating_template, validate_references, validate_ratings)
from metrics import summarize, lexical_hints, bootstrap, quantile
from ui import reference_form, rating_form, embedded


def fixture():
    refs = []; gen = []; caps = []
    for i in range(32):
        eid = 'TEST_ONLY_{:02d}'.format(i)
        text = 'A red car stands beside two trees. Example {}.'.format(i)
        refs.append(dict(example_id=eid, cohort='dev', official_split='train',
                         text=text, text_sha256=text_sha(text)))
        caps.append(dict(example_id=eid, image_snr_db=0, output_text='A red car beside two trees.'))
        for arm, ctrl in MODES.values():
            gen.append(dict(example_id=eid, cohort='dev', official_split='train', regime='both0',
                arm=arm, control=ctrl, source_text=text, source_text_sha256=text_sha(text),
                source_text_at_RX=False, source_mask_at_RX=False, source_image_at_RX=False,
                training_updates=0, generation_provenance='V25_VERIFIED_DEV_REUSE',
                image_snr_db=0, text_snr_db=0,
                received_caption='A blue car beside one tree.', normalized_wer=.5))
    return assemble(refs, gen, caps), (refs, gen, caps)


def references(material):
    doc = reference_template(material, 'p' * 64)
    doc.update(reviewer='TEST_ONLY_REFERENCE_REVIEWER', human_attested=True, prior_candidate_exposure='no')
    for r in doc['items']:
        r['facts'] = [dict(fact_id='f01', category='objects', visibility='potentially_visual',
                          statement='There is a car.', source_quote='car'),
                      dict(fact_id='f02', category='colors', visibility='potentially_visual',
                          statement='The car is red.', source_quote='red car'),
                      dict(fact_id='f03', category='counts', visibility='potentially_visual',
                          statement='There are two trees.', source_quote='two trees')]
    return doc


def ratings(material, refs):
    doc = rating_template(material, refs, 'r' * 64)
    doc.update(reviewer='TEST_ONLY_CANDIDATE_REVIEWER', human_attested=True)
    for r, candidate in zip(doc['items'], material['candidates']):
        for j in r['judgments']:
            retained = j['fact_id'] == 'f01' or candidate['method'] in ('caption', 'visual_matched')
            j['label'] = 'retained' if retained else 'missing'
            j['candidate_quote'] = 'car' if retained else ''
    return doc


class Contracts(unittest.TestCase):
    def setUp(self):
        self.material, self.parts = fixture()
        self.refs = references(self.material)

    def test_all_32_by_five_candidates_and_deterministic_hidden_case_ids(self):
        self.assertEqual(len(self.material['references']), 32)
        self.assertEqual(len(self.material['candidates']), 160)
        self.assertEqual(self.material, assemble(*self.parts))
        self.assertEqual(len({r['case_id'] for r in self.material['candidates']}), 160)

    def test_duplicate_missing_generation_and_caption_rejected(self):
        a, b, c = copy.deepcopy(self.parts)
        with self.assertRaises(ValueError):
            assemble(a, b[:-1], c)
        with self.assertRaises(ValueError):
            assemble(a, b+[b[0]], c)
        with self.assertRaises(ValueError):
            assemble(a, b, c+[c[0]])

    def test_source_leak_or_changed_reference_and_wrong_channel_rejected(self):
        for field, value in [('source_text_at_RX', True), ('text_snr_db', -5), ('source_text', 'changed')]:
            a, b, c = copy.deepcopy(self.parts); b[0][field] = value
            with self.assertRaises(ValueError):
                assemble(a, b, c)

    def test_pending_reference_cannot_be_scored_or_approved(self):
        raw = reference_template(self.material, 'p'*64)
        with self.assertRaises(ValueError):
            validate_references(raw, self.material, 'p'*64)
        raw = references(self.material); raw['items'][0]['facts'] = []
        with self.assertRaises(ValueError):
            validate_references(raw, self.material, 'p'*64)

    def test_reference_exact_evidence_binding_duplicates_and_categories(self):
        for mutate in ('quote', 'duplicate', 'binding', 'unknown', 'missing_reference'):
            raw = copy.deepcopy(self.refs)
            if mutate == 'quote': raw['items'][0]['facts'][0]['source_quote'] = 'a secret dog'
            if mutate == 'duplicate': raw['items'][0]['facts'][1]['fact_id'] = 'f01'
            if mutate == 'binding': raw['prepared_sha256'] = 'wrong'
            if mutate == 'unknown': raw['items'][0]['facts'][0]['category'] = 'invented'
            if mutate == 'missing_reference': raw['items'].pop()
            with self.assertRaises(ValueError):
                validate_references(raw, self.material, 'p'*64)
        validate_references(self.refs, self.material, 'p'*64)

    def test_every_case_and_fact_requires_explicit_human_judgment(self):
        raw = ratings(self.material, self.refs)
        for mutate in ('missing_case', 'duplicate_case', 'missing_fact', 'empty_label', 'no_attestation'):
            x = copy.deepcopy(raw)
            if mutate == 'missing_case': x['items'].pop()
            if mutate == 'duplicate_case': x['items'][0] = copy.deepcopy(x['items'][1])
            if mutate == 'missing_fact': x['items'][0]['judgments'].pop()
            if mutate == 'empty_label': x['items'][0]['judgments'][0]['label'] = ''
            if mutate == 'no_attestation': x['human_attested'] = False
            with self.assertRaises(ValueError):
                validate_ratings(x, self.material, self.refs, 'r'*64)

    def test_ratings_evidence_and_addition_spans_cannot_reference_other_text(self):
        raw = ratings(self.material, self.refs)
        raw['items'][0]['judgments'][0]['candidate_quote'] = 'this is not in the candidate'
        with self.assertRaises(ValueError):
            validate_ratings(raw, self.material, self.refs, 'r'*64)
        raw = ratings(self.material, self.refs)
        raw['items'][0]['unsupported'] = [dict(statement='dog', candidate_quote='dog', note='not in source')]
        with self.assertRaises(ValueError):
            validate_ratings(raw, self.material, self.refs, 'r'*64)

    def test_macro_micro_counts_category_and_opportunity_not_training_success(self):
        raw = validate_ratings(ratings(self.material, self.refs), self.material, self.refs, 'r'*64)
        s = summarize(self.material, self.refs, raw)
        by = {r['method']:r for r in s['groups']}
        self.assertEqual(by['self']['total_reference_facts'], 96)
        self.assertAlmostEqual(by['self']['macro_retention_lower'], 1/3)
        self.assertEqual(by['caption']['macro_retention_lower'], 1)
        self.assertIsNone(by['self']['by_category']['relations']['retention_lower'])
        self.assertEqual(s['opportunity']['supported_missing_facts'], 64)
        self.assertEqual(s['opportunity']['helpable_descriptions'], 32)
        self.assertEqual(s['opportunity']['decision'], 'WORTH_PROPOSING_A_SMALL_CORRECTION_PILOT')
        self.assertFalse(s['automatic_training_approval']); self.assertFalse(s['semantic_success'])
        self.assertEqual(s['training_updates'], 0)

    def test_uncertainty_keeps_bounds_and_blocks_opportunity(self):
        raw = ratings(self.material, self.refs)
        j=raw['items'][0]['judgments'][1]; j.update(label='uncertain', candidate_quote='', note='TEST_ONLY ambiguous')
        validated = validate_ratings(raw, self.material, self.refs, 'r'*64)
        s=summarize(self.material,self.refs,validated)
        self.assertEqual(s['opportunity']['decision'],'PENDING_ADJUDICATION')
        self.assertEqual(s['opportunity']['uncertain_judgments'],1)
        row=next(r for r in s['per_case'] if r['case_id']==raw['items'][0]['case_id'])
        self.assertGreater(row['retention_upper'],row['retention_lower'])

    def test_literal_hints_are_not_labels_or_entity_bound_facts(self):
        hints=lexical_hints(self.material)
        self.assertEqual(hints[0]['literal_overlap_hints']['colors'],['red'])
        self.assertEqual(hints[0]['literal_overlap_hints']['counts'],['two'])
        self.assertNotIn('score',hints[0]);self.assertNotIn('label',hints[0])

    def test_source_only_page_and_candidate_method_key_is_not_embedded(self):
        source=reference_form(self.material,reference_template(self.material,'p'*64))
        self.assertNotIn('A blue car',source)
        candidate=rating_form(self.material,self.refs,rating_template(self.material,self.refs,'r'*64))
        self.assertNotIn('historical_normalized_WER',candidate)
        self.assertNotIn('visual_shuffled',candidate);self.assertNotIn('visual_matched',candidate)
        self.assertNotIn('fetch(',source+candidate)
        self.assertNotIn('innerHTML',source+candidate)
        escaped=embedded({'x':'</script><img src=x onerror=bad()>&\u2028'})
        self.assertNotIn('</script',escaped);self.assertEqual(json.loads(escaped)['x'],'</script><img src=x onerror=bad()>&\u2028')

    def test_exclusive_atomic_json_nonfinite_duplicates_and_path_guards(self):
        with tempfile.TemporaryDirectory(prefix='V27_TEST_ONLY_',dir=PROJECT/'experiment_results') as temp:
            root=Path(temp);p=root/'test.json';atomic(p,{'a':1})
            with self.assertRaises(FileExistsError):atomic(p,{'a':2})
            self.assertEqual(read(p),{'a':1})
            with self.assertRaises(ValueError):atomic(root/'bad.json',{'a':float('nan')})
            with self.assertRaises(ValueError):local(root,'../escape')
            with self.assertRaises(FileExistsError):new_output(root)
            linked=root/'link';linked.symlink_to(p)
            with self.assertRaises(ValueError):local(root,'link')
            with self.assertRaises(ValueError):Budget(-1)
            with (root/'duplicate.json').open('x') as f:f.write('{"a":1,"a":2}')
            with self.assertRaises(ValueError):read(root/'duplicate.json')
            with (root/'nonfinite.json').open('x') as f:f.write('{"a":NaN}')
            with self.assertRaises(ValueError):read(root/'nonfinite.json')

    def test_bootstrap_is_paired_deterministic_and_quantile_correct(self):
        self.assertEqual(quantile([0,1,2,3],.25),.75)
        self.assertEqual(bootstrap([1,1,1]),[1.,1.])
        self.assertEqual(bootstrap([0,1,2]),bootstrap([0,1,2]))
        self.assertIsNone(bootstrap([1,2]))

    def test_no_training_entry_or_heavy_dependency_import(self):
        p=subprocess.run([sys.executable,'-B',str(FOLDER/'run.py'),'train'],capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0)
        for n in ('run.py','core.py','parent.py','facts.py','metrics.py','report.py','ui.py'):
            source=(FOLDER/n).read_text()
            self.assertNotIn('import torch',source);self.assertNotIn('import transformers',source)
        self.assertIsInstance(source_check()['files'],dict)

    def test_TEST_ONLY_full_three_stage_collection_and_resealed_score_corruption(self):
        import run as engine
        import report
        import collect as archive
        from parent import PINS
        with tempfile.TemporaryDirectory(prefix='V27_TEST_ONLY_',dir=PROJECT/'experiment_results') as temp:
            base=Path(temp);original=base/'fake_parent';original.mkdir()
            def fake_copy(src,dst):
                dst.mkdir();atomic(dst/'TEST_ONLY.json',{'not_real_parent':True})
            with mock.patch.object(engine,'check_parent',return_value=(original,{'pinned_files':PINS,'TEST_ONLY':True})), \
                 mock.patch.object(engine,'copy_evidence',side_effect=fake_copy), \
                 mock.patch.object(engine,'materialize',return_value=self.material), \
                 mock.patch.object(report,'materialize',return_value=self.material):
                prepared,_=engine.prepare(original,False,base/'prepared',Budget(2))
                raw=references(self.material);raw['prepared_sha256']=sha(prepared/'PREPARED.json')
                atomic(base/'refs.json',raw)
                reviewed,_=engine.seal_references(prepared,base/'refs.json',base/'review',Budget(2))
                normalized=read(reviewed/'REFERENCES.json');rated=ratings(self.material,normalized)
                rated['review_sha256']=sha(reviewed/'REVIEW.json');atomic(base/'ratings.json',rated)
                scored,result=engine.score(reviewed,base/'ratings.json',base/'scored',Budget(2))
                self.assertEqual(result['status'],'REVIEW_COMPLETE')
                self.assertFalse(result['semantic_success'])
                self.assertEqual(result['result']['opportunity']['supported_missing_facts'],64)
                package=archive.collect(scored,base/'TEST_ONLY_records.tar.gz',Budget(2))
                self.assertTrue(package.is_file())
                # Even recomputing the receipt cannot make invented metrics pass the independent verifier.
                invented=read(scored/'RESULT.json');invented['groups'][0]['macro_retention_lower']=.123
                (scored/'RESULT.json').write_text(json.dumps(invented))
                receipt=read(scored/'COMPLETE.json');receipt['files']['RESULT.json']=sha(scored/'RESULT.json')
                (scored/'COMPLETE.json').write_text(json.dumps(receipt))
                with self.assertRaises(ValueError):report.verify(scored)


if __name__=='__main__':
    unittest.main(verbosity=2)
