"""Synthetic independently sealed matrices: negative integrity/metrics/HTML cases."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
import numpy as np
from docci_baseline_v14.core import PROJECT, load, atomic, sha, read, rows, array_hash, geometry, DECODE
from docci_baseline_v14.report import verify, render
from docci_text_continue_v11.metrics import metrics


def dump(path, values):
    with path.open('w', encoding='utf-8') as f:
        for row in values:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')


def seal(root):
    files = {p.relative_to(root).as_posix():sha(p) for p in root.rglob('*')
             if p.is_file() and p.name not in ('LIGHT_MANIFEST.json', 'RUN_FINISHED.json')}
    atomic(root/'LIGHT_MANIFEST.json', files)
    atomic(root/'RUN_FINISHED.json', dict(manifest_sha256=sha(root/'LIGHT_MANIFEST.json')))


def fixture(root):
    root.mkdir()
    c = load(profile='smoke')
    atomic(root/'config.json', c)
    atomic(root/'design.json', dict(mode='calibrate', task='text', text_arms=['legacy64']))
    entries = [dict(example_id='qual_dev_'+str(i), official_split='qual_dev',
               text='A <dog> '+str(i), text_sha256=hashlib.sha256(('A <dog> '+str(i)).encode()).hexdigest(), source_tokens=5)
               for i in range(2)]
    atomic(root/'text/cache_validation_identity.json', dict(records=entries))
    atomic(root/'text/identity.json', dict(backbone_signature_before='synthetic_frozen', backbone_signature_after='synthetic_frozen'))
    source = root/'source_snapshot'
    source.mkdir()
    (source/'fixture.txt').write_text('synthetic; not real model evidence')
    atomic(source/'MANIFEST.json', dict(files={'fixture.txt':sha(source/'fixture.txt')}, dependencies={}))
    output_names = ['text/native_validation', 'text/historical_reference/validation']
    for name in output_names:
        folder = root/name
        folder.mkdir(parents=True)
        (folder/'frames').mkdir()
        native = name.endswith('native_validation')
        controls = ['native'] if native else ['matched','shuffled','zero']
        matrix = dict(task='text', cohort='validation', example_ids=[e['example_id'] for e in entries],
                      controls=controls, strategies=list(DECODE), native_privileged_NOT_transmission=native,
                      geometry=None if native else geometry('text'), LPIPS_status='NOT_MEASURED')
        atomic(folder/'matrix.json', matrix)
        frames = []
        if not native:
            for i, e in enumerate(entries):
                a = np.full(18432, 1 if i==0 else -1, np.float32)
                file = folder/'frames'/(e['example_id']+'.npz')
                with file.open('xb') as f:
                    np.savez_compressed(f, observed=a)
                frames.append(dict(file=file.relative_to(folder).as_posix(), sha=sha(file), arr=array_hash(a)))
        output = []
        for i,e in enumerate(entries):
            for control in controls:
                for strategy in DECODE:
                    text = e['text'] if control in ('native','matched') else 'Other source'
                    chunks = [[2,0,4,5,2]]+[[2,0,2]]*8
                    donor = (i+1)%2
                    row = dict(e, task='text', cohort='validation', condition='ideal_clean', snr_db=None,
                        source_text=e['text'], source_text_sha256=e['text_sha256'], strategy=strategy,
                        received_caption=text, control=control, native=native,
                        frame_file=None if native else frames[i]['file'], frame_sha256=None if native else frames[i]['sha'],
                        tx_array_sha256=None if native else frames[i]['arr'],
                        donor_example_id=entries[donor]['example_id'] if control=='shuffled' else None,
                        received_array_sha256=None if native else frames[donor if control=='shuffled' else i]['arr'],
                        input_truncated=False, chunk_generated_token_ids=chunks, generated_token_ids=[4,5],
                        stopped_by_capacity=False, **metrics(e['text'],text))
                    output.append(row)
        dump(folder/'rows.jsonl', output)
        atomic(folder/'EVAL_COMPLETE.json', dict(rows=len(output), rows_sha256=sha(folder/'rows.jsonl'), matrix_sha256=sha(folder/'matrix.json')))
    atomic(root/'run_identity.json', dict(schema='docci-baseline-v14', outputs=output_names,
           source_manifest_sha256=sha(source/'MANIFEST.json')))
    seal(root)
    return root


def main_fixture(root):
    import shutil
    fixture(root)
    atomic(root/'design.json',dict(mode='smoke',task='text',text_arms=['legacy64']))
    selected = read(root/'text/cache_validation_identity.json')['records']
    train = [dict(selected[i%2],example_id='train_'+str(i),official_split='train') for i in range(8)]
    atomic(root/'text/cache_train_identity.json',dict(records=train))
    for cohort in ('validation','diagnostic_train'):
        dst = root/'text/legacy64'/cohort
        shutil.copytree(root/'text/historical_reference/validation',dst)
        if cohort == 'diagnostic_train':
            for i,e in enumerate(selected):
                (dst/'frames'/(e['example_id']+'.npz')).rename(dst/'frames'/(train[i]['example_id']+'.npz'))
            m = read(dst/'matrix.json')
            m.update(cohort=cohort,example_ids=[e['example_id'] for e in train[:2]])
            atomic(dst/'matrix.json',m)
            rr = rows(dst/'rows.jsonl')
            for r in rr:
                i = [e['example_id'] for e in selected].index(r['example_id'])
                r.update(example_id=train[i]['example_id'],official_split='train',cohort=cohort,
                         frame_file='frames/'+train[i]['example_id']+'.npz')
                if r['donor_example_id'] is not None:
                    r['donor_example_id']=train[(i+1)%2]['example_id']
            dump(dst/'rows.jsonl',rr)
            atomic(dst/'EVAL_COMPLETE.json',dict(rows=len(rr),rows_sha256=sha(dst/'rows.jsonl'),matrix_sha256=sha(dst/'matrix.json')))
    train_folder = root/'text/legacy64/train'
    train_folder.mkdir()
    atomic(train_folder/'trainability.json',dict(total_updates=2,unique_training_ids=[e['example_id'] for e in train]))
    atomic(train_folder/'learning_audit.json',dict(passed=True,groups={'codec':dict(parameters=1,nonzero_gradient=True,changed=True)}))
    atomic(train_folder/'TRAIN_COMPLETE.json',dict(updates=2,unique_examples_seen=1,actual_disk_roundtrip=True))
    atomic(root/'text/legacy64/identity.json',dict(initial_hashes={'p':'a'},final_hashes={'p':'b'}))
    dump(train_folder/'history.jsonl',[dict(update=i+1,total_updates=2,batch_example_ids=['train_0'],
        unique_examples_seen=1,loss=1.,gradient_norm=1.,valid_target_tokens=21) for i in range(2)])
    identity = read(root/'run_identity.json')
    identity['outputs']=['text/native_validation','text/legacy64/diagnostic_train','text/legacy64/validation']
    atomic(root/'run_identity.json',identity)
    seal(root)
    return root


class Reports(unittest.TestCase):
    def setUp(self):
        (PROJECT/'releases').mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=str(PROJECT/'releases'), prefix='v14_report_')
        self.root = fixture(Path(self.tmp.name)/'run')

    def tearDown(self):
        self.tmp.cleanup()

    def mutate(self, change):
        folder = self.root/'text/historical_reference/validation'
        rr = rows(folder/'rows.jsonl')
        change(rr)
        dump(folder/'rows.jsonl',rr)
        done = read(folder/'EVAL_COMPLETE.json')
        done.update(rows=len(rr), rows_sha256=sha(folder/'rows.jsonl'))
        atomic(folder/'EVAL_COMPLETE.json',done)
        seal(self.root)

    def test_full_positive_matrix_independent_metrics_and_html(self):
        data = verify(self.root)
        self.assertIn('NOT_QUALITY_SUCCESS', data['status'])
        page = render(data)
        self.assertIn('&lt;dog&gt;',page)
        self.assertNotIn('A <dog>',page)

    def test_wrong_metric_rejected_even_with_resealed_hash(self):
        self.mutate(lambda rr: rr[0].update(normalized_wer=.123))
        with self.assertRaisesRegex(ValueError,'metrics'):
            verify(self.root)

    def test_incomplete_control_matrix_rejected(self):
        self.mutate(lambda rr: rr.pop())
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            verify(self.root)

    def test_duplicate_row_rejected(self):
        self.mutate(lambda rr: rr.__setitem__(1,copy.deepcopy(rr[0])))
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            verify(self.root)

    def test_finite_clean_SNR_not_allowed(self):
        self.mutate(lambda rr: rr[0].update(snr_db=40.))
        with self.assertRaisesRegex(ValueError,'channel'):
            verify(self.root)

    def test_wrong_donor_rejected(self):
        def change(rr):
            next(r for r in rr if r['control']=='shuffled')['donor_example_id']='qual_dev_0'
        self.mutate(change)
        with self.assertRaisesRegex(ValueError,'donor'):
            verify(self.root)

    def test_changed_original_text_rejected(self):
        self.mutate(lambda rr: rr[0].update(source_text='different original'))
        with self.assertRaisesRegex(ValueError,'Original text'):
            verify(self.root)

    def test_wrong_generated_concatenation_rejected(self):
        self.mutate(lambda rr: rr[0].update(generated_token_ids=[5,4]))
        with self.assertRaisesRegex(ValueError,'concatenation'):
            verify(self.root)

    def test_original_manifest_tamper_rejected(self):
        (self.root/'source_snapshot/fixture.txt').write_text('changed')
        with self.assertRaisesRegex(ValueError,'Light file'):
            verify(self.root)

    def test_failed_run_is_not_completed(self):
        atomic(self.root/'FAILED.json',dict(reason='synthetic'))
        with self.assertRaisesRegex(ValueError,'Failed run'):
            verify(self.root)

    def test_real_collect_archive_restore_and_no_original_writes(self):
        import tarfile
        from docci_baseline_v14.collect import collect
        before = {p.relative_to(self.root).as_posix():sha(p) for p in self.root.rglob('*') if p.is_file()}
        out = Path(self.tmp.name)/'records.tar.gz'
        collect(self.root,out,1)
        after = {p.relative_to(self.root).as_posix():sha(p) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        restored = Path(self.tmp.name)/'restore'
        restored.mkdir()
        with tarfile.open(out) as tar:
            for member in tar.getmembers():
                self.assertTrue(member.isfile())
                self.assertNotIn('..',Path(member.name).parts)
                target = restored/member.name
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as f:
                    f.write(tar.extractfile(member).read())
        verify(restored/(out.name+'.contents')/'records')
        with self.assertRaises(FileExistsError):
            collect(self.root,out,1)

    def test_fact_inventory_is_source_first_and_blinded(self):
        from docci_baseline_v14.audit import prepare, blind
        out = Path(self.tmp.name)/'facts'
        prepare(self.root,out)
        facts = rows(out/'source_facts.jsonl')
        self.assertTrue(all(r['reviewed'] is False and r['facts']==[] for r in facts))
        with self.assertRaises(ValueError):
            blind(self.root,out/'source_facts.jsonl',Path(self.tmp.name)/'badblind')
        for r in facts:
            r.update(reviewed=True,facts=[dict(id='F1',text='a dog',visually_verifiable=True)])
        dump(out/'source_facts.jsonl',facts)
        target = Path(self.tmp.name)/'blind'
        blind(self.root,out/'source_facts.jsonl',target)
        self.assertNotIn('experiment', rows(target/'blind_review.jsonl')[0])
        self.assertTrue((target/'PRIVATE_KEY_DO_NOT_GIVE_REVIEWER.json').is_file())

    def test_human_scoring_refuses_pending_and_retains_unclear_denominator(self):
        from docci_baseline_v14.audit import prepare,blind,score
        source=Path(self.tmp.name)/'facts'
        prepare(self.root,source)
        rr=rows(source/'source_facts.jsonl')
        for r in rr:
            r.update(reviewed=True,facts=[dict(id='F1',text='dog',visually_verifiable=True),dict(id='F2',text='outdoors',visually_verifiable=True)])
        dump(source/'source_facts.jsonl',rr)
        folder=Path(self.tmp.name)/'blind'
        blind(self.root,source/'source_facts.jsonl',folder)
        with self.assertRaises(ValueError):
            score(folder,Path(self.tmp.name)/'score.json')
        review=rows(folder/'blind_review.jsonl')
        for r in review:
            r.update(reviewed=True,source_unsupported_claims=[],generated_claim_count=1)
            r['facts'][0]['judgment']='retained'
            r['facts'][1]['judgment']='unclear'
        dump(folder/'blind_review.jsonl',review)
        score(folder,Path(self.tmp.name)/'score.json')
        data=read(Path(self.tmp.name)/'score.json')
        self.assertTrue(all(g['mean_fact_retention']==.5 for g in data['groups'].values()))

    def test_collection_must_not_write_inside_original(self):
        from docci_baseline_v14.collect import collect
        with self.assertRaisesRegex(ValueError,'outside immutable'):
            collect(self.root,self.root/'bad_records.tar.gz',1)

    def test_training_history_positive_and_tampered_schedule(self):
        root = main_fixture(Path(self.tmp.name)/'main')
        self.assertIn('VERIFIED',verify(root)['status'])
        path = root/'text/legacy64/train/history.jsonl'
        hist = rows(path)
        hist[0]['batch_example_ids']=['train_1']
        dump(path,hist)
        seal(root)
        with self.assertRaisesRegex(ValueError,'presentations'):
            verify(root)

    def test_valid_target_count_not_free_padding(self):
        root = main_fixture(Path(self.tmp.name)/'main')
        path = root/'text/legacy64/train/history.jsonl'
        hist = rows(path)
        hist[0]['valid_target_tokens']=100
        dump(path,hist)
        seal(root)
        with self.assertRaisesRegex(ValueError,'token'):
            verify(root)

    def test_frozen_backbone_audit_required(self):
        p = self.root/'text/identity.json'
        atomic(p,dict(backbone_signature_before='a',backbone_signature_after='b'))
        seal(self.root)
        with self.assertRaisesRegex(ValueError,'Frozen backbone'):
            verify(self.root)


if __name__ == '__main__':
    unittest.main(verbosity=2)
