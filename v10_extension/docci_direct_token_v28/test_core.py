"""CPU-only contracts and adversarial verifier fixtures. No real-data claims."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_direct_token_v28.core import (PROJECT, FOLDER, load, read, sha, safe, seed,
    geometry, schedule, training_snr, trials, WIDTH, PARENTS)
from docci_direct_token_v28.channel import noise, constants, array_sha
from docci_direct_token_v28.report import stats_check, verify_eval, verify_train, aggregate
from docci_text_continue_v11.metrics import metrics


class Contracts(unittest.TestCase):
    def test_protocol_and_no_resource_inflation(self):
        self.assertEqual(geometry('image')['total_complex_uses'], 4128)
        self.assertEqual(geometry('text')['total_complex_uses'], 9248)
        self.assertEqual(geometry('image')['repetitions'], 1)
        self.assertEqual(geometry('text')['common_reals'], 0)
        self.assertFalse(geometry('image')['FSO'])
        self.assertEqual(set(load()['low_snr_db']), {-8, -5, -2, 0, 5, 10, 20})

    def test_finite_config_and_wrong_protocol_rejected(self):
        with tempfile.TemporaryDirectory(prefix='v28-test-', dir=str(PROJECT)) as temp:
            for key, value in [('low_snr_db', [-5]), ('reserve_free_gib', float('nan')),
                               ('text_lr', .01), ('text_accumulation', 0)]:
                c = read(FOLDER/'config.json'); c[key] = value
                file = Path(temp)/'c.json'; file.write_text(json.dumps(c))
                with self.assertRaises(ValueError): load(file)

    def test_schedule_matching_and_negative_snr_coverage(self):
        c = load(profile='full'); stream = schedule(c, 'text')
        self.assertEqual(len(stream), 2560)
        self.assertEqual(len(schedule(c, 'image')), 10240)
        self.assertEqual(stream, schedule(c, 'text'))
        values = {training_snr(c, 'low_snr', e, 'train_'+str(i)) for e in range(3) for i in range(100)}
        self.assertEqual(values, set(c['low_snr_db']))
        self.assertFalse(set(c['normal_snr_db']) & {-8, -5, -2})

    def test_awgn_zero_clean_energy_and_shared_draws(self):
        self.assertTrue(np.all(noise(8192, None, 42) == 0))
        a, b = noise(200000, -5, 42), noise(200000, 0, 42)
        self.assertAlmostEqual(float(np.mean(a.astype(float)**2)), 10**.5, delta=.04)
        np.testing.assert_allclose(a, b*np.float32(10**.25), atol=1e-6)
        for invalid in [float('nan'), float('inf'), -11, 21]:
            with self.assertRaises(ValueError): noise(8, invalid, 1)

    def test_pilots_and_actual_frame_tampering(self):
        tx = np.ones(8192, np.float32); eps, s = constants('image', -5, 123)
        s.update(signal_power=1., realized_snr_db=10*np.log10(1/s['noise_power']),
            tx_sha256=array_sha(tx), observed_sha256=array_sha(tx+eps))
        stats_check(s, 'image', -5, 123, tx)
        bad = copy.deepcopy(s); bad['pilot_noise_power'] += .01
        with self.assertRaises(ValueError): stats_check(bad, 'image', -5, 123, tx)
        with self.assertRaises(ValueError): stats_check(s, 'image', -5, 123, tx*.9)

    def test_safe_output_scope(self):
        for value in [PROJECT, PROJECT/'../oops', PROJECT.parent/'oops']:
            with self.assertRaises(ValueError): safe(value)

    def fixture(self, root):
        c = load(profile='smoke'); root.mkdir(); (root/'frames').mkdir()
        val = [dict(example_id='qual_dev_'+str(i), text='A red car.', text_sha256='fake-'+str(i),
                    source_tokens=6) for i in range(2)]
        values = []
        for i, source in enumerate(val):
            with (root/'frames'/(source['example_id']+'.npz')).open('wb') as f:
                np.savez(f, tx=np.ones(WIDTH['text'], np.float32))
            for snr, repeat, control in trials(c):
                donor = val[(i+1) % 2]['example_id'] if control == 'shuffled' else source['example_id']
                epsilon, stats = constants('text', snr, seed(c['seed'], 'eval', 'text', donor, snr, repeat))
                tx = np.ones(WIDTH['text'], np.float32)
                stats.update(signal_power=1., realized_snr_db=None if snr is None else 10*np.log10(1/stats['noise_power']),
                    tx_sha256=array_sha(tx), observed_sha256=array_sha(tx+epsilon))
                r = dict(example_id=source['example_id'], official_split='qual_dev', task='text', arm='frozen',
                    snr_db=snr, repeat=repeat, control=control, donor_id=donor, stats=stats,
                    receiver_wire_sha256=array_sha(np.zeros_like(tx)) if control == 'zero' else stats['observed_sha256'],
                    geometry=geometry('text'), input_truncated=False, source_text=source['text'],
                    source_text_sha256=source['text_sha256'], source_tokens=source['source_tokens'],
                    received_caption='A red car.', chunk_generated_token_ids=[[2, 0, 7, 2]]*9,
                    stopped_by_capacity=False, **metrics(source['text'], 'A red car.'))
                values.append(r)
        identity = dict(task='text', arm='frozen', config=c, ids=[r['example_id'] for r in val],
                        geometry=geometry('text'), checkpoint_sha256=PARENTS['text']['sha256'])
        (root/'identity.json').write_text(json.dumps(identity))
        self.write_eval(root, values)
        return c, val, values

    def write_eval(self, root, values):
        (root/'rows.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in values))
        (root/'EVAL_COMPLETE.json').write_text(json.dumps(dict(rows=len(values),
            rows_sha256=sha(root/'rows.jsonl'), identity_sha256=sha(root/'identity.json'))))

    def test_evaluation_full_frame_verifier_and_negative_cases(self):
        with tempfile.TemporaryDirectory(prefix='v28-test-', dir=str(PROJECT)) as temp:
            root = Path(temp)/'evaluation'; c, val, values = self.fixture(root)
            verify_eval(root, 'text', 'frozen', c, val, lambda: None, True)
            for mutation in ('missing', 'duplicate', 'wrong_metric', 'wrong_noise', 'wrong_donor'):
                bad = copy.deepcopy(values)
                if mutation == 'missing': bad.pop()
                elif mutation == 'duplicate': bad[-1] = bad[0]
                elif mutation == 'wrong_metric': bad[0]['normalized_wer'] = .1
                elif mutation == 'wrong_noise': bad[0]['stats']['noise_seed'] += 1
                else: bad[0]['donor_id'] = val[1]['example_id']
                self.write_eval(root, bad)
                with self.assertRaises(ValueError): verify_eval(root, 'text', 'frozen', c, val, lambda: None, True)

    def test_bootstrap_clusters_noise_repeats_within_description(self):
        c = load(profile='pilot'); allrows = {}
        for arm, error in [('frozen', .8), ('normal_range', .7), ('low_snr', .6)]:
            allrows[arm] = [dict(example_id=str(i), snr_db=s, repeat=r, control='matched', psnr=20+(1-error), ssim=.5, LPIPS=None)
                for i in range(2) for s in [None]+c['eval_snr_db'] for r in range(3)]
        result = aggregate(allrows, c, 'image')
        self.assertEqual(result['effects']['-5']['low_vs_normal_range']['independent_descriptions'], 2)
        self.assertAlmostEqual(result['effects']['-5']['low_vs_normal_range']['mean'], .1)
        self.assertFalse(result['semantic_success'])

    def test_partial_collect_excludes_weights_and_arrays(self):
        import tarfile
        from docci_direct_token_v28.collect import collect
        with tempfile.TemporaryDirectory(prefix='v28-test-', dir=str(PROJECT)) as temp:
            root = Path(temp)/'failed'; root.mkdir()
            (root/'PIPELINE.json').write_text(json.dumps(dict(status='FAILED')))
            (root/'last.pth').write_bytes(b'fake-weight-not-a-real-model')
            (root/'frame.npz').write_bytes(b'fake-array')
            dest = Path(temp)/'records.tar.gz'
            metadata = collect(root, dest, partial=True)
            self.assertTrue(metadata['diagnostics_only'])
            self.assertFalse(metadata['summary']['completed'])
            with tarfile.open(dest) as t:
                self.assertFalse(any(m.name.endswith(('.pth', '.npz')) for m in t.getmembers()))
                self.assertTrue(all(m.isfile() for m in t.getmembers()))
            self.assertTrue(dest.with_name(dest.name+'.sha256').is_file())
            with self.assertRaises(FileExistsError): collect(root, dest, partial=True)

    def test_training_verifier_rejects_loss_and_tx_rx_audit_corruption(self):
        with tempfile.TemporaryDirectory(prefix='v28-test-', dir=str(PROJECT)) as temp:
            root = Path(temp); c = load(profile='smoke'); task = 'image'; arm = 'low_snr'
            train = [dict(example_id='train_'+str(i)) for i in range(c['train_samples'])]
            hist = []
            for step, (epoch, ix) in enumerate(schedule(c, task)):
                observed = []
                for i in ix:
                    eid = train[i]['example_id']; views = []
                    for view, snr, weight in [('clean', None, .25), ('noisy', training_snr(c, arm, epoch, eid), .75)]:
                        tx = np.ones(WIDTH[task], np.float32)
                        eps, s = constants(task, snr, seed(c['seed'], 'train', task, epoch, eid, view))
                        s.update(signal_power=1., realized_snr_db=None if snr is None else 10*np.log10(1/s['noise_power']),
                            tx_sha256=array_sha(tx), observed_sha256=array_sha(tx+eps))
                        views.append(dict(view=view, weight=weight, **s))
                    observed.append(dict(example_id=eid, views=views))
                total = len(schedule(c, task))
                hist.append(dict(update=step+1, total_updates=total, epoch=epoch, denominator=len(ix),
                    batch_ids=[train[i]['example_id'] for i in ix], loss=.3, clean_loss=.3, noisy_loss=.3,
                    lr=c['image_lr']*(.1+.9*(1-step/total)), gradient_norm=.1, observations=observed))
            def write(name, value): (root/name).write_text(json.dumps(value))
            write('identity.json', dict(task=task, arm=arm, parent_sha256=PARENTS[task]['sha256'], config=c,
                train_ids=[x['example_id'] for x in train], two_views=True, cross_modal=False, trainable_parameters=2))
            write('TRAIN_COMPLETE.json', dict(updates=2, fixed_final=True, actual_disk_roundtrip=True))
            audit = dict(frozen_unchanged=True, backbone_before='a', backbone_after='a',
                         groups={n:dict(parameters=1, gradient=True, changed=True) for n in ('encoder','decoder')})
            write('learning_audit.json', audit); write('REPLAY.json', dict(passed=True))
            (root/'history.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in hist))
            verify_train(root, task, arm, c, train, lambda: None, False)
            bad = copy.deepcopy(audit); bad['groups']['encoder']['gradient'] = False
            write('learning_audit.json', bad)
            with self.assertRaises(ValueError): verify_train(root, task, arm, c, train, lambda: None, False)
            write('learning_audit.json', audit); hist[0]['loss'] = .4
            (root/'history.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in hist))
            with self.assertRaises(ValueError): verify_train(root, task, arm, c, train, lambda: None, False)

    def test_collection_failure_recoverable_without_training(self):
        from unittest.mock import patch
        from docci_direct_token_v28.collect import collect
        with tempfile.TemporaryDirectory(prefix='v28-test-', dir=str(PROJECT)) as temp:
            root = Path(temp)/'complete'; root.mkdir()
            (root/'PIPELINE.json').write_text(json.dumps(dict(status='COLLECTION_FAILED', stages={'train':'PRODUCED'})))
            with patch('docci_direct_token_v28.collect.verify', return_value=dict(completed=True, status='COMPLETE', tasks={})):
                collect(root, Path(temp)/'collect02.tar.gz')
            self.assertEqual(read(root/'PIPELINE.json')['status'], 'COMPLETE')
            self.assertTrue(read(root/'VERIFIED_COMPLETE.json')['independent_verified'])


if __name__ == '__main__': unittest.main(verbosity=2)
