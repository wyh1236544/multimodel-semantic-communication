"""TEST_ONLY fabricated storage and CPU orchestration; never neural evidence."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_segment_bridge_v17 import core, report
from docci_segment_bridge_v17.core import atomic, sha, read, batches, condition, noise_seed, controls, eval_conditions, geometry
from docci_rx_bridge_v16 import core as old
from docci_rx_bridge_v16.run import snapshot as old_snapshot
from docci_rx_bridge_v16.collect import copy_light
from docci_rx_bridge_v16.test_report import row as synthetic_row


def save_lines(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n')


def old_cache(base):
    cache = base/'old_cache'; cache.mkdir()
    old_snapshot(cache, old.check_source())
    atomic(cache/'identity.json', dict(action='cache', smoke=False, TEST_ONLY=True, source_manifest_sha256=sha(old.FOLDER/'MANIFEST.json')))
    atomic(cache/'config.json', old.load())
    atomic(cache/'source_identity.json', dict(annotation_sha256=old.DOCCI_SHA256, official_counts=dict(train=9647, test=5000, qual_dev=100, qual_test=100)))
    atomic(cache/'parents.json', {k: dict(sha256=v['sha256']) for k, v in old.PARENTS.items()})
    for task in ('text', 'image'):
        atomic(cache/(task+'_frozen_audit.json'), dict(before='TEST_ONLY', after='TEST_ONLY', unchanged=True))
    (cache/'images').mkdir()
    digest = hashlib.sha256(b'a red car').hexdigest()
    entries = [dict(example_id=('train_' if i < 9647 else 'qual_dev_')+str(i), official_split='train' if i < 9647 else 'qual_dev',
               text='a red car', text_sha256=digest, source_tokens=5, text_file='TEST_ONLY.npz', image_file='TEST_ONLY.npz',
               text_file_sha256='TEST_ONLY_T', image_file_sha256='TEST_ONLY_I') for i in range(9747)]
    atomic(cache/'index.json', dict(records=entries))
    baseline = []
    for r in entries[:8]+entries[9647:]:
        v = synthetic_row(r['example_id']); v.update(official_split=r['official_split'], source_text_sha256=digest)
        # Faithfully retain the actual historical producer's omitted source_tokens.
        v.pop('source_tokens')
        baseline.append(v)
    save_lines(cache/'baseline.jsonl', baseline); old.seal(cache)
    return cache, entries


def stats(eid, channel, phase='validation', epoch=0):
    nominal = None if channel == 'clean' else int(channel.split('_')[1])
    power = 0. if nominal is None else 10.**(-nominal/10.)
    gate = np.array([.99]+[.001]*8, np.float32)
    return dict(nominal_snr_db=nominal, realized_snr_db=nominal, noise_variance=power,
                signal_power=1., noise_power=power, noise_seed=noise_seed(phase, epoch, eid, channel),
                tx_sha256='TEST_ONLY_WIRE', observed_sha256='TEST_ONLY_WIRE' if nominal is None else 'TEST_ONLY_OBS',
                gate_sha256=hashlib.sha256(gate.tobytes()).hexdigest(),
                memory_sha256=hashlib.sha256(np.zeros((9, 32, 768), np.float32).tobytes()).hexdigest(),
                gate_recomputed_from_received_memory=True, clean_gate_reused=False)


def inputs_fixture(pipeline, cache, entries):
    root = pipeline/'smoke_inputs'; c = core.load(True)
    core.snapshot(root, dict(stage='inputs', smoke=True, cache=str(cache), dimensions=768,
                            cache_index_sha256=sha(cache/'index.json'), TEST_ONLY=True))
    atomic(root/'config.json', c)
    copy_light(cache, root/'cache_reference', lambda: None)
    atomic(root/'cache_receipt.json', dict(arrays_verified=True, smoke=False,
           light_manifest_sha256=sha(cache/'LIGHT_MANIFEST.json'), TEST_ONLY_FABRICATED_RECEIPT=True))
    counts = np.zeros((8, 9), np.int64); counts[:, 0] = 3
    teacher = np.zeros((8, 10, 768), np.float32); teacher[:, 0] = 1; teacher[:, 9] = 1
    with (root/'teacher.npz').open('xb') as f:
        np.savez_compressed(f, teacher=teacher, counts=counts)
    atomic(root/'TEACHER.json', dict(file='teacher.npz', sha256=sha(root/'teacher.npz'),
           shape=[8, 10, 768], supervision_only=True, validation_teachers=False,
           index=[dict(example_id=r['example_id'], text_sha256=r['text_sha256']) for r in entries[:8]]))
    (root/'observations').mkdir(); (root/'images').mkdir()
    obs, baseline = [], []
    for channel in eval_conditions(True):
        for r in entries[9647:9649]:
            name = 'observations/'+r['example_id']+'__'+channel+'.npz'
            with (root/name).open('xb') as f:
                np.savez_compressed(f, memory=np.zeros((9, 32, 768), np.float32), gate=np.array([.99]+[.001]*8, np.float32))
            entry = dict(example_id=r['example_id'], official_split='qual_dev', source_text=r['text'], source_text_sha256=r['text_sha256'],
                         source_tokens=5, condition=channel, file=name, sha256=sha(root/name), stats=stats(r['example_id'], channel))
            obs.append(entry)
            v = synthetic_row(r['example_id']); v.update(entry)
            baseline.append(v)
    atomic(root/'observations.json', obs); save_lines(root/'baseline.jsonl', baseline)
    atomic(root/'frozen_audit.json', dict(before='TEST_ONLY', after='TEST_ONLY', unchanged=True))
    core.seal(root)
    return root


def arm_fixture(root, inputs, entries, arm):
    c = core.load(True); train = entries[:8]; ids = [r['example_id'] for r in train]
    core.snapshot(root, dict(stage='arm', arm=arm, smoke=True, inputs=str(inputs),
                  inputs_light_sha256=sha(inputs/'LIGHT_MANIFEST.json'), TEST_ONLY=True))
    atomic(root/'config.json', c)
    atomic(root/'trainability.json', dict(TRAIN_ids=ids, ce_updates=5, alignment_updates=2,
                                        trainable_parameters=core.expected_parameters()))
    history = []
    for step, (epoch, ix) in enumerate(list(batches(8, 1, 1, 42))[:5]):
        channels = [condition(arm, 'ce', epoch, ids[i]) for i in ix]
        history.append(dict(update=step+1, total_updates=5, epoch=epoch+1, batch_ids=[ids[i] for i in ix], valid_target_tokens=21,
                            loss=1., CE=1., gradient_norm=1., lr=c['ce_lr']*(.1+.9*(1-step/5)),
                            conditions=channels, observations=[stats(ids[i], channel, 'ce', epoch) for i, channel in zip(ix, channels)],
                            donor_ids=[ids[(i+1+core.seed('ce-donor', epoch, ids[i]) % 7) % 8] for i in ix]))
    save_lines(root/'history.jsonl', history)
    align = [dict(update=step+1, batch_ids=[ids[i] for i in ix], conditions=[condition(arm, 'align', epoch, ids[i]) for i in ix],
                  loss=1., contrast=1., segment=1., matching=1.) for step, (epoch, ix) in enumerate(batches(8, 4, 1, 42))]
    save_lines(root/'alignment.jsonl', align)
    atomic(root/'learning_audit.json', dict(passed=True, groups={k: dict(changed=True, nonzero_gradient=True,
           parameters=n) for k, n in core.parameter_groups().items()}))
    atomic(root/'frozen_audit.json', dict(before='TEST_ONLY', after='TEST_ONLY', unchanged=True))
    with (root/'last.pth').open('xb') as f:f.write(b'TEST_ONLY_NOT_A_NEURAL_CHECKPOINT')
    atomic(root/'TRAIN_COMPLETE.json', dict(updates=5, alignment_updates=2, unique_training_ids=5,
           actual_disk_roundtrip=True, fixed_final=True, checkpoint_sha256=sha(root/'last.pth'), TEST_ONLY=True))
    subset = entries[9647:9649]
    atomic(root/'matrix.json', dict(ids=[r['example_id'] for r in subset], controls=list(controls(True)), conditions=list(eval_conditions(True)), geometry=geometry(arm)))
    by = {(r['example_id'], r['condition']): r for r in read(inputs/'observations.json')}
    output = []
    for channel in eval_conditions(True):
        for i, r in enumerate(subset):
            for control in controls(True):
                donor = subset[1-i] if control.startswith('donor') else r
                v = synthetic_row(r['example_id'])
                v.update(arm=arm, control=control, condition=channel, source_text_sha256=r['text_sha256'],
                     off_reused=control == 'off',
                     source_tokens=5, official_split='qual_dev', oracle=core.kind(arm) == 'oracle_clip',
                     own_observation_sha256=by[r['example_id'], channel]['sha256'], own_stats=by[r['example_id'], channel]['stats'],
                     first_free_content_probability=np.array([.99]+[.001]*8, np.float32).astype(float).tolist(),
                     auxiliary_donor_id=donor['example_id'] if control.startswith('donor') else None,
                     auxiliary_cache_sha256=donor[('text' if core.kind(arm) == 'self' else 'image')+'_file_sha256'])
                output.append(v)
    save_lines(root/'rows.jsonl', output); atomic(root/'EVAL_COMPLETE.json', dict(rows=len(output)))
    core.seal(root)


class Storage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='v17_TEST_ONLY_', dir=str(core.PROJECT/'releases'))
        self.base = Path(self.tmp.name); self.cache, self.entries = old_cache(self.base)
        self.index_patch = patch.object(report, 'CACHE_INDEX', sha(self.cache/'index.json')); self.index_patch.start()
        self.root = self.base/'pipeline'; self.root.mkdir()
        self.inputs = inputs_fixture(self.root, self.cache, self.entries)
        for arm in core.ARMS:arm_fixture(self.root/('smoke_'+arm), self.inputs, self.entries, arm)
        self.state = dict(status='SMOKE_COMPLETE', TEST_ONLY=True,
                         stages={name: 'VERIFIED' for name in ['smoke_inputs']+['smoke_'+a for a in core.ARMS]})
        atomic(self.root/'PIPELINE.json', self.state)

    def tearDown(self):
        self.index_patch.stop(); self.tmp.cleanup()

    def test_real_CPU_verifier_and_collection_restore(self):
        from docci_segment_bridge_v17.collect import collect
        before = {p: sha(p) for p in self.root.rglob('*') if p.is_file()}
        verified = report.pipeline(self.root)
        self.assertFalse(verified['full_semantic_success']); self.assertEqual(len(verified['completed_stages']), 5)
        archive = self.base/'records.tar.gz'; collect(self.root, archive)
        self.assertTrue(all(sha(p) == digest for p, digest in before.items()))
        restored = self.base/'restored'
        with tarfile.open(archive) as tar:
            for m in tar.getmembers():
                self.assertTrue(m.isfile()); self.assertNotIn('..', Path(m.name).parts)
                self.assertNotIn(Path(m.name).suffix, ('.npz', '.pth'))
                dst = restored/m.name; dst.parent.mkdir(parents=True, exist_ok=True)
                with dst.open('xb') as f:f.write(tar.extractfile(m).read())
        bundle = restored/'v17_records'
        for name, digest in read(bundle/'ARCHIVE_MANIFEST.json').items():self.assertEqual(sha(bundle/name), digest)
        light = report.pipeline(bundle/'pipeline', arrays=False)
        self.assertEqual(light['arm_results'], dict(verified['arm_results'], **{
            k: dict(v, arrays_verified=False) for k, v in verified['arm_results'].items()}))
        with self.assertRaises(FileExistsError):collect(self.root, archive)

    def test_teacher_tampering_and_light_boundary(self):
        with (self.inputs/'teacher.npz').open('ab') as f:f.write(b'TEST_ONLY corruption')
        with self.assertRaises(ValueError):report.pipeline(self.root)
        self.assertFalse(report.pipeline(self.root, arrays=False)['arrays_verified'])

    def test_resealed_wrong_donor_and_wrong_channel_rejected(self):
        path = self.root/'smoke_D_oracle_mixed'; rows = core.lines(path/'rows.jsonl')
        rows[1]['auxiliary_donor_id'] = rows[1]['example_id']
        save_lines(path/'rows.jsonl', rows); core.seal(path)
        with self.assertRaises(ValueError):report.pipeline(self.root)

    def test_resealed_wrong_training_token_denominator_rejected(self):
        path = self.root/'smoke_C_self_mixed'; rows = core.lines(path/'history.jsonl')
        rows[0]['valid_target_tokens'] = 1
        save_lines(path/'history.jsonl', rows); core.seal(path)
        with self.assertRaises(ValueError):report.pipeline(self.root)

    def test_complete_label_cannot_hide_missing_main_arms(self):
        atomic(self.root/'PIPELINE.json', dict(self.state, status='COMPLETE'))
        with self.assertRaises(ValueError):report.pipeline(self.root)

    def test_partial_does_not_promote_smoke_to_main(self):
        atomic(self.root/'PIPELINE.json', dict(self.state, status='PARTIAL_BUDGET'))
        v = report.pipeline(self.root)
        self.assertEqual(v['pipeline_status'], 'PARTIAL_BUDGET'); self.assertEqual(v['comparisons'], {})

    def test_partial_input_is_saved_but_never_rendered_as_result(self):
        from docci_segment_bridge_v17.collect import collect
        incomplete = self.root/'inputs'; incomplete.mkdir()
        with (incomplete/'baseline.jsonl').open('x') as f:f.write('{ TEST_ONLY interrupted JSON')
        state = copy.deepcopy(self.state); state['status'] = 'PARTIAL_BUDGET'; state['stages']['inputs'] = 'UNVERIFIED_PARTIAL'
        atomic(self.root/'PIPELINE.json', state)
        archive = self.base/'partial_records.tar.gz'; collect(self.root, archive)
        with tarfile.open(archive) as tar:
            text = tar.extractfile('v17_records/pipeline/index.html').read().decode()
            self.assertIn('PARTIAL_BUDGET', text); self.assertNotIn('interrupted JSON', text)


class OwnedProcess(unittest.TestCase):
    def test_stop_only_its_owned_child(self):
        from docci_segment_bridge_v17.launch import stop_owned
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'], start_new_session=True)
        try:
            stop_owned(child); self.assertIsNotNone(child.poll())
        finally:
            if child.poll() is None:child.kill(); child.wait()


if __name__ == '__main__':
    unittest.main(verbosity=2)
