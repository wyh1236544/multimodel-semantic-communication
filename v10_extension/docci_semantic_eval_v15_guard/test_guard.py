"""Synthetic readiness/idempotence contracts, not V14 training-quality evidence."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import guard
import install as installer


class GuardContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=str(guard.PROJECT))
        self.root = Path(self.tmp.name)
        self.real_project = guard.PROJECT
        self.patch = mock.patch.object(guard, 'PROJECT', self.root)
        self.patch.start()
        self.run = self.root / 'experiment_results/v14_text_pilot_01'

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def ready(self):
        main = self.run / 'main'
        main.mkdir(parents=True)
        path = self.root / 'v10_extension/docci_baseline_v14'
        path.mkdir(parents=True)
        shutil.copyfile(self.real_project / 'v10_extension/docci_baseline_v14/MANIFEST.json', path / 'MANIFEST.json')
        (main / 'source_snapshot').mkdir()
        shutil.copyfile(path / 'MANIFEST.json', main / 'source_snapshot/MANIFEST.json')
        guard.write(self.run / 'launch_identity.json', dict(action='train', task='text', profile='pilot',
             text_arms=guard.ARMS, physical_gpu=1, smoke_weights_reused=False))
        guard.write(self.run / 'LAUNCH_FINISHED.json', dict(final_root=str(main), engineering_success_NOT_recovery_claim=True))
        guard.write(main / 'design.json', dict(mode='train', task='text', text_arms=guard.ARMS))
        guard.write(main / 'LIGHT_MANIFEST.json', {'SYNTHETIC_fixture': 'not_real_frame_verification'})
        guard.write(main / 'RUN_FINISHED.json', dict(manifest_sha256=guard.sha(main / 'LIGHT_MANIFEST.json')))
        guard.write(main / 'run_identity.json', dict(mode='train', task='text', outputs=guard.expected_outputs(),
            source_manifest_sha256=guard.sha(path / 'MANIFEST.json')))
        groups = {}
        for name in guard.expected_outputs():
            controls = ['native'] if name == 'text/native_validation' else ['matched', 'shuffled', 'zero']
            groups[name] = {s + '/' + c: {'n': 100} for s in ('greedy', 'deployment') for c in controls}
        receipt = dict(status='BASELINE_RECORDS_VERIFIED_NOT_QUALITY_SUCCESS', mode='train', FSO=False, groups=groups)
        text = ''.join('VERIFY ' + p + '\n' for p in guard.expected_outputs()) + json.dumps(receipt)
        (self.run / 'main_verify.log').write_text(text)
        return receipt

    def edit(self, path, key, value):
        doc = guard.read(path)
        doc[key] = value
        path.write_text(json.dumps(doc))

    def v13(self):
        p = self.root / '_v13_server_cpu_offline_01/prepared'
        p.mkdir(parents=True)
        guard.write(p / 'PREPARED.json', {'schema': 'SYNTHETIC_TEST_ONLY'})
        return p

    def test_incomplete_or_not_started_does_not_create_run_or_output(self):
        self.assertEqual(guard.inspect(self.run)['state'], 'NOT_READY')
        self.assertFalse(self.run.exists())
        with mock.patch.object(guard, 'source_check'), mock.patch.object(guard, 'load_v15') as load:
            self.assertEqual(guard.prepare_once(self.run, self.root / 'missing')['state'], 'NOT_READY')
            load.assert_not_called()
        self.assertFalse(guard.output_for(self.run).exists())

    def test_main_finished_without_outer_verification_is_not_ready(self):
        (self.run / 'main').mkdir(parents=True)
        guard.write(self.run / 'main/RUN_FINISHED.json', {'complete': True})
        self.assertEqual(guard.inspect(self.run)['state'], 'NOT_READY')

    def test_failure_preserved_even_if_finished_marker_exists(self):
        self.ready()
        guard.write(self.run / 'FAILED.json', {'reason': 'SYNTHETIC'})
        self.assertEqual(guard.inspect(self.run)['state'], 'FAILED_PRESERVED')

    def test_ready_exact_text_pilot_and_receipt(self):
        self.ready()
        result = guard.inspect(self.run)
        self.assertEqual(result['state'], 'READY_FOR_READ_ONLY_IMPORT')
        self.assertEqual(len(result['inputs']), 6)

    def test_smoke_image_or_different_arms_rejected(self):
        self.ready()
        p = self.run / 'launch_identity.json'
        original = p.read_bytes()
        for key, value in [('action', 'smoke'), ('task', 'image'), ('profile', 'full'), ('text_arms', ['legacy64'])]:
            p.write_bytes(original)
            self.edit(p, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError): guard.inspect(self.run)

    def test_changed_manifest_or_source_or_final_root_rejected(self):
        self.ready()
        for path, key in [(self.run / 'main/RUN_FINISHED.json', 'manifest_sha256'),
                          (self.run / 'main/run_identity.json', 'source_manifest_sha256'),
                          (self.run / 'LAUNCH_FINISHED.json', 'final_root')]:
            before = path.read_bytes()
            self.edit(path, key, 'changed')
            with self.subTest(key=key), self.assertRaises(ValueError): guard.inspect(self.run)
            path.write_bytes(before)

    def test_report_incomplete_missing_or_smoke_counts_rejected(self):
        receipt = self.ready()
        p = self.run / 'main_verify.log'
        original = p.read_bytes()
        p.write_text('unverified')
        with self.assertRaises(ValueError): guard.inspect(self.run)
        receipt['groups']['text/legacy64/validation']['greedy/matched']['n'] = 2
        p.write_text(''.join('VERIFY ' + n + '\n' for n in guard.expected_outputs()) + json.dumps(receipt))
        with self.assertRaises(ValueError): guard.inspect(self.run)
        p.write_bytes(original)

    def test_prepare_once_and_repeated_call_reuses_without_producer(self):
        self.ready()
        prepared = self.v13()
        calls = []
        def producer(old, output, main):
            calls.append(str(main))
            output.mkdir()
            guard.write(output / 'SYNTHETIC.json', {})
        verifier = mock.Mock()
        with mock.patch.object(guard, 'source_check'):
            first = guard.prepare_once(self.run, prepared, producer, verifier)
            again = guard.prepare_once(self.run, prepared, producer, verifier)
        self.assertEqual(first['state'], 'IMPORTED_ONCE')
        self.assertEqual(again['state'], 'ALREADY_IMPORTED_NO_REPEAT')
        self.assertEqual(len(calls), 1)
        self.assertEqual(verifier.call_count, 2)

    def test_existing_partial_refuses_work_and_does_not_cleanup(self):
        self.ready()
        prepared = self.v13()
        out = guard.output_for(self.run)
        out.mkdir()
        guard.write(out / 'IMPORT_STARTED.json', {'pid': 0})
        producer = mock.Mock()
        with mock.patch.object(guard, 'source_check'):
            result = guard.prepare_once(self.run, prepared, producer, mock.Mock())
        self.assertEqual(result['state'], 'OUTPUT_RESERVED_OR_PARTIAL')
        producer.assert_not_called()
        self.assertTrue((out / 'IMPORT_STARTED.json').exists())

    def test_concurrent_second_caller_cannot_reenter_producer(self):
        self.ready()
        prepared = self.v13()
        entered = []
        def producer(old, output, main):
            inner = guard.prepare_once(self.run, prepared, mock.Mock(side_effect=AssertionError('duplicate')), mock.Mock())
            entered.append(inner['state'])
            output.mkdir()
        with mock.patch.object(guard, 'source_check'):
            result = guard.prepare_once(self.run, prepared, producer, mock.Mock())
        self.assertEqual(result['state'], 'IMPORTED_ONCE')
        self.assertEqual(entered, ['OUTPUT_RESERVED_OR_PARTIAL'])

    def test_producer_failure_is_preserved_not_retried(self):
        self.ready()
        prepared = self.v13()
        producer = mock.Mock(side_effect=RuntimeError('synthetic stop'))
        with mock.patch.object(guard, 'source_check'):
            with self.assertRaises(RuntimeError): guard.prepare_once(self.run, prepared, producer, mock.Mock())
            result = guard.prepare_once(self.run, prepared, producer, mock.Mock())
        self.assertEqual(producer.call_count, 1)
        self.assertEqual(result['state'], 'OUTPUT_RESERVED_OR_PARTIAL')
        self.assertTrue((guard.output_for(self.run) / 'IMPORT_FAILED.json').exists())

    def test_changed_original_inputs_not_reused(self):
        self.ready()
        prepared = self.v13()
        with mock.patch.object(guard, 'source_check'):
            guard.prepare_once(self.run, prepared, lambda a, b, d: b.mkdir(), mock.Mock())
            (prepared / 'PREPARED.json').write_text('{}')
            with self.assertRaises(ValueError): guard.prepare_once(self.run, prepared, mock.Mock(), mock.Mock())

    def test_wrong_scope_main_directory_and_symlink_rejected(self):
        self.ready()
        with self.assertRaises(ValueError): guard.inspect(self.run / 'main')
        link = self.root / 'experiment_results/link'
        link.symlink_to(self.run, target_is_directory=True)
        with self.assertRaises(ValueError): guard.inspect(link)

    def test_production_has_no_process_training_gpu_network_entry(self):
        tree = ast.parse((guard.FOLDER / 'guard.py').read_text())
        imported = [n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)]
        self.assertFalse(set(imported) & {'subprocess', 'torch', 'transformers', 'requests', 'socket'})
        self.assertNotIn('train', [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)])


class InstallerContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=str(guard.PROJECT))
        self.root = Path(self.tmp.name)
        self.bundle, self.project = self.root / 'bundle', self.root / 'project'
        self.project.mkdir()
        self.name = 'v10_extension/docci_semantic_eval_v15_guard/guard.py'
        p = self.bundle / 'payload' / self.name
        p.parent.mkdir(parents=True)
        p.write_text('SYNTHETIC_INSTALLER_FIXTURE')
        guard.write(self.bundle / 'PAYLOAD.json', dict(files={self.name: guard.sha(p)}, dependencies={}))

    def tearDown(self): self.tmp.cleanup()

    def test_additive_install_and_identical_repeat_zero_writes(self):
        self.assertEqual(installer.install(self.bundle, self.project), 1)
        before = (self.project / self.name).stat().st_mtime_ns
        self.assertEqual(installer.install(self.bundle, self.project), 0)
        self.assertEqual((self.project / self.name).stat().st_mtime_ns, before)

    def test_conflict_aborts_before_any_new_file(self):
        other = 'v10_extension/docci_semantic_eval_v15/common.py'
        p = self.bundle / 'payload' / other
        p.parent.mkdir(parents=True)
        p.write_text('approved')
        doc = guard.read(self.bundle / 'PAYLOAD.json')
        doc['files'][other] = guard.sha(p)
        (self.bundle / 'PAYLOAD.json').write_text(json.dumps(doc))
        dest = self.project / other
        dest.parent.mkdir(parents=True)
        dest.write_text('user changes')
        with self.assertRaises(FileExistsError): installer.install(self.bundle, self.project)
        self.assertFalse((self.project / self.name).exists())
        self.assertEqual(dest.read_text(), 'user changes')

    def test_forbidden_v14_path_and_symlink_refused(self):
        (self.bundle / 'PAYLOAD.json').write_text(json.dumps(dict(files={'v10_extension/docci_baseline_v14/run.py': 'x'}, dependencies={})))
        with self.assertRaises(ValueError): installer.install(self.bundle, self.project)
        with self.assertRaises(ValueError): installer.target(self.project, '../outside')


if __name__ == '__main__':
    unittest.main(verbosity=2)
