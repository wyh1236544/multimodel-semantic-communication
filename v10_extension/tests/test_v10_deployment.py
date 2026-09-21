"""Additive deployment and write-boundary tests; no external writes or models."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from v10.paths import PROJECT_ROOT, write_path, local_environment
from deployment.install_bundle import RELEASE, MANIFEST, install_archive, safe_target
import launch_v10


def make_bundle(root, payload):
    archive = root/'fixture.pyz'
    manifest = {'release': RELEASE, 'files': [
        {'path': p, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        for p, data in payload.items()]}
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('__main__.py', '# synthetic fixture')
        for name, data in payload.items():
            z.writestr('payload/'+name, data)
        z.writestr('payload/'+MANIFEST, json.dumps(manifest))
    return archive


class DeploymentTests(unittest.TestCase):
    def test_atomic_output_boundary_and_stale_temporary_preservation(self):
        from v10.io import atomic_json, begin_run
        with self.assertRaises(ValueError):
            atomic_json(PROJECT_ROOT.parent/'not_created.json', {'x': 1})
        with self.assertRaises(ValueError):
            begin_run(PROJECT_ROOT.parent/'not_created', None, None, PROJECT_ROOT)
        with tempfile.TemporaryDirectory(prefix='atomic_') as tmp:
            target = Path(tmp)/'record.json'
            atomic_json(target, {'first': 1})
            stale = target.with_name(target.name+'.tmp')
            stale.write_bytes(b'prior failed write')
            with self.assertRaises(FileExistsError):
                atomic_json(target, {'second': 2})
            self.assertEqual(json.loads(target.read_text()), {'first': 1})
            self.assertEqual(stale.read_bytes(), b'prior failed write')

    def test_additive_install_and_idempotent_retry(self):
        with tempfile.TemporaryDirectory(prefix='deploy_') as tmp:
            root = Path(tmp)/'multimodel'
            root.mkdir()
            old = root/'train.py'
            old.write_bytes(b'preserve original training code')
            payload = {'run_v10.sh': b'# new runner', 'v10_extension/a.py': b'# new model'}
            archive = make_bundle(root, payload)
            before = {p.relative_to(root).as_posix() for p in root.rglob('*')}
            self.assertTrue(install_archive(archive, root, dry_run=True)['dry_run'])
            self.assertEqual(before, {p.relative_to(root).as_posix() for p in root.rglob('*')})
            report = install_archive(archive, root)
            self.assertEqual(report['overwritten_files'], 0)
            times = {name: (root/name).stat().st_mtime_ns for name in payload}
            self.assertEqual(install_archive(archive, root)['new_files'], 0)
            self.assertEqual(times, {name: (root/name).stat().st_mtime_ns for name in payload})
            self.assertEqual(old.read_bytes(), b'preserve original training code')

    def test_collision_aborts_before_any_extraction(self):
        with tempfile.TemporaryDirectory(prefix='collision_') as tmp:
            root = Path(tmp)/'multimodel'
            root.mkdir()
            archive = make_bundle(root, {'v10_extension/a.py': b'new', 'run_v10.sh': b'new'})
            (root/'run_v10.sh').write_bytes(b'old user file')
            with self.assertRaises(FileExistsError):
                install_archive(archive, root)
            self.assertFalse((root/'v10_extension').exists())
            self.assertEqual((root/'run_v10.sh').read_bytes(), b'old user file')

    def test_unsafe_archive_paths_are_rejected(self):
        for value in ('../oops', '/tmp/file', 'models/model.py', 'v10_extension/../train.py',
                      'v10_extension\\oops', 'v10_extension//oops', 'C:/outside'):
            with self.assertRaises(ValueError):
                safe_target(PROJECT_ROOT, value)

    def test_output_outside_project_and_root_rejected(self):
        for target in (PROJECT_ROOT, PROJECT_ROOT.parent/'outside_v10.json', PROJECT_ROOT/'../escape'):
            with self.assertRaises(ValueError):
                write_path(target)
        self.assertEqual(write_path(PROJECT_ROOT/'experiment_results/new_run').parent.name, 'experiment_results')
        with patch('pathlib.Path.is_symlink', return_value=True):
            with self.assertRaises(ValueError):
                write_path(PROJECT_ROOT/'symlink/new.json')

    def test_launcher_cwd_inputs_and_no_path_override(self):
        with patch.dict(os.environ, {}, clear=False):
            for name in ('INITIAL_CHECKPOINT', 'DATA_ROOT', 'PRETRAINED_ROOT', 'SPLIT_MANIFEST'):
                os.environ.pop(name, None)
            old = Path.cwd()
            try:
                os.chdir(PROJECT_ROOT)
                plan = launch_v10.make_plan(['smoke', 'experiment_results/v10/not_created', '--dry-run'])
                self.assertEqual(plan['cwd'], str(PROJECT_ROOT))
                self.assertIn(str(PROJECT_ROOT/'v10_extension/train_v10.py'), plan['command'])
                self.assertIn(str(PROJECT_ROOT/'v10_extension/configs/v10_multichannel.json'), plan['command'])
                self.assertTrue(plan['dry_run'])
                with self.assertRaises(ValueError):
                    launch_v10.make_plan(['smoke', 'experiment_results/new', '--output-dir=/tmp/escape'])
                with self.assertRaises(ValueError):
                    launch_v10.make_plan(['train', str(PROJECT_ROOT.parent/'outside')])
                with self.assertRaises(ValueError):
                    launch_v10.make_plan(['splits', '--output', '../outside.json'])
            finally:
                os.chdir(old)

    def test_child_cache_environment_is_local_and_does_not_modify_parent(self):
        before = dict(os.environ)
        with tempfile.TemporaryDirectory(prefix='cache_') as tmp:
            root = Path(tmp).resolve()
            env = local_environment(root)
            for name in ('TMPDIR', 'HF_HOME', 'TORCH_HOME', 'CUDA_CACHE_PATH', 'XDG_CACHE_HOME'):
                self.assertTrue(Path(env[name]).is_dir())
                Path(env[name]).relative_to(root)
            self.assertEqual(env['PYTHONDONTWRITEBYTECODE'], '1')
            self.assertEqual(before, dict(os.environ))


if __name__ == '__main__':
    unittest.main()
