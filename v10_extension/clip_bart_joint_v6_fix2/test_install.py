"""Report fix installation contracts; no neural runtime or dependencies."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from install import install, sha, MODULE, NAMES

SOURCE = Path(__file__).resolve().parent
PROJECT = SOURCE.parents[1]


class Installer(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='joint_v6_fix2_install_', dir=str(PROJECT/'releases'))
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.root = self.parent/'multimodel'
        self.root.mkdir()
        self.patch = json.loads((SOURCE/'PATCH_MANIFEST.json').read_text())
        manifest = json.loads((SOURCE/'payload/MANIFEST.json').read_text())
        for name in self.patch['required_sources']:
            dest = self.root/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT/name, dest)
        target = self.root/MODULE
        target.mkdir()
        for name in manifest['files']:
            original = SOURCE/'previous'/name if name in NAMES else PROJECT/MODULE/name
            shutil.copyfile(original, target/name)
        shutil.copyfile(SOURCE/'previous/MANIFEST.json', target/'MANIFEST.json')
        handoff = self.root/'v10_extension/docs/HANDOFF.md'
        handoff.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/'previous/HANDOFF.md', handoff)
        # Unrelated historical result sentinel must not be touched.
        for attempt in ('01','02'):
            results = self.root/('experiment_results/v10_clip_bart_joint_v6_'+attempt)
            results.mkdir(parents=True)
            (results/'FAILED.json').write_text('{"original_failure": true}\n')
        self.before = {str(f.relative_to(self.root)): f.read_bytes() for f in self.root.rglob('*') if f.is_file()}

    def untouched(self):
        self.assertEqual(self.before, {str(f.relative_to(self.root)): f.read_bytes() for f in self.root.rglob('*') if f.is_file()})

    def test_install_backup_preserve_and_idempotent(self):
        install(SOURCE, self.root)
        records = self.root/'v10_extension/deployment_records'
        backups = list(records.glob('clip_bart_joint_v6_fix2_*'))
        self.assertEqual(len(backups), 1)
        for name in NAMES:
            self.assertEqual((backups[0]/name).read_bytes(), self.before[MODULE+'/'+name])
            self.assertEqual(sha((self.root/MODULE/name).read_bytes()), self.patch['replacements'][name]['after'])
        self.assertEqual((backups[0]/'HANDOFF.md').read_bytes(), self.before['v10_extension/docs/HANDOFF.md'])
        for name, data in self.before.items():
            if name not in [MODULE+'/'+n for n in NAMES]+['v10_extension/docs/HANDOFF.md']:
                self.assertEqual((self.root/name).read_bytes(), data)
        after = {str(f.relative_to(self.root)): f.read_bytes() for f in self.root.rglob('*') if f.is_file()}
        install(SOURCE, self.root)
        self.assertEqual(after, {str(f.relative_to(self.root)): f.read_bytes() for f in self.root.rglob('*') if f.is_file()})

    def test_dry_run_no_writes(self):
        install(SOURCE, self.root, True); self.untouched()

    def test_unknown_report_preserved(self):
        p = self.root/MODULE/'report.py'; p.write_bytes(b'user edit\n')
        self.before[MODULE+'/report.py'] = p.read_bytes()
        with self.assertRaises(FileExistsError): install(SOURCE, self.root)
        self.untouched()

    def test_configuration_drift_preserved(self):
        p = self.root/MODULE/'config.json'; p.write_bytes(p.read_bytes()+b' ')
        self.before[MODULE+'/config.json'] = p.read_bytes()
        with self.assertRaises(ValueError): install(SOURCE, self.root)
        self.untouched()

    def test_historical_dependency_drift_preserved(self):
        name = next(iter(self.patch['required_sources']))
        p = self.root/name; p.write_bytes(p.read_bytes()+b'# drift\n')
        self.before[name] = p.read_bytes()
        with self.assertRaises(ValueError): install(SOURCE, self.root)
        self.untouched()

    def test_mixed_version_rejected(self):
        p = self.root/MODULE/'report.py'; shutil.copyfile(SOURCE/'payload/report.py', p)
        self.before[MODULE+'/report.py'] = p.read_bytes()
        with self.assertRaises(FileExistsError): install(SOURCE, self.root)
        self.untouched()

    def test_symlink_target_rejected(self):
        p = self.root/MODULE/'report.py'; outside = self.parent/'outside.py'
        shutil.copyfile(p, outside); original = outside.read_bytes()
        p.unlink(); p.symlink_to(outside)
        with self.assertRaises(ValueError): install(SOURCE, self.root)
        self.assertEqual(outside.read_bytes(), original)
        self.assertFalse((self.root/'v10_extension/deployment_records').exists())

    def test_patch_payload_tamper_rejected(self):
        clone = self.parent/'altered_patch'; shutil.copytree(SOURCE, clone, ignore=shutil.ignore_patterns('__pycache__'))
        p = clone/'payload/report.py'; p.write_bytes(p.read_bytes()+b'# altered\n')
        with self.assertRaises(ValueError): install(clone, self.root)
        self.untouched()


if __name__ == '__main__':
    unittest.main(verbosity=2)
