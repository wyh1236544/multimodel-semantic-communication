"""Synthetic CPU render/collection tests, NOT actual model performance."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT))
sys.path.insert(0, str(EXT.parent))
from docci_semantic_probe_v12_collect_fix1 import collect as fixed

TEST_ROOT = fixed.core.PROJECT / '.cache/v10/docci_semantic_probe_v12_collect_fix1_tests'


def fixture(root, mode, tasks):
    result = dict(task_results={mode: {}})
    if 'text' in tasks:
        result['task_results'][mode]['text'] = dict(arms={}, WER_B1_minus_B0=dict(mean=0.01))
        for arm in ('B0_ce', 'B1_ce_kd'):
            at = root / mode / 'text' / arm / 'validation'
            at.mkdir(parents=True)
            row = dict(example_id='a', control='matched', source_text='<source> & text', received_caption='<received> & text')
            (at / 'rows.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
    if 'image' in tasks:
        result['task_results'][mode]['image'] = dict(decision='SYNTHETIC_TEST_ONLY', LPIPS='NOT_MEASURED')
        for tag in ('P0_parent', 'A0_image_only_evaluation', 'A1_oracle_text_evaluation'):
            at = root / mode / 'image' / tag
            at.mkdir(parents=True)
            row = dict(example_id='a', tag=tag, condition='clean', control='none', source_text='<oracle>',
                image_file=tag + '/test.png', source_view_file='source.png', psnr=20., ssim=.5)
            (at / 'rows.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
        # The renderer only links these files; record-validation is tested separately.
        (root / mode / 'image' / 'source.png').write_bytes(b'synthetic-placeholder-NOT-a-model-image')
    return result


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob('*') if p.is_file()}


class CollectionRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEST_ROOT.mkdir(parents=True, exist_ok=True)

    def assert_restored(self):
        self.assertFalse(hasattr(fixed.report, 'json'))
        self.assertIs(fixed.original.render, fixed.report.render)

    def test_original_bug_reproduced_for_both_nonempty_task_branches(self):
        for task in ('text', 'image'):
            with self.subTest(task=task), tempfile.TemporaryDirectory(dir=str(TEST_ROOT)) as d:
                root = Path(d)
                result = fixture(root, 'main', [task])
                self.assert_restored()
                with self.assertRaisesRegex(NameError, 'json'):
                    fixed.report.render(root, result)

    def test_actual_legacy_packaging_with_all_render_modes_and_task_combinations(self):
        for mode in ('smoke', 'main'):
            for tasks in (['text'], ['image'], ['text', 'image']):
                with self.subTest(mode=mode,tasks=tasks), tempfile.TemporaryDirectory(dir=str(TEST_ROOT)) as d:
                    root = Path(d) / 'v10_synthetic'; root.mkdir()
                    result = fixture(root, mode, tasks)
                    (root / 'last.pth').write_bytes(b'not-pickle; must be excluded')
                    before = hashes(root); destination = root.parent / 'records.tar.gz'
                    # Only GPU-record verification is mocked. Original collector I/O,
                    # original renderer and source checks all run for real.
                    with mock.patch.object(fixed.original, 'verify', return_value=result) as verifier:
                        fixed.collect(root, destination, 1)
                        verifier.assert_called_once_with(root)
                    self.assertEqual(before, hashes(root)); self.assert_restored()
                    self.assertNotIn('torch', sys.modules)
                    with tarfile.open(destination) as archive:
                        names = archive.getnames(); prefix = 'v10_docci_semantic_probe_v12_records/'
                        page = archive.extractfile(prefix+'index.html').read().decode('utf-8')
                        self.assertIn('未经过传输', page)
                        self.assertNotIn('last.pth', ' '.join(names))
                        self.assertIn(prefix+'COLLECTOR_FIX1.json', names)
                        self.assertIn(prefix+'collector_fix1_source/collect.py', names)
                        if 'text' in tasks:
                            self.assertIn('&lt;source&gt; &amp; text', page)
                            self.assertIn('&lt;received&gt; &amp; text', page)
                            self.assertNotIn('<received>', page)
                        if 'image' in tasks:
                            self.assertIn('SYNTHETIC_TEST_ONLY', page)
                            self.assertIn('mask25 不是 FSO', page)

    def test_verification_failure_propagates_without_staging_or_source_edits(self):
        with tempfile.TemporaryDirectory(dir=str(TEST_ROOT)) as d:
            root = Path(d) / 'run'; root.mkdir(); before = hashes(root)
            destination = root.parent / 'fail.tar.gz'
            with mock.patch.object(fixed.original,'verify',side_effect=ValueError('invalid records')):
                with self.assertRaisesRegex(ValueError,'invalid records'):
                    fixed.collect(root, destination)
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name(destination.name+'.contents').exists())
            self.assertEqual(before,hashes(root)); self.assert_restored()

    def test_existing_archive_or_staging_is_preserved(self):
        for kind in ('archive','contents'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=str(TEST_ROOT)) as d:
                root=Path(d)/'run'; root.mkdir(); destination=root.parent/'existing.tar.gz'
                target=destination if kind=='archive' else destination.with_name(destination.name+'.contents')
                target.write_bytes(b'original evidence')
                with mock.patch.object(fixed.original,'verify') as verifier:
                    with self.assertRaises(FileExistsError): fixed.collect(root,destination)
                    verifier.assert_not_called()
                self.assertEqual(target.read_bytes(),b'original evidence'); self.assert_restored()

    def test_renderer_failure_still_propagates_and_restores_runtime_globals(self):
        with tempfile.TemporaryDirectory(dir=str(TEST_ROOT)) as d:
            root=Path(d)/'run'; root.mkdir(); result=fixture(root,'main',['text'])
            with mock.patch.object(fixed.original,'verify',return_value=result), mock.patch.object(fixed.report,'jsonlines',side_effect=ValueError('damaged render input')):
                with self.assertRaisesRegex(ValueError,'damaged render input'):
                    fixed.collect(root,root.parent/'renderfail.tar.gz')
            self.assert_restored()

    def test_unchanged_verifier_and_real_source_contract(self):
        self.assertIs(fixed.original.verify,fixed.report.verify)
        manifest=fixed.check_sources()
        self.assertTrue(manifest['original_V12_manifest_sha256'])
        self.assert_restored()


if __name__ == '__main__':
    unittest.main(verbosity=2)
