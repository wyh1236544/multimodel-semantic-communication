"""CPU regression uses the REAL original producer expression, not richer fixtures."""
import ast
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import shutil
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_rx_bridge_v16.core import PROJECT, FOLDER as ORIGINAL, atomic, read, sha, seal, lines, load
from docci_rx_bridge_v16 import report
from docci_rx_bridge_v16.test_integration import fixture
from docci_text_continue_v11.metrics import metrics
from docci_rx_bridge_v16_fix1.compat import installed, write_receipt, ORIGINAL_MANIFEST
from docci_rx_bridge_v16_fix1.launch import resumed_config


def producer_cache(base):
    root, cache = fixture(base)
    tree = ast.parse((ORIGINAL/'cache.py').read_text())
    appends = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
               isinstance(n.func, ast.Attribute) and n.func.attr == 'append' and
               isinstance(n.func.value, ast.Name) and n.func.value.id == 'baseline']
    if len(appends) != 1:
        raise AssertionError('Pinned production baseline producer changed')
    code = compile(ast.Expression(appends[0].args[0]), '<trusted-production-baseline-expression>', 'eval')
    output = []
    for r in lines(cache/'baseline.jsonl'):
        generated = {k: r[k] for k in ('received_caption', 'generated_token_ids',
                     'chunk_generated_token_ids', 'stopped_by_capacity')}
        source = dict(example_id=r['example_id'], official_split=r['official_split'],
                      text=r['source_text'], text_sha256=r['source_text_sha256'])
        value = eval(code, {'__builtins__': {'dict': dict}, 'row': source,
                           'result': generated, 'metrics': metrics})
        if 'source_tokens' in value:
            raise AssertionError('Regression must reproduce exact missing-column original writer')
        output.append(value)
    (cache/'baseline.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in output))
    # Synthetic fixture only; mirrors a cache before its failed verification.
    (cache/'VERIFIED.json').unlink()
    seal(cache)
    return root, cache


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='v16_fix1_test_', dir=str(PROJECT/'releases'))
        self.base = Path(self.tmp.name)
        self.root, self.cache = producer_cache(self.base)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reproduce_KeyError_then_verify_without_original_writes(self):
        with self.assertRaisesRegex(KeyError, 'source_tokens'):
            report.verify(self.cache)
        before = {p: sha(p) for p in self.cache.rglob('*') if p.is_file()}
        with installed() as verify:
            result = verify(self.cache)
        self.assertTrue(result['arrays_verified'])
        self.assertEqual(result['compatibility_fix1']['normalizations'][0]['normalized_rows'], 10)
        self.assertTrue(all(sha(p) == digest for p, digest in before.items()))
        self.assertEqual(sha(ORIGINAL/'MANIFEST.json'), ORIGINAL_MANIFEST)

    def test_full_training_record_collection_restore_and_provenance(self):
        from docci_rx_bridge_v16 import collect
        with installed() as verify:
            result = verify(self.root)
            self.assertFalse(result['full_semantic_success'])
            out = self.base/'records.tar.gz'
            before = {p: sha(p) for p in self.root.rglob('*') if p.is_file()}
            collect.collect(self.root, out)
            self.assertTrue(all(sha(p) == digest for p, digest in before.items()))
        with tarfile.open(out) as tar:
            names = tar.getnames()
            self.assertTrue(any('compatibility_snapshot/' in n for n in names))
            self.assertFalse(any(Path(n).suffix in ('.pth', '.npz', '.pt') for n in names))
            summary = json.load(tar.extractfile('v16_records/summary.json'))
            self.assertIn('compatibility_fix1', summary)

    def test_wrong_present_count_is_NOT_repaired(self):
        rows = lines(self.cache/'baseline.jsonl')
        rows[0]['source_tokens'] = 9
        (self.cache/'baseline.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        seal(self.cache)
        with installed() as verify:
            with self.assertRaises(ValueError):
                verify(self.cache)

    def test_wrong_index_count_rejected_by_original_frame_check(self):
        ix = read(self.cache/'index.json')
        ix['records'][0]['source_tokens'] = 9
        atomic(self.cache/'index.json', ix)
        seal(self.cache)
        with installed() as verify:
            with self.assertRaisesRegex(ValueError, 'Source coverage'):
                verify(self.cache)

    def test_mismatched_source_hash_rejected(self):
        rows = lines(self.cache/'baseline.jsonl')
        rows[0]['source_text'] = 'unrelated synthetic source'
        (self.cache/'baseline.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        seal(self.cache)
        with installed() as verify:
            with self.assertRaises(ValueError):
                verify(self.cache)

    def test_context_restores_globals_after_failure(self):
        from docci_rx_bridge_v16 import run, collect
        before = report.lines, report.verify, run.snapshot, collect.verify, collect.copy_light
        with self.assertRaises(RuntimeError):
            with installed():
                raise RuntimeError('synthetic abort')
        self.assertEqual(before, (report.lines, report.verify, run.snapshot, collect.verify, collect.copy_light))

    def test_receipt_exclusive_and_wrong_receipt_preserved(self):
        with installed() as verify:
            result = verify(self.cache)
        write_receipt(self.cache, result)
        digest = sha(self.cache/'VERIFIED.json')
        write_receipt(self.cache, result)
        self.assertEqual(digest, sha(self.cache/'VERIFIED.json'))
        with self.assertRaises(FileExistsError):
            write_receipt(self.cache, dict(result, arrays_verified=False))
        self.assertEqual(digest, sha(self.cache/'VERIFIED.json'))

    def test_length_NOT_estimated_from_generated_caption(self):
        rows = lines(self.cache/'baseline.jsonl')
        rows[0]['received_caption'] = ''
        (self.cache/'baseline.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with installed():
            augmented = report.lines(self.cache/'baseline.jsonl')
        self.assertEqual(augmented[0]['source_tokens'], 5)
        self.assertNotIn('source_tokens', lines(self.cache/'baseline.jsonl')[0])

    def test_resumed_effective_configuration_identical(self):
        c = read(self.cache/'config.json')
        self.assertEqual(load(resumed_config(c), smoke=True), c)
        with self.assertRaises(ValueError):
            resumed_config(dict(c, train_samples=16))

    def test_resume_routes_existing_cache_without_recreation(self):
        # Orchestration-only mock: real CPU report children; GPU child replaced
        # by copied TEST_ONLY fixtures, never considered neural/GPU evidence.
        from docci_rx_bridge_v16_fix1 import launch
        from docci_rx_bridge_v16_fix1.compat import snapshot
        old_popen = launch.subprocess.Popen
        commands = []
        class Done:
            pid = 999999
            def wait(self, timeout=None):
                return 0
            def poll(self):
                return 0
        def child(cmd, **kwargs):
            commands.append(cmd)
            if str(ORIGINAL/'launch.py') in cmd or any(Path(x).name == 'test_fix.py' for x in cmd):
                return Done()
            if 'run' in cmd:
                at = cmd.index('run')
                self.assertEqual(cmd[at+1], 'smoke_train')
                target = Path(cmd[at+2])
                shutil.copytree(self.root, target)
                snapshot(target)
                seal(target)
                return Done()
            return old_popen(cmd, **kwargs)
        output = self.base/'resumed'
        before = sha(self.cache/'baseline.jsonl')
        argv = ['launch.py', 'resume-smoke', str(self.cache), str(output), '--max-hours', '1']
        with patch.object(sys, 'argv', argv), patch.object(launch.subprocess, 'Popen', child):
            launch.main()
        self.assertEqual(before, sha(self.cache/'baseline.jsonl'))
        self.assertTrue(read(output/'LAUNCH_FINISHED.json')['cache_reused'])
        self.assertTrue((output/'smoke_train/VERIFIED.json').is_file())
        run_commands = [c for c in commands if 'run' in c]
        self.assertEqual(len(run_commands), 1)
        self.assertNotIn('cache', run_commands[0])

    def test_resume_rejects_existing_original_training_before_children(self):
        from docci_rx_bridge_v16_fix1 import launch
        (self.cache.parent/'smoke_train').mkdir()
        output = self.base/'must_not_exist'
        argv = ['launch.py', 'resume-smoke', str(self.cache), str(output)]
        with patch.object(sys, 'argv', argv), patch.object(launch.subprocess, 'Popen') as child:
            with self.assertRaises(FileExistsError):
                launch.main()
            child.assert_not_called()
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
