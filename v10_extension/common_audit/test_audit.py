"""Synthetic software tests only; no semantic-performance claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import audit
import install
import zipfile
import hashlib


def identity(ablation):
    return {'status': 'complete', 'checkpoint_sha256': 'fixture-checkpoint',
            'split_sha256': 'fixture-split', 'config': {'fixture': True},
            'source_sha256': {'fixture': 'source'},
            'args': dict(ablation=ablation, seed=42, split='validation',
                         samples=2, repeats=2, profiles=['awgn'], snr_list=[0.0], no_amp=True)}


class AuditTests(unittest.TestCase):
    def test_installer_idempotency_collision_and_traversal(self):
        with tempfile.TemporaryDirectory(dir=str(audit.ROOT)) as temp:
            root = Path(temp) / 'multimodel'
            root.mkdir()
            (root / 'run_v10.sh').touch()
            archive = Path(temp) / 'fixture.pyz'
            name = 'v10_extension/common_audit/fixture.txt'
            def pack(entries):
                with zipfile.ZipFile(archive, 'w') as z:
                    z.writestr('__main__.py', '# fixture, never executed')
                    z.writestr('MANIFEST.json', json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in entries.items()}))
                    for k, v in entries.items():
                        z.writestr(k, v)
            pack({name: b'fixture'})
            self.assertEqual(install.install(archive, root), 1)
            self.assertEqual(install.install(archive, root), 0)
            other = 'v10_extension/common_audit/new.txt'
            pack({other: b'new', name: b'different'})
            with self.assertRaises(ValueError):
                install.install(archive, root)
            self.assertFalse((root / other).exists())
            self.assertEqual((root / name).read_bytes(), b'fixture')
            pack({'v10_extension/common_audit/../../outside.txt': b'bad'})
            with self.assertRaises(ValueError):
                install.install(archive, root)

    def test_pair_rejects_wrong_seed_weight_and_incomplete(self):
        a, b = identity('common_only'), identity('common_only_random')
        audit.validate_pair(a, b)
        for field, value in [('status', 'running'), ('checkpoint_sha256', 'different')]:
            bad = copy.deepcopy(b)
            bad[field] = value
            with self.assertRaises(ValueError):
                audit.validate_pair(a, bad)
        b['args']['seed'] = 43
        with self.assertRaises(ValueError):
            audit.validate_pair(a, b)

    def test_base_hashes(self):
        self.assertGreater(len(audit.check_sources()), 10)

    def test_report_pair_direction_archive_and_html_escape(self):
        import compare_v10
        with tempfile.TemporaryDirectory(dir=str(audit.ROOT)) as temp:
            out = Path(temp)
            for arm, ablation, metric in [('real', 'common_only', .5),
                                          ('random', 'common_only_random', .1)]:
                d = out / arm
                d.mkdir()
                audit.save(d / 'run_identity.json', identity(ablation))
                summary = {'common_to_text_retrieval_r1': metric, 'retrieval_gallery_size': 2}
                audit.save(d / 'summary.json', {'summary': {'awgn/0.0': summary}})
                rows = []
                for image in [1, 2]:
                    for repeat in [0, 1]:
                        rows.append(dict(image_id=image, profile='awgn', nominal_snr_db=0.0,
                                         repeat=repeat, physical_unit='complex', forward_uses=352,
                                         reverse_rf_real_uses=64, tx_average_power=1,
                                         psnr=metric, ssim=metric, bleu2=metric,
                                         common_cosine=metric, common_caption_bleu2=metric,
                                         image_file='images/fixture.png',
                                         source_caption_evaluation_only='<script>bad</script>',
                                         common_caption='fixture only'))
                with (d / 'samples.jsonl').open('x', encoding='utf-8') as f:
                    f.write('\n'.join(json.dumps(r) for r in rows))
            # Fixture lacks a real manifest/checkpoint. Only bypass artifact verification;
            # actual matching, measured budgets and bootstrap/report code run unchanged.
            with patch('verify_v10_run.verify'), patch.object(compare_v10, 'verify'):
                audit.report(out, {'missing': out / 'missing'})
                value = audit.read(out / 'audit_summary.json')
                delta = value['real_minus_random']['summary']['awgn/0.0']['delta_candidate_minus_baseline']
                self.assertAlmostEqual(delta['common_cosine']['mean'], .4)
                self.assertEqual(value['real_minus_random']['summary']['awgn/0.0']['independent_image_units'], 2)
                self.assertFalse(value['existing_full_absolute']['missing']['available'])
                page = (out / 'captions.html').read_text(encoding='utf-8')
                self.assertNotIn('<script>', page)
                self.assertIn('&lt;script&gt;', page)
                self.assertTrue((out / 'return_results.zip').is_file())
                with self.assertRaises(FileExistsError):
                    audit.report(out, {})
                path = out / 'random' / 'samples.jsonl'
                text = path.read_text(encoding='utf-8').replace('"forward_uses": 352', '"forward_uses": 353')
                path.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    compare_v10.compare(out / 'random', out / 'real')


if __name__ == '__main__':
    unittest.main(verbosity=2)
