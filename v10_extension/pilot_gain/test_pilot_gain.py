"""Stdlib synthetic report tests; not real GPU/model evaluation."""
import ast
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT))
from pilot_gain.run import STRATEGIES, parser, validate
from pilot_gain.report import make_report, write_tables


class ReportTests(unittest.TestCase):
    def setUp(self):
        cache = EXT.parent/'.cache/v10/pilot_gain_tests'
        cache.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='pilot_gain_test_', dir=str(cache)))
        self.args = dict(samples=2, repeats=2, seed=42, profiles=['awgn'], snr_list=[10.],
                         data_root='data', vae_path='vae', text_model_path='clip')
        self.put(self.root/'plan.json', {'args': self.args})
        for strategy, mode in STRATEGIES.items():
            d = self.root/strategy
            d.mkdir()
            args = dict(self.args, split='validation', no_amp=True, ablation='full')
            self.put(d/'COMPLETE.json', {'rows': 4, 'expected_rows': 4})
            self.put(d/'run_identity.json', dict(status='complete', args=args,
                config={'v10': dict(fixed_mode=mode, csi='pilot', equalizer='mmse')},
                checkpoint_sha256='ckpt', split_sha256='split', source_sha256={'a.py': 'source'}))
            self.put(d/'summary.json', {'diffusion_executed': False, 'source_caption_used_by_receiver': False,
                'summary': {'awgn/10.0': {'common_to_text_retrieval_r1': .5}}})
            rows = []
            for i in (1, 2):
                for r in (0, 1):
                    rows.append(dict(image_id=i, profile='awgn', nominal_snr_db=10., repeat=r,
                        strategy=strategy, psnr=20.+(2 if strategy == 'adaptive' else 0),
                        ssim=.7, bleu2=.3, token_acc=.4, common_caption_bleu2=.2,
                        common_cosine=.8, header_rejected=0.,
                        requested_mode={None: 'reconstruction', 0: 'reconstruction', 1: 'hybrid', 2: 'generation'}[mode],
                        sender_mode='reconstruction', source_caption_evaluation_only='source '+str(i),
                        pilot_draw_sha256=str(i)+str(r), payload_rng_sha256='rng', payload_shape=[1, 288, 16],
                        physical_unit='rf_complex_symbol', payload_uses=2304, pilot_uses=32, header_uses=64,
                        forward_uses=2400, reverse_rf_real_uses=64))
            self.rows(d, rows)

    def tearDown(self):
        shutil.rmtree(str(self.root))  # exact disposable test-owned directory only

    def put(self, path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    def rows(self, directory, rows):
        (directory/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')

    def mutate(self, callback):
        d = self.root/'fixed_hybrid'
        rows = [json.loads(s) for s in (d/'samples.jsonl').read_text().splitlines()]
        callback(rows)
        self.rows(d, rows)

    def test_known_gain_and_cluster_interval(self):
        out = make_report(self.root, 100)
        e = out['groups']['awgn/10.0']['gains']['fixed_hybrid']['psnr']
        self.assertEqual(e['mean_delta'], 2.)
        self.assertEqual(e['ci95_pointwise'], [2., 2.])
        self.assertEqual(out['unique_images'], 2)
        write_tables(self.root, out)
        self.assertTrue((self.root/'REPORT.md').is_file())

    def test_missing_sample_rejected(self):
        self.mutate(lambda rows: rows.pop())
        with self.assertRaises(ValueError): make_report(self.root, 100)

    def test_duplicate_rejected(self):
        self.mutate(lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])))
        with self.assertRaises(ValueError): make_report(self.root, 100)

    def test_unpaired_noise_rejected(self):
        self.mutate(lambda rows: rows[0].update(payload_rng_sha256='different'))
        with self.assertRaises(ValueError): make_report(self.root, 100)

    def test_resource_mismatch_rejected(self):
        self.mutate(lambda rows: rows[0].update(forward_uses=9999))
        with self.assertRaises(ValueError): make_report(self.root, 100)

    def test_wrong_requested_mode_rejected(self):
        self.mutate(lambda rows: rows[0].update(requested_mode='generation'))
        with self.assertRaises(ValueError): make_report(self.root, 100)

    def test_checkpoint_mismatch_rejected(self):
        p = self.root/'fixed_hybrid/run_identity.json'
        v = json.loads(p.read_text()); v['checkpoint_sha256'] = 'different'; self.put(p, v)
        with self.assertRaises(ValueError): make_report(self.root, 100)

    def test_nonfinite_rejected(self):
        self.mutate(lambda rows: rows[0].update(psnr=float('nan')))
        with self.assertRaises(ValueError): make_report(self.root, 100)

    def test_wrong_lattice_rejected(self):
        self.mutate(lambda rows: rows[0].update(nominal_snr_db=5.))
        with self.assertRaises(ValueError): make_report(self.root, 100)

    def test_python38_grammar(self):
        for p in Path(__file__).parent.glob('*.py'):
            ast.parse(p.read_text(), feature_version=(3, 8))


if __name__ == '__main__':
    unittest.main(verbosity=2)
