"""Local non-Torch tests; no claim of pretrained/GPU numeric execution."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
EXT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXT))
from bart_clip_joint_v1.core import load, TEMPLATE, offline_snapshot, digest, quality_gate, make_row, file_under


class CoreContracts(unittest.TestCase):
    def test_fixed_budget(self):
        c = load(); self.assertEqual((c['common_slots']+c['image_slots']+c['text_slots'])*c['width']//2, 2304)
    def altered(self, key, value):
        c = load(); c[key] = value
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'config.json'; p.write_text(json.dumps(c))
            with self.assertRaises(ValueError): load(p)
    def test_unknown(self): self.altered('new_setting', 1)
    def test_bool(self): self.altered('train_samples', True)
    def test_nonfinite(self): self.altered('learning_rate', float('inf'))
    def test_capacity(self): self.altered('max_tokens', 77)
    def test_no_ce(self): self.altered('feature_warmup_updates', 1200)
    def test_fraction(self): self.altered('memory_margin_gate', 1.)
    def test_limits(self): self.altered('diagnostic_samples', 100)
    def test_gate_uses_memory(self):
        c = load(); g = {k: dict(covered_wer=v) for k, v in [('matched', .1), ('zero', .7), ('shuffled', .9)]}
        self.assertTrue(quality_gate(g, c)['passed']); g['shuffled']['covered_wer'] = .1
        self.assertFalse(quality_gate(g, c)['passed'])
    def test_wer_not_bounded_accuracy(self):
        item = dict(image_id=1, text='one', covered_text='one', truncated=False, source_sha256='sha', source_utf8_bytes=3)
        r = make_row(item, 'test', 'clean', 'two three four'); self.assertGreater(r['covered_wer'], 1.)
    def test_python38_and_receiver_signature(self):
        for p in Path(__file__).parent.glob('*.py'): ast.parse(p.read_text(), feature_version=(3, 8))
        tree = ast.parse(Path(__file__).with_name('model.py').read_text())
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'receive')
        self.assertEqual([n.arg for n in fn.args.args], ['self', 'wire', 'reliability', 'pilot_snr', 'available', 'text_only'])
    def test_offline_missing_no_download(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError): offline_snapshot(d, 'bart')
    def test_unsafe_path(self):
        with self.assertRaises(ValueError): file_under(Path('.'), '../escape')
    def test_offline_hash_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = dict(model_type='bart', d_model=768, vocab_size=50265, encoder_layers=6, decoder_layers=6,
                       bos_token_id=0, eos_token_id=2, pad_token_id=1, decoder_start_token_id=2)
            (root/'config.json').write_text(json.dumps(cfg))
            for n in ('pytorch_model.bin', 'vocab.json', 'merges.txt'): (root/n).write_text('fixture, NOT model weights')
            (root/'SHA256SUMS').write_text(digest(root/'pytorch_model.bin')+'  pytorch_model.bin\n')
            self.assertTrue(offline_snapshot(root, 'bart')['offline_only'])
            (root/'pytorch_model.bin').write_text('drift')
            with self.assertRaises(ValueError): offline_snapshot(root, 'bart')


if __name__ == '__main__': unittest.main(verbosity=2)
