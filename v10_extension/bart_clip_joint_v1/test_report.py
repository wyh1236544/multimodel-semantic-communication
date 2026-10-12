"""Synthetic result audit tests; fixtures are not experimental observations."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
EXT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXT))
from bart_clip_joint_v1 import VERSION
from bart_clip_joint_v1.core import load, make_row, quality_gate
from bart_clip_joint_v1.report import checked_groups, inspect, seal, verify


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))


def fixture(root):
    cfg = load(); cfg.update(diagnostic_updates=8, feature_warmup_updates=2)
    items = [dict(image_id=i, text=t, covered_text=t, truncated=False, source_sha256=str(i), source_utf8_bytes=len(t)) for i, t in [(1, 'A cat sleeps.'), (2, 'Two dogs run.')]]
    rows = [make_row(item, arm, 'clean', item['text'] if arm == 'matched' else '' if arm == 'zero' else items[1-i]['text']) for arm in ('matched', 'zero', 'shuffled') for i, item in enumerate(items)]
    path = root/'overfit_controls.jsonl'; path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    write(root/'overfit_gate.json', quality_gate(checked_groups(path), cfg))
    write(root/'run_identity.json', dict(version=VERSION, status='diagnostic_ready', wall_seconds=1.))
    write(root/'design.json', dict(version=VERSION, config=cfg, smoke=False, stage='diagnostic',
        train_ids=[1, 2], validation_ids=[], payload_complex_symbols=2304, analog_repeats=1, raw_utf8_wire_bytes=0))
    write(root/'frozen_verification.json', dict(bart_and_reference_unchanged=True, old_unchanged=True, clip_unchanged=True, no_test_source_used=True, gradients_verified=True))
    write(root/'real_backbone_gradient_gate.json', dict(passed=True, groups={'fixture': True}))
    stage = root/'text_private_overfit'; stage.mkdir()
    write(stage/'learning_audit.json', dict(passed=True, frozen_unchanged=True, groups={'fixture':dict(parameters=1, nonzero_gradient=True, changed=True)}))
    (stage/'history.jsonl').write_text('\n'.join(json.dumps(dict(step=i)) for i in range(1, 9))+'\n')
    write(stage/'TRAIN_COMPLETE.json', dict(steps=8, optimizer_updates=4))
    return rows


class ResultContracts(unittest.TestCase):
    def test_recompute_and_seal(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); fixture(root); inspect(root); seal(root)
            self.assertEqual(verify(root), 'diagnostic_ready')
    def test_hash_drift(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); fixture(root); seal(root); (root/'overfit_controls.jsonl').write_text('')
            with self.assertRaises(ValueError): verify(root)
    def test_wrong_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); rows = fixture(root); rows[0]['covered_wer'] = .8
            (root/'overfit_controls.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaises(ValueError): inspect(root)
    def test_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); rows = fixture(root); rows.append(rows[0]); (root/'overfit_controls.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaises(ValueError): inspect(root)
    def test_false_formal_completion(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); fixture(root); write(root/'run_identity.json', dict(version=VERSION, status='complete'))
            with self.assertRaises(ValueError): inspect(root)
    def test_missing_gradient(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); fixture(root); write(root/'real_backbone_gradient_gate.json', dict(passed=False))
            with self.assertRaises(ValueError): inspect(root)
    def test_frozen_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); fixture(root); write(root/'frozen_verification.json', {})
            with self.assertRaises(ValueError): inspect(root)
    def test_incomplete_updates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); fixture(root); write(root/'text_private_overfit/TRAIN_COMPLETE.json', dict(steps=8, optimizer_updates=3))
            with self.assertRaises(ValueError): inspect(root)


if __name__ == '__main__': unittest.main(verbosity=2)
