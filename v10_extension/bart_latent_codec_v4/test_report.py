"""Synthetic report positives/negatives, NOT neural success evidence."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
EXT = Path(__file__).resolve().parents[1]; PROJECT = EXT.parent; sys.path.insert(0, str(EXT))
from bart_latent_codec_v4 import VERSION, ARMS
from bart_latent_codec_v4.core import load, row, gate, geometry, batch_indices, digest
from bart_latent_codec_v4.report import verify, finish, groups
from bart_latent_codec_v4.collect import collect


def save(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f: json.dump(v, f)


def write_rows(p, rows):
    with p.open('w', encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r)+'\n')


class Reports(unittest.TestCase):
    def location(self): return tempfile.TemporaryDirectory(prefix='v4_report_test_', dir=str(PROJECT/'releases'))
    def fixture(self, root, native_good=True):
        c = load(); c.update(diagnostic_samples=2, batch_size=2, diagnostic_updates=8)
        width = 64; ids = [11, 12]
        items = [dict(image_id=i, annotation_id=i+20, split='train', text=t, covered_text=t, covered_tokens=5,
            source_sha256='fixture-only', source_utf8_bytes=len(t.encode()), truncated=False)
            for i, t in zip(ids, ['a cat sits', 'a dog runs'])]
        snap = root/'source_snapshot/model.py'; snap.parent.mkdir()
        with snap.open('w') as f: f.write('# synthetic report fixture\n')
        save(root/'run_identity.json', dict(version=VERSION, status='screen_complete', config=c, width=width,
            smoke=False, expand=False, source_sha256={'model.py': digest(snap)}))
        save(root/'design.json', dict(config=c, width=width, arms=list(ARMS), text_only=True, decoder_frozen=True,
            protocol=geometry(width), diagnostic_view=dict(image_ids=ids)))
        save(root/'bart_interface_audit.json', dict(passed=True, incremental_max_logit_error=0., cached_source_max_logit_error=0.))
        save(root/'frozen_verification.json', dict(passed=True, BART_unchanged=True, original_data_split_unchanged=True))
        nr = [row(item, 'native', 'matched', dict(text=item['text'] if native_good else '',
            stopped_by_capacity=False, receiver_failure=False), width, source_memory_oracle_train_only=True) for item in items]
        write_rows(root/'native.jsonl', nr)
        from bart_latent_codec_v4.core import summarize
        ns = summarize(nr)
        save(root/'native_gate.json', dict(passed=native_good, groups=ns, smoke_quality_bypass=False))
        statuses = {}
        for arm in ARMS:
            if not native_good:
                statuses[arm] = dict(status='native_stop', diagnostic_passed=False, expanded=False); continue
            folder = root/arm; folder.mkdir(); tf = folder/'diagnostic'; tf.mkdir()
            save(folder/'initial_state_hashes.json', {'query': 'same synthetic hash'})
            save(tf/'TRAIN_COMPLETE.json', dict(updates=8, checkpoint_sha256='weights excluded fixture', wall_seconds=1))
            history = [dict(update=i+1, total_updates=8, batch_image_ids=[ids[j] for j in batch_indices(2, 2, i, c['seed'])], proxy=[]) for i in range(8)]
            write_rows(tf/'history.jsonl', history)
            audit = {k: dict(tensors=1, nonzero_gradient=True, changed=True) for k in ('compressor', 'tx_normalization', 'receiver', 'memory')}
            save(tf/'learning_audit.json', dict(passed=True, frozen_unchanged=True, groups=audit))
            save(tf/'reload_audit.json', dict(passed=True)); save(tf/'forward_consistency.json', dict(tolerance=c['forward_tolerance'], errors=dict(wire=0., memory=0.)))
            save(tf/'trainability.json', dict(updates=8, batch_size=2, example_presentations=16, decoder_unfrozen=False, original_position_MSE=False))
            rows = []
            for control in ('matched', 'zero', 'shuffled', 'matched_beam4'):
                for i, item in enumerate(items):
                    text = '' if control == 'zero' else items[1-i]['text'] if control == 'shuffled' else item['text']
                    rows.append(row(item, arm, control, dict(text=text, stopped_by_capacity=False, receiver_failure=False), width,
                        donor_image_id=ids[1-i] if control == 'shuffled' else None))
            write_rows(folder/'diagnostic_controls.jsonl', rows)
            g = gate(groups(rows), c); save(folder/'diagnostic_gate.json', g)
            statuses[arm] = dict(status='diagnostic_passed', diagnostic_passed=True, expanded=False)
        save(root/'arm_status.json', statuses)
    def test_positive_and_negative_native_screen(self):
        for native_good in (True, False):
            with self.location() as d:
                root = Path(d); self.fixture(root, native_good); finish(root); self.assertTrue(verify(root)['passed'])
    def test_modified_metrics_rejected(self):
        with self.location() as d:
            root = Path(d); self.fixture(root); p = root/ARMS[0]/'diagnostic_controls.jsonl'
            from bart_latent_codec_v4.report import lines
            rows = lines(p); rows[0]['covered_wer'] = .5; write_rows(p, rows)
            with self.assertRaises(ValueError): verify(root, False)
    def test_missing_row_rejected(self):
        with self.location() as d:
            root = Path(d); self.fixture(root); p = root/ARMS[0]/'diagnostic_controls.jsonl'
            from bart_latent_codec_v4.report import lines
            write_rows(p, lines(p)[:-1])
            with self.assertRaises(ValueError): verify(root, False)
    def test_missing_update_rejected(self):
        with self.location() as d:
            root = Path(d); self.fixture(root); p = root/ARMS[0]/'diagnostic/history.jsonl'
            from bart_latent_codec_v4.report import lines
            write_rows(p, lines(p)[:-1])
            with self.assertRaises(ValueError): verify(root, False)
    def test_different_initialization_rejected(self):
        with self.location() as d:
            root = Path(d); self.fixture(root); save(root/ARMS[1]/'initial_state_hashes.json', {'query': 'different'})
            with self.assertRaises(ValueError): verify(root, False)
    def test_illegal_expansion_rejected(self):
        with self.location() as d:
            root = Path(d); self.fixture(root); p = root/'arm_status.json'; s = json.loads(p.read_text()); s[ARMS[0]]['expanded'] = True; save(p, s)
            with self.assertRaises(ValueError): verify(root, False)
    def test_failure_not_success(self):
        with self.location() as d:
            root = Path(d); self.fixture(root); finish(root); save(root/'FAILED.json', {'failure': True})
            with self.assertRaises(ValueError): verify(root)
    def test_archive_roundtrip_excludes_weights_and_refuses_overwrite(self):
        import tarfile
        with self.location() as d:
            root = Path(d)/'records'; root.mkdir(); self.fixture(root); finish(root)
            archive = Path(d)/'records.tar.gz'; collect(root, archive)
            dest = Path(d)/'restore'; dest.mkdir()
            with tarfile.open(archive) as tar:
                self.assertFalse(any(n.endswith('.pth') for n in tar.getnames())); tar.extractall(str(dest))
            self.assertTrue(verify(dest/'v10_bart_latent_codec_v4_records')['passed'])
            with self.assertRaises(FileExistsError): collect(root, archive)
    def test_seal_detects_extra_and_missing_files(self):
        with self.location() as d:
            root = Path(d); self.fixture(root); finish(root)
            with (root/'unexpected.txt').open('w') as f: f.write('extra')
            with self.assertRaises(ValueError): verify(root)

    def expanded_negative(self, root):
        self.fixture(root)
        from bart_latent_codec_v4.report import read
        identity = read(root/'run_identity.json'); c = identity['config']
        c.update(train_samples=8, val_samples=2, clean_updates=8, continue_updates=8)
        identity.update(expand=True, config=c); save(root/'run_identity.json', identity)
        design = read(root/'design.json'); design.update(config=c,
            main_train_view=dict(image_ids=list(range(11, 19))), validation_view=dict(image_ids=[21, 22]))
        save(root/'design.json', design)
        folder = root/ARMS[0]; clean = folder/'clean'; clean.mkdir(); tr = list(range(11, 19))
        for name in ('TRAIN_COMPLETE.json', 'learning_audit.json', 'reload_audit.json', 'forward_consistency.json', 'trainability.json'):
            save(clean/name, read(folder/'diagnostic'/name))
        write_rows(clean/'history.jsonl', [dict(update=i+1, total_updates=8,
            batch_image_ids=[tr[j] for j in batch_indices(8, 2, i, c['seed'])], proxy=[]) for i in range(8)])
        items = [dict(image_id=i, annotation_id=i+20, split='validation', text='a cat sits', covered_text='a cat sits',
            covered_tokens=5, source_sha256='fixture-only', source_utf8_bytes=10, truncated=False) for i in [21, 22]]
        rows = [row(item, ARMS[0], control, dict(text='', stopped_by_capacity=False, receiver_failure=False), 64,
            donor_image_id=43-item['image_id'] if control == 'shuffled' else None)
            for control in ('matched', 'zero', 'shuffled', 'matched_beam4') for item in items]
        write_rows(folder/'validation_controls.jsonl', rows); vg = gate(groups(rows), c, False)
        save(folder/'validation_gate.json', vg)
        s = read(root/'arm_status.json'); s[ARMS[0]].update(status='clean_validation_stop', expanded=True, validation_passed=False)
        save(root/'arm_status.json', s)

    def test_negative_validation_is_valid_stop(self):
        with self.location() as d:
            root = Path(d); self.expanded_negative(root); self.assertTrue(verify(root, False)['passed'])

    def test_no_noise_training_after_failed_validation(self):
        with self.location() as d:
            root = Path(d); self.expanded_negative(root); (root/ARMS[0]/'fso_continue').mkdir()
            with self.assertRaises(ValueError): verify(root, False)


if __name__ == '__main__': unittest.main(verbosity=2)
