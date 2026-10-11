"""NumPy channel/reference + independent artifact validator negative tests."""
import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
EXT = Path(__file__).resolve().parents[1]
PROJECT = EXT.parent
sys.path.insert(0, str(EXT))
from fso_jscc_v1 import ARMS, VERSION
from fso_jscc_v1.core import load, draw_gain, observe, numpy_channel, train_condition, seed_for, kernels, digest, TEMPLATE
from fso_jscc_v1.report import matrix, verify, paired_ci, write_report
from optical_realism_v3.spec import load_plan, resources
from optical_realism_v3.transport import RealisticTransport
from optical_link_v1.config import load as optical_load


class ChannelTests(unittest.TestCase):
    def test_gamma_gamma_is_intensity_not_field(self):
        cfg = load(); h, d = draw_gain(cfg, 'fso_strong', 14)
        self.assertAlmostEqual(abs(h)**2, d['intensity_turbulence']*d['pointing_intensity']*d['path_intensity'])

    def test_path_loss_reduces_true_snr_not_noise(self):
        cfg = load(); worse = copy.deepcopy(cfg); worse['profiles']['fso_moderate']['loss_db'] = 10
        a = observe(cfg, 'fso_moderate', 0., 4, 7); b = observe(worse, 'fso_moderate', 0., 4, 7)
        self.assertEqual(a[1], b[1]); self.assertAlmostEqual(abs(a[0])**2/abs(b[0])**2, 10.)

    def test_field_is_one_per_frame_and_seeded(self):
        cfg = load(); self.assertEqual(draw_gain(cfg, 'fso_moderate', 3), draw_gain(cfg, 'fso_moderate', 3))
        self.assertNotEqual(draw_gain(cfg, 'fso_moderate', 3)[0], draw_gain(cfg, 'fso_moderate', 4)[0])

    def test_pilot_estimate_is_not_truth(self):
        d = observe(load(), 'fso_strong', -5, 3, 4)
        self.assertNotEqual(d[0], d[2]); self.assertTrue(0 < d[4] < 1)

    def test_clean_and_nonfinite_config(self):
        x = np.random.default_rng(7).normal(size=(288, 16)).astype(np.float32)
        y, r, _ = numpy_channel(x, load(), 'clean', 40., 3, 4)
        self.assertTrue(np.allclose(x, y, atol=1e-6)); self.assertEqual(r, 1.)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'config.json'; cfg = load(); cfg['encoder_lr'] = float('nan')
            path.write_text(json.dumps(cfg));
            with self.assertRaises(ValueError):
                load(path)

    def test_training_mix_bounds_and_low_snr_weight(self):
        values = [train_condition(load(), 0, p)[1] for p in range(2000)]
        self.assertTrue(min(values) >= -5 and max(values) <= 20)
        self.assertTrue(.45 < np.mean(np.array(values) <= 5) < .55)

    def test_rrc_energy_and_length(self):
        h = kernels(load()); self.assertEqual(len(h), 129); self.assertAlmostEqual(float(h@h), 1., places=6)

    def test_repeated_noise_is_paired_not_retransmission(self):
        x = np.ones((288, 16), np.float32)
        y, _, a = numpy_channel(x, load(), 'fso_moderate', 0., 2, 3)
        z, _, b = numpy_channel(x, load(), 'fso_moderate', 0., 2, 3)
        self.assertTrue(np.array_equal(y, z)); self.assertEqual(a, b)

    def test_real_v3_single_shot_charged_samples(self):
        plan = load_plan(repeats=1)[0]; cases = optical_load()[1]
        optical = cases['good']; optical['waveform']['policy'] = plan['digital_policy']
        budget = resources(optical, plan)
        t = RealisticTransport(optical, 1, 2, -9., budget['slot_samples'], plan['laser_max_dbm'], plan['hardware'])
        t.probe(); y, rel, stats = t.continuous(np.ones((288, 16), np.float32), 1, 'lmmse')
        self.assertEqual(y.shape, (288, 16)); self.assertEqual(stats['used_samples'], budget['analog_samples'])
        self.assertEqual(stats['analog_repeats'], 1); self.assertEqual(len(t.events), 2)
        self.assertIsNone(stats['pre_fec_ber']); self.assertEqual(stats['raw_utf8_wire_bytes'], 0)


class ArtifactTests(unittest.TestCase):
    def fixture(self, directory):
        cfg = load(); cfg.update(val_samples=2, physical_val_samples=1, draws=1, physical_draws=1,
                                 train_samples=2, epochs=1, grad_accum=1, warmup_optimizer_steps=0, save_images=0)
        design = dict(version=VERSION, config=cfg, arms=list(ARMS), split='validation',
            validation_ids=[101, 102], train_ids=[201, 202], validation_source_sha256={'101': 'source101', '102': 'source102'},
            proxy_conditions=[dict(id='fso_moderate_snr0', profile='fso_moderate', snr=0)],
            physical_conditions=[dict(id='moderate_p24')], physical_resource=dict(analog_samples=20),
            train_forward_steps=2, optimizer_steps_per_trained_arm=2)
        (directory/'design.json').write_text(json.dumps(design))
        rows, events = [], []
        for key in sorted(matrix(design)):
            arm, image, engine, condition, draw = key
            row = dict(arm=arm, image_id=image, engine=engine, condition=condition, draw=draw,
                source_sha256='source'+str(image), field_seed=3, noise_seed=4, psnr=20.+ARMS.index(arm),
                ssim=.8, token_acc=.6, normalized_wer=.7, bleu2=.4,
                analog_repeats=1, raw_utf8_wire_bytes=0, pre_fec_ber=None, post_fec_ber=None, receiver_available=True)
            if engine == 'proxy':
                row.update(payload_complex_symbols=2304, forward_complex_symbols=2560)
            else:
                row.update(event_count=2, used_samples=20, slot_samples=100,
                    launch_signal_energy_j=1e-12, signal_energy_budget_j=1e-10)
                for purpose in ('probe', 'features'):
                    events.append(dict(arm=arm, image_id=image, engine=engine, condition=condition, draw=draw,
                        purpose=purpose, awg_samples=8, guard_samples=2))
            rows.append(row)
        (directory/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (directory/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
        for arm in ARMS:
            target = directory/('train_'+arm); target.mkdir()
            trained = arm != ARMS[0]
            (target/'COMPLETE.json').write_text(json.dumps(dict(arm=arm, trained=trained, steps=2 if trained else 0,
                optimizer_steps=2 if trained else 0, checkpoint_sha256=None,
                nonzero_gradient_steps=dict(source_encoder=1, tx_mapping=1, rx_mapping=1, decoder=1),
                parameter_update_squared_l2_by_group=dict(source_encoder=1, tx_mapping=1, rx_mapping=1, decoder=1))))
            if trained:
                (target/'history.jsonl').write_text(''.join(json.dumps(dict(arm=arm, forward_step=i,
                    optimizer_step=i, loss=1., epoch=1, image_id=200+i, nominal_snr_db=0.,
                    profile='clean' if arm == ARMS[1] else 'awgn' if arm == ARMS[2] else 'fso_moderate'))+'\n' for i in (1, 2)))
        return rows

    def temp(self):
        folder = PROJECT/'releases'; folder.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix='fso_jscc_test_', dir=str(folder))

    def test_complete_matrix_and_cluster_difference(self):
        with self.temp() as tmp:
            d = Path(tmp); rows = self.fixture(d); verify(d)
            ci = paired_ci(rows); self.assertEqual(ci['mean'], 1.); self.assertEqual(ci['ci95'], [1., 1.])
            result = write_report(d); self.assertTrue(result['no_paper_performance_claim'])
            self.assertIn('R=1', (d/'report.html').read_text())

    def test_missing_duplicate_and_fake_ber_rejected(self):
        for mode in ('missing', 'duplicate', 'ber', 'seed', 'repeats'):
            with self.temp() as tmp:
                d = Path(tmp); rows = self.fixture(d)
                if mode == 'missing':
                    rows.pop()
                elif mode == 'duplicate':
                    rows[-1] = rows[0]
                elif mode == 'ber':
                    rows[0]['pre_fec_ber'] = 0.
                elif mode == 'seed':
                    rows[0]['noise_seed'] = 999
                else:
                    rows[0]['analog_repeats'] = 2
                (d/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.assertRaises(ValueError):
                    verify(d)

    def test_frozen_missing_encoder_gradient_and_unpaid_samples_rejected(self):
        for mode in ('frozen', 'gradient', 'resource'):
            with self.temp() as tmp:
                d = Path(tmp); rows = self.fixture(d)
                if mode == 'resource':
                    rows = [dict(r, used_samples=21) if r['engine'] == 'waveform_v3' else r for r in rows]
                    (d/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                else:
                    target = d/('train_'+(ARMS[0] if mode == 'frozen' else ARMS[3]))/'COMPLETE.json'
                    record = json.loads(target.read_text())
                    if mode == 'frozen':
                        record['trained'] = True
                    else:
                        record['nonzero_gradient_steps']['source_encoder'] = 0
                    target.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    verify(d)

    def test_image_clusters_not_channel_rows_are_bootstrap_unit(self):
        rows = []
        for arm in ARMS[2:]:
            for image in (1, 2):
                for draw in range(30):
                    rows.append(dict(arm=arm, image_id=image, engine='proxy', condition='test', draw=draw,
                        psnr=0. if arm == ARMS[2] else float(image)))
        result = paired_ci(rows)
        self.assertEqual(result['images'], 2); self.assertAlmostEqual(result['mean'], 1.5)

    def test_completion_hashes_and_collected_bundle(self):
        from fso_jscc_v1.collect import collect
        with self.temp() as tmp:
            root = Path(tmp); d = root/'records'; d.mkdir()
            rows = self.fixture(d); write_report(d)
            (d/'run_identity.json').write_text(json.dumps(dict(status='complete', synthetic_test_only=True)))
            (d/'checkpoint_load.json').write_text(json.dumps(dict(synthetic_test_only=True)))
            files = [f for f in d.rglob('*') if f.is_file()]
            (d/'COMPLETE.json').write_text(json.dumps(dict(version=VERSION, rows=len(rows),
                result_sha256={str(f.relative_to(d)): digest(f) for f in files})))
            verify(d, complete=True)
            archive = root/'synthetic_records.tar.gz'; collect(d, archive)
            self.assertTrue(archive.is_file()); self.assertTrue((root/'synthetic_records.tar.gz.sha256').is_file())
            with (d/'samples.jsonl').open('a') as stream:
                stream.write('\n')
            with self.assertRaises(ValueError):
                verify(d, complete=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
