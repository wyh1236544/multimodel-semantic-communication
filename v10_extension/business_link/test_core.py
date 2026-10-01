"""NumPy/stdlib contracts; not real COCO, checkpoint, CUDA or hardware results."""
import copy
from pathlib import Path
import sys
import unittest
import json
import tempfile
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from business_link.engine import Transport, wire_packet, read_wire, field_draw
from business_link.metrics import text_metrics
from business_link.report import make_report
from physical_link.digital import encode_source, decode_source, bits_bytes, bytes_bits
from business_link.run import freeze_business


class TransportTests(unittest.TestCase):
    def test_wire_crc_length_and_header(self):
        packet = wire_packet(b'abc', 'text', 0, 1, 32)
        self.assertEqual(read_wire(packet, 'text', 32), (0, 1, b'abc'))
        self.assertIsNone(read_wire(packet, 'features', 32))
        self.assertIsNone(read_wire(packet[:-1], 'text', 32))
        damaged = bytearray(packet); damaged[14] ^= 1
        self.assertIsNone(read_wire(bytes(damaged), 'text', 32))

    def test_actual_utf8_waveform_all_digital_policies(self):
        data = ('中文 optical text '*4).encode()
        for policy in ('16qam_none', '16qam_conv', '16qam_repeat2', 'qpsk_conv'):
            t = Transport({'packet_bytes': 64}, .4+.7j, 1e-5, 33)
            t.probe()
            recovered, ok, record = t.transfer(data, 'text', policy)
            self.assertTrue(ok, policy)
            self.assertEqual(recovered, data)
            self.assertGreater(record['packets_planned'], 1)
            totals = t.totals()
            self.assertEqual(t.pulse.rs, 16e9)
            self.assertEqual(t.pulse.sps, 4)
            self.assertEqual(totals['post_errors'], 0)
            self.assertAlmostEqual(totals['awg_samples']/64e9, totals['waveform_seconds'], places=20)
            self.assertAlmostEqual(totals['scheduled_seconds'],
                (totals['awg_samples']+totals['guard_samples'])/64e9)

    def test_feature_object_waveform_and_crc_erasure(self):
        x = np.random.default_rng(3).normal(size=128)
        bits, expected, _ = encode_source(x)
        t = Transport({'packet_bytes': 64}, 1j, 0, 41)
        received, ok, _ = t.transfer(bits_bytes(bits), 'features', '16qam_conv')
        self.assertTrue(ok)
        restored, accepted = decode_source(bytes_bits(received), len(x))
        self.assertTrue(accepted)
        np.testing.assert_array_equal(restored, expected)
        failed = Transport({'packet_bytes': 64}, 0, 2, 41)
        received, ok, _ = failed.transfer(bits_bytes(bits), 'features', '16qam_conv')
        self.assertFalse(ok)
        self.assertEqual(received, b'')
        self.assertGreater(failed.totals()['post_errors'], 0)

    def test_redundancy_increases_cost_same_task(self):
        costs = []
        for policy in ('16qam_none', '16qam_conv', '16qam_repeat2'):
            t = Transport({}, 1, 0, 2)
            received, ok, _ = t.transfer(b'the same business bytes', 'text', policy)
            self.assertTrue(ok)
            costs.append(t.totals()['scheduled_seconds'])
        self.assertLess(costs[0], costs[1]); self.assertLess(costs[1], costs[2])

    def test_failed_retries_charged_not_oracle_repaired(self):
        with self.assertRaises(ValueError):
            Transport({'max_attempts': 2}, 1, .1, 0)
        t = Transport({'max_attempts': 2, 'ideal_ack_arq': True, 'packet_bytes': 32}, 0, 2, 5)
        received, ok, record = t.transfer(b'lost text', 'text', '16qam_conv')
        self.assertFalse(ok); self.assertEqual(received, b'')
        self.assertEqual(record['packet_attempts'], 2)
        self.assertEqual(t.totals()['ack_seconds'], 128/1e6)
        self.assertAlmostEqual(t.totals()['arq_wait_seconds'], 4*1000/299792458.)
        # Same field for both attempts; only their noise stream changes.
        self.assertEqual(t.gain, 0)

    def test_fixed_field_and_invalid_settings(self):
        profile = {'family': 'fso', 'alpha': 4., 'beta': 2., 'xi': 1.8, 'a0': .9}
        a, irradiance = field_draw(profile, 81)
        self.assertEqual(field_draw(profile, 81)[0], a)
        self.assertAlmostEqual(abs(a)**2, irradiance)
        for settings in ({'sps': 1}, {'rolloff': 2}, {'packet_bytes': 0}, {'max_attempts': 5}):
            with self.assertRaises(ValueError):
                Transport(settings, 1, .1, 2)

    def test_text_failures_not_omitted(self):
        self.assertEqual(text_metrics('你好 world', '你好 world')['text_cer'], 0.)
        self.assertEqual(text_metrics('你好 world', '')['text_cer'], 1.)
        self.assertEqual(text_metrics('one two', '')['text_wer'], 1.)


def fixture():
    design = {'image_ids': [1, 2], 'profiles': ['fso_moderate'], 'snr_list': [10],
        'policies': ['16qam_conv'], 'receivers': ['baseline', 'received_text'],
        'repeats': 1, 'business_bytes': 300, 'task_psnr_threshold': 20.,
        'settings': {'max_attempts': 1}}
    rows = []
    for image_id in (1, 2):
        for receiver in design['receivers']:
            rows.append({'image_id': image_id, 'profile': 'fso_moderate', 'nominal_snr_db': 10,
                'repeat': 0, 'policy': '16qam_conv', 'receiver': receiver,
                'input_sha256': str(image_id), 'business_bytes': 100*image_id,
                'field_identity': 'same', 'payload_sha256': 'same-source', 'sender_mode': 'reconstruction',
                'sequential_link_seconds': 3., 'one_way_propagation_seconds': 1.,
                'serialization_and_control_seconds': 2.,
                'pre_bits': 10 if image_id == 1 else 90, 'pre_errors': 1 if image_id == 1 else 0,
                'post_bits': 10, 'post_errors': 0, 'packet_attempts': 1, 'crc_failures': 0,
                'psnr': 21.+(receiver == 'received_text'), 'ssim': .7, 'text_cer': 0., 'text_wer': 0.,
                'text_exact_match_diagnostic': True, 'fusion_used': receiver == 'received_text',
                'task_quality_success': image_id == 1, 'feature_object_crc_accepted': True,
                'text_crc_accepted': True, 'input_delivery_complete': True,
                'waveform_seconds': 1., 'inference_plus_channel_simulation_wall_seconds': .1})
    return rows, design


class ReportTests(unittest.TestCase):
    def test_coverage_bit_weighting_and_censored_completion(self):
        rows, design = fixture()
        report = make_report(rows, design)
        group = next(iter(report['summary'].values()))
        self.assertAlmostEqual(group['pre_fec_ber'], .01)  # NOT mean(.1, 0)
        self.assertEqual(group['scheduled_attempt_time_seconds'], 5.)  # one flight per burst
        self.assertIsNone(group['completion_time_seconds'])
        self.assertEqual(group['quality_business_goodput_bps'], 160.)
        self.assertEqual(next(iter(report['received_text_fusion_gains'].values()))['delta_psnr_db'], 1.)
        for broken in (rows[:-1], rows+[rows[0]]):
            with self.assertRaises(ValueError): make_report(broken, design)

    def test_unpaired_or_changed_input_rejected(self):
        rows, design = fixture()
        for field in ('field_identity', 'payload_sha256', 'input_sha256'):
            bad = copy.deepcopy(rows); bad[1][field] = 'changed'
            with self.assertRaises(ValueError): make_report(bad, design)
        bad = copy.deepcopy(rows); bad[1]['pre_errors'] = 2
        with self.assertRaises(ValueError): make_report(bad, design)


class BusinessManifestTests(unittest.TestCase):
    def test_byte_budget_freeze_reuse_and_changed_input(self):
        project = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix='business_test_', dir=str(project)) as temp:
            folder = Path(temp)
            jpeg = folder/'source.jpg'
            jpeg.write_bytes(b'synthetic-file-reference')
            class Image:
                def numpy(self): return np.zeros((3, 2, 2), np.float32)
            class Dataset:
                records = [{'image_id': 1}, {'image_id': 2}]
                changed = False
                def __len__(self): return 2
                def __getitem__(self, index):
                    return {'image_id': index+1, 'image': Image(),
                            'text': 'abc' if not self.changed else 'xyz', 'path': str(jpeg)}
            dataset = Dataset()
            args = SimpleNamespace(samples=2, business_bytes=20, reuse_business_manifest=None)
            items, manifest = freeze_business(dataset, args, 'split')
            self.assertEqual(len(items), 1)
            self.assertEqual(manifest['business_bytes'], 15)
            frozen = folder/'frozen.json'
            frozen.write_text(json.dumps(manifest))
            args.reuse_business_manifest = str(frozen)
            self.assertEqual(freeze_business(dataset, args, 'split')[1]['tasks'], manifest['tasks'])
            dataset.changed = True
            with self.assertRaises(ValueError): freeze_business(dataset, args, 'split')
            args.reuse_business_manifest, args.business_bytes, args.samples = None, None, 3
            with self.assertRaises(ValueError): freeze_business(dataset, args, 'split')


if __name__ == '__main__':
    unittest.main(verbosity=2)
