"""Synthetic CPU contracts only; real saved parents and pretrained GPU remain smoke gates."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_pretrained_checks_v22 import core as c
from docci_pretrained_checks_v22 import digital as d
from docci_pretrained_checks_v22.report import matrix, paired

class Contracts(unittest.TestCase):
    def test_noise_common_across_arms_and_no_free_negative_snr_training(self):
        x = c.noise('x', -5, 0, 'image', 8192)
        self.assertTrue(np.array_equal(x, c.noise('x', -5, 0, 'image', 8192)))
        self.assertFalse(np.array_equal(x, c.noise('x', -5, 1, 'image', 8192)))
        self.assertEqual(float(np.max(np.abs(c.noise('x', None, 0, 'image', 20)))), 0.)
        self.assertGreater(float(np.mean(x*x)), 2.)
        with self.assertRaises(ValueError): c.noise('x', -10, 0, 'image', 3)

    def test_real_donors_and_duplicate_rejection(self):
        m = c.donor_map(['a', 'b', 'c'])
        self.assertTrue(all(a != b for a, b in m.items()))
        for ids in (['a'], ['a', 'a']):
            with self.assertRaises(ValueError): c.donor_map(ids)

    def test_no_geometry_resize_inside_metric(self):
        a = np.zeros((512, 512, 3), np.uint8)
        b = a.copy(); b[:10] = 255
        self.assertTrue(c.metric(a, b, [0, 10, 512, 512])['identical'])
        self.assertGreater(c.metric(a, b, [0, 0, 512, 512])['MSE'], 0)
        with self.assertRaises(ValueError): c.metric(a, b[:100], [0, 0, 512, 512])

    def test_path_escape_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory(dir=str(c.PROJECT/'experiment_results')) as temp:
            p = Path(temp)
            for n in ('../escape', '/absolute'):
                with self.assertRaises(ValueError): c.local(p, n)
            (p/'link').symlink_to(p.parent)
            with self.assertRaises(ValueError): c.local(p, 'link/anything')

    def test_result_byte_and_file_set_verification(self):
        with tempfile.TemporaryDirectory(dir=str(c.PROJECT/'experiment_results')) as temp:
            p = Path(temp); (p/'a.json').write_text('{}')
            c.complete(p, 'inventory', {})
            self.assertEqual(c.verify_files(p)['stage'], 'inventory')
            (p/'unexpected').write_text('x')
            with self.assertRaises(ValueError): c.verify_files(p)

    def test_asset_lock_detects_weight_change(self):
        with tempfile.TemporaryDirectory(dir=str(c.PROJECT/'experiment_results')) as temp:
            p = Path(temp); (p/'weights.bin').write_bytes(b'test-not-real-weights')
            lock = c.seal_tree(p); c.verify_tree(lock)
            (p/'weights.bin').write_bytes(b'changed')
            with self.assertRaises(ValueError): c.verify_tree(lock)

    def test_actual_packet_roundtrip_all_mcs(self):
        payload = b'Actual synthetic image payload.\0'*9
        for profile in d.PROFILES:
            with self.subTest(profile=profile):
                signal, accounting, _ = d.encode(payload, 0, profile)
                rx = d.receive(signal, profile)
                self.assertEqual(rx['payload'], payload)
                self.assertTrue(rx['header_crc'] and rx['packet_crc'])
                self.assertEqual(len(signal), 4128)
                self.assertAlmostEqual(float(np.mean(np.abs(signal)**2)), 1., places=6)
                self.assertEqual(accounting['header_symbols'], 648)

    def test_maximum_fitting_length_and_one_byte_overflow(self):
        for profile in d.PROFILES:
            n = d.capacity(profile)
            d.allocation(n, profile)
            with self.assertRaises(ValueError): d.allocation(n+1, profile)

    def test_all_16qam_points_gray_hard_decode(self):
        x = d.modulate(d.QAM_LABELS.ravel(), 4)
        hard = (d.llrs(x, .001, 4) < 0).astype(np.uint8)
        self.assertEqual(len(set(x)), 16)
        np.testing.assert_array_equal(hard, d.QAM_LABELS.ravel())
        self.assertAlmostEqual(float(np.mean(np.abs(x)**2)), 1.)

    def test_header_failure_not_source_repair(self):
        signal, _, _ = d.encode(b'a test', 0, 'qpsk12')
        signal[32:680] = 0
        rx = d.receive(signal, 'qpsk12')
        self.assertEqual(rx['status'], 'HEADER_REJECT')
        self.assertEqual(rx['payload'], b'')
        self.assertIsNone(rx['decoded_bits'])

    def test_image_noise_same_samples_as_continuous(self):
        signal, _, _ = d.encode(b'a test', 0, 'qpsk12')
        y = d.channel(signal, 'e', 0, 2)
        real = np.stack(((y[32:]-signal[32:]).real, (y[32:]-signal[32:]).imag), 1).ravel()*np.sqrt(2)
        np.testing.assert_allclose(real, c.noise('e', 0, 2, 'image', 8192), atol=1e-12)

    def test_jpeg_actual_source_rate_fit_and_receive(self):
        rng = np.random.default_rng(5)
        image = Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8))
        with tempfile.TemporaryDirectory(dir=str(c.PROJECT/'experiment_results')) as temp:
            payload, choice = d.source_encode(image, 'qpsk12', 'jpeg', Path(temp))
            self.assertLessEqual(len(payload), d.capacity('qpsk12'))
            self.assertIn(choice['encoded_size'], (512, 256, 128, 64, 32))
            self.assertEqual(d.reconstruct(payload, 0).size, (512, 512))
            frame, _, _ = d.encode(payload, 0, 'qpsk12')
            self.assertEqual(d.receive(frame, 'qpsk12')['payload'], payload)

    def test_bpg_no_silent_jpeg_substitute(self):
        with self.assertRaises(ValueError):
            d.source_encode(Image.new('RGB', (512, 512)), 'qpsk12', 'bpg', c.PROJECT)

    def test_receiver_signature_has_no_truth_or_true_length(self):
        import inspect
        self.assertEqual(list(inspect.signature(d.receive).parameters), ['y', 'profile'])

    def test_matrix_missing_duplicate_and_extra_are_failures(self):
        r = dict(example_id='a', snr_db=0, repeat=0, control='x')
        key = ((('a', 0, 0), 'x'))
        matrix([r], {key})
        for rs in ([], [r, r]):
            with self.assertRaises(ValueError): matrix(rs, {key})

    def test_bootstrap_clusters_repeats_within_image(self):
        a = {(eid, rep):dict(LPIPS=.3, psnr=20.) for eid in ('a', 'b') for rep in (0, 1, 2)}
        b = {k:dict(LPIPS=.4, psnr=19.) for k in a}
        effect = paired(a, b, 'LPIPS', .05/3)
        self.assertAlmostEqual(effect['mean'], .1)
        self.assertEqual(effect['images'], 2)
        self.assertTrue(effect['repeats_averaged_within_image'])
        with self.assertRaises(ValueError): paired(a, {}, 'LPIPS')

    def test_prompt_block_interface_without_torch_import(self):
        # Inspect code policy here; authentic tokenizer/text-encoder work is a GPU smoke gate.
        from docci_pretrained_checks_v22.neural import prompt_embeddings
        import inspect
        body = inspect.getsource(prompt_embeddings)
        self.assertIn('truncation=False', body)
        self.assertIn('clip_blocks', body)
        self.assertNotIn('truncation=True', body)

    def test_full_prompt_chunk_recovery_and_explicit_overflow(self):
        for n in (0, 74, 75, 76, 150, 1200):
            body = list(range(10, 10+n))
            ids, plan = c.clip_blocks(body, 1, 2, 0)
            recovered = []
            for i, row in enumerate(ids):
                self.assertEqual(len(row), 77)
                recovered.extend(row[1:1+min(75, n-i*75)])
            self.assertEqual(recovered, body)
            self.assertFalse(plan['truncated'])
        with self.assertRaises(ValueError): c.clip_blocks(list(range(1201)), 1, 2, 0)

    def test_synthetic_cpu_stage_and_portable_gallery(self):
        from types import SimpleNamespace
        from docci_pretrained_checks_v22.run import digital
        from docci_pretrained_checks_v22.report import verify, render
        with tempfile.TemporaryDirectory(dir=str(c.PROJECT/'experiment_results')) as temp:
            folder = Path(temp); base = folder/'prepared'; out = folder/'digital'
            base.mkdir(); out.mkdir(); (base/'receiver').mkdir(); (base/'truth').mkdir()
            examples, data = [], []
            for j, eid in enumerate(('synthetic_a', 'synthetic_b')):
                text = 'Synthetic test description '+str(j)
                image = Image.new('RGB', (512, 512), (40+j*50, 20, 120))
                truth = base/'truth'/(eid+'.png'); image.save(truth)
                examples.append(dict(example_id=eid, source_image='truth/'+truth.name,
                    source_image_sha256=c.sha(truth), source_text=text,
                    source_text_sha256=c.digest(text.encode()), rectangle=[0, 0, 512, 512]))
                for snr in c.CONFIG['snr_db']:
                    path = base/'receiver'/(c.case_name(eid, snr, 0)+'.png'); image.save(path)
                    data.append(dict(example_id=eid, snr_db=snr, repeat=0,
                        donor_id='synthetic_b' if j==0 else 'synthetic_a', received_image='receiver/'+path.name,
                        received_image_sha256=c.sha(path), received_caption=text,
                        received_caption_sha256=c.digest(text.encode()), source_text_at_RX=False,
                        post_noise_renormalization=False, image_payload_symbols=4096, text_payload_symbols=9216))
            c.atomic(base/'index.json', dict(examples=examples, repeats=1, profile='smoke', SYNTHETIC_ONLY=True))
            (base/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data))
            c.complete(base, 'prepare', dict(rows=6, SYNTHETIC_ONLY=True))
            digital(SimpleNamespace(prepared=base, codec='jpeg', bpg_encoder=None, bpg_decoder=None,
                bpg_encoder_sha256=None, bpg_decoder_sha256=None), out, c.Budget(.1))
            self.assertEqual(verify(out, base)['rows'], 24)
            render(SimpleNamespace(prepared=base, stages=[out], output=folder/'report'))
            page = (folder/'report/index.html').read_text()
            self.assertIn('images/', page)
            self.assertNotIn('src="'+str(folder), page)
            self.assertFalse(c.read(folder/'report/summary.json')['semantic_success'])
            self.assertTrue((folder/'report/image_fact_review.csv').is_file())
            if (c.FOLDER/'MANIFEST.json').is_file():
                from docci_pretrained_checks_v22.collect import collect
                collect(SimpleNamespace(prepared=base, stages=[out], destination=folder/'records.tar.gz', max_hours=.1))
                self.assertTrue((folder/'records.tar.gz.sha256').is_file())
            # Rehashing a bad producer's output must not bypass semantic contracts.
            bad = c.rows(out/'rows.jsonl')
            bad[0]['source_image_at_RX'] = True
            (out/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in bad))
            c.complete(out, 'digital', dict(rows=len(bad)))
            with self.assertRaises(ValueError): verify(out, base)

    def test_no_training_or_network_in_stage_parser(self):
        from docci_pretrained_checks_v22.run import parser
        p = parser()
        with self.assertRaises(SystemExit): p.parse_args(['train', 'new'])

if __name__=='__main__':
    unittest.main(verbosity=2)
