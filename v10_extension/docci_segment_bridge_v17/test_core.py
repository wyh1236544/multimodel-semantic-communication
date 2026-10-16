"""CPU protocol/statistics contracts; no neural performance claims."""
import ast
import copy
from pathlib import Path
import sys
import unittest
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_segment_bridge_v17.core import (ARMS, RUN_ORDER, FOLDER, load, condition, noise_seed,
    donor_index, controls, batches, expected_parameters, parameter_groups, geometry)
from docci_segment_bridge_v17.report import channel_stats, compare
from docci_rx_bridge_v16.test_report import row


class Protocol(unittest.TestCase):
    def test_fixed_counts_budget_and_priority(self):
        c = load()
        self.assertEqual((c['train_samples'], c['val_samples'], c['ce_epochs']), (9647, 100, 6))
        self.assertEqual(len(list(batches(9647, 8, 6, 42))), 7236)
        self.assertEqual(len(list(batches(9647, 64, 2, 42))), 302)
        self.assertEqual(set(RUN_ORDER), set(ARMS))
        self.assertEqual(RUN_ORDER[:2], ('C_self_mixed', 'D_oracle_mixed'))
        self.assertEqual(expected_parameters(), 4429826)
        self.assertEqual(sum(parameter_groups().values()), expected_parameters())

    def test_noisy_pair_common_conditions_and_noise(self):
        for epoch in range(6):
            for i in range(30):
                eid = 'train_'+str(i)
                a = condition(ARMS[2], 'ce', epoch, eid)
                b = condition(ARMS[3], 'ce', epoch, eid)
                self.assertEqual(a, b)
                self.assertEqual(condition(ARMS[0], 'ce', epoch, eid), 'clean')
                self.assertEqual(noise_seed('ce', epoch, eid, a), noise_seed('ce', epoch, eid, b))
        self.assertNotEqual(noise_seed('ce', 0, 'x', 'awgn_0'), noise_seed('ce', 1, 'x', 'awgn_0'))

    def test_five_real_distinct_donors(self):
        for i in range(100):
            donors = [donor_index(i, 100, c) for c in controls(False) if c.startswith('donor')]
            self.assertEqual(len(set(donors)), 5)
            self.assertNotIn(i, donors)
        with self.assertRaises(ValueError):
            donor_index(0, 17, 'donor17')

    def test_geometry_oracle_not_equal_resource_claim(self):
        self.assertTrue(geometry('B_oracle_clean')['oracle'])
        self.assertFalse(geometry('A_self_clean')['oracle'])
        self.assertFalse(geometry('D_oracle_mixed')['waveform'])
        self.assertFalse(geometry('D_oracle_mixed')['teacher_or_source_mask_at_receiver'])

    def test_stats_reject_clean_gate_reuse_and_wrong_seed(self):
        s = dict(nominal_snr_db=10, noise_variance=.1, noise_seed=noise_seed('validation', 0, 'x', 'awgn_10'),
                 realized_snr_db=10., signal_power=1., noise_power=.1,
                 gate_recomputed_from_received_memory=True, clean_gate_reused=False)
        channel_stats(s, 'awgn_10', 'x')
        for change in ({'clean_gate_reused': True}, {'noise_seed': 123}, {'noise_power': 0.}, {'signal_power': 2.}):
            with self.assertRaises(ValueError):
                channel_stats(dict(s, **change), 'awgn_10', 'x')

    def test_clean_SNR_is_none_not_zero(self):
        s = dict(nominal_snr_db=None, noise_variance=0., noise_seed=noise_seed('validation', 0, 'x', 'clean'),
                 realized_snr_db=None, signal_power=1., noise_power=0., observed_sha256='same', tx_sha256='same',
                 gate_recomputed_from_received_memory=True, clean_gate_reused=False)
        channel_stats(s, 'clean', 'x')
        with self.assertRaises(ValueError):
            channel_stats(dict(s, nominal_snr_db=0), 'clean', 'x')

    def test_donor_average_unit_and_numerical_is_not_fact_success(self):
        visual, own = [], []
        for i in range(100):
            eid = 'example_'+str(i)
            for control in controls(False):
                r = row(eid, bad=control.startswith('donor'))
                r.update(condition='clean', control=control)
                visual.append(r)
            r = row(eid, bad=True); r.update(condition='clean', control='matched'); own.append(r)
        r = compare(visual, own, 'clean')
        self.assertTrue(r['numerical_screen_pass'])
        self.assertFalse(r['full_semantic_success'])
        self.assertEqual(r['vs_mean_of_fixed_donors']['pairs'], 100)
        for x in visual:
            if x['control'] == 'matched':
                x['empty_false_activation'] = 1
        self.assertFalse(compare(visual, own, 'clean')['numerical_screen_pass'])

    def test_actual_producer_schema_and_receiver_signatures(self):
        tree = ast.parse((FOLDER/'model.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SegmentBridge')
        repair = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'repair')
        self.assertEqual([a.arg for a in repair.args.args], ['self', 'memory', 'aux', 'gate', 'enabled'])
        producer = ast.parse((FOLDER/'evaluate.py').read_text())
        rowdict = next(n.value for n in ast.walk(producer) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'row' for t in n.targets))
        names = {k.arg for k in rowdict.keywords}
        self.assertTrue({'source_tokens', 'own_stats', 'own_observation_sha256', 'empty_slots', 'off_reused'} <= names)
        self.assertNotIn('source_mask', [a.arg for a in repair.args.args])


if __name__ == '__main__':
    unittest.main(verbosity=2)
