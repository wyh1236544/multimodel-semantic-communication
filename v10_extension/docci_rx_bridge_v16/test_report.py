import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_rx_bridge_v16.core import PROJECT, atomic, load, geometry
from docci_rx_bridge_v16.report import paired, compare, verify_metrics, validate_eval, render
from docci_text_continue_v11.metrics import metrics


def row(eid, control='matched', bad=False):
    source = 'a red car'
    text = 'a blue car' if bad else source
    generated = [[2, 0, 4, 5, 6, 2]]+[[2, 0, 2] for _ in range(8)]
    return dict(example_id=eid, official_split='qual_dev', control=control, arm='rx_clip',
                source_text=source, source_text_sha256='source', source_tokens=5,
                received_caption=text, chunk_generated_token_ids=generated,
                generated_token_ids=[4, 5, 6], stopped_by_capacity=False,
                first_free_content_probability=[.99]+[.001]*8, empty_slots=list(range(1, 9)),
                nonempty_slots=[0], empty_false_activation=0, nonempty_suppressed=0,
                condition='ideal_clean', snr_db=None, oracle=False, input_truncated=False,
                own_text_cache_sha256='t'+eid, auxiliary_cache_sha256='i'+eid,
                auxiliary_donor_id=None, **metrics(source, text))


class ReportContracts(unittest.TestCase):
    def test_independent_metrics_and_capacity(self):
        r = row('a')
        verify_metrics(r)
        for key, value in [('normalized_wer', .123), ('generated_token_ids', [9]), ('stopped_by_capacity', True)]:
            bad = dict(r, **{key: value})
            with self.assertRaises(ValueError):
                verify_metrics(bad)

    def test_paired_direction_and_no_duplicate_pairs(self):
        good, bad = [row('a'), row('b')], [row('a', bad=True), row('b', bad=True)]
        result = paired(good, bad, 100)
        self.assertGreater(result['gain_pp'], 30)
        self.assertGreater(result['paired95CI_pp'][0], 0)
        with self.assertRaises(ValueError):
            paired(good+[good[0]], bad, 10)

    def test_numerical_screen_never_semantic_success(self):
        candidate, own = [], []
        for eid in ('a', 'b'):
            for control in ('matched', 'shuffled', 'zero', 'off'):
                candidate.append(row(eid, control, control != 'matched'))
                own.append(row(eid, control, True))
        result = compare(candidate, own, dict(load(), bootstrap_draws=100))
        self.assertTrue(result['numerical_screen_pass'])
        self.assertFalse(result['full_semantic_success'])
        self.assertEqual(result['facts'], 'PENDING')
        candidate[0]['empty_false_activation'] = 1
        self.assertFalse(compare(candidate, own, dict(load(), bootstrap_draws=100))['numerical_screen_pass'])

    def test_real_donor_own_memory_and_complete_matrix(self):
        with tempfile.TemporaryDirectory(prefix='v16_report_', dir=str(PROJECT)) as d:
            folder = Path(d)
            entries = [dict(example_id=x, official_split='qual_dev', text='a red car',
                       text_sha256='source', source_tokens=5, text_file_sha256='t'+x,
                       image_file_sha256='i'+x) for x in ('a', 'b')]
            baseline, rows = [row('a'), row('b')], []
            for i, eid in enumerate(('a', 'b')):
                for control in ('matched', 'shuffled', 'zero', 'off'):
                    r = row(eid, control)
                    if control == 'shuffled':
                        donor = ('a', 'b')[1-i]
                        r.update(auxiliary_donor_id=donor, auxiliary_cache_sha256='i'+donor)
                    rows.append(r)
            atomic(folder/'matrix.json', dict(ids=['a', 'b'], controls=['matched', 'shuffled', 'zero', 'off'], arm='rx_clip', geometry=geometry('rx_clip')))
            def save():
                (folder/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            save()
            validate_eval(folder, entries, baseline, 'rx_clip')
            rows[1]['auxiliary_donor_id'] = 'a'
            save()
            with self.assertRaises(ValueError):
                validate_eval(folder, entries, baseline, 'rx_clip')
            rows[1]['auxiliary_donor_id'] = 'b'
            rows[-1]['own_text_cache_sha256'] = 'ta'
            save()
            with self.assertRaises(ValueError):
                validate_eval(folder, entries, baseline, 'rx_clip')

    def test_html_escape_and_empty_result(self):
        with tempfile.TemporaryDirectory(prefix='v16_render_', dir=str(PROJECT)) as d:
            s = render(Path(d), {'status': '<script>bad</script>'})
            self.assertIn('&lt;script&gt;', s)
            self.assertNotIn('<script>', s)


if __name__ == '__main__':
    unittest.main(verbosity=2)
