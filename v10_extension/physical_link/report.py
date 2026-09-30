"""Audit row identities and fading pairing before any exploratory comparison."""
import json
import math
from collections import defaultdict
from pathlib import Path

METRICS = ('psnr', 'ssim', 'bleu2', 'pre_fec_ber', 'post_fec_ber', 'crc_accepted',
           'quantization_mse', 'source_clip_fraction', 'forward_seconds',
           'forward_awg_samples', 'forward_uses', 'forward_energy_power_seconds', 'source_packet_bits', 'coded_bits',
           'semantic_quantized_goodput_bps', 'sequential_semantic_goodput_bps',
           'sequential_link_seconds', 'reverse_seconds', 'tx_average_power', 'sample_average_power',
           'dac_clip_fraction', 'header_rejected', 'effective_nominal_snr_db')


def identity(row):
    return (row['image_id'], row['profile'], row['nominal_snr_db'], row['repeat'])


def make_report(root, design):
    args = design['args']
    runs, reference, input_identity = {}, None, None
    rows_total = 0
    conditions = []
    expected = args['samples'] * args['repeats'] * len(args['profiles']) * len(args['snr_list'])
    for item in design['runs']:
        directory = Path(item['output'])
        complete = json.loads((directory / 'COMPLETE.json').read_text())
        summary = json.loads((directory / 'summary.json').read_text())
        run_identity = json.loads((directory / 'run_identity.json').read_text())
        inputs = (run_identity['checkpoint_sha256'], run_identity['split_sha256'])
        if input_identity is None:
            input_identity = inputs
        if inputs != input_identity:
            raise ValueError('Checkpoint/split identity differs')
        if (summary.get('split') != 'validation' or summary.get('source_caption_used_by_receiver') is not False
                or summary.get('diffusion_executed') is not False or summary.get('rows') != expected):
            raise ValueError('Split/leakage/completeness contract violated')
        rows = [json.loads(line) for line in (directory / 'samples.jsonl').read_text().splitlines() if line.strip()]
        keyed = {identity(row): row for row in rows}
        if len(rows) != expected or len(keyed) != expected or complete['rows'] != expected:
            raise ValueError('Incomplete or duplicate rows: ' + str(directory))
        ids = {r['image_id'] for r in rows}
        lattice = {(i, profile, snr, repeat) for i in ids for profile in args['profiles']
                   for snr in args['snr_list'] for repeat in range(args['repeats'])}
        if len(ids) != args['samples'] or set(keyed) != lattice:
            raise ValueError('Wrong image/condition/repeat lattice')
        if reference is None:
            reference = keyed
        if keyed.keys() != reference.keys():
            raise ValueError('Mismatched evaluation image/condition identities')
        for key, row in keyed.items():
            if not math.isclose(row['optical_irradiance_gain'], reference[key]['optical_irradiance_gain'], rel_tol=1e-6):
                raise ValueError('Fading is not paired across rates/variants')
            if row.get('forward_uses') is not None and row['forward_uses'] != row['payload_uses'] + row['pilot_uses'] + row['header_uses']:
                raise ValueError('Incorrect physical resource accounting')
        runs[(item['rate'], item['variant'])] = keyed
        rows_total += len(rows)
        groups = defaultdict(list)
        for row in rows:
            groups[(row['profile'], row['nominal_snr_db'])].append(row)
        for (profile, snr), group in groups.items():
            if len(group) != args['samples'] * args['repeats']:
                raise ValueError('Wrong condition sample count')
            result = {'variant': item['variant'], 'symbol_rate': item['rate'], 'profile': profile,
                      'reference_snr_db': snr, 'rows': len(group)}
            for metric in METRICS:
                vals = [row[metric] for row in group if row.get(metric) is not None]
                if vals and not all(math.isfinite(float(x)) for x in vals):
                    raise ValueError('Nonfinite metric: ' + metric)
                result[metric] = sum(vals) / len(vals) if vals else None
            conditions.append(result)
    differences = []
    for (rate, variant), keyed in runs.items():
        baseline = runs.get((rate, 'analog'))
        if variant == 'analog' or baseline is None:
            continue
        grouped = defaultdict(list)
        for key, row in keyed.items():
            grouped[(row['profile'], row['nominal_snr_db'])].append((row, baseline[key]))
        for (profile, snr), pairs in grouped.items():
            diff = {'variant_minus': 'analog', 'variant': variant, 'symbol_rate': rate,
                    'profile': profile, 'reference_snr_db': snr}
            for metric in ('psnr', 'ssim', 'bleu2'):
                diff[metric] = sum(a[metric] - b[metric] for a, b in pairs) / len(pairs)
            differences.append(diff)
    return {'rows': rows_total, 'conditions': conditions, 'paired_differences': differences,
            'checkpoint_sha256': input_identity[0], 'split_sha256': input_identity[1],
            'fading_pairing_verified': True,
            'interpretation': 'exploratory zero-shot validation; not equal frame duration/energy, not measured hardware bitrate'}
