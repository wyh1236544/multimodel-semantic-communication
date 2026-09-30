"""Strict, no-cherry-picking audit of the isolated optical-link pilot."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from link_compare.rate import serialization


def _mean(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def _load_json(path):
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def make_report(root, design):
    root = Path(root)
    args = design['args']
    expected_rows = args['samples'] * args['repeats'] * len(args['profiles']) * len(args['snr_list'])
    conditions, raw = [], {}
    for planned in design['runs']:
        folder = Path(planned['output'])
        complete = _load_json(folder / 'COMPLETE.json')
        summary = _load_json(folder / 'summary.json')
        if complete.get('rows') != expected_rows or complete.get('expected_rows') != expected_rows or summary.get('rows') != expected_rows:
            raise RuntimeError('Incomplete evaluation in ' + str(folder))
        if summary.get('split') != 'validation' or summary.get('source_caption_used_by_receiver') is not False or summary.get('diffusion_executed') is not False:
            raise RuntimeError('Evaluation split/leakage status differs: ' + str(folder))
        with (folder / 'samples.jsonl').open(encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if len(rows) != expected_rows:
            raise RuntimeError('samples.jsonl has an unexpected row count: ' + str(folder))
        variant = planned['variant']
        rate = float(planned['forward_symbol_rate_sps'])
        for profile in args['profiles']:
            for reference_snr, simulated_snr in zip(planned['reference_snr_db'], planned['simulated_nominal_snr_db']):
                selected = [r for r in rows if r['profile'] == profile and math.isclose(float(r['nominal_snr_db']), simulated_snr, abs_tol=1e-5)]
                if len(selected) != args['samples'] * args['repeats']:
                    raise RuntimeError('Incomplete condition: {} {} {}'.format(variant, profile, simulated_snr))
                ids = {(int(r['image_id']), int(r['repeat'])) for r in selected}
                if len(ids) != len(selected):
                    raise RuntimeError('Duplicate image/repeat in ' + str(folder))
                resources = {(int(r['payload_uses']), int(r['pilot_uses']), int(r['header_uses']),
                              int(r['forward_uses']), int(r['reverse_rf_real_uses']), str(r['physical_unit'])) for r in selected}
                if len(resources) != 1:
                    raise RuntimeError('Inconsistent physical resources in ' + str(folder))
                payload, pilots, header, forward, reverse, unit = resources.pop()
                if payload + pilots + header != forward:
                    raise RuntimeError('Forward resource accounting error in ' + str(folder))
                expected_unit = 'optical_intensity_slot' if variant == 'imdd_continuous' else 'coherent_optical_complex_symbol'
                if unit != expected_unit:
                    raise RuntimeError('Unexpected physical unit in ' + str(folder))
                if reverse != 64:
                    raise RuntimeError('Unexpected reverse control resource in ' + str(folder))
                key = profile + '/' + str(simulated_snr)
                aggregate = summary['summary'].get(key)
                if aggregate is None or aggregate.get('rows') != len(selected):
                    raise RuntimeError('Missing summary group: ' + key)
                time_cost = serialization(forward, rate, reverse, args['feedback_symbol_rate'])
                bits_per_symbol = {'coherent_qpsk': 2, 'coherent_16qam': 4}.get(variant)
                result = {
                    'variant': variant, 'profile': profile, 'reference_snr_db': reference_snr,
                    'simulated_nominal_snr_db': simulated_snr, 'forward_symbol_rate_sps': rate,
                    'forward_uses': forward, 'payload_uses': payload, 'pilot_uses': pilots,
                    'header_uses': header, 'reverse_rf_real_uses': reverse,
                    'physical_unit': unit,
                    'forward_serialization_ms': 1e3 * time_cost['forward_seconds'],
                    'reverse_serialization_ms': 1e3 * time_cost['reverse_seconds'],
                    'ideal_sequential_pairs_per_second': time_cost['ideal_sequential_pairs_per_second'],
                    'gross_constellation_label_bits_per_frame': payload * bits_per_symbol if bits_per_symbol else None,
                    'real_application_bit_rate_bps': None,
                    'psnr_db': aggregate['psnr'], 'psnr_p05_db': aggregate['psnr_p05'],
                    'ssim': aggregate['ssim'], 'bleu2': aggregate['bleu2'],
                    'token_acc': aggregate['token_acc'],
                    'common_retrieval_r1': aggregate['common_to_text_retrieval_r1'],
                    'header_rejected': aggregate['header_rejected'],
                    'tx_average_power': _mean(selected, 'tx_average_power'),
                    'quantization_mse': _mean(selected, 'quantization_mse'),
                    'constellation_symbol_error_rate': _mean(selected, 'constellation_symbol_error_rate'),
                    'sample_count': len(selected), 'unique_images': len({r['image_id'] for r in selected}),
                }
                conditions.append(result)
                raw[(variant, rate, profile, float(reference_snr))] = {
                    (int(r['image_id']), int(r['repeat'])): r for r in selected
                }
    paired = []
    for rate in args['symbol_rates']:
        for profile in args['profiles']:
            for reference_snr in args['snr_list']:
                base = raw.get(('coherent_continuous', float(rate), profile, float(reference_snr)))
                if base is None:
                    continue
                for variant in ('coherent_qpsk', 'coherent_16qam'):
                    other = raw.get((variant, float(rate), profile, float(reference_snr)))
                    if other is None:
                        continue
                    if set(other) != set(base):
                        raise RuntimeError('Finite-alphabet and continuous samples are not paired')
                    paired.append({'variant_minus_coherent_continuous': variant,
                                   'profile': profile, 'reference_snr_db': reference_snr,
                                   'forward_symbol_rate_sps': rate,
                                   'pairs': len(base),
                                   'mean_delta_psnr_db': sum(float(other[k]['psnr']) - float(base[k]['psnr']) for k in base) / len(base),
                                   'mean_delta_bleu2': sum(float(other[k]['bleu2']) - float(base[k]['bleu2']) for k in base) / len(base),
                                   'mean_delta_tx_power': sum(float(other[k]['tx_average_power']) - float(base[k]['tx_average_power']) for k in base) / len(base)})
    fields = list(conditions[0])
    with (root / 'rate_summary.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(conditions)
    return {'status': 'pilot_completed_not_hardware_validated', 'conditions': conditions,
            'paired_coherent_constellation_differences': paired,
            'rows': expected_rows * len(design['runs']),
            'rate_definition': 'Assumed forward symbols/s, fixed power + white thermal-noise PSD; SNR decreases 10log10(rate/reference_rate).',
            'time_definition': 'Serialization-only upper bound; excludes VAE/CLIP inference, synchronization, guards, optical front-end and real hardware.',
            'comparison_boundary': 'Within coherent variants: same checkpoint/data/seeds/rate/noise model, but actual transmit power may differ and is reported. IM/DD versus coherent: different SNR definitions, no matched-photon-budget superiority claim.',
            'digital_boundary': 'QPSK/16-QAM are uncoded hard constellation constraints, not a source codec + FEC baseline. Gross labels are not application throughput.'}
