#!/usr/bin/env python3
"""Paired V10 validation pilot: IM/DD versus coherent continuous/finite alphabets.

Run from multimodel project root with the server's existing torch environment.
No checkpoint, frozen split, legacy result or base model file is modified.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

from link_compare.rate import effective_snr_db

VARIANTS = ('imdd_continuous', 'coherent_continuous', 'coherent_qpsk', 'coherent_16qam')
CONSTELLATIONS = {'coherent_continuous': 'continuous',
                  'coherent_qpsk': 'qpsk', 'coherent_16qam': '16qam'}


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--config', default=str(EXTENSION_ROOT / 'configs/v10_multichannel.json'))
    for name in ('checkpoint', 'split-manifest', 'output-dir', 'data-root', 'vae-path', 'text-model-path'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--profiles', nargs='+', choices=('fso_weak', 'fso_moderate', 'fso_strong'), default=['fso_moderate'])
    p.add_argument('--snr-list', nargs='+', type=float, default=[0., 10., 20.], help='Reference-rate nominal SNR; optical and coherent definitions differ')
    p.add_argument('--symbol-rates', nargs='+', type=float, default=[1e6, 1e7], help='Hypothetical forward symbols or optical-intensity slots per second')
    p.add_argument('--reference-symbol-rate', type=float, default=1e6)
    p.add_argument('--feedback-symbol-rate', type=float, default=1e6, help='Assumed RF real-symbol rate')
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--save-images', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--min-free-gb', type=float, default=8.)
    p.add_argument('--min-free-disk-gb', type=float, default=8.)
    p.add_argument('--max-hours', type=float, default=0.5, help='Maximum for one variant/rate evaluation')
    p.add_argument('--max-total-hours', type=float, default=2., help='Cooperative time limit, checked between evaluations')
    p.add_argument('--dry-run', action='store_true', help='Print the plan without importing torch or writing anything')
    return p


def validate(args):
    if len(set(args.variants)) != len(args.variants) or len(set(args.symbol_rates)) != len(args.symbol_rates):
        raise ValueError('Duplicate variant or symbol rate')
    if min(args.symbol_rates + [args.reference_symbol_rate, args.feedback_symbol_rate]) <= 0:
        raise ValueError('Symbol rates must be positive')
    if not all(math.isfinite(x) for x in args.snr_list + args.symbol_rates + [args.reference_symbol_rate, args.feedback_symbol_rate]):
        raise ValueError('SNR and symbol rates must be finite')
    if min(args.samples, args.repeats, args.max_hours, args.max_total_hours) <= 0 or args.samples < 2:
        raise ValueError('Invalid bounded evaluation settings; at least two images required')
    if args.save_images < 0 or args.num_workers < 0:
        raise ValueError('Invalid image/worker settings')
    if len(set(args.snr_list)) != len(args.snr_list):
        raise ValueError('Duplicate SNR')


def rate_tag(rate):
    return 'sps_' + ('{:.0f}'.format(rate) if rate.is_integer() else str(rate).replace('.', 'p'))


def plan(args):
    results = []
    for rate in args.symbol_rates:
        for variant in args.variants:
            results.append({'variant': variant, 'forward_symbol_rate_sps': rate,
                            'reference_snr_db': args.snr_list,
                            'simulated_nominal_snr_db': [round(effective_snr_db(s, rate, args.reference_symbol_rate), 6) for s in args.snr_list],
                            'output': str(Path(args.output_dir) / rate_tag(rate) / variant)})
    return results


def main(argv=None):
    args = parser().parse_args(argv)
    validate(args)
    planned = plan(args)
    if args.dry_run:
        print(json.dumps({'label': 'UNRUN_PLANNED_ONLY', 'rows': planned}, indent=2, ensure_ascii=False))
        return 0

    import torch  # noqa: F401 - checked only on the user's configured server environment
    from unittest.mock import patch
    import eval_v10
    import v10.model as v10_model
    from v10.channels import LinkFrame
    from v10.io import atomic_json
    from v10.paths import write_path
    from link_compare.coherent import CoherentFSOFrame
    from link_compare.report import make_report

    root = write_path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    code_files = (Path(__file__), Path(__file__).with_name('coherent.py'),
                  Path(__file__).with_name('rate.py'), Path(__file__).with_name('report.py'))
    code_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files}
    design = {'status': 'running', 'purpose': 'exploratory_validation_pilot_not_final_test',
              'args': vars(args), 'source_sha256': code_hashes, 'runs': planned,
              'warnings': [
                  'IM/DD intensity-SNR and coherent Es/N0 are different; same nominal dB is not equal optical power.',
                  'QPSK/16-QAM are hard constellation constraints on pretrained semantic symbols, without source-bit compression or FEC.',
                  'All variants use the SAME unchanged checkpoint; finite alphabets are zero-shot diagnostics, not trained digital baselines.',
                  'Rate sweep assumes fixed received power and white thermal-noise PSD, noise variance proportional to symbol rate.',
                  'Symbol rates and reverse feedback rate are HYPOTHETICAL inputs, not measured hardware throughput.',
                  'Coherent channel includes random block-constant optical phase and pilot CSI, but not phase drift/shot noise/pulse shaping.',
              ]}
    atomic_json(root / 'design.json', design)
    deadline = time.monotonic() + args.max_total_hours * 3600
    try:
        for item in planned:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Total pilot time budget exhausted')
            child = Path(item['output'])
            child.parent.mkdir(exist_ok=True)
            eval_args = copy.deepcopy(args)
            eval_args.output_dir = str(child)
            eval_args.snr_list = item['simulated_nominal_snr_db']
            eval_args.split = 'validation'
            eval_args.ablation = 'full'
            eval_args.allow_settings_change = False
            eval_args.no_amp = True
            eval_args.max_hours = min(args.max_hours, remaining / 3600)
            variant = item['variant']
            if variant == 'imdd_continuous':
                factory = LinkFrame
            else:
                alphabet = CONSTELLATIONS[variant]

                def factory(*a, **kw):
                    return CoherentFSOFrame(*a, constellation=alphabet, **kw)

            print('Running {} at {:g} symbols/s, reference SNR {} dB'.format(
                variant, item['forward_symbol_rate_sps'], args.snr_list), flush=True)
            with patch.object(v10_model, 'LinkFrame', factory):
                eval_v10.main(eval_args)
        report = make_report(root, design)
        atomic_json(root / 'rate_summary.json', report)
        design['status'] = 'complete'
        atomic_json(root / 'design.json', design)
        atomic_json(root / 'COMPLETE.json', {'runs': len(planned), 'conditions': len(report['conditions']),
                                            'rows': report['rows']})
    except BaseException as exc:
        design['status'] = 'failed_or_interrupted'
        design['error'] = '{}: {}'.format(type(exc).__name__, exc)
        atomic_json(root / 'failure.json', design)
        raise
    print('Complete: {}'.format(root / 'rate_summary.json'), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
