#!/usr/bin/env python3
"""Bounded V2 validation pilot. Never trains or writes old experiment outputs."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VARIANTS = ('analog', '16qam_none', '16qam_conv', 'qpsk_conv')


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--config', default=str(ROOT / 'configs/v10_multichannel.json'))
    for name in ('checkpoint', 'split-manifest', 'output-dir', 'data-root', 'vae-path', 'text-model-path'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--profiles', nargs='+', choices=('fso_weak', 'fso_moderate', 'fso_strong'), default=['fso_moderate'])
    p.add_argument('--snr-list', nargs='+', type=float, default=[0., 10., 20.])
    p.add_argument('--symbol-rates', nargs='+', type=float, default=[8e9, 14e9])
    p.add_argument('--reference-symbol-rate', type=float, default=14e9)
    p.add_argument('--sample-rate', type=float, default=64e9)
    p.add_argument('--feedback-symbol-rate', type=float, default=1e6, help='Assumed reverse RF real symbols/s')
    p.add_argument('--rolloff', type=float, default=.01)
    p.add_argument('--span', type=int, default=128)
    p.add_argument('--qbits', type=int, choices=(4, 6, 8), default=8)
    p.add_argument('--quantizer-limit', type=float, default=4.)
    p.add_argument('--dac-bits', type=int, choices=(8, 10, 12, 16), default=8)
    p.add_argument('--analog-bandwidth', type=float, default=None, help='ASSUMED combined frontend 3-dB bandwidth, Hz; not hardware calibration')
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS[:3]))
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--save-images', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--min-free-gb', type=float, default=8.)
    p.add_argument('--min-free-disk-gb', type=float, default=8.)
    p.add_argument('--max-hours', type=float, default=.5)
    p.add_argument('--max-total-hours', type=float, default=2.)
    p.add_argument('--dry-run', action='store_true')
    return p


def validate(args):
    from physical_link.waveform import Pulse
    numbers = args.symbol_rates + args.snr_list + [args.sample_rate, args.reference_symbol_rate, args.feedback_symbol_rate, args.quantizer_limit, args.max_hours, args.max_total_hours]
    if not all(math.isfinite(x) for x in numbers):
        raise ValueError('Nonfinite settings')
    if min(args.sample_rate, args.reference_symbol_rate, args.feedback_symbol_rate, args.quantizer_limit, args.max_hours, args.max_total_hours) <= 0:
        raise ValueError('Nonpositive settings')
    if args.analog_bandwidth is not None and (not math.isfinite(args.analog_bandwidth) or args.analog_bandwidth <= 0):
        raise ValueError('Invalid analog bandwidth')
    for group in (args.variants, args.symbol_rates, args.snr_list, args.profiles):
        if len(set(group)) != len(group):
            raise ValueError('Duplicate planned condition')
    if args.samples < 2 or args.repeats < 1 or min(args.save_images, args.num_workers) < 0:
        raise ValueError('Invalid sample settings')
    for rate in args.symbol_rates:
        Pulse(args.sample_rate, rate, args.rolloff, args.span)


def main(argv=None):
    args = parser().parse_args(argv)
    validate(args)
    planned = [{'variant': variant, 'rate': rate,
                'output': str(Path(args.output_dir) / ('baud_' + format(rate, '.0f')) / variant)}
               for rate in args.symbol_rates for variant in args.variants]
    if args.dry_run:
        print(json.dumps({'status': 'PLANNED_NOT_RUN', 'runs': planned,
                          'expected_rows': len(planned) * args.samples * args.repeats * len(args.profiles) * len(args.snr_list)}, indent=2))
        return 0
    from v10.paths import local_environment
    os.environ.pop('SUDO_PW', None)
    os.environ.update(local_environment())
    import eval_v10
    import v10.model as model
    from v10.paths import write_path
    from v10.io import atomic_json
    from unittest.mock import patch
    from physical_link.channel import WaveformFSOFrame
    from physical_link.report import make_report
    target = write_path(args.output_dir)
    target.mkdir(parents=True, exist_ok=False)
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted(Path(__file__).parent.glob('*.py'))}
    design = {'status': 'running', 'args': vars(args), 'runs': planned, 'source_sha256': hashes,
              'split': 'validation', 'training': False, 'receiver': 'coherent_iq_no_kk',
              'limitations': [
                  'Same pretrained checkpoint, no waveform/digital fine-tuning; not JPEG/H264 traditional source codec.',
                  'Independent real-feature quantization, public fixed range; CRC-failed whole payload erased, no ARQ.',
                  'K7 rate-1/2 (171,133 octal) terminated convolutional code with 1024-bit blocks; NOT LDPC.',
                  'Pilot scalar channel/phase estimate, public sample-start synchronization; no CMA/Volterra/CFO/clock drift.',
                  '256-symbol preamble on each of probe/header/payload; filter guards and FEC padding charged.',
                  'Analog has token-power UEP; digital has uniform FEC/no UEP. Not a strict equal-resource UEP comparison.',
                  'Fixed received power + white noise PSD idealization: Es/N0 shifts -10log10(Rs/Rref). No calibrated optical budget.',
                  'Digital/analog frame durations and energies differ; report actual resources, do not claim equal latency.',
                  'DAC/ADC clock is assumed equal 64G; actual ADC/laser/photodetector/LO bandwidth and response uncalibrated.',
              ]}
    atomic_json(target / 'design.json', design)
    started = time.monotonic()
    try:
        for item in planned:
            remaining = args.max_total_hours * 3600 - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError('Total pilot budget exhausted')
            child = Path(item['output'])
            child.parent.mkdir(exist_ok=True)
            evaluation = copy.deepcopy(args)
            evaluation.output_dir = str(child)
            evaluation.split, evaluation.ablation = 'validation', 'full'
            evaluation.no_amp, evaluation.allow_settings_change = True, False
            evaluation.max_hours = min(args.max_hours, remaining / 3600)
            # Keep REFERENCE SNR in eval sample_seed: identical fading/phase at
            # every rate and variant. Actual noise scaling happens in factory.
            def factory(*a, **kw):
                return WaveformFSOFrame(*a, **kw, variant=item['variant'], rs=item['rate'],
                    fs=args.sample_rate, rolloff=args.rolloff, span=args.span,
                    qbits=args.qbits, quantizer_limit=args.quantizer_limit,
                    dac_bits=args.dac_bits, bandwidth=args.analog_bandwidth,
                    reference_rate=args.reference_symbol_rate, feedback_rate=args.feedback_symbol_rate)
            print('Running {} at {:g} GBaud'.format(item['variant'], item['rate'] / 1e9), flush=True)
            with patch.object(model, 'LinkFrame', factory):
                eval_v10.main(evaluation)
        report = make_report(target, design)
        atomic_json(target / 'physical_summary.json', report)
        design['status'] = 'complete'
        atomic_json(target / 'design.json', design)
        atomic_json(target / 'COMPLETE.json', {'rows': report['rows'], 'runs': len(planned),
                                            'wall_seconds': time.monotonic() - started})
    except BaseException as exc:
        atomic_json(target / 'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
    print('Complete: ' + str(target / 'physical_summary.json'), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
