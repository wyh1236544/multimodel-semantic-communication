"""Four paired validation evaluations; no training, no old-output overwrite."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT))
STRATEGIES = {'adaptive': None, 'fixed_reconstruction': 0,
              'fixed_hybrid': 1, 'fixed_generation': 2}


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--config', default=str(EXT/'configs/v10_multichannel.json'))
    for name in ('checkpoint', 'split-manifest', 'output-dir', 'data-root',
                 'vae-path', 'text-model-path'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--profiles', nargs='+', choices=('awgn', 'rayleigh', 'fso_weak', 'fso_moderate', 'fso_strong'),
                   default=['awgn', 'fso_moderate', 'fso_strong'])
    p.add_argument('--snr-list', nargs='+', type=float, default=[0., 5., 10., 20.])
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--save-images', type=int, default=4)
    p.add_argument('--seed', type=int, default=42, help='One reproducible draw identity, NOT multiple training seeds')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--min-free-gb', type=float, default=8.)
    p.add_argument('--max-total-hours', type=float, default=2.)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--dry-run', action='store_true')
    return p


def validate(a):
    if a.samples < 2 or a.repeats < 1 or a.save_images < 0 or a.bootstrap < 100:
        raise ValueError('Need >=2 images, >=1 repeat, nonnegative image budget, >=100 bootstrap draws')
    if not math.isfinite(a.max_total_hours) or a.max_total_hours <= 0:
        raise ValueError('Invalid wall-time budget')
    if not math.isfinite(a.min_free_gb) or a.min_free_gb < 0:
        raise ValueError('Invalid GPU budget')
    for values in (a.profiles, a.snr_list):
        if len(set(values)) != len(values):
            raise ValueError('Duplicate conditions')
    if not all(math.isfinite(x) for x in a.snr_list):
        raise ValueError('Nonfinite SNR')


def main(argv=None):
    a = parser().parse_args(argv)
    validate(a)
    from v10.paths import write_path, local_environment
    target = write_path(a.output_dir)
    plan = {'status': 'PLANNED_NOT_RUN', 'args': vars(a), 'strategies': STRATEGIES,
            'expected_rows_per_strategy': a.samples*a.repeats*len(a.profiles)*len(a.snr_list),
            'split': 'validation', 'training': False, 'csi': 'pilot', 'equalizer': 'mmse',
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(Path(__file__).parent.glob('*.py'))}}
    if a.dry_run:
        print(json.dumps(plan, indent=2))
        return
    for name in ('config', 'checkpoint', 'split_manifest'):
        if not Path(getattr(a, name)).is_file():
            raise FileNotFoundError(getattr(a, name))
    target.mkdir(parents=True, exist_ok=False)
    def record(name, value):
        with (target/name).open('x', encoding='utf-8') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
    record('plan.json', plan)
    env = local_environment()
    env['PYTHONPATH'] = str(EXT)
    started = time.monotonic()
    try:
        for strategy in STRATEGIES:
            remaining = a.max_total_hours*3600-(time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError('Total suite budget exhausted')
            cmd = [sys.executable, '-B', str(Path(__file__).with_name('worker.py')),
                   '--strategy', strategy, '--config', a.config, '--checkpoint', a.checkpoint,
                   '--split-manifest', a.split_manifest, '--output-dir', str(target/strategy),
                   '--data-root', a.data_root, '--vae-path', a.vae_path,
                   '--text-model-path', a.text_model_path, '--split', 'validation',
                   '--samples', str(a.samples), '--repeats', str(a.repeats),
                   '--save-images', str(a.save_images), '--seed', str(a.seed), '--device', a.device,
                   '--num-workers', '0', '--no-amp', '--csi', 'pilot', '--equalizer', 'mmse',
                   '--allow-settings-change', '--min-free-gb', str(a.min_free_gb),
                   '--max-hours', str(remaining/3600), '--profiles']+a.profiles+['--snr-list']+[str(x) for x in a.snr_list]
            print('START {}: log={}'.format(strategy, target/(strategy+'.log')), flush=True)
            with (target/(strategy+'.log')).open('x', encoding='utf-8') as log:
                subprocess.run(cmd, cwd=str(EXT.parent), env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=remaining)
            print('DONE {} elapsed={:.1f}s'.format(strategy, time.monotonic()-started), flush=True)
        from pilot_gain.report import make_report
        report = make_report(target, a.bootstrap)
        record('gain_summary.json', report)
        from pilot_gain.report import write_tables
        write_tables(target, report)
        record('COMPLETE.json', {'status': 'complete', 'paired_rows_per_strategy': report['paired_rows'],
                                'wall_seconds': time.monotonic()-started, 'training': False})
        print('COMPLETE: '+str(target/'gain_summary.json'), flush=True)
    except BaseException as exc:
        record('FAILED.json', {'error': str(exc), 'type': type(exc).__name__,
                               'wall_seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    main()
