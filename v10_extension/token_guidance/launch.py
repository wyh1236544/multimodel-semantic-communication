"""User-started project-local run with new outputs, GPU1 and a hard overall budget."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
EXT = Path(__file__).resolve().parents[1]
PROJECT = EXT.parent
sys.path.insert(0, str(EXT))
from v10.paths import local_environment, write_path


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('action', choices=('check', 'smoke', 'quick'))
    p.add_argument('output', nargs='?')
    p.add_argument('--max-hours', type=float)
    args, rest = p.parse_known_args()
    exe = os.environ.get('PYTHON_BIN', sys.executable)
    env = local_environment(PROJECT)
    env.setdefault('CUDA_VISIBLE_DEVICES', '1')
    if not Path(exe).is_file():
        raise FileNotFoundError('Explicit interpreter not found')
    folder = Path(__file__).resolve().parent
    if args.action == 'check':
        if args.output or rest or args.max_hours is not None:
            p.error('check accepts no additional arguments')
        for filename in ('test_core.py', 'test_model.py'):
            subprocess.run([exe, '-B', str(folder/filename)], cwd=str(PROJECT), env=env, check=True)
        print('CPU contract tests passed. Real checkpoint/GPU/COCO effectiveness still untested.')
        return
    if not args.output:
        p.error('Specify a NEW experiment_results directory')
    output = write_path(args.output)
    if output.exists():
        raise FileExistsError('Existing results preserved: '+str(output))
    hours = args.max_hours if args.max_hours is not None else (.5 if args.action == 'smoke' else 3.)
    if not 0 < hours <= 12:
        raise ValueError('Overall wall-time budget must be >0 and <=12 hours')
    defaults = dict(checkpoint=PROJECT/'experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth',
                    config=EXT/'configs/v10_multichannel.json', split_manifest=EXT/'manifests/ln_v10_splits.json',
                    data_root=PROJECT/'COCO2017', vae_path=PROJECT/'pretrained/sd-vae-ft-mse',
                    text_model_path=PROJECT/'pretrained/clip-vit-base-patch32')
    command = [exe, '-B', '-u', str(folder/'run.py')]
    for key, value in defaults.items():
        command += ['--'+key.replace('_', '-'), str(value)]
    command += ['--output-dir', str(output), '--max-hours', str(hours)]
    if args.action == 'smoke':
        command += ['--smoke']
    if any(x.startswith('--output-dir') or x.startswith('--smoke') or x.startswith('--max-hours') for x in rest):
        p.error('Output, smoke flag and budget are controlled by launch')
    command += rest
    print('GPU visibility={} | {} | four independent short fine-tunings | hard cap={}h'.format(env['CUDA_VISIBLE_DEVICES'], args.action, hours), flush=True)
    log = output.with_name(output.name+'.log')
    if log.exists():
        raise FileExistsError('Existing log preserved: '+str(log))
    log.parent.mkdir(parents=True, exist_ok=True)
    # Keep terminal output visible and a persistent log without a shell pipeline.
    with log.open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=str(PROJECT), env=env, stdout=stream, stderr=subprocess.STDOUT)
        print('PID={} log={}'.format(process.pid, log), flush=True)
        try:
            code = process.wait(timeout=hours*3600)
        except BaseException:
            # This is our exact child PID only, never a process-name kill.
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code:
        raise SystemExit('Run failed (exit {}), preserved log: {}'.format(code, log))
    if not (output/'COMPLETE.json').is_file():
        raise RuntimeError('Exit zero but missing COMPLETE marker')
    print('Finished. Read '+str(output/'REPORT.md'), flush=True)


if __name__ == '__main__':
    main()
