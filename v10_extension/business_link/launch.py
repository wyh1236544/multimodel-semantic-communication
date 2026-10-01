"""Offline project-local launch; use the EXISTING server interpreter and GPU 1."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
EXT = Path(__file__).resolve().parents[1]
PROJECT = EXT.parent
sys.path.insert(0, str(EXT))
from v10.paths import local_environment, write_path


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('action', choices=('check', 'smoke', 'compare'))
    p.add_argument('output', nargs='?')
    args, rest = p.parse_known_args()
    exe = os.environ.get('PYTHON_BIN', sys.executable)
    if not Path(exe).is_file():
        raise FileNotFoundError('Interpreter not found: '+exe)
    env = local_environment(PROJECT)
    env.setdefault('CUDA_VISIBLE_DEVICES', '1')
    folder = Path(__file__).resolve().parent
    if args.action == 'check':
        if args.output or rest:
            p.error('check accepts no output/extra arguments')
        for filename in ('test_core.py', 'test_adapter.py'):
            subprocess.run([exe, '-B', str(folder/filename)], cwd=str(PROJECT), env=env, check=True)
        print('Both numerical and torch CPU adapter suites passed; GPU/checkpoint performance NOT validated.')
        return
    if not args.output:
        p.error('Use a NEW output directory inside multimodel')
    output = write_path(args.output)
    if output.exists():
        raise FileExistsError('Existing output preserved: '+str(output))
    command = [exe, '-B', '-u', str(folder/'run.py'), '--output-dir', str(output)]
    if args.action == 'smoke':
        command += ['--samples', '2', '--profiles', 'awgn', 'fso_moderate', '--snr-list', '20',
                    '--policies', '16qam_conv', '--save-images', '2', '--max-hours', '.5']
    command += rest
    print('Physical GPU visibility: '+env['CUDA_VISIBLE_DEVICES'], flush=True)
    print('No training. Existing checkpoint and data are read-only.', flush=True)
    subprocess.run(command, cwd=str(PROJECT), env=env, check=True)


if __name__ == '__main__':
    main()
