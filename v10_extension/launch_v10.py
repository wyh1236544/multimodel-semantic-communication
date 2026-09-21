"""All V10 commands run with cwd=the existing multimodel project."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from v10.paths import PROJECT_ROOT, write_path, local_environment

CODE = Path(__file__).resolve().parent
ENTRY = {'check': 'check_v10.py', 'splits': 'prepare_v10_splits.py',
         'smoke': 'train_v10.py', 'train': 'train_v10.py', 'eval': 'eval_v10.py',
         'generate': 'generate_v10.py', 'compare': 'compare_v10.py',
         'verify': 'verify_v10_run.py', 'package': 'verify_v10_package.py'}


def usage():
    print('Run inside /data1/zw/WYH/multimodel:\n'
          '  bash run_v10.sh package | check [NEW_REPORT] | splits [args]\n'
          '  bash run_v10.sh smoke NEW_RUN [args]\n'
          '  bash run_v10.sh train NEW_RUN [args]\n'
          '  bash run_v10.sh eval CHECKPOINT NEW_RUN [args]\n'
          '  bash run_v10.sh generate EVAL_DIR NEW_RUN [args]\n'
          '  bash run_v10.sh compare BASELINE CANDIDATE NEW_REPORT [args]\n'
          '  bash run_v10.sh verify RUN_DIR\n'
          'Append --dry-run to inspect the resolved command without writing files.\n'
          'Use ACTION --help for the underlying script options.')


def make_plan(argv):
    args = list(argv)
    action = args.pop(0) if args else 'help'
    if action in ('help', '-h', '--help'):
        return {'help': True}
    if action not in ENTRY:
        raise ValueError('Unknown action: '+action)
    command = [sys.executable, '-B', '-u', str(CODE/ENTRY[action])]
    if args == ['--help'] or args == ['-h']:
        return dict(command=command+args, action=action, script_help=True)
    dry = '--dry-run' in args
    args = [a for a in args if a != '--dry-run']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    stamp += '_'+str(os.getpid())
    initial = os.environ.get('INITIAL_CHECKPOINT', str(PROJECT_ROOT/'experiment_results/ln_v9a_feedback_generation_seed42/seed42.pth'))
    data = os.environ.get('DATA_ROOT', str(PROJECT_ROOT/'COCO2017'))
    pretrained = Path(os.environ.get('PRETRAINED_ROOT', str(PROJECT_ROOT/'pretrained')))
    manifest = os.environ.get('SPLIT_MANIFEST', str(CODE/'manifests/ln_v10_splits.json'))
    run = None

    def take(description):
        if not args or args[0].startswith('-'):
            raise ValueError('Missing '+description+'; see bash run_v10.sh help')
        return args.pop(0)

    if action == 'check':
        output = take('new report') if args and not args[0].startswith('-') else str(CODE/('verification/server_check_'+stamp+'.json'))
        base = ['--output', output]
    elif action == 'splits':
        base = ['--data-root', data, '--output', manifest]
    elif action in ('train', 'smoke', 'eval'):
        checkpoint = take('checkpoint') if action == 'eval' else initial
        run = take('new run directory')
        base = ['--config', str(CODE/'configs/v10_multichannel.json'), '--checkpoint', checkpoint,
                '--split-manifest', manifest, '--output-dir', run, '--data-root', data,
                '--vae-path', str(pretrained/'sd-vae-ft-mse'),
                '--text-model-path', str(pretrained/'clip-vit-base-patch32')]
        if action == 'smoke':
            base += ['--smoke-test', '--max-hours', '0.5']
    elif action == 'generate':
        source, run = take('evaluation directory'), take('new output directory')
        base = ['--eval-dir', source, '--output-dir', run,
                '--diffusion-model-path', str(pretrained/'stable-diffusion-v1-5')]
    elif action == 'compare':
        base = ['--baseline', take('baseline'), '--candidate', take('candidate'),
                '--output', take('new comparison report')]
    elif action == 'verify':
        base = [take('run directory')]
    else:
        base = []
    # Paths controlled by position/environment must not be secretly overridden
    # by later argparse options. Other path arguments are validated below.
    protected = {'--output', '--output-dir', '--checkpoint', '--split-manifest',
                 '--data-root', '--vae-path', '--text-model-path', '--eval-dir',
                 '--diffusion-model-path', '--baseline', '--candidate'}
    for value in args:
        if value.split('=', 1)[0] in protected:
            raise ValueError('Use the positional argument or documented environment variable for '+value)
    combined = base+args
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for option in protected | {'--config', '--extra-excluded-json'}:
        parser.add_argument(option)
    paths, _ = parser.parse_known_args(combined)
    for name in ('output', 'output_dir'):
        value = getattr(paths, name)
        if value:
            target = write_path(value)
            if target.exists():
                raise FileExistsError('Use a NEW output; existing path preserved: '+str(target))
    # The existing datasets/checkpoints are read-only inputs; even their paths
    # need to be located under the project (read-only data symlinks are allowed).
    for name, value in vars(paths).items():
        if value and name not in ('output', 'output_dir'):
            path = Path(os.path.abspath(value))
            try:
                path.relative_to(PROJECT_ROOT)
            except ValueError:
                raise ValueError('Use an input path under multimodel for '+name)
    if action == 'verify':
        Path(os.path.abspath(base[0])).relative_to(PROJECT_ROOT)
    log = str(write_path(PROJECT_ROOT/('logs/v10/'+action+'_'+stamp+'.log')))
    return dict(action=action, cwd=str(PROJECT_ROOT), command=command+combined,
                dry_run=dry, log=log, verify_run=run if action in ('train', 'smoke', 'eval') else None)


def execute(argv):
    os.chdir(PROJECT_ROOT)
    plan = make_plan(argv)
    if plan.get('help'):
        usage()
        return 0
    if plan.get('script_help'):
        return subprocess.call(plan['command'], cwd=str(PROJECT_ROOT), env=local_environment())
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if plan['dry_run']:
        return 0
    env = local_environment()
    log = write_path(plan['log'])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(plan, ensure_ascii=False)+'\n')
        stream.flush()
        child = subprocess.Popen(plan['command'], cwd=str(PROJECT_ROOT), env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding='utf-8', errors='replace', bufsize=1)
        try:
            for line in child.stdout:
                print(line, end='', flush=True)
                stream.write(line)
                stream.flush()
            code = child.wait()
        except KeyboardInterrupt:
            # Only this launcher-owned child; never enumerate/kill other tasks.
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            stream.write('\nInterrupted by user; no success claimed.\n')
            return 130
        stream.write('\nchild_exit_code={}\n'.format(code))
        if code:
            return code
        if plan['verify_run']:
            result = subprocess.run([sys.executable, '-B', str(CODE/'verify_v10_run.py'), plan['verify_run']],
                                    cwd=str(PROJECT_ROOT), env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            print(result.stdout, end='', flush=True)
            stream.write(result.stdout)
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(execute(sys.argv[1:]))
