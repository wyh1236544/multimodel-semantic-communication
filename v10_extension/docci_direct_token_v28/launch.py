"""Isolated offline workflow. Checks -> real smoke -> paired experiment -> verify."""
import argparse
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_direct_token_v28.core import PROJECT, FOLDER, ARMS, PARENTS, atomic, read, sha, safe, load, check_source
from docci_direct_token_v28.data import arguments


def stop_owned(child):
    if child.poll() is not None: return
    if os.getpgid(child.pid) != child.pid: raise RuntimeError('Unexpected child process-group ownership')
    os.killpg(child.pid, signal.SIGTERM)
    try: child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if child.poll() is None and os.getpgid(child.pid) == child.pid: os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action', choices=('inventory', 'check', 'smoke', 'pipeline'))
    p.add_argument('output', nargs='?'); p.add_argument('--task', choices=('image', 'text', 'both'), default='image')
    p.add_argument('--profile', choices=('pilot', 'full'), default='pilot')
    p.add_argument('--config'); p.add_argument('--gpu', choices=('0', '1'), default='1')
    p.add_argument('--max-hours', type=float, default=12); arguments(p); a = p.parse_args()
    if (a.action in ('smoke', 'pipeline')) != bool(a.output): p.error('Fresh output only for smoke/pipeline')
    if not math.isfinite(a.max_hours) or not .25 <= a.max_hours <= 48: p.error('Hard cap 0.25..48 hours')
    if a.output and safe(a.output).exists(): raise FileExistsError('Existing run preserved before tests')
    m = check_source(); c = load(a.config, a.profile)
    tasks = ['image', 'text'] if a.task == 'both' else [a.task]
    if a.action == 'inventory':
        result = {}
        for task in tasks:
            pp = Path(getattr(a, task+'_checkpoint')); cache = Path(getattr(a, task+'_features'))
            result[task] = dict(checkpoint=str(pp), exists=pp.is_file(),
                pinned_sha256=PARENTS[task]['sha256'], features=str(cache),
                metadata_exists=(cache/'cache_train_identity.json').is_file(),
                reference_exists=(PROJECT/PARENTS[task]['reference']).is_file())
        print(__import__('json').dumps(dict(assets=result, training=False, GPU_allocated=False,
            note='Existence is not integrity, compatibility or performance proof'), indent=2))
        return
    env = {k: v for k, v in os.environ.items() if k != 'SUDO_PW'}
    env.update(CUDA_VISIBLE_DEVICES=a.gpu, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONPATH=os.pathsep.join((str(EXT), str(PROJECT))), V28_REQUIRE_TORCH='1')
    exe = os.environ.get('PYTHON_BIN') or sys.executable; deadline = time.monotonic()+a.max_hours*3600
    def invoke(cmd, log=None):
        remaining = deadline-time.monotonic()
        if remaining <= 0: raise TimeoutError('Global launch budget exhausted')
        child = None; f = log.open('x', encoding='utf-8') if log else None
        try:
            child = subprocess.Popen(cmd, cwd=str(PROJECT), env=env, stdout=f,
                stderr=subprocess.STDOUT if f else None, start_new_session=True)
            print('Owned PID={}{}'.format(child.pid, ' log='+str(log) if log else ''), flush=True)
            if child.wait(timeout=remaining): raise RuntimeError('Stage failed; preserved; no automatic retry')
        finally:
            if child is not None and child.poll() is None: stop_owned(child)
            if f: f.close()
    for test in ('test_core.py', 'test_torch.py'):
        invoke([exe, '-B', str(FOLDER/test)])
    print('CPU/Torch contracts passed; actual saved-weight CUDA smoke still required. No installs/downloads.', flush=True)
    if a.action == 'check': return
    root = safe(a.output)
    if root.exists(): raise FileExistsError('Existing run preserved; choose a NEW output')
    if shutil.disk_usage(PROJECT).free < c['minimum_disk_gib']*2**30: raise RuntimeError('Disk startup reserve not met')
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(a.config or FOLDER/'config.json', root/'config.json')
    for base, files in [('source_snapshot', dict(m['files'], **{'MANIFEST.json': sha(FOLDER/'MANIFEST.json')})),
                        ('dependency_snapshot', m['dependencies'])]:
        for name in files:
            target = root/base/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile((FOLDER if base == 'source_snapshot' else PROJECT)/name, target)
    profiles = ['smoke'] if a.action == 'smoke' else ['smoke', a.profile]
    state = dict(status='RUNNING', tasks=tasks, profiles=profiles, stages={}, physical_gpu=a.gpu,
        no_cross_modal=True, AWGN_only=True, max_hours=a.max_hours, installed_environment_unchanged=True)
    atomic(root/'PIPELINE.json', state)
    try:
        for profile in profiles:
            for task in tasks:
                group = root/profile/task; group.mkdir(parents=True)
                for label in ['inputs']+['train_'+x for x in ARMS]+['eval_'+x for x in ('frozen',)+ARMS]:
                    key = profile+'/'+task+'/'+label
                    state['stages'][key] = 'RUNNING'; atomic(root/'PIPELINE.json', state)
                    atomic(root/'progress.json', dict(stage=key, state='RUNNING'))
                    action = 'audit' if label == 'inputs' else 'train' if label.startswith('train_') else 'evaluate'
                    remaining = (deadline-time.monotonic()-600)/3600
                    if remaining <= 0: raise TimeoutError('Reserved 10 minutes for independent verification/collection')
                    cmd = [exe, '-B', '-u', str(FOLDER/'run.py'), action, str(group/label), '--task', task,
                        '--profile', profile, '--gpu', a.gpu, '--config', str(root/'config.json'), '--max-hours', str(remaining)]
                    for name in ('image_features', 'text_features', 'image_checkpoint', 'text_checkpoint', 'bart_path', 'vae_path', 'lpips_assets'):
                        value = getattr(a, name)
                        if value is not None: cmd += ['--'+name.replace('_', '-'), value]
                    if action != 'audit':
                        arm = label.split('_', 1)[1]; cmd += ['--inputs', str(group/'inputs'), '--arm', arm]
                        if action == 'evaluate' and arm != 'frozen': cmd += ['--checkpoint', str(group/('train_'+arm)/'last.pth')]
                    invoke(cmd, group/(label+'.log'))
                    state['stages'][key] = 'PRODUCED_NOT_YET_INDEPENDENTLY_VERIFIED'
                    atomic(root/'PIPELINE.json', state)
            if profile == 'smoke':
                invoke([exe, '-B', '-u', str(FOLDER/'report.py'), str(root), '--full', '--only-profile', 'smoke'],
                       root/'smoke_independent_verify.log')
                atomic(root/'SMOKE_GATE.json', dict(passed=True, actual_saved_parents=True,
                    TX_RX_updated=True, independent_CPU_verified=True, performance_success=False))
                state['smoke_gate_passed'] = True; atomic(root/'PIPELINE.json', state)
        state['target_status'] = 'SMOKE_COMPLETE' if a.action == 'smoke' else 'COMPLETE'
        state['status'] = 'TRAIN_AND_EVAL_FINISHED'
        atomic(root/'PIPELINE.json', state)
        invoke([exe, '-B', '-u', str(FOLDER/'collect.py'), str(root), str(root.with_name(root.name+'_records.tar.gz')),
                '--full'], root/'collection.log')
        print('V28 '+state['target_status']+'; send PRIVATE light records. Arrays/weights remain on server.', flush=True)
    except BaseException as exc:
        state['status'] = ('COLLECTION_FAILED' if state['status'] == 'TRAIN_AND_EVAL_FINISHED' else
            'PARTIAL_BUDGET' if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) else 'FAILED')
        state['exception'] = dict(type=type(exc).__name__, message=str(exc))
        atomic(root/'PIPELINE.json', state)
        print('V28 stopped. Keep all files; collect --partial for diagnostics. No automatic retry.', flush=True)
        raise


if __name__ == '__main__': main()
