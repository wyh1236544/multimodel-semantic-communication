"""One owned eight-hour maximum pipeline; reserves time for PRIVATE collection."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_segment_bridge_v17.core import PROJECT, FOLDER, ARMS, RUN_ORDER, safe, atomic, check_source, snapshot, read


def stop_owned(child):
    if child.poll() is not None:
        return
    group = os.getpgid(child.pid)
    if group == child.pid:
        os.killpg(group, signal.SIGTERM)
    else:
        child.terminate()
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if group == child.pid and os.getpgid(child.pid) == child.pid:
            os.killpg(child.pid, signal.SIGKILL)
        else:
            child.kill()
        child.wait()


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('action', choices=('check', 'smoke', 'pipeline')); p.add_argument('output', nargs='?')
    p.add_argument('--cache', default=str(PROJECT/'experiment_results/v16_cache_01/main'))
    p.add_argument('--max-hours', type=float, default=8)
    p.add_argument('--bart-path', default=str(PROJECT/'pretrained/bart-base'))
    p.add_argument('--text-checkpoint')
    a = p.parse_args()
    if (a.action == 'check') != (a.output is None) or not .5 <= a.max_hours <= 8:
        p.error('check has no output; other actions need NEW output; finite0.5..8h cap')
    check_source()
    exe = os.environ.get('PYTHON_BIN') or sys.executable
    if not Path(exe).is_file():
        raise FileNotFoundError('Existing offline interpreter required')
    env = {k: v for k, v in os.environ.items() if k != 'SUDO_PW'}
    env.update(CUDA_VISIBLE_DEVICES='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1', V17_REQUIRE_TORCH='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONPATH=os.pathsep.join((str(EXT), str(PROJECT))))
    deadline = time.monotonic()+a.max_hours*3600
    reserve = 900 if a.action == 'pipeline' else 300
    root = None
    state = dict(status='RUNNING', stages={}, execution_order=list(RUN_ORDER), max_hours=a.max_hours, collection_reserve_seconds=reserve,
                 no_V14_retraining=True, no_original_cache_regeneration=True, no_automatic_actual_or_FSO=True)
    def invoke(cmd, log=None, collection=False):
        available = deadline-time.monotonic()-(0 if collection else reserve)
        if available <= 0:
            raise TimeoutError('Global budget reserved for collection')
        stream = log.open('x', encoding='utf-8') if log else None
        child = None
        try:
            child = subprocess.Popen(cmd, cwd=str(PROJECT), env=env, stdout=stream,
                    stderr=subprocess.STDOUT if stream else None, start_new_session=True)
            print('Owned PID={}{}'.format(child.pid, ' log='+str(log) if log else ''), flush=True)
            if child.wait(timeout=available):
                raise RuntimeError('Owned child failed; no automatic retry; preserve logs')
        finally:
            if child is not None and child.poll() is None:
                stop_owned(child)
            if stream:
                stream.close()
    for name in ('test_core.py', 'test_storage.py', 'test_model.py'):
        invoke([exe, '-B', str(FOLDER/name)])
    print('CPU contracts passed; actual saved-model GPU gates are next. No dependency changes.', flush=True)
    if a.action == 'check':
        return
    root = safe(a.output)
    snapshot(root, dict(stage='pipeline', action=a.action, physical_gpu=1, pid=os.getpid(),
                        max_hours=a.max_hours, private_results=True, implementation_not_performance_claim=True))
    names = ['smoke_inputs']+['smoke_'+arm for arm in ARMS]
    if a.action == 'pipeline':
        names += ['inputs']+list(ARMS)
    state['stages'] = {name: 'NOT_RUN' for name in names}
    atomic(root/'PIPELINE.json', state)
    def run(action, label, smoke=False, arm=None):
        state['stages'][label] = 'RUNNING'; atomic(root/'PIPELINE.json', state)
        remaining = (deadline-time.monotonic()-reserve)/3600
        cmd = [exe, '-B', '-u', str(FOLDER/'run.py'), action, str(root/label), '--max-hours', str(remaining),
               '--bart-path', a.bart_path]
        inputs = root/('smoke_inputs' if smoke else 'inputs')
        if smoke:
            cmd += ['--smoke']
        if a.text_checkpoint:
            cmd += ['--text-checkpoint', a.text_checkpoint]
        if action == 'prepare':
            cmd += ['--cache', str(Path(a.cache).resolve()), '--cache-receipt', str(root/'cache_verified.json')]
        else:
            cmd += ['--inputs', str(inputs), '--arm', arm]
        invoke(cmd, root/(label+'.log'))
        report = [exe, '-B', '-u', str(FOLDER/'report.py'), str(root/label)]
        if action != 'prepare':
            report += ['--inputs', str(inputs)]
        invoke(report, root/(label+'_verify.log'))
        state['stages'][label] = 'VERIFIED'; atomic(root/'PIPELINE.json', state)
    failed = None
    try:
        invoke([exe, '-B', '-u', str(FOLDER/'audit_cache.py'), str(Path(a.cache).resolve()),
                str(root/'cache_verified.json')], root/'cache_audit.log')
        run('prepare', 'smoke_inputs', True)
        for arm in ARMS:
            run('train', 'smoke_'+arm, True, arm)
        # Independent smoke pair identity checks before any full-stage preparation.
        invoke([exe, '-B', '-u', str(FOLDER/'report.py'), str(root), '--light', '--receipt', 'SMOKE_VERIFIED.json'], root/'smoke_pairs_verify.log')
        if a.action == 'pipeline':
            run('prepare', 'inputs')
            for arm in RUN_ORDER:
                run('train', arm, arm=arm)
        state['status'] = 'COMPLETE' if a.action == 'pipeline' else 'SMOKE_COMPLETE'
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        state['status'] = 'PARTIAL_BUDGET'; failed = exc
    except BaseException as exc:
        state['status'] = 'FAILED'; failed = exc
    for name, value in list(state['stages'].items()):
        if value == 'RUNNING':
            state['stages'][name] = 'UNVERIFIED_PARTIAL'
    if failed is not None:
        state['exception'] = dict(type=type(failed).__name__, message=str(failed))
        atomic(root/'FAILED_OR_BUDGET.json', state['exception'])
    atomic(root/'PIPELINE.json', state)
    # The provisional smoke-only root receipt must not be overwritten by a full one.
    archive = root.with_name(root.name+'_records.tar.gz')
    try:
        invoke([exe, '-B', '-u', str(FOLDER/'collect.py'), str(root), str(archive),
                '--max-minutes', str(min(180., max(.01, (deadline-time.monotonic())/60.)))],
               root/'collection.log', collection=True)
        atomic(root/'COLLECTED.json', dict(archive=str(archive), pipeline_status=state['status'],
                                          complete=state['status'] == 'COMPLETE'))
        print('V17 {}. Send PRIVATE {} and its .sha256; no auto actual/FSO.'.format(state['status'], archive), flush=True)
    except BaseException as exc:
        state['status'] = 'COLLECTION_FAILED'
        state['collection_error'] = dict(type=type(exc).__name__, message=str(exc))
        atomic(root/'PIPELINE.json', state)
        atomic(root/'COLLECTION_FAILED.json', dict(type=type(exc).__name__, message=str(exc)))
        raise
    if failed is not None:
        raise RuntimeError('Pipeline stopped as {}; partial records retained, not success'.format(state['status']))


if __name__ == '__main__':
    main()
