"""Attach V15 ONCE to an existing, completed V14 text run. Never starts training."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parents[1]
ARMS = ['legacy64', 'released64']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe(path, root=PROJECT):
    p = Path(path).absolute()
    if Path(root) not in p.parents:
        raise ValueError('Path must stay below existing project root')
    for q in (p,) + tuple(p.parents):
        if q.is_symlink():
            raise ValueError('Symlink path rejected')
    return p


def source_check():
    m = read(FOLDER / 'MANIFEST.json')
    for root, mapping in ((FOLDER, m['files']), (PROJECT, m['dependencies'])):
        for name, expected in mapping.items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Invalid manifest path')
            p = safe(root / name)
            if sha(p) != expected:
                raise ValueError('Source/dependency changed: ' + name)
    return m


def load_v15():
    source_check()
    folder = PROJECT / 'v10_extension/docci_semantic_eval_v15'
    # No files from experiment output are imported. These installed modules are pinned.
    for name in ('common', 'run', 'challenge'):
        mod = sys.modules.get(name)
        if mod is not None and Path(mod.__file__).resolve().parent != folder:
            raise RuntimeError('Ambiguous Python module identity: ' + name)
    sys.path.insert(0, str(folder))
    return importlib.import_module('run')


def expected_outputs():
    return ['text/native_validation'] + [p for a in ARMS for p in (
        'text/' + a + '/diagnostic_train', 'text/' + a + '/validation')]


def report_receipt(path):
    """V14 prints known VERIFY lines, then one JSON document; no code execution."""
    text = Path(path).read_text(encoding='utf-8')
    prefix = ''.join('VERIFY ' + p + '\n' for p in expected_outputs())
    if not text.startswith(prefix):
        raise ValueError('Missing/unexpected V14 independent-verifier output')
    result = json.loads(text[len(prefix):])
    if (result.get('status') != 'BASELINE_RECORDS_VERIFIED_NOT_QUALITY_SUCCESS'
            or result.get('mode') != 'train' or result.get('FSO') is not False
            or set(result.get('groups', {})) != set(expected_outputs())):
        raise ValueError('V14 independent-verifier receipt differs')
    for name in expected_outputs():
        if not name.endswith('validation'):
            continue
        controls = ['native'] if name == 'text/native_validation' else ['matched', 'shuffled', 'zero']
        group = result['groups'][name]
        expected = {s + '/' + c for s in ('greedy', 'deployment') for c in controls}
        if set(group) != expected or any(type(v.get('n')) is not int or v['n'] != 100 for v in group.values()):
            raise ValueError('Need complete100-example greedy/deployment validation, not smoke')
    return result


def inspect(training_run):
    """Cheap state check. Absence of completion does NOT prove a process is alive."""
    run = safe(training_run)
    if run.parent != PROJECT / 'experiment_results':
        raise ValueError('Pass the outer existing experiment_results/V14_run directory, not main/smoke')
    for fail in (run / 'FAILED.json', run / 'main/FAILED.json', run / 'smoke/FAILED.json'):
        fail = safe(fail)
        if fail.exists():
            return dict(state='FAILED_PRESERVED', reason='Failure evidence exists; no retry, training or evaluation started', run=str(run))
    if not (run / 'LAUNCH_FINISHED.json').is_file():
        return dict(state='NOT_READY', reason='Outer completion absent. It may still run or have stopped; no training/polling started', run=str(run))
    launch = read(run / 'launch_identity.json')
    if (launch.get('action') != 'train' or launch.get('task') != 'text'
            or launch.get('profile') != 'pilot' or launch.get('text_arms') != ARMS
            or launch.get('physical_gpu') != 1 or launch.get('smoke_weights_reused') is not False):
        raise ValueError('Not the existing V14 two-arm text pilot; no automatic new run')
    main = safe(run / 'main')
    finished = read(run / 'LAUNCH_FINISHED.json')
    if (Path(finished['final_root']).absolute() != main
            or finished.get('engineering_success_NOT_recovery_claim') is not True):
        raise ValueError('Launcher did not finish the expected main run')
    design = read(main / 'design.json')
    if design.get('mode') != 'train' or design.get('task') != 'text' or design.get('text_arms') != ARMS:
        raise ValueError('Wrong main design')
    main_done = read(main / 'RUN_FINISHED.json')
    if sha(main / 'LIGHT_MANIFEST.json') != main_done['manifest_sha256']:
        raise ValueError('V14 light manifest changed/incomplete')
    identity = read(main / 'run_identity.json')
    approved = sha(PROJECT / 'v10_extension/docci_baseline_v14/MANIFEST.json')
    if (identity.get('mode') != 'train' or identity.get('task') != 'text'
            or identity.get('outputs') != expected_outputs()
            or identity.get('source_manifest_sha256') != approved
            or sha(main / 'source_snapshot/MANIFEST.json') != approved):
        raise ValueError('Executed V14 source/task identity changed')
    report_receipt(run / 'main_verify.log')
    names = ['launch_identity.json', 'LAUNCH_FINISHED.json', 'main/RUN_FINISHED.json',
             'main/LIGHT_MANIFEST.json', 'main/run_identity.json', 'main_verify.log']
    return dict(state='READY_FOR_READ_ONLY_IMPORT', run=str(run), main=str(main),
        inputs={name: sha(run / name) for name in names},
        scope='Outer launcher+saved CPU-verifier receipt checked; full V14 neural/frame/history verification NOT rerun')


def output_for(run):
    return PROJECT / 'experiment_results' / ('v15_from_' + Path(run).name)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def prepare_once(training_run, v13_prepared, prepare_fn=None, verify_fn=None):
    source_check()
    state = inspect(training_run)
    if state['state'] != 'READY_FOR_READ_ONLY_IMPORT':
        return state
    prepared = safe(v13_prepared)
    # Pin exact original inputs as well as this adapter; no original outputs rewritten.
    identity = dict(v14=state['inputs'], v13_prepared_sha256=sha(prepared / 'PREPARED.json'),
                    guard_source_sha256=sha(FOLDER / 'MANIFEST.json'), training_run=state['run'])
    output = safe(output_for(state['run']))
    if output.exists():
        done = safe(output / 'IMPORT_FINISHED.json')
        if not done.is_file():
            return dict(state='OUTPUT_RESERVED_OR_PARTIAL', output=str(output),
                        reason='Another import or preserved failed attempt may exist; no repeat or auto cleanup')
        previous = read(done)
        if previous['identity'] != identity:
            raise ValueError('Existing output belongs to different input/source bytes; preserve it')
        verifier = verify_fn or load_v15().verify_prepared
        verifier(output / 'prepared')
        return dict(state='ALREADY_IMPORTED_NO_REPEAT', output=str(output),
                    reason='Same complete frozen import verified; no re-preparation, training or model scoring')
    # Atomic mkdir is the cross-process claim. Losing contenders NEVER start work.
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return dict(state='OUTPUT_RESERVED_OR_PARTIAL', output=str(output), reason='Concurrent caller reserved output; no duplicate')
    write(output / 'IMPORT_STARTED.json', dict(identity=identity, pid=os.getpid(),
          operation='READ_ONLY_TEXT_IMPORT_ONLY', no_training_or_model_scoring=True))
    try:
        producer = prepare_fn or load_v15().prepare
        producer(prepared, output / 'prepared', state['main'])
        after = inspect(training_run)
        if after != state or sha(prepared / 'PREPARED.json') != identity['v13_prepared_sha256']:
            raise ValueError('Inputs changed during import; preserve partial output')
        verifier = verify_fn or load_v15().verify_prepared
        verifier(output / 'prepared')
        write(output / 'IMPORT_FINISHED.json', dict(identity=identity,
              status='READ_ONLY_IMPORT_COMPLETE_NOT_MODEL_QUALITY_PASS',
              historical_quality='UNCHANGED', V14_BERTScore='NOT_MEASURED', human_facts='PENDING',
              no_training_or_GPU_or_download=True, no_repeat_scoring_of_old_V13=True))
    except BaseException as exc:
        write(output / 'IMPORT_FAILED.json', dict(exception=type(exc).__name__, message=str(exc),
              partial_preserved=True, automatic_retry=False))
        raise
    return dict(state='IMPORTED_ONCE', output=str(output),
                reason='V14 text attached to V15; no training, BERTScore inference or download')


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('action', choices=('check', 'status', 'prepare'))
    p.add_argument('--training-run', default='experiment_results/v14_text_pilot_01')
    p.add_argument('--v13-prepared', default='_v13_server_cpu_offline_01/prepared')
    a = p.parse_args()
    source_check()
    if a.action == 'check':
        print('Guard source verified; V14/V15 preserved. No training/GPU/job started.')
        return
    result = inspect(a.training_run) if a.action == 'status' else prepare_once(a.training_run, a.v13_prepared)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if result['state'] == 'NOT_READY':
        raise SystemExit(2)
    if result['state'] == 'FAILED_PRESERVED':
        raise SystemExit(3)
    if result['state'] == 'OUTPUT_RESERVED_OR_PARTIAL':
        raise SystemExit(4)


if __name__ == '__main__':
    main()
