"""Package approved original V15 plus additive guard, preserving all old bytes."""
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

from guard import FOLDER, PROJECT, sha, write


def invoke(path, *args, expected=0):
    result = subprocess.run([sys.executable, '-B', str(path)] + list(args), cwd=str(path.parents[2])
        if path.name == 'guard.py' else str(PROJECT), env={k:v for k,v in __import__('os').environ.items() if k != 'SUDO_PW'},
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    if result.returncode != expected:
        raise RuntimeError(result.stdout)
    return result.stdout


def main():
    output = PROJECT / 'releases/v10_docci_semantic_eval_v15_guard.tar.gz'
    if output.exists():
        raise FileExistsError('Existing source release preserved')
    v15 = PROJECT / 'v10_extension/docci_semantic_eval_v15'
    original = json.loads((v15 / 'MANIFEST.json').read_text())
    payload = {('v10_extension/docci_semantic_eval_v15/' + n): v15 / n for n in list(original['files']) + ['MANIFEST.json']}
    dependencies = dict(original['dependencies'])
    for name, p in payload.items():
        dependencies[name] = sha(p)
    for name in ('MANIFEST.json', 'launch.py', 'report.py'):
        p = PROJECT / 'v10_extension/docci_baseline_v14' / name
        dependencies[p.relative_to(PROJECT).as_posix()] = sha(p)
    files = sorted(p for p in FOLDER.iterdir() if p.suffix in ('.py', '.md') and p.is_file())
    manifest = dict(schema='v15-existing-v14-run-guard', files={p.name: sha(p) for p in files}, dependencies=dependencies,
        training='NEVER_STARTED', writes_V14=False, new_BERTScore_inference=False,
        default_training_run='experiment_results/v14_text_pilot_01', old_V15_manifest_sha256=sha(v15 / 'MANIFEST.json'))
    mf = FOLDER / 'MANIFEST.json'
    if mf.exists():
        if json.loads(mf.read_text()) != manifest:
            raise ValueError('Unreleased manifest differs; inspect before rebuilding')
    else:
        write(mf, manifest)
    for p in files + list(payload.values()):
        if p.suffix == '.py': ast.parse(p.read_text(), feature_version=(3, 8))
    testlog = invoke(FOLDER / 'test_guard.py')
    print(testlog, end='')
    oldlog = invoke(v15 / 'test_eval.py')
    files += [mf]
    payload.update({p.relative_to(PROJECT).as_posix(): p for p in files})
    archive_files = {('payload/' + n): p.read_bytes() for n, p in payload.items()}
    archive_files['install.py'] = (FOLDER / 'install.py').read_bytes()
    outside_dependencies = {n:s for n,s in dependencies.items() if n not in payload}
    archive_files['PAYLOAD.json'] = (json.dumps(dict(files={n:sha(p) for n,p in payload.items()},
        dependencies=outside_dependencies, source_only=True, writes_V14_or_runs=False), indent=2) + '\n').encode()
    prefix = '_v10_docci_semantic_eval_v15_guard/'
    output.parent.mkdir(exist_ok=True)
    with output.open('xb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w', format=tarfile.USTAR_FORMAT) as tf:
            for name, data in sorted(archive_files.items()):
                info = tarfile.TarInfo(prefix + name)
                info.size, info.mode = len(data), 0o644
                tf.addfile(info, io.BytesIO(data))
    with tempfile.TemporaryDirectory(dir=str(PROJECT / 'releases')) as temp:
        temp = Path(temp)
        with tarfile.open(output) as tf:
            if set(tf.getnames()) != {prefix + n for n in archive_files}:
                raise ValueError('Archive inventory mismatch')
            for m in tf.getmembers():
                data = tf.extractfile(m).read()
                if not m.isfile() or data != archive_files[m.name[len(prefix):]]:
                    raise ValueError('Archive bytes/type mismatch')
                p = temp / m.name
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open('xb') as f: f.write(data)
        restored = temp / 'project'
        restored.mkdir()
        for name in outside_dependencies:
            p = restored / name
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / name, p)
        installer = temp / prefix / 'install.py'
        invoke(installer, '--project-root', str(restored))
        repeated = invoke(installer, '--project-root', str(restored))
        if 'Installed 0 missing' not in repeated: raise ValueError('Installer was not idempotent')
        restored_guard = restored / 'v10_extension/docci_semantic_eval_v15_guard/guard.py'
        invoke(restored_guard, 'check')
        invoke(restored_guard, 'status', expected=2)
        invoke(restored_guard, 'prepare', expected=2)
        if (restored / 'experiment_results').exists():
            raise ValueError('NOT_READY created work or training output')
        invoke(restored_guard.with_name('test_guard.py'))
    with output.with_name(output.name + '.sha256').open('x') as f:
        f.write(sha(output) + '  ' + output.name + '\n')
    write(output.with_name(output.name + '.verification.json'), dict(bytes=output.stat().st_size,
        sha256=sha(output), regular_members=len(archive_files), source_only=True, Python38_parse=True,
        new_synthetic_test_methods=sum(isinstance(n, ast.FunctionDef) and n.name.startswith('test_')
            for n in ast.walk(ast.parse((FOLDER / 'test_guard.py').read_text()))),
        old_V15_synthetic_tests=20, archive_member_bytes_verified=True, isolated_install_tests='PASSED',
        isolated_repeat_install='ZERO_WRITES', isolated_status_and_prepare_pending='NO_OUTPUT_CREATED_EXIT2',
        actual_user_V14_completion_or_integration='NOT_VERIFIED; no new completed text records supplied',
        V14_changed=False, old_V15_changed=False, training='NONE', GPU='NOT_USED'))
    print('Verified additive guard release: ' + str(output))


if __name__ == '__main__':
    main()
