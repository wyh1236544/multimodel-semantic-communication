"""Build fresh source-only release after tests; never changes historical code."""
import ast
import gzip
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

from common import FOLDER, PROJECT, read, sha, write


def main():
    archive = PROJECT / 'releases/v10_docci_semantic_eval_v15.tar.gz'
    if archive.exists():
        raise FileExistsError('Published release is immutable; make a new version')
    files = sorted(p for p in FOLDER.iterdir() if p.is_file() and p.suffix in ('.py', '.md', '.json') and p.name != 'MANIFEST.json')
    for p in files:
        if p.suffix == '.py': ast.parse(p.read_text(encoding='utf-8'), feature_version=(3, 8))
    dep = PROJECT / 'v10_extension/docci_semantic_eval_v13'
    dependencies = sorted(list(dep.glob('*.py')) + [dep / 'protocol.json'])
    manifest = dict(schema='v15-source-only', python_min='3.8', training_changes=False,
         files={p.name: sha(p) for p in files},
         dependencies={p.relative_to(PROJECT).as_posix(): sha(p) for p in dependencies},
         weights_or_raw_data=False, optional_CAPTURE_AlignScore_runtime='NOT_IMPLEMENTED_NOT_VALIDATED')
    path = FOLDER / 'MANIFEST.json'
    if path.exists():
        if read(path) != manifest: raise ValueError('Unreleased manifest differs; inspect before rebuilding')
    else:
        write(path, manifest)
    result = subprocess.run([sys.executable, '-B', str(FOLDER / 'test_eval.py')], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end='')
    if result.returncode: raise RuntimeError('Tests failed; no source release created')
    prefix = 'v10_extension/docci_semantic_eval_v15/'
    files += [path]
    archive.parent.mkdir(exist_ok=True)
    with archive.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w', format=tarfile.USTAR_FORMAT) as tf:
            for p in files:
                data = p.read_bytes()
                info = tarfile.TarInfo(prefix + p.name)
                info.size, info.mode = len(data), 0o644
                tf.addfile(info, io.BytesIO(data))
    with tempfile.TemporaryDirectory(dir=str(PROJECT / 'releases')) as tmp:
        restored = Path(tmp)
        with tarfile.open(archive) as tf:
            if tf.getnames() != [prefix + p.name for p in files]: raise ValueError('Archive inventory differs')
            for member, p in zip(tf.getmembers(), files):
                import hashlib
                data = tf.extractfile(member).read()
                if not member.isfile() or hashlib.sha256(data).hexdigest() != sha(p): raise ValueError('Archive bytes differ')
                target = restored / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as f: f.write(data)
        for p in dependencies:
            target = restored / p.relative_to(PROJECT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
        restored_test = subprocess.run([sys.executable, '-B', str(restored / prefix / 'test_eval.py')], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if restored_test.returncode: raise RuntimeError(restored_test.stdout)
    with archive.with_name(archive.name + '.sha256').open('x') as f:
        f.write(sha(archive) + '  ' + archive.name + '\n')
    write(archive.with_name(archive.name + '.verification.json'), dict(bytes=archive.stat().st_size,
        sha256=sha(archive), regular_members=len(files), Python38_parse=True,
        synthetic_tests='PASSED; see test source, not model performance', isolated_restore_tests='PASSED',
        actual_neural_scoring='NOT_EXECUTED', CAPTURE_AlignScore='NOT_EXECUTED', human='PENDING', training='NONE'))
    print('Verified source release: ' + str(archive))


if __name__ == '__main__':
    main()
