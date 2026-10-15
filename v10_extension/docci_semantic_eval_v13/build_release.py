"""Build/verify an additive source-only tar; no historical sources or results modified."""
import ast
import gzip
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from core import FOLDER, digest, encode, file_hash, save


def main():
    project = FOLDER.parents[1]
    archive = project / 'releases/v10_docci_semantic_eval_v13.tar.gz'
    manifest = FOLDER / 'MANIFEST.json'
    if archive.exists() or manifest.exists():
        raise FileExistsError('Existing release preserved; version new changes explicitly')
    sources = sorted(p for p in FOLDER.iterdir() if p.is_file() and p.suffix in ('.py', '.md', '.json'))
    for file in sources:
        if file.suffix == '.py':
            ast.parse(file.read_text(encoding='utf-8'), feature_version=(3, 8))
    result = subprocess.run([sys.executable, '-B', str(FOLDER / 'test_eval.py')], check=True,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    save(manifest, dict(schema='v13-source-only', python_min='3.8', training_changes=False,
         files={p.name: file_hash(p) for p in sources}, protocol_sha256=file_hash(FOLDER / 'protocol.json')))
    archive.parent.mkdir(exist_ok=True)
    prefix = 'v10_extension/docci_semantic_eval_v13/'
    with archive.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w', format=tarfile.USTAR_FORMAT) as tar:
            for file in sources + [manifest]:
                data = file.read_bytes()
                info = tarfile.TarInfo(prefix + file.name)
                info.size, info.mode = len(data), 0o644
                tar.addfile(info, io.BytesIO(data))
    with tarfile.open(archive) as tar, tempfile.TemporaryDirectory(dir=str(project / 'releases')) as tmp:
        restored = Path(tmp)
        if set(tar.getnames()) != {prefix + p.name for p in sources + [manifest]}:
            raise ValueError('Archive inventory differs')
        for member in tar.getmembers():
            if not member.isfile() or member.name != prefix + Path(member.name).name:
                raise ValueError('Unexpected archive path/type')
            data = tar.extractfile(member).read()
            if digest(data) != file_hash(FOLDER / Path(member.name).name):
                raise ValueError('Archive bytes differ')
            destination = restored / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as handle:
                handle.write(data)
        subprocess.run([sys.executable, '-B', str(restored / prefix / 'test_eval.py')], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with archive.with_name(archive.name + '.sha256').open('x') as handle:
        handle.write(file_hash(archive) + '  ' + archive.name + '\n')
    receipt = dict(archive=archive.name, bytes=archive.stat().st_size, sha256=file_hash(archive),
         files=len(sources) + 1, python38_parse=True, synthetic_test_methods=14,
         isolated_restore_tests_passed=True, source_inventory_verified=True,
         real_pretrained_BERTScore='NOT_EXECUTED', human_facts='PENDING',
         training='NOT_EXECUTED', GPU='NOT_USED', server_environment='UNCHANGED')
    save(project / 'releases/v10_docci_semantic_eval_v13_verification.json', receipt)
    with (project / 'releases/v10_docci_semantic_eval_v13_tests.txt').open('x') as handle:
        handle.write(result.stdout)
    print(encode(receipt).decode(), end='')


if __name__ == '__main__':
    main()
