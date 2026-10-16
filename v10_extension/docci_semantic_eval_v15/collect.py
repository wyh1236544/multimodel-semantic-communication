"""Verify and privately package immutable V15 results; no 180s report subprocess."""
import argparse
import gzip
import io
from pathlib import Path
import tarfile

from common import FOLDER, PROJECT, inside, read, sha, verify_seal, write
from run import verify_prepared


def collect(prepared, output, bert=None, challenge=None):
    verify_prepared(prepared)
    sources = [('prepared', Path(prepared))]
    for name, path in (('bertscore', bert), ('challenge', challenge)):
        if path:
            m = verify_seal(path, 'COMPLETE.json')
            if m['prepared_sha256'] != sha(Path(prepared) / 'PREPARED.json'):
                raise ValueError('Mixed preparations in collection')
            sources.append((name, Path(path)))
    destination = Path(output).absolute()
    if PROJECT not in destination.parents or destination.exists() or destination.with_name(destination.name + '.sha256').exists():
        raise ValueError('Use a NEW archive inside the project')
    for p in (destination.parent,) + tuple(destination.parents):
        if p.is_symlink():
            raise ValueError('Output symlink forbidden')
    files = {}
    for name, folder in sources:
        if folder.resolve() == destination.parent or folder.resolve() in destination.parents:
            raise ValueError('Archive must be outside immutable input directories')
        for p in sorted(folder.rglob('*')):
            if p.is_symlink():
                raise ValueError('No symlink inputs')
            if p.is_file():
                files[name + '/' + p.relative_to(folder).as_posix()] = p
    # Source only; no weights, raw data repository, cache, or unrelated output tree.
    for name in list(read(FOLDER / 'MANIFEST.json')['files']) + ['MANIFEST.json']:
        files['source_snapshot/' + name] = inside(FOLDER, name)
    inventory = {name: sha(path) for name, path in files.items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w', format=tarfile.USTAR_FORMAT) as tf:
            for name, p in files.items():
                data = p.read_bytes()
                import hashlib
                if hashlib.sha256(data).hexdigest() != inventory[name]:
                    raise ValueError('Input changed while collecting')
                info = tarfile.TarInfo('v15_semantic_records/' + name)
                info.size, info.mode = len(data), 0o644
                tf.addfile(info, io.BytesIO(data))
    with tarfile.open(destination, 'r:gz') as tf:
        if len(tf.getmembers()) != len(files):
            raise ValueError('Collection inventory differs')
        for m in tf.getmembers():
            import hashlib
            if not m.isfile() or hashlib.sha256(tf.extractfile(m).read()).hexdigest() != inventory[m.name[len('v15_semantic_records/'):]]:
                raise ValueError('Collection bytes differ')
    with destination.with_name(destination.name + '.sha256').open('x') as f:
        f.write(sha(destination) + '  ' + destination.name + '\n')
    write(destination.with_name(destination.name + '.receipt.json'), dict(archive_sha256=sha(destination),
         files=len(files), bytes=destination.stat().st_size, private_raw_text=True,
         public_GitHub_upload=False, member_sha256=inventory))
    print('PRIVATE records verified: ' + str(destination), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--bert')
    p.add_argument('--challenge')
    a = p.parse_args()
    collect(a.prepared, a.output, a.bert, a.challenge)
