"""Build source-only tar, checksum and verify extracted bytes inside project."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile


def main():
    folder = Path(__file__).resolve().parent
    project = folder.parents[1]
    release = project/'releases'
    release.mkdir(exist_ok=True)
    archive = release/'v10_pilot_gain_v1.tar.gz'
    if archive.exists():
        raise FileExistsError(archive)
    files = sorted(p for p in folder.iterdir() if p.suffix in ('.py', '.md'))
    manifest = {p.relative_to(project).as_posix(): {'bytes': p.stat().st_size,
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
    mbytes = (json.dumps({'files': manifest, 'source_only': True}, indent=2)+'\n').encode()
    with (folder/'MANIFEST.json').open('xb') as f:
        f.write(mbytes)
    with archive.open('xb') as f:
        with gzip.GzipFile(fileobj=f, mode='wb', mtime=0, filename='') as gz:
            with tarfile.open(fileobj=gz, mode='w') as tar:
                for p in files+[folder/'MANIFEST.json']:
                    data = p.read_bytes()
                    info = tarfile.TarInfo(p.relative_to(project).as_posix())
                    info.size = len(data); info.mode = 0o644
                    tar.addfile(info, io.BytesIO(data))
    restore = Path(tempfile.mkdtemp(prefix='pilot_gain_restore_', dir=str(release)))
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.name not in list(manifest)+['v10_extension/pilot_gain/MANIFEST.json']:
                raise ValueError('Unexpected archive target')
        tar.extractall(str(restore))
    for name, item in manifest.items():
        if hashlib.sha256((restore/name).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Restore checksum mismatch')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with archive.with_name(archive.name+'.sha256').open('x') as f:
        f.write(digest+'  '+archive.name+'\n')
    print(json.dumps({'archive': str(archive), 'sha256': digest, 'source_files': len(files),
                      'restore_verified': str(restore)}, indent=2))


if __name__ == '__main__':
    main()
