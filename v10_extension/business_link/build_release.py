"""Deterministic source tar; verify a real extraction and additive installation."""
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
from install import install


def main():
    folder = Path(__file__).resolve().parent
    project = folder.parents[1]
    release = project/'releases'
    release.mkdir(exist_ok=True)
    archive = release/'v10_business_link_v3.tar.gz'
    if archive.exists() or (folder/'MANIFEST.json').exists():
        raise FileExistsError('Preserve existing release/manifest; use a new version')
    files = sorted(p for p in folder.iterdir() if p.suffix in ('.py', '.md'))
    for path in files:
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path), feature_version=(3, 8))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    required = ['v10_extension/physical_link/digital.py', 'v10_extension/physical_link/waveform.py',
                'v10_extension/v10/model.py', 'v10_extension/v10/protocol.py']
    manifest = {'schema': 'business-link-v3-source', 'source_only': True,
                'files': {p.name: digest(p) for p in files},
                'required_sources': {name: digest(project/name) for name in required}}
    with (folder/'MANIFEST.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    with archive.open('xb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0, filename='') as gz:
            with tarfile.open(fileobj=gz, mode='w') as tar:
                for path in files+[folder/'MANIFEST.json']:
                    data = path.read_bytes()
                    info = tarfile.TarInfo('_v10_business_link_v3/'+path.name)
                    info.size, info.mode = len(data), 0o644
                    tar.addfile(info, io.BytesIO(data))
    restore = Path(tempfile.mkdtemp(prefix='business_v3_restore_', dir=str(release)))
    with tarfile.open(archive, 'r:gz') as tar:
        names = {'_v10_business_link_v3/'+p.name for p in files+[folder/'MANIFEST.json']}
        if set(tar.getnames()) != names or any(not m.isfile() for m in tar.getmembers()):
            raise ValueError('Unexpected archive members')
        tar.extractall(str(restore))
    source = restore/'_v10_business_link_v3'
    for name, expected in manifest['files'].items():
        if digest(source/name) != expected:
            raise ValueError('Archive restoration hash mismatch')
    synthetic = restore/'synthetic_multimodel'
    synthetic.mkdir()
    for name in required:
        path = synthetic/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write((project/name).read_bytes())
    install(source, synthetic)
    install(source, synthetic, verify_only=True)
    conflict = synthetic/'v10_extension/business_link/run.py'
    # No destructive test rewrite: a distinct synthetic root with a conflict.
    blocked = restore/'conflict_multimodel'
    blocked.mkdir()
    for name in required:
        path = blocked/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream: stream.write((project/name).read_bytes())
    target = blocked/'v10_extension/business_link'
    target.mkdir()
    with (target/'run.py').open('xb') as stream: stream.write(b'existing-user-file\n')
    try:
        install(source, blocked)
    except FileExistsError:
        pass
    else:
        raise ValueError('Conflict was not rejected')
    if (target/'run.py').read_bytes() != b'existing-user-file\n':
        raise ValueError('Conflict file was modified')
    checksum = digest(archive)
    with archive.with_name(archive.name+'.sha256').open('x') as stream:
        stream.write(checksum+'  '+archive.name+'\n')
    print(json.dumps({'archive': str(archive), 'sha256': checksum, 'source_files': len(files),
        'restore_and_additive_install_verified': str(restore), 'python38_parse': True}, indent=2))


if __name__ == '__main__':
    main()
