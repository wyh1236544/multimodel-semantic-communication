"""Additive source installer; validate all inputs/conflicts before touching old files."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(root, name):
    path = Path(name)
    if path.is_absolute() or not path.parts or any(p in ('.', '..') for p in path.parts):
        raise ValueError('Unsafe relative install path')
    current = root
    for part in path.parts:
        current = current/part
        if current.is_symlink():
            raise ValueError('Symlink install path refused')
    return current


def install(source, root, verify_only=False):
    source, root = Path(source).resolve(strict=True), Path(root).resolve(strict=True)
    manifest = json.loads(safe(source, 'MANIFEST.json').read_text(encoding='utf-8'))
    if manifest['schema'] != 'fso-fluctuating-v2-source' or not manifest['source_only']:
        raise ValueError('Wrong release')
    for name, expected in manifest['required_sources'].items():
        path = safe(root, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Prerequisite differs/missing; nothing installed: '+name)
    for name, expected in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Flat source package required')
        if digest(safe(source, name)) != expected:
            raise ValueError('Payload hash mismatch: '+name)
    handoff = safe(root, 'v10_extension/docs/HANDOFF.md')
    old = handoff.read_bytes()
    entry = safe(source, 'SHARED_HANDOFF_ENTRY.md').read_text(encoding='utf-8').rstrip()+'\n'
    if entry.splitlines()[0] in old.decode('utf-8') and entry not in old.decode('utf-8'):
        raise FileExistsError('Divergent existing handoff preserved')
    target = safe(root, 'v10_extension/fso_fluctuating_v2')
    if target.exists():
        for name, expected in manifest['files'].items():
            if not safe(target, name).is_file() or digest(target/name) != expected:
                raise FileExistsError('Existing source differs; preserved: '+str(target/name))
        if digest(safe(target, 'MANIFEST.json')) != digest(source/'MANIFEST.json'):
            raise FileExistsError('Existing manifest differs; preserved')
    elif verify_only:
        raise FileNotFoundError('New module is not installed')
    elif not verify_only:
        target.mkdir(exist_ok=False)
        for name in list(manifest['files'])+['MANIFEST.json']:
            with (target/name).open('xb') as stream:
                stream.write((source/name).read_bytes())
    if verify_only and entry not in old.decode('utf-8'):
        raise ValueError('Handoff entry missing')
    if entry not in old.decode('utf-8'):
        records = safe(root, 'v10_extension/deployment_records'); records.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = records/('HANDOFF_before_fso_fluctuating_v2_'+stamp+'.md')
        with backup.open('xb') as stream:
            stream.write(old)
        temporary = safe(root, 'v10_extension/docs/HANDOFF.fso-fluctuating-v2-'+stamp+'.tmp')
        with temporary.open('xb') as stream:
            stream.write(old+b'\n'+entry.encode()); stream.flush(); os.fsync(stream.fileno())
        os.replace(str(temporary), str(handoff))
        with (records/('fso_fluctuating_v2_'+stamp+'.json')).open('x') as stream:
            json.dump(dict(backup=str(backup.relative_to(root)), before_sha256=hashlib.sha256(old).hexdigest(),
                           after_sha256=digest(handoff), release_manifest_sha256=digest(source/'MANIFEST.json')), stream, indent=2)
    for name, expected in manifest['files'].items():
        if digest(target/name) != expected:
            raise ValueError('Post-install hash mismatch')
    print('FSO fluctuating V2 installed/verified; old source, weights, split and results preserved. Handoff backed up before append.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--project-root', required=True)
    p.add_argument('--verify-only', action='store_true'); a = p.parse_args()
    install(Path(__file__).resolve().parent, Path(a.project_root), a.verify_only)

