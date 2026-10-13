"""All-conflict preflight, additive source installation, recoverable handoff."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def safe(root, name):
    p = Path(name)
    if p.is_absolute() or not p.parts or any(v in ('.', '..') for v in p.parts): raise ValueError('Unsafe path')
    cursor = root
    for v in p.parts:
        cursor = cursor/v
        if cursor.is_symlink(): raise ValueError('Symlink destination rejected')
    return cursor


def install(source, root, verify_only=False):
    source = Path(source).resolve(strict=True); root = Path(root).resolve(strict=True)
    manifest = json.loads(safe(source, 'MANIFEST.json').read_text())
    if manifest['schema'] != 'clip-bart-joint-v6-source' or manifest['source_only'] is not True: raise ValueError('Wrong source release')
    for name, sha in manifest['required_sources'].items():
        f = safe(root, name)
        if not f.is_file() or digest(f) != sha: raise ValueError('Old prerequisite differs; nothing installed: '+name)
    for name, sha in manifest['files'].items():
        if Path(name).name != name or digest(safe(source, name)) != sha: raise ValueError('Invalid payload hash/path')
    target = safe(root, 'v10_extension/clip_bart_joint_v6'); handoff = safe(root, 'v10_extension/docs/HANDOFF.md')
    old = handoff.read_bytes(); entry = (source/'SHARED_HANDOFF_ENTRY.md').read_text().rstrip()+'\n'
    if entry.splitlines()[0] in old.decode() and entry not in old.decode(): raise FileExistsError('Divergent shared handoff preserved')
    if target.exists():
        if not target.is_dir(): raise FileExistsError('Non-directory target')
        for name, sha in manifest['files'].items():
            if not safe(target, name).is_file() or digest(target/name) != sha: raise FileExistsError('Existing user source preserved: '+name)
        if digest(safe(target, 'MANIFEST.json')) != digest(source/'MANIFEST.json'): raise FileExistsError('Existing manifest differs')
    elif verify_only: raise FileNotFoundError('Module not installed')
    else:
        target.mkdir(exist_ok=False)
        for name in list(manifest['files'])+['MANIFEST.json']:
            with (target/name).open('xb') as f: f.write((source/name).read_bytes())
    if verify_only and entry not in old.decode(): raise ValueError('Shared handoff entry absent')
    if entry not in old.decode():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        records = safe(root, 'v10_extension/deployment_records'); records.mkdir(exist_ok=True)
        backup = records/('HANDOFF_before_clip_bart_joint_v6_'+stamp+'.md')
        with backup.open('xb') as f: f.write(old)
        temporary = safe(root, 'v10_extension/docs/HANDOFF.bart-latent-v4-'+stamp+'.tmp')
        with temporary.open('xb') as f: f.write(old+b'\n'+entry.encode()); f.flush(); os.fsync(f.fileno())
        os.replace(str(temporary), str(handoff))
        with (records/('clip_bart_joint_v6_'+stamp+'.json')).open('x') as f:
            json.dump(dict(backup=str(backup.relative_to(root)), before_sha256=hashlib.sha256(old).hexdigest(),
                after_sha256=digest(handoff), release_manifest_sha256=digest(source/'MANIFEST.json')), f, indent=2)
    if any(digest(target/n) != sha for n, sha in manifest['files'].items()): raise ValueError('Post-install source differs')
    print('CLIP+BART joint V6 installed/verified. Old source/data/weights/results preserved. No training.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--project-root', required=True); p.add_argument('--verify-only', action='store_true')
    a = p.parse_args(); install(Path(__file__).resolve().parent, a.project_root, a.verify_only)
