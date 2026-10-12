"""Additive, hash-checked installation with byte-preserved handoff backup."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(root,name):
    p=Path(name)
    if p.is_absolute() or not p.parts or any(x in ('.','..') for x in p.parts): raise ValueError('Unsafe path')
    cursor=root
    for part in p.parts:
        cursor=cursor/part
        if cursor.is_symlink(): raise ValueError('Symlink destination rejected')
    return cursor


def install(source,root,verify_only=False):
    source=Path(source).resolve(strict=True); root=Path(root).resolve(strict=True)
    manifest=json.loads(safe(source,'MANIFEST.json').read_text())
    if manifest['schema']!='text-recovery-v3-source' or manifest['source_only'] is not True: raise ValueError('Wrong release')
    for name,sha in manifest['required_sources'].items():
        file=safe(root,name)
        if not file.is_file() or digest(file)!=sha: raise ValueError('Old dependency differs; nothing installed: '+name)
    for name,sha in manifest['files'].items():
        if Path(name).name!=name or digest(safe(source,name))!=sha: raise ValueError('Payload path/hash mismatch')
    target=safe(root,'v10_extension/text_recovery_v3'); handoff=safe(root,'v10_extension/docs/HANDOFF.md')
    old=handoff.read_bytes(); entry=(source/'SHARED_HANDOFF_ENTRY.md').read_text().rstrip()+'\n'
    if entry.splitlines()[0] in old.decode() and entry not in old.decode(): raise FileExistsError('Divergent handoff preserved')
    if target.exists():
        if not target.is_dir(): raise FileExistsError('Non-directory module target')
        for name,sha in manifest['files'].items():
            if not safe(target,name).is_file() or digest(target/name)!=sha: raise FileExistsError('Existing user file preserved: '+name)
        if digest(safe(target,'MANIFEST.json'))!=digest(source/'MANIFEST.json'): raise FileExistsError('Existing manifest differs')
    elif verify_only: raise FileNotFoundError('Module not installed')
    else:
        target.mkdir(exist_ok=False)
        for name in list(manifest['files'])+['MANIFEST.json']:
            with (target/name).open('xb') as f: f.write((source/name).read_bytes())
    if verify_only and entry not in old.decode(): raise ValueError('Shared handoff entry absent')
    if entry not in old.decode():
        records=safe(root,'v10_extension/deployment_records'); records.mkdir(exist_ok=True)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup=records/('HANDOFF_before_text_recovery_v3_'+stamp+'.md')
        with backup.open('xb') as f: f.write(old)
        temp=safe(root,'v10_extension/docs/HANDOFF.text-recovery-v3-'+stamp+'.tmp')
        with temp.open('xb') as f: f.write(old+b'\n'+entry.encode()); f.flush(); os.fsync(f.fileno())
        os.replace(str(temp),str(handoff))
        with (records/('text_recovery_v3_'+stamp+'.json')).open('x') as f:
            json.dump(dict(backup=str(backup.relative_to(root)),before_sha256=hashlib.sha256(old).hexdigest(),
                after_sha256=digest(handoff),release_manifest_sha256=digest(source/'MANIFEST.json')),f,indent=2)
    for name,sha in manifest['files'].items():
        if digest(target/name)!=sha: raise ValueError('Post-install mismatch')
    print('Text V3 installed/verified; old source, weights, data and results preserved. No training.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--project-root',required=True); p.add_argument('--verify-only',action='store_true')
    a=p.parse_args(); install(Path(__file__).resolve().parent,a.project_root,a.verify_only)
