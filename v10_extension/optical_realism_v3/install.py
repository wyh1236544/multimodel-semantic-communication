"""Additive checked installation. All conflicts checked before any source write."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(root,name):
    path=Path(name)
    if path.is_absolute() or not path.parts or any(p in ('.','..') for p in path.parts):
        raise ValueError('Unsafe relative deployment path')
    current=root
    for part in path.parts:
        current=current/part
        if current.is_symlink(): raise ValueError('Symlink installation path refused')
    return current


def install(source,root,verify_only=False):
    source,root=Path(source).resolve(strict=True),Path(root).resolve(strict=True)
    manifest=json.loads(safe(source,'MANIFEST.json').read_text(encoding='utf-8'))
    if manifest['schema']!='optical-realism-v3-source' or not manifest['source_only']:
        raise ValueError('Wrong source-only release')
    if not safe(root,'v10_extension').is_dir(): raise ValueError('Existing multimodel V10 root required')
    for name,expected in manifest['required_sources'].items():
        path=safe(root,name)
        if not path.is_file() or digest(path)!=expected:
            raise ValueError('Missing/changed prerequisite; NOTHING installed: '+str(path))
    for name,expected in manifest['files'].items():
        if Path(name).name!=name: raise ValueError('Flat source package required')
        path=safe(source,name)
        if not path.is_file() or digest(path)!=expected: raise ValueError('Source hash mismatch: '+name)
    handoff=safe(root,'v10_extension/docs/HANDOFF.md')
    if not handoff.is_file(): raise ValueError('Existing shared handoff required')
    old=handoff.read_bytes()
    entry=safe(source,'SHARED_HANDOFF_ENTRY.md').read_text(encoding='utf-8').rstrip()+'\n'
    if entry.splitlines()[0] in old.decode('utf-8') and entry not in old.decode('utf-8'):
        raise FileExistsError('Divergent existing handoff entry preserved')
    if verify_only and entry not in old.decode('utf-8'): raise ValueError('Shared handoff entry not installed')
    target=safe(root,'v10_extension/optical_realism_v3')
    if target.exists():
        for name,expected in manifest['files'].items():
            if not safe(target,name).is_file() or digest(target/name)!=expected:
                raise FileExistsError('Existing source differs; preserved: '+str(target/name))
        if digest(safe(target,'MANIFEST.json'))!=digest(source/'MANIFEST.json'):
            raise FileExistsError('Existing source manifest differs; preserved')
    elif verify_only: raise FileNotFoundError('Optical realism V3 not installed')
    else:
        target.mkdir(exist_ok=False)
        for name in list(manifest['files'])+['MANIFEST.json']:
            with (target/name).open('xb') as stream: stream.write((source/name).read_bytes())
    for name,expected in manifest['files'].items():
        if digest(target/name)!=expected: raise ValueError('Post-install source verification failed')
    if entry not in old.decode('utf-8'):
        records=safe(root,'v10_extension/deployment_records'); records.mkdir(exist_ok=True)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup=records/('HANDOFF_before_optical_realism_v3_'+stamp+'.md')
        with backup.open('xb') as stream: stream.write(old)
        temporary=safe(root,'v10_extension/docs/HANDOFF.optical-realism-v3-'+stamp+'.tmp')
        with temporary.open('xb') as stream:
            stream.write(old+b'\n'+entry.encode('utf-8')); stream.flush(); os.fsync(stream.fileno())
        os.replace(str(temporary),str(handoff))
        with (records/('optical_realism_v3_'+stamp+'.json')).open('x',encoding='utf-8') as stream:
            json.dump(dict(shared_handoff_backup=str(backup.relative_to(root)),before_sha256=hashlib.sha256(old).hexdigest(),after_sha256=digest(handoff)),stream,indent=2)
    print('Optical realism V3 source verified. Old code/weights/results unchanged; handoff backed up before append.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--project-root',required=True)
    p.add_argument('--verify-only',action='store_true'); a=p.parse_args()
    install(Path(__file__).resolve().parent,Path(a.project_root),a.verify_only)

