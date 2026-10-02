"""Additive manifest-checked installation. Refuse conflicts before any write."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(root,name):
    parts=Path(name).parts
    if not parts or Path(name).is_absolute() or any(part in ('.','..') for part in parts):
        raise ValueError('Unsafe relative path')
    result=root
    for part in parts:
        result=result/part
        if result.is_symlink(): raise ValueError('Linked installation path rejected')
    return result


def install(source,root,verify_only=False):
    source,root=Path(source).resolve(strict=True),Path(root).resolve(strict=True)
    manifest=json.loads((source/'MANIFEST.json').read_text(encoding='utf-8'))
    if manifest['schema']!='optical-link-v1-source': raise ValueError('Unexpected source package')
    if not safe(root,'v10_extension').is_dir(): raise ValueError('Need existing multimodel/V10 root')
    for name,expected in manifest['required_sources'].items():
        path=safe(root,name)
        if not path.is_file() or digest(path)!=expected:
            raise ValueError('Prerequisite missing/changed; no files installed: '+str(path))
    for name,expected in manifest['files'].items():
        if Path(name).name!=name: raise ValueError('Flat package filenames required')
        path=safe(source,name)
        if not path.is_file() or digest(path)!=expected: raise ValueError('Package hash mismatch: '+name)
    handoff=safe(root,'v10_extension/docs/HANDOFF.md')
    if not handoff.is_file(): raise ValueError('Existing shared HANDOFF.md required')
    old=handoff.read_text(encoding='utf-8')
    entry=(source/'SHARED_HANDOFF_ENTRY.md').read_text(encoding='utf-8').rstrip()+'\n'
    if entry.splitlines()[0] in old and entry not in old:
        raise FileExistsError('Divergent optical handoff entry preserved; inspect before install')
    if verify_only and entry not in old:
        raise ValueError('Shared optical handoff entry missing')
    target=safe(root,'v10_extension/optical_link_v1')
    if target.exists():
        for name,expected in manifest['files'].items():
            path=safe(target,name)
            if not path.is_file() or digest(path)!=expected:
                raise FileExistsError('Existing optical_link_v1 differs; preserved: '+str(path))
        if digest(safe(target,'MANIFEST.json'))!=digest(source/'MANIFEST.json'):
            raise FileExistsError('Existing manifest differs; preserved')
        print('Identical existing optical V1 source verified: '+str(target))
    else:
        if verify_only: raise FileNotFoundError('Optical V1 is not installed')
        target.mkdir(exist_ok=False)
        for name in list(manifest['files'])+['MANIFEST.json']:
            with (target/name).open('xb') as stream: stream.write((source/name).read_bytes())
    for name,expected in manifest['files'].items():
        if digest(target/name)!=expected: raise ValueError('Post-install verification failed')
    if entry not in old:
        records=safe(root,'v10_extension/deployment_records')
        records.mkdir(exist_ok=True)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup=safe(root,'v10_extension/deployment_records/HANDOFF_before_optical_v1_'+stamp+'.md')
        old_bytes=handoff.read_bytes()
        with backup.open('xb') as stream: stream.write(old_bytes)
        new=old_bytes+b'\n'+entry.encode('utf-8')
        temporary=safe(root,'v10_extension/docs/HANDOFF.optical-v1-'+stamp+'.tmp')
        with temporary.open('xb') as stream: stream.write(new); stream.flush(); os.fsync(stream.fileno())
        os.replace(str(temporary),str(handoff))
        record=safe(root,'v10_extension/deployment_records/optical_v1_'+stamp+'.json')
        with record.open('x',encoding='utf-8') as stream:
            json.dump(dict(shared_handoff_backup=str(backup.relative_to(root)),
                before_sha256=hashlib.sha256(old_bytes).hexdigest(), after_sha256=digest(handoff)),stream,indent=2)
    print('Optical V1 and backed-up shared handoff verified; old code/weights/results untouched.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',required=True); p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    install(Path(__file__).resolve().parent,Path(args.project_root),args.verify_only)
