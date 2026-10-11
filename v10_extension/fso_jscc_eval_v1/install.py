"""Additive verified installer: existing sources/weights/results are never replaced."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(root,name):
    path=Path(name)
    if path.is_absolute() or not path.parts or any(p in ('.','..') for p in path.parts): raise ValueError('Unsafe install path')
    current=root
    for part in path.parts:
        current=current/part
        if current.is_symlink(): raise ValueError('Symlink path refused: '+str(current))
    return current


def install(source,root,verify_only=False):
    source,root=Path(source).resolve(strict=True),Path(root).resolve(strict=True)
    manifest=json.loads(safe(source,'MANIFEST.json').read_text(encoding='utf-8'))
    if manifest['schema']!='fso-jscc-expanded-eval-v1-source' or not manifest['source_only']: raise ValueError('Wrong release')
    for name,expected in manifest['required_sources'].items():
        path=safe(root,name)
        if not path.is_file() or digest(path)!=expected: raise ValueError('Dependency differs/missing; nothing installed: '+name)
    for name,expected in manifest['files'].items():
        if Path(name).name!=name or digest(safe(source,name))!=expected: raise ValueError('Flat source/hash mismatch: '+name)
    handoff=safe(root,'v10_extension/docs/HANDOFF.md'); old=handoff.read_bytes(); old_text=old.decode('utf-8')
    entry=safe(source,'SHARED_HANDOFF_ENTRY.md').read_text(encoding='utf-8').rstrip()+'\n'
    if entry.splitlines()[0] in old_text and entry not in old_text: raise FileExistsError('Divergent existing handoff preserved')
    target=safe(root,'v10_extension/fso_jscc_eval_v1')
    if target.exists():
        for name,expected in manifest['files'].items():
            if not safe(target,name).is_file() or digest(target/name)!=expected: raise FileExistsError('Existing source differs, preserved: '+str(target/name))
        if not safe(target,'MANIFEST.json').is_file() or digest(target/'MANIFEST.json')!=digest(source/'MANIFEST.json'): raise FileExistsError('Existing manifest differs, preserved')
    elif verify_only: raise FileNotFoundError('New evaluator not installed')
    if verify_only and entry not in old_text: raise ValueError('Shared handoff missing')
    # Resolve every remaining writable path before creating the new module.
    records=safe(root,'v10_extension/deployment_records')
    if records.exists() and not records.is_dir(): raise FileExistsError('Deployment history path is not a directory')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    temporary=safe(root,'v10_extension/docs/HANDOFF.fso-jscc-eval-v1-'+stamp+'.tmp')
    if not target.exists():
        target.mkdir(exist_ok=False)
        for name in list(manifest['files'])+['MANIFEST.json']:
            with (target/name).open('xb') as stream: stream.write((source/name).read_bytes())
    if entry not in old_text:
        records.mkdir(exist_ok=True); backup=records/('HANDOFF_before_fso_jscc_eval_v1_'+stamp+'.md')
        with backup.open('xb') as stream: stream.write(old)
        with temporary.open('xb') as stream:
            stream.write(old+b'\n'+entry.encode()); stream.flush(); os.fsync(stream.fileno())
        os.replace(str(temporary),str(handoff))
        with (records/('fso_jscc_eval_v1_'+stamp+'.json')).open('x') as stream:
            json.dump(dict(backup=str(backup.relative_to(root)),before_sha256=hashlib.sha256(old).hexdigest(),after_sha256=digest(handoff),release_manifest_sha256=digest(source/'MANIFEST.json')),stream,indent=2)
    for name,expected in manifest['files'].items():
        if digest(target/name)!=expected: raise ValueError('Post-install hash mismatch')
    print('Expanded evaluator installed/verified. NO training. Old source/weights/split/results preserved; handoff backed up before append.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False); p.add_argument('--project-root',required=True); p.add_argument('--verify-only',action='store_true'); a=p.parse_args()
    install(Path(__file__).resolve().parent,Path(a.project_root),a.verify_only)
