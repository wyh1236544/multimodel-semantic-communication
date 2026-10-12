"""Additive install with complete hash/conflict preflight and recoverable handoff."""
import argparse
import hashlib
import json
import os
from datetime import datetime,timezone
from pathlib import Path


def digest(file): return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def safe(root,name):
    p=Path(name)
    if p.is_absolute() or not p.parts or any(x in ('.','..') for x in p.parts): raise ValueError('Unsafe install path')
    current=root
    for part in p.parts:
        current=current/part
        if current.is_symlink(): raise ValueError('Symlink install path refused')
    return current


def install(source,root,verify_only=False):
    source=Path(source).resolve(strict=True); root=Path(root).resolve(strict=True)
    m=json.loads(safe(source,'MANIFEST.json').read_text())
    if m['schema']!='joint-jscc-architecture-v1-source' or not m['source_only']: raise ValueError('Wrong release')
    for name,sha in m['required_sources'].items():
        file=safe(root,name)
        if not file.is_file() or digest(file)!=sha: raise ValueError('Prerequisite missing/different; NOTHING installed: '+name)
    for name,sha in m['files'].items():
        if Path(name).name!=name or digest(safe(source,name))!=sha: raise ValueError('Payload path/hash mismatch')
    handoff=safe(root,'v10_extension/docs/HANDOFF.md'); old=handoff.read_bytes()
    entry=safe(source,'SHARED_HANDOFF_ENTRY.md').read_text().rstrip()+'\n'
    if entry.splitlines()[0] in old.decode() and entry not in old.decode(): raise FileExistsError('Divergent handoff preserved')
    records=safe(root,'v10_extension/deployment_records')
    target=safe(root,'v10_extension/joint_jscc_arch_v1')
    if target.exists():
        for name,sha in m['files'].items():
            if not safe(target,name).is_file() or digest(target/name)!=sha: raise FileExistsError('Existing source differs; preserved: '+str(target/name))
        if digest(safe(target,'MANIFEST.json'))!=digest(source/'MANIFEST.json'): raise FileExistsError('Existing manifest differs')
    elif verify_only: raise FileNotFoundError('New module not installed')
    else:
        target.mkdir(exist_ok=False)
        for name in list(m['files'])+['MANIFEST.json']:
            with (target/name).open('xb') as stream: stream.write((source/name).read_bytes())
    if verify_only and entry not in old.decode(): raise ValueError('Handoff entry missing')
    if entry not in old.decode():
        records.mkdir(exist_ok=True); stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup=records/('HANDOFF_before_joint_jscc_arch_v1_'+stamp+'.md')
        with backup.open('xb') as stream: stream.write(old)
        temporary=safe(root,'v10_extension/docs/HANDOFF.joint-jscc-arch-v1-'+stamp+'.tmp')
        with temporary.open('xb') as stream:
            stream.write(old+b'\n'+entry.encode()); stream.flush(); os.fsync(stream.fileno())
        os.replace(str(temporary),str(handoff))
        with (records/('joint_jscc_arch_v1_'+stamp+'.json')).open('x') as stream:
            json.dump(dict(backup=str(backup.relative_to(root)),before_sha256=hashlib.sha256(old).hexdigest(),
                           after_sha256=digest(handoff),release_manifest_sha256=digest(source/'MANIFEST.json')),stream,indent=2)
    for name,sha in m['files'].items():
        if digest(target/name)!=sha: raise ValueError('Post-install hash mismatch')
    print('Joint JSCC architecture V1 installed/verified. Historical source, weights, results and split preserved; HANDOFF backed up before append.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--project-root',required=True); p.add_argument('--verify-only',action='store_true'); a=p.parse_args()
    install(Path(__file__).resolve().parent,Path(a.project_root),a.verify_only)

