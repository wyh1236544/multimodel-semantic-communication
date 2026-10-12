"""Additive preflighted install; historical files preserved, handoff backup first."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(root,name):
    p=Path(name)
    if p.is_absolute() or not p.parts or any(x in ('.','..') for x in p.parts): raise ValueError('Unsafe install path')
    cursor=root
    for part in p.parts:
        cursor=cursor/part
        if cursor.is_symlink(): raise ValueError('Symlink install target refused')
    return cursor


def install(source,root,verify_only=False):
    source=Path(source).resolve(strict=True); root=Path(root).resolve(strict=True)
    m=json.loads(safe(source,'MANIFEST.json').read_text())
    if m['schema']!='bart-clip-joint-v1-source' or m['source_only'] is not True: raise ValueError('Wrong release')
    for n,sha in m['required_sources'].items():
        f=safe(root,n)
        if not f.is_file() or digest(f)!=sha: raise ValueError('Prerequisite differs; nothing installed: '+n)
    for n,sha in m['files'].items():
        if Path(n).name!=n or digest(safe(source,n))!=sha: raise ValueError('Payload path/hash differs')
    target=safe(root,'v10_extension/bart_clip_joint_v1'); records=safe(root,'v10_extension/deployment_records')
    handoff=safe(root,'v10_extension/docs/HANDOFF.md'); old=handoff.read_bytes()
    entry=(source/'SHARED_HANDOFF_ENTRY.md').read_text().rstrip()+'\n'
    if entry.splitlines()[0] in old.decode() and entry not in old.decode(): raise FileExistsError('Divergent handoff preserved')
    if target.exists():
        for n,sha in m['files'].items():
            if not safe(target,n).is_file() or digest(target/n)!=sha: raise FileExistsError('Existing source differs; preserved: '+str(target/n))
        if digest(safe(target,'MANIFEST.json'))!=digest(source/'MANIFEST.json'): raise FileExistsError('Existing manifest differs')
    elif verify_only: raise FileNotFoundError('Module not installed')
    else:
        target.mkdir(exist_ok=False)
        for n in list(m['files'])+['MANIFEST.json']:
            with (target/n).open('xb') as f: f.write((source/n).read_bytes())
    if verify_only and entry not in old.decode(): raise ValueError('Shared handoff entry missing')
    if entry not in old.decode():
        records.mkdir(exist_ok=True); stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup=records/('HANDOFF_before_bart_clip_joint_v1_'+stamp+'.md')
        with backup.open('xb') as f: f.write(old)
        temp=safe(root,'v10_extension/docs/HANDOFF.bart-clip-joint-v1-'+stamp+'.tmp')
        with temp.open('xb') as f: f.write(old+b'\n'+entry.encode()); f.flush(); os.fsync(f.fileno())
        os.replace(str(temp),str(handoff))
        with (records/('bart_clip_joint_v1_'+stamp+'.json')).open('x') as f:
            json.dump(dict(backup=str(backup.relative_to(root)),before_sha256=hashlib.sha256(old).hexdigest(),
                after_sha256=digest(handoff),release_manifest_sha256=digest(source/'MANIFEST.json')),f,indent=2)
    for n,sha in m['files'].items():
        if digest(target/n)!=sha: raise ValueError('Post-install mismatch')
    print('CLIP+BART C/I/T source installed/verified; old source/weights/results preserved; handoff backed up.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--project-root',required=True); p.add_argument('--verify-only',action='store_true'); a=p.parse_args()
    install(Path(__file__).resolve().parent,a.project_root,a.verify_only)
