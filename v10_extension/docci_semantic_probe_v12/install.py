"""Additive source-only installation. Preflight every conflict before first write."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def at(root,name):
    p=Path(name)
    if p.is_absolute() or not p.parts or any(x in ('.','..') for x in p.parts): raise ValueError('Unsafe installation path')
    target=root
    for part in p.parts:
        target=target/part
        if target.is_symlink(): raise ValueError('Symlink installation rejected')
    return target


def install(source,root,verify_only=False):
    source=Path(source).resolve(strict=True); root=Path(root).resolve(strict=True)
    m=json.loads(at(source,'MANIFEST.json').read_text())
    if m['schema']!='docci-semantic-probe-v12-source' or m['source_only'] is not True: raise ValueError('Wrong release')
    allowed=set(m['files'])|{'MANIFEST.json'}
    if {f.name for f in source.iterdir() if f.is_file()}!=allowed: raise ValueError('Unexpected/missing source payload')
    for name,sha in m['files'].items():
        if Path(name).name!=name or digest(at(source,name))!=sha: raise ValueError('Payload hash differs')
    for name,sha in m['required_sources'].items():
        f=at(root,name)
        if not f.is_file() or digest(f)!=sha: raise ValueError('Historical source differs; nothing installed: '+name)
    target=at(root,'v10_extension/docci_semantic_probe_v12'); handoff=at(root,'v10_extension/docs/HANDOFF.md')
    records=at(root,'v10_extension/deployment_records')
    if not handoff.is_file(): raise FileNotFoundError('Existing shared HANDOFF required')
    if records.exists() and not records.is_dir(): raise FileExistsError('Deployment record conflict')
    old=handoff.read_bytes(); entry=at(source,'SHARED_HANDOFF_ENTRY.md').read_bytes()
    if not entry.endswith(b'\n'): raise ValueError('Incomplete handoff entry')
    if entry.splitlines()[0] in old and entry not in old: raise FileExistsError('Divergent handoff entry preserved')
    if target.exists():
        if not target.is_dir(): raise FileExistsError('User target preserved')
        for name in allowed:
            f=at(target,name)
            if not f.is_file() or digest(f)!=digest(source/name): raise FileExistsError('User source preserved: '+name)
    elif verify_only: raise FileNotFoundError('V12 not installed')
    if verify_only:
        if entry not in old: raise ValueError('Shared entry absent')
        print('Installed V12 source verified. No training or dependency changes.'); return
    if not target.exists():
        target.mkdir(exist_ok=False)
        for name in sorted(allowed):
            with (target/name).open('xb') as f: f.write((source/name).read_bytes())
    if entry not in old:
        records.mkdir(exist_ok=True); stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup=records/('HANDOFF_before_docci_semantic_probe_v12_'+stamp+'.md')
        with backup.open('xb') as f: f.write(old)
        temporary=at(root,'v10_extension/docs/HANDOFF.v12-'+stamp+'.part')
        with temporary.open('xb') as f: f.write(old+b'\n'+entry); f.flush(); os.fsync(f.fileno())
        os.replace(str(temporary),str(handoff))
        with (records/('docci_semantic_probe_v12_'+stamp+'.json')).open('x') as f:
            json.dump(dict(before_sha256=hashlib.sha256(old).hexdigest(),after_sha256=digest(handoff),
                backup=str(backup.relative_to(root)),source_manifest_sha256=digest(source/'MANIFEST.json')),f,indent=2)
    print('V12 installed additively. Old source/data/weights/results preserved. No training or environment changes.')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--project-root',required=True); p.add_argument('--verify-only',action='store_true')
    a=p.parse_args(); install(Path(__file__).resolve().parent,a.project_root,a.verify_only)
