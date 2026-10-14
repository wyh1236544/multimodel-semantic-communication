"""New-directory-only release installation; preflight all conflicts before writes."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(root,name):
    p=Path(name)
    if p.is_absolute() or not p.parts or any(x in ('.','..') for x in p.parts): raise ValueError('Unsafe installation path')
    at=root
    for part in p.parts:
        at=at/part
        if at.is_symlink(): raise ValueError('Symlink installation destination rejected')
    return at


def install(source,root,verify_only=False):
    source=Path(source).resolve(strict=True); root=Path(root).resolve(strict=True)
    manifest=json.loads(safe(source,'MANIFEST.json').read_text())
    if manifest['schema']!='docci-reconstruction-v10-source' or manifest['source_only'] is not True: raise ValueError('Wrong source release')
    for name,sha in manifest['required_sources'].items():
        file=safe(root,name)
        if not file.is_file() or digest(file)!=sha: raise ValueError('Historical prerequisite differs; nothing installed: '+name)
    allowed=set(manifest['files'])|{'MANIFEST.json'}
    if {f.name for f in source.iterdir() if f.is_file()}!=allowed: raise ValueError('Unexpected/missing source payload')
    for name,sha in manifest['files'].items():
        if Path(name).name!=name or digest(safe(source,name))!=sha: raise ValueError('Payload name/hash differs')
    target=safe(root,'v10_extension/docci_reconstruction_v10'); handoff=safe(root,'v10_extension/docs/HANDOFF.md'); records=safe(root,'v10_extension/deployment_records')
    if not handoff.is_file(): raise FileNotFoundError('Existing shared HANDOFF required')
    if records.exists() and not records.is_dir(): raise FileExistsError('Deployment records conflict preserved')
    old=handoff.read_bytes(); entry=safe(source,'SHARED_HANDOFF_ENTRY.md').read_bytes()
    if not entry.endswith(b'\n'): raise ValueError('Handoff needs final newline')
    if entry.splitlines()[0] in old and entry not in old: raise FileExistsError('Divergent shared handoff preserved')
    if target.exists():
        if not target.is_dir(): raise FileExistsError('Existing user target preserved')
        for name,sha in manifest['files'].items():
            file=safe(target,name)
            if not file.is_file() or digest(file)!=sha: raise FileExistsError('Existing user source preserved: '+name)
        if not safe(target,'MANIFEST.json').is_file() or digest(target/'MANIFEST.json')!=digest(source/'MANIFEST.json'): raise FileExistsError('Existing manifest differs')
    elif verify_only: raise FileNotFoundError('New module not installed')
    if verify_only:
        if entry not in old: raise ValueError('Shared handoff entry absent')
        print('V10 installed source hashes verified. No training or old-file mutation.'); return
    if not target.exists():
        target.mkdir(exist_ok=False)
        for name in sorted(allowed):
            with (target/name).open('xb') as f: f.write((source/name).read_bytes())
    if entry not in old:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'); records.mkdir(exist_ok=True)
        backup=records/('HANDOFF_before_docci_reconstruction_v10_'+stamp+'.md')
        with backup.open('xb') as f: f.write(old)
        temporary=safe(root,'v10_extension/docs/HANDOFF.docci-v10-'+stamp+'.tmp')
        with temporary.open('xb') as f: f.write(old+b'\n'+entry); f.flush(); os.fsync(f.fileno())
        os.replace(str(temporary),str(handoff))
        with (records/('docci_reconstruction_v10_'+stamp+'.json')).open('x') as f: json.dump(dict(backup=str(backup.relative_to(root)),before_sha256=hashlib.sha256(old).hexdigest(),after_sha256=digest(handoff),release_manifest_sha256=digest(source/'MANIFEST.json')),f,indent=2)
    if any(digest(target/n)!=sha for n,sha in manifest['files'].items()): raise ValueError('Installed source differs')
    print('DOCCI independent text/image V10 installed. Historical source/data/weights/results preserved. No training or dependency changes.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--project-root',required=True); p.add_argument('--verify-only',action='store_true')
    a=p.parse_args(); install(Path(__file__).resolve().parent,a.project_root,a.verify_only)
