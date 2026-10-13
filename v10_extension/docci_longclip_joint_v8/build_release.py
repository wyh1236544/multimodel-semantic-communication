"""Deterministic source-only release, real restore and mutation-negative checks."""
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from install import install


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blob(path):
    data=Path(path).read_bytes(); return hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()


def main():
    folder=Path(__file__).resolve().parent; project=folder.parents[1]; release=project/'releases'; release.mkdir(exist_ok=True)
    archive=release/'v10_docci_longclip_joint_v8.tar.gz'
    if archive.exists() or (folder/'MANIFEST.json').exists(): raise FileExistsError('Published sources preserved; use a new revision')
    files=sorted(f for f in folder.iterdir() if f.is_file() and f.suffix in ('.py','.json','.md','.txt','.svg'))
    for f in files:
        if f.suffix=='.py': ast.parse(f.read_text(),filename=str(f),feature_version=(3,8))
    parent=project/'v10_extension/bidirectional_rx_v7'; required=dict(json.loads((parent/'MANIFEST.json').read_text())['required_sources'])
    for file in parent.iterdir():
        if file.is_file() and file.suffix in ('.py','.json','.md'): required[str(file.relative_to(project))]=digest(file)
    for name in ('utils/__init__.py','utils/metrics.py','utils/checkpoint.py','utils/config.py'): required[name]=digest(project/name)
    if any(digest(project/name)!=sha for name,sha in required.items()): raise ValueError('Historical prerequisite changed')
    manifest=dict(schema='docci-longclip-joint-v8-source',source_only=True,python_min='3.8',files={f.name:digest(f) for f in files},required_sources=required,
        data_and_pretrained_weights_included=False,local_torch_pretrained_gpu_tests='NOT_EXECUTED',server_cpu_and_authentic_256_512_smoke_required=True)
    with (folder/'MANIFEST.json').open('x') as f: json.dump(manifest,f,indent=2); f.write('\n')
    published=files+[folder/'MANIFEST.json']
    with archive.open('xb') as f,gzip.GzipFile(fileobj=f,mode='wb',filename='',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
        for file in published:
            data=file.read_bytes(); member=tarfile.TarInfo('_v10_docci_longclip_joint_v8/'+file.name); member.size=len(data); member.mode=0o644
            tar.addfile(member,io.BytesIO(data))
    restore=Path(tempfile.mkdtemp(prefix='docci_longclip_joint_v8_restore_',dir=str(release)))
    with tarfile.open(archive) as tar:
        expected={'_v10_docci_longclip_joint_v8/'+f.name for f in published}
        if set(tar.getnames())!=expected or any(not m.isfile() for m in tar.getmembers()): raise ValueError('Unsafe source archive entries')
        # Explicit verified single-component files; no extractall links/traversal.
        source=restore/'_v10_docci_longclip_joint_v8'; source.mkdir()
        for member in tar.getmembers():
            with (source/Path(member.name).name).open('xb') as f: f.write(tar.extractfile(member).read())
    if any(digest(source/f.name)!=digest(f) for f in published): raise ValueError('Archive restored bytes differ')
    def fixture(label):
        root=restore/label; root.mkdir()
        for name in required:
            dest=root/name; dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f: f.write((project/name).read_bytes())
        handoff=root/'v10_extension/docs/HANDOFF.md'; handoff.parent.mkdir(parents=True,exist_ok=True)
        with handoff.open('xb') as f: f.write(b'# Exact previous user handoff\n')
        return root
    target=fixture('multimodel'); install(source,target); install(source,target,True); install(source,target)
    backups=list((target/'v10_extension/deployment_records').glob('HANDOFF_before_docci_longclip_joint_v8_*.md'))
    if len(backups)!=1 or backups[0].read_bytes()!=b'# Exact previous user handoff\n': raise ValueError('Exact backup/reinstall check failed')
    if any(digest(target/name)!=sha for name,sha in required.items()): raise ValueError('Historical source mutated by install')
    for name in ('test_core.py','test_report.py'):
        subprocess.run([sys.executable,'-B',str(target/'v10_extension/docci_longclip_joint_v8'/name)],cwd=str(target),check=True)
    conflict=fixture('conflict'); dest=conflict/'v10_extension/docci_longclip_joint_v8'; dest.mkdir()
    with (dest/'run.py').open('xb') as f: f.write(b'user-owned source\n')
    try: install(source,conflict)
    except FileExistsError: pass
    else: raise ValueError('User conflict not rejected')
    if (dest/'run.py').read_bytes()!=b'user-owned source\n' or (conflict/'v10_extension/docs/HANDOFF.md').read_bytes()!=b'# Exact previous user handoff\n': raise ValueError('Conflict changed user files')
    drift=fixture('drift'); key=next(iter(required))
    with (drift/key).open('ab') as f: f.write(b'user drift')
    try: install(source,drift)
    except ValueError: pass
    else: raise ValueError('Prerequisite drift accepted')
    if (drift/'v10_extension/docci_longclip_joint_v8').exists(): raise ValueError('Drift failure installed payload')
    link=fixture('symlink'); empty=restore/'outside'; empty.mkdir()
    (link/'v10_extension/docci_longclip_joint_v8').symlink_to(empty,target_is_directory=True)
    try: install(source,link)
    except ValueError: pass
    else: raise ValueError('Symlink target accepted')
    if list(empty.iterdir()): raise ValueError('Symlink destination mutated')
    conflict_handoff=fixture('handoff-conflict'); handoff=conflict_handoff/'v10_extension/docs/HANDOFF.md'
    entry=(source/'SHARED_HANDOFF_ENTRY.md').read_bytes()
    with handoff.open('ab') as f: f.write(b'\n'+entry.splitlines()[0]+b'\nuser alternate entry\n')
    before=handoff.read_bytes()
    try: install(source,conflict_handoff)
    except FileExistsError: pass
    else: raise ValueError('Divergent handoff accepted')
    if handoff.read_bytes()!=before or (conflict_handoff/'v10_extension/docci_longclip_joint_v8').exists(): raise ValueError('Divergent handoff mutated target')
    local_backup=project/'v10_extension/deployment_records/HANDOFF_before_docci_longclip_joint_v8_20261013_local.md'
    local_handoff=project/'v10_extension/docs/HANDOFF.md'
    if not local_backup.is_file() or local_handoff.read_bytes()!=local_backup.read_bytes()+b'\n'+entry: raise ValueError('Exact local backup/append not verified')
    sha=digest(archive)
    with archive.with_name(archive.name+'.sha256').open('x') as f: f.write(sha+'  '+archive.name+'\n')
    result=dict(archive=str(archive),archive_sha256=sha,bytes=archive.stat().st_size,source_files=len(published),prerequisite_files=len(required),
        python38_parse=True,local_core_tests=8,local_report_tests=5,restore_install_reinstall_conflict_drift_symlink_handoff_backup=True,
        local_torch_transformers_pretrained_GPU_DOCCI='NOT_EXECUTED; unavailable; server actual gates required',restore_directory=str(restore))
    with (release/'v10_docci_longclip_joint_v8_build_verification.json').open('x') as f: json.dump(result,f,indent=2)
    publication=dict(files=[dict(path=str(f.relative_to(project)),content=f.read_text(),git_blob_sha1=blob(f)) for f in published+[local_backup,local_handoff]],
        prerequisite_git_blob_sha1={n:blob(project/n) for n in required},previous_handoff_git_blob_sha1=blob(local_backup),archive_sha256=sha)
    with (release/'v10_docci_longclip_joint_v8_publication_payload.json').open('x',encoding='utf-8') as f: json.dump(publication,f,ensure_ascii=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
