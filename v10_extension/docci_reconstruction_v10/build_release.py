"""Deterministic source package, real install/restore and atomic-publication data."""
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
    archive=release/'v10_docci_reconstruction_v10.tar.gz'
    if archive.exists() or (folder/'MANIFEST.json').exists(): raise FileExistsError('Published source preserved; use a new revision')
    files=sorted(f for f in folder.iterdir() if f.is_file() and f.suffix in ('.py','.json','.md','.txt','.svg'))
    for file in files:
        if file.suffix=='.py': ast.parse(file.read_text(),filename=str(file),feature_version=(3,8))
    parent=project/'v10_extension/docci_modal_v9'; required=dict(json.loads((parent/'MANIFEST.json').read_text())['required_sources'])
    for file in parent.iterdir():
        if file.is_file() and file.suffix in ('.py','.json','.md'): required[str(file.relative_to(project))]=digest(file)
    for name in ('utils/__init__.py','utils/metrics.py','utils/checkpoint.py','utils/config.py'): required[name]=digest(project/name)
    backup=project/'v10_extension/deployment_records/HANDOFF_before_docci_reconstruction_v10_20261014_local.md'; handoff=project/'v10_extension/docs/HANDOFF.md'; entry=(folder/'SHARED_HANDOFF_ENTRY.md').read_bytes()
    if not backup.is_file() or handoff.read_bytes()!=backup.read_bytes()+b'\n'+entry: raise ValueError('Exact local old HANDOFF backup/append differs')
    if any(digest(project/name)!=sha for name,sha in required.items()): raise ValueError('Historical prerequisite changed')
    # Test BEFORE creating any final source release artifact.
    for name in ('test_core.py','test_report.py','test_model.py'): subprocess.run([sys.executable,'-B',str(folder/name)],cwd=str(project),check=True)
    manifest=dict(schema='docci-reconstruction-v10-source',source_only=True,python_min='3.8',files={f.name:digest(f) for f in files},required_sources=required,
        data_and_pretrained_weights_included=False,local_torch_pretrained_gpu_tests='NOT_EXECUTED',server_cpu_and_authentic_256_512_smoke_required=True)
    with (folder/'MANIFEST.json').open('x') as f: json.dump(manifest,f,indent=2); f.write('\n')
    published=files+[folder/'MANIFEST.json']
    restore=Path(tempfile.mkdtemp(prefix='docci_reconstruction_v10_restore_',dir=str(release))); source=restore/'_v10_docci_reconstruction_v10'; source.mkdir()
    for file in published:
        with (source/file.name).open('xb') as f: f.write(file.read_bytes())
    def fixture(label):
        root=restore/label; root.mkdir()
        for name in required:
            dest=root/name; dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f: f.write((project/name).read_bytes())
        file=root/'v10_extension/docs/HANDOFF.md'; file.parent.mkdir(parents=True,exist_ok=True)
        with file.open('xb') as f: f.write(b'# Exact previous user handoff\n')
        return root
    target=fixture('multimodel'); install(source,target); install(source,target,True); install(source,target)
    backups=list((target/'v10_extension/deployment_records').glob('HANDOFF_before_docci_reconstruction_v10_*.md'))
    if len(backups)!=1 or backups[0].read_bytes()!=b'# Exact previous user handoff\n': raise ValueError('Exact backup/reinstall failed')
    for name in ('test_core.py','test_report.py'): subprocess.run([sys.executable,'-B',str(target/'v10_extension/docci_reconstruction_v10'/name)],cwd=str(target),check=True)
    if any(digest(target/name)!=sha for name,sha in required.items()): raise ValueError('Historical source mutated by install')
    conflict=fixture('conflict'); dest=conflict/'v10_extension/docci_reconstruction_v10'; dest.mkdir()
    with (dest/'run.py').open('xb') as f: f.write(b'user-owned source\n')
    try: install(source,conflict)
    except FileExistsError: pass
    else: raise ValueError('User source conflict accepted')
    if (dest/'run.py').read_bytes()!=b'user-owned source\n' or (conflict/'v10_extension/docs/HANDOFF.md').read_bytes()!=b'# Exact previous user handoff\n': raise ValueError('Conflict mutated old data')
    drift=fixture('drift'); key=next(iter(required))
    with (drift/key).open('ab') as f: f.write(b'user drift')
    try: install(source,drift)
    except ValueError: pass
    else: raise ValueError('Prerequisite drift accepted')
    if (drift/'v10_extension/docci_reconstruction_v10').exists(): raise ValueError('Drift failure installed anything')
    link=fixture('symlink'); outside=restore/'outside'; outside.mkdir(); (link/'v10_extension/docci_reconstruction_v10').symlink_to(outside,target_is_directory=True)
    try: install(source,link)
    except ValueError: pass
    else: raise ValueError('Symlink destination accepted')
    if list(outside.iterdir()): raise ValueError('Symlink target mutated')
    divergent=fixture('divergent-handoff'); file=divergent/'v10_extension/docs/HANDOFF.md'
    with file.open('ab') as f: f.write(b'\n'+entry.splitlines()[0]+b'\nuser alternate entry\n')
    before=file.read_bytes()
    try: install(source,divergent)
    except FileExistsError: pass
    else: raise ValueError('Divergent handoff accepted')
    if file.read_bytes()!=before or (divergent/'v10_extension/docci_reconstruction_v10').exists(): raise ValueError('Divergent handoff mutated target')
    with archive.open('xb') as out,gzip.GzipFile(fileobj=out,mode='wb',filename='',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
        for file in published:
            data=file.read_bytes(); member=tarfile.TarInfo('_v10_docci_reconstruction_v10/'+file.name); member.size=len(data); member.mode=0o644
            tar.addfile(member,io.BytesIO(data))
    checked=restore/'archive_roundtrip'; checked.mkdir()
    with tarfile.open(archive) as tar:
        expected={'_v10_docci_reconstruction_v10/'+f.name for f in published}
        if set(tar.getnames())!=expected or any(not m.isfile() for m in tar.getmembers()): raise ValueError('Unsafe source archive')
        for member in tar.getmembers():
            with (checked/Path(member.name).name).open('xb') as f: f.write(tar.extractfile(member).read())
    if any(digest(checked/f.name)!=digest(f) for f in published): raise ValueError('Archive restored bytes differ')
    sha=digest(archive)
    with archive.with_name(archive.name+'.sha256').open('x') as f: f.write(sha+'  '+archive.name+'\n')
    result=dict(archive=str(archive),archive_sha256=sha,bytes=archive.stat().st_size,source_files=len(published),prerequisite_files=len(required),python38_parse=True,
        local_core_tests=8,local_report_tests=5,restore_install_reinstall_conflict_drift_symlink_handoff_backup=True,
        local_torch_transformers_pretrained_GPU_DOCCI='NOT_EXECUTED; unavailable; actual server gates mandatory',restore_directory=str(restore))
    with (release/'v10_docci_reconstruction_v10_build_verification.json').open('x') as f: json.dump(result,f,indent=2)
    publication=dict(files=[dict(path=str(f.relative_to(project)),content=f.read_text(),git_blob_sha1=blob(f)) for f in published+[backup,handoff]],
        prerequisite_git_blob_sha1={n:blob(project/n) for n in required},previous_handoff_git_blob_sha1=blob(backup),archive_sha256=sha)
    with (release/'v10_docci_reconstruction_v10_publication_payload.json').open('x',encoding='utf-8') as f: json.dump(publication,f,ensure_ascii=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
