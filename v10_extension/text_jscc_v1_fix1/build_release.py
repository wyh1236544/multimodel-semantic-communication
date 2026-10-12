"""Deterministic source-only release, isolated restore/install and Git blob manifest."""
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
from install import install


def main():
    folder=Path(__file__).resolve().parent; project=folder.parents[1]; release=project/'releases'; release.mkdir(exist_ok=True)
    archive=release/'v10_text_jscc_v1_fix1.tar.gz'
    if archive.exists() or (folder/'MANIFEST.json').exists(): raise FileExistsError('Release exists; preserve and use a new version')
    files=sorted(f for f in folder.iterdir() if f.suffix in ('.py','.json','.md'))
    for f in files:
        if f.suffix=='.py': ast.parse(f.read_text(),filename=str(f),feature_version=(3,8))
    digest=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    required=dict(json.loads((project/'v10_extension/joint_jscc_arch_v1/MANIFEST.json').read_text())['required_sources'])
    for f in sorted((project/'v10_extension/joint_jscc_arch_v1').iterdir()):
        if f.suffix in ('.py','.json','.md'): required[str(f.relative_to(project))]=digest(f)
    if any(digest(project/n)!=sha for n,sha in required.items()): raise ValueError('Historical dependency drift')
    manifest=dict(schema='text-only-jscc-v1-fix1-source',source_only=True,python_min='3.8',files={f.name:digest(f) for f in files},required_sources=required)
    with (folder/'MANIFEST.json').open('x') as f: json.dump(manifest,f,indent=2); f.write('\n')
    with archive.open('xb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',filename='',mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
                for file in files+[folder/'MANIFEST.json']:
                    data=file.read_bytes(); entry=tarfile.TarInfo('_v10_text_jscc_v1_fix1/'+file.name)
                    entry.size,entry.mode=len(data),0o644; tar.addfile(entry,io.BytesIO(data))
    restore=Path(tempfile.mkdtemp(prefix='text_jscc_v1_fix1_restore_',dir=str(release)))
    with tarfile.open(archive,'r:gz') as tar:
        if set(tar.getnames())!={'_v10_text_jscc_v1_fix1/'+f.name for f in files+[folder/'MANIFEST.json']} or any(not m.isfile() for m in tar.getmembers()): raise ValueError('Wrong archive members')
        tar.extractall(str(restore))
    source=restore/'_v10_text_jscc_v1_fix1'
    def synthetic(name):
        root=restore/name; root.mkdir()
        for n in required:
            file=root/n; file.parent.mkdir(parents=True,exist_ok=True)
            with file.open('xb') as f: f.write((project/n).read_bytes())
        handoff=root/'v10_extension/docs/HANDOFF.md'; handoff.parent.mkdir(parents=True,exist_ok=True)
        with handoff.open('xb') as f: f.write(b'# Exact original handoff\n')
        return root
    root=synthetic('multimodel'); install(source,root); install(source,root,True); install(source,root)
    backups=list((root/'v10_extension/deployment_records').glob('HANDOFF_before_text_jscc_v1_fix1_*.md'))
    if len(backups)!=1 or backups[0].read_bytes()!=b'# Exact original handoff\n': raise ValueError('Backup/reinstall mismatch')
    if any(digest(root/n)!=sha for n,sha in required.items()): raise ValueError('Historical source mutated')
    conflict=synthetic('conflict'); target=conflict/'v10_extension/text_jscc_v1_fix1'; target.mkdir()
    with (target/'run.py').open('xb') as f: f.write(b'user-owned source\n')
    try: install(source,conflict)
    except FileExistsError: pass
    else: raise ValueError('Existing user source conflict accepted')
    if (target/'run.py').read_bytes()!=b'user-owned source\n': raise ValueError('User source mutated')
    wrong=synthetic('wrong_dependency'); key=next(iter(required))
    with (wrong/key).open('ab') as f: f.write(b'drift')
    try: install(source,wrong)
    except ValueError: pass
    else: raise ValueError('Prerequisite drift accepted')
    if (wrong/'v10_extension/text_jscc_v1_fix1').exists(): raise ValueError('Dependency preflight wrote payload')
    sha=digest(archive)
    with archive.with_name(archive.name+'.sha256').open('x') as f: f.write(sha+'  '+archive.name+'\n')
    result=dict(archive=str(archive),sha256=sha,bytes=archive.stat().st_size,source_files=len(files)+1,
        required_sources=len(required),python38_parse=True,restore_install_reinstall_conflict_dependency_backup_checks=True,
        local_non_torch_unique_tests=31,Torch_pretrained_GPU='NOT_EXECUTED_LOCALLY',restore_directory=str(restore))
    with (release/'v10_text_jscc_v1_fix1_build_verification.json').open('x') as f: json.dump(result,f,indent=2)
    def blob(f):
        data=f.read_bytes(); return hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()
    payload=dict(files=[dict(path=str(f.relative_to(project)),content=f.read_text(),git_blob_sha1=blob(f)) for f in files+[folder/'MANIFEST.json']],
        prerequisite_git_blob_sha1={n:blob(project/n) for n in required},shared_handoff_entry=(folder/'SHARED_HANDOFF_ENTRY.md').read_text(),archive_sha256=sha)
    with (release/'v10_text_jscc_v1_fix1_publication_payload.json').open('x',encoding='utf-8') as f: json.dump(payload,f,ensure_ascii=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
