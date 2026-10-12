"""Deterministic source release + real restore/conflict/prerequisite checks."""
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
    archive=release/'v10_text_recovery_v3.tar.gz'
    if archive.exists() or (folder/'MANIFEST.json').exists(): raise FileExistsError('Release preserved; use new version')
    files=sorted(f for f in folder.iterdir() if f.suffix in ('.py','.json','.md'))
    for f in files:
        if f.suffix=='.py': ast.parse(f.read_text(),filename=str(f),feature_version=(3,8))
    digest=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    parent=project/'v10_extension/bart_coco_modal_v2'
    required=dict(json.loads((parent/'MANIFEST.json').read_text())['required_sources'])
    for f in parent.iterdir():
        if f.suffix in ('.py','.json','.md'): required[str(f.relative_to(project))]=digest(f)
    if any(digest(project/n)!=sha for n,sha in required.items()): raise ValueError('Historical prerequisite drift')
    manifest=dict(schema='text-recovery-v3-source',source_only=True,python_min='3.8',
        files={f.name:digest(f) for f in files},required_sources=required)
    with (folder/'MANIFEST.json').open('x') as f: json.dump(manifest,f,indent=2); f.write('\n')
    with archive.open('xb') as f,gzip.GzipFile(fileobj=f,mode='wb',filename='',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
        for file in files+[folder/'MANIFEST.json']:
            data=file.read_bytes(); member=tarfile.TarInfo('_v10_text_recovery_v3/'+file.name)
            member.size=len(data); member.mode=0o644; tar.addfile(member,io.BytesIO(data))
    restore=Path(tempfile.mkdtemp(prefix='text_recovery_v3_restore_',dir=str(release)))
    with tarfile.open(archive,'r:gz') as tar:
        if set(tar.getnames())!={'_v10_text_recovery_v3/'+f.name for f in files+[folder/'MANIFEST.json']} or any(not m.isfile() for m in tar.getmembers()): raise ValueError('Bad source tar members')
        tar.extractall(str(restore))
    source=restore/'_v10_text_recovery_v3'
    if any(digest(source/f.name)!=digest(f) for f in files+[folder/'MANIFEST.json']): raise ValueError('Archive roundtrip hash mismatch')
    def fixture(name):
        root=restore/name; root.mkdir()
        for n in required:
            dest=root/n; dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f: f.write((project/n).read_bytes())
        handoff=root/'v10_extension/docs/HANDOFF.md'; handoff.parent.mkdir(parents=True,exist_ok=True)
        with handoff.open('xb') as f: f.write(b'# Exact original handoff\n')
        return root
    root=fixture('multimodel'); install(source,root); install(source,root,True); install(source,root)
    backups=list((root/'v10_extension/deployment_records').glob('HANDOFF_before_text_recovery_v3_*.md'))
    if len(backups)!=1 or backups[0].read_bytes()!=b'# Exact original handoff\n': raise ValueError('Handoff backup/reinstall mismatch')
    if any(digest(root/n)!=sha for n,sha in required.items()): raise ValueError('Old dependency modified')
    conflict=fixture('conflict'); dest=conflict/'v10_extension/text_recovery_v3'; dest.mkdir()
    with (dest/'run.py').open('xb') as f: f.write(b'user owned\n')
    try: install(source,conflict)
    except FileExistsError: pass
    else: raise ValueError('Conflict not rejected')
    if (dest/'run.py').read_bytes()!=b'user owned\n': raise ValueError('Conflict file modified')
    wrong=fixture('wrong_dependency'); key=next(iter(required))
    with (wrong/key).open('ab') as f: f.write(b'drift')
    try: install(source,wrong)
    except ValueError: pass
    else: raise ValueError('Dependency drift not rejected')
    if (wrong/'v10_extension/text_recovery_v3').exists(): raise ValueError('Drift wrote payload')
    symlink=fixture('symlink'); empty=restore/'empty'; empty.mkdir()
    (symlink/'v10_extension/text_recovery_v3').symlink_to(empty,target_is_directory=True)
    try: install(source,symlink)
    except ValueError: pass
    else: raise ValueError('Symlink accepted')
    if list(empty.iterdir()): raise ValueError('Symlink destination changed')
    sha=digest(archive)
    with archive.with_name(archive.name+'.sha256').open('x') as f: f.write(sha+'  '+archive.name+'\n')
    result=dict(archive=str(archive),archive_sha256=sha,bytes=archive.stat().st_size,source_files=len(files)+1,
        prerequisite_files=len(required),python38_parse=True,restore_install_reinstall_conflict_dependency_symlink_backup=True,
        new_non_torch_tests=23,inherited_non_torch_regressions=53,
        local_Torch_pretrained_GPU='NOT_EXECUTED; unavailable',restore_directory=str(restore))
    with (release/'v10_text_recovery_v3_build_verification.json').open('x') as f: json.dump(result,f,indent=2)
    def blob(file):
        data=file.read_bytes(); return hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()
    published=files+[folder/'MANIFEST.json']
    backup=project/'v10_extension/deployment_records/HANDOFF_before_text_recovery_v3_20261012_local.md'
    handoff=project/'v10_extension/docs/HANDOFF.md'
    if not backup.is_file() or (folder/'SHARED_HANDOFF_ENTRY.md').read_text().rstrip() not in handoff.read_text(): raise ValueError('Local handoff backup/append missing')
    published += [backup,handoff]
    payload=dict(files=[dict(path=str(f.relative_to(project)),content=f.read_text(),git_blob_sha1=blob(f)) for f in published],
        prerequisite_git_blob_sha1={n:blob(project/n) for n in required},archive_sha256=sha)
    with (release/'v10_text_recovery_v3_publication_payload.json').open('x',encoding='utf-8') as f: json.dump(payload,f,ensure_ascii=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
