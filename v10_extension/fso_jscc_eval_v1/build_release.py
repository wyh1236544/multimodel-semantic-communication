"""Source-only reproducible release with dependency/conflict/backup restore tests."""
import ast
import gzip
import hashlib
import io
import json
import tarfile
import tempfile
from pathlib import Path
from install import install


def main():
    folder=Path(__file__).resolve().parent; project=folder.parents[1]; release=project/'releases'; release.mkdir(exist_ok=True)
    archive=release/'v10_fso_jscc_eval_v1.tar.gz'
    outputs=[archive,archive.with_name(archive.name+'.sha256'),folder/'MANIFEST.json',release/'v10_fso_jscc_eval_v1_build_verification.json',release/'v10_fso_jscc_eval_v1_publication_payload.json']
    if any(p.exists() for p in outputs): raise FileExistsError('Immutable release exists; preserve it and use a new version')
    files=sorted(f for f in folder.iterdir() if f.suffix in ('.py','.json','.md'))
    for f in files:
        if f.suffix=='.py': ast.parse(f.read_text(encoding='utf-8'),filename=str(f),feature_version=(3,8))
    required=[]
    for name in ('v10','models','utils','data','losses','configs','token_guidance','business_link','physical_link','optical_link_v1','final_compare_v1','optical_realism_v2','optical_realism_v3'):
        required.extend(str(f.relative_to(project)) for f in sorted((project/'v10_extension'/name).rglob('*')) if f.suffix in ('.py','.json','.yaml') and f.name!='MANIFEST.json' and '__pycache__' not in f.parts)
    required.extend(str(f.relative_to(project)) for f in sorted((project/'v10_extension').glob('*v10*.py')))
    # Original FSO identity hashes include its source docs and release manifest.
    required.extend(str(f.relative_to(project)) for f in sorted((project/'v10_extension/fso_jscc_v1').iterdir()) if f.suffix in ('.py','.json','.md'))
    required=sorted(set(required)); digest=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    manifest=dict(schema='fso-jscc-expanded-eval-v1-source',source_only=True,python_min='3.8',training_performed=False,
        files={f.name:digest(f) for f in files},required_sources={n:digest(project/n) for n in required})
    with (folder/'MANIFEST.json').open('x',encoding='utf-8') as f: json.dump(manifest,f,indent=2); f.write('\n')
    def payload_archive(destination):
        with destination.open('xb') as stream:
            with gzip.GzipFile(fileobj=stream,mode='wb',filename='',mtime=0) as gz:
                with tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
                    for f in files+[folder/'MANIFEST.json']:
                        data=f.read_bytes(); info=tarfile.TarInfo('_v10_fso_jscc_eval_v1/'+f.name); info.size,info.mode=len(data),0o644
                        tar.addfile(info,io.BytesIO(data))
    payload_archive(archive)
    restore=Path(tempfile.mkdtemp(prefix='fso_jscc_eval_v1_restore_',dir=str(release)))
    twin=restore/'deterministic_copy.tar.gz'; payload_archive(twin)
    if digest(twin)!=digest(archive): raise ValueError('Non-deterministic source archive')
    with tarfile.open(archive,'r:gz') as tar:
        expected={'_v10_fso_jscc_eval_v1/'+f.name for f in files+[folder/'MANIFEST.json']}
        if set(tar.getnames())!=expected or any(not m.isfile() for m in tar.getmembers()): raise ValueError('Unsafe archive member set')
        tar.extractall(str(restore))
    source=restore/'_v10_fso_jscc_eval_v1'
    def synthetic(name):
        root=restore/name; root.mkdir()
        for n in required:
            target=root/n; target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f: f.write((project/n).read_bytes())
        handoff=root/'v10_extension/docs/HANDOFF.md'; handoff.parent.mkdir(parents=True,exist_ok=True)
        with handoff.open('xb') as f: f.write(b'# Old synthetic history preserved\n')
        return root
    root=synthetic('multimodel'); install(source,root); install(source,root,verify_only=True); install(source,root)
    backups=list((root/'v10_extension/deployment_records').glob('HANDOFF_before_fso_jscc_eval_v1_*.md'))
    if len(backups)!=1 or backups[0].read_bytes()!=b'# Old synthetic history preserved\n': raise ValueError('Backup/reinstall invariant failed')
    if any(digest(root/n)!=digest(project/n) for n in required): raise ValueError('Old dependency modified')
    blocked=synthetic('source_conflict'); target=blocked/'v10_extension/fso_jscc_eval_v1'; target.mkdir()
    with (target/'run.py').open('xb') as f: f.write(b'user source preserved\n')
    try: install(source,blocked)
    except FileExistsError: pass
    else: raise ValueError('Source conflict not refused')
    if (target/'run.py').read_bytes()!=b'user source preserved\n' or (blocked/'v10_extension/docs/HANDOFF.md').read_bytes()!=b'# Old synthetic history preserved\n': raise ValueError('Conflict changed old files')
    wrong=synthetic('wrong_dependency')
    with (wrong/required[0]).open('ab') as f: f.write(b'changed')
    try: install(source,wrong)
    except ValueError: pass
    else: raise ValueError('Wrong dependency not refused')
    if (wrong/'v10_extension/fso_jscc_eval_v1').exists(): raise ValueError('Writes before dependency preflight')
    divergent=synthetic('divergent_handoff')
    with (divergent/'v10_extension/docs/HANDOFF.md').open('ab') as f: f.write((source/'SHARED_HANDOFF_ENTRY.md').read_text().splitlines()[0].encode()+b'\nuser-owned wording\n')
    try: install(source,divergent)
    except FileExistsError: pass
    else: raise ValueError('Divergent shared handoff not refused')
    if (divergent/'v10_extension/fso_jscc_eval_v1').exists(): raise ValueError('Writes before shared handoff preflight')
    sha=digest(archive)
    with archive.with_name(archive.name+'.sha256').open('x') as f: f.write(sha+'  '+archive.name+'\n')
    result=dict(archive=str(archive),sha256=sha,source_files=len(files)+1,prerequisite_files=len(required),python38_parse=True,
        deterministic_restore_install_reinstall_conflict_dependency_backup_checks=True,restore_directory=str(restore),Torch_real_checkpoints_COCO_GPU='NOT_EXECUTED_LOCALLY')
    with (release/'v10_fso_jscc_eval_v1_build_verification.json').open('x') as f: json.dump(result,f,indent=2)
    def blob(f):
        data=f.read_bytes(); return hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()
    publication=dict(files=[dict(path=str(f.relative_to(project)),content=f.read_text(encoding='utf-8'),git_blob_sha1=blob(f)) for f in files+[folder/'MANIFEST.json']],
        prerequisite_git_blob_sha1={n:blob(project/n) for n in required},shared_handoff_entry=(folder/'SHARED_HANDOFF_ENTRY.md').read_text(encoding='utf-8'),archive_sha256=sha)
    with (release/'v10_fso_jscc_eval_v1_publication_payload.json').open('x',encoding='utf-8') as f: json.dump(publication,f,ensure_ascii=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
