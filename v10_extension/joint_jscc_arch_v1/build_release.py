"""Source-only deterministic archive, restore/install conflicts and Git publication data."""
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
    folder=Path(__file__).resolve().parent; project=folder.parents[1]
    release=project/'releases'; release.mkdir(exist_ok=True)
    archive=release/'v10_joint_jscc_arch_v1.tar.gz'
    if archive.exists() or (folder/'MANIFEST.json').exists(): raise FileExistsError('Immutable release exists; use a new version')
    files=sorted(f for f in folder.iterdir() if f.suffix in ('.py','.json','.md'))
    for f in files:
        if f.suffix=='.py': ast.parse(f.read_text(),filename=str(f),feature_version=(3,8))
    required=[]
    for name in ('v10','models','utils','data','losses','configs','token_guidance','business_link','physical_link',
                 'optical_link_v1','final_compare_v1','optical_realism_v2','optical_realism_v3','fso_jscc_v1',
                 'fso_jscc_eval_v1','fso_fluctuating_v2','vae_tail_ablation_v1','full_unfreeze_v1'):
        required += [str(f.relative_to(project)) for f in sorted((project/'v10_extension'/name).rglob('*'))
                     if f.suffix in ('.py','.json','.yaml') and f.name!='MANIFEST.json' and '__pycache__' not in f.parts]
    required+=['v10_extension/eval_v10.py']
    digest=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    manifest=dict(schema='joint-jscc-architecture-v1-source',source_only=True,python_min='3.8',
        files={f.name:digest(f) for f in files},required_sources={n:digest(project/n) for n in required})
    with (folder/'MANIFEST.json').open('x') as stream: json.dump(manifest,stream,indent=2); stream.write('\n')
    with archive.open('xb') as stream:
        with gzip.GzipFile(fileobj=stream,mode='wb',filename='',mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
                for f in files+[folder/'MANIFEST.json']:
                    data=f.read_bytes(); entry=tarfile.TarInfo('_v10_joint_jscc_arch_v1/'+f.name)
                    entry.size,entry.mode=len(data),0o644; tar.addfile(entry,io.BytesIO(data))
    restore=Path(tempfile.mkdtemp(prefix='joint_jscc_arch_v1_restore_',dir=str(release)))
    with tarfile.open(archive,'r:gz') as tar:
        if set(tar.getnames())!={'_v10_joint_jscc_arch_v1/'+f.name for f in files+[folder/'MANIFEST.json']} or any(not m.isfile() for m in tar.getmembers()): raise ValueError('Wrong archive members')
        tar.extractall(str(restore))
    source=restore/'_v10_joint_jscc_arch_v1'
    def synthetic(name):
        root=restore/name; root.mkdir()
        for n in required:
            file=root/n; file.parent.mkdir(parents=True,exist_ok=True)
            with file.open('xb') as stream: stream.write((project/n).read_bytes())
        handoff=root/'v10_extension/docs/HANDOFF.md'; handoff.parent.mkdir(parents=True,exist_ok=True)
        with handoff.open('xb') as stream: stream.write(b'# Preserved synthetic history\n')
        return root
    root=synthetic('multimodel'); install(source,root); install(source,root,verify_only=True); install(source,root)
    backups=list((root/'v10_extension/deployment_records').glob('HANDOFF_before_joint_jscc_arch_v1_*.md'))
    if len(backups)!=1 or backups[0].read_bytes()!=b'# Preserved synthetic history\n': raise ValueError('Handoff backup/reinstall failed')
    for name,sha in manifest['required_sources'].items():
        if digest(root/name)!=sha: raise ValueError('Historical dependency modified by install')
    conflict=synthetic('conflict'); target=conflict/'v10_extension/joint_jscc_arch_v1'; target.mkdir()
    with (target/'run.py').open('xb') as stream: stream.write(b'user-source\n')
    try: install(source,conflict)
    except FileExistsError: pass
    else: raise ValueError('Existing user source conflict not refused')
    if (target/'run.py').read_bytes()!=b'user-source\n': raise ValueError('User source overwritten')
    wrong=synthetic('wrong_dependency')
    with (wrong/required[0]).open('ab') as stream: stream.write(b'changed')
    try: install(source,wrong)
    except ValueError: pass
    else: raise ValueError('Dependency mismatch not refused')
    if (wrong/'v10_extension/joint_jscc_arch_v1').exists(): raise ValueError('Source installed before prerequisite preflight')
    sha=digest(archive)
    with archive.with_name(archive.name+'.sha256').open('x') as stream: stream.write(sha+'  '+archive.name+'\n')
    result=dict(archive=str(archive),sha256=sha,source_files=len(files)+1,required_sources=len(required),
        python38_parse=True,restore_install_reinstall_conflict_dependency_backup_checks=True,restore_directory=str(restore),
        Torch_pretrained_GPU='NOT_EXECUTED_LOCALLY')
    with (release/'v10_joint_jscc_arch_v1_build_verification.json').open('x') as stream: json.dump(result,stream,indent=2)
    def blob(f):
        data=f.read_bytes(); return hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()
    publication=dict(files=[dict(path=str(f.relative_to(project)),content=f.read_text(),git_blob_sha1=blob(f)) for f in files+[folder/'MANIFEST.json']],
        prerequisite_git_blob_sha1={n:blob(project/n) for n in required},shared_handoff_entry=(folder/'SHARED_HANDOFF_ENTRY.md').read_text(),archive_sha256=sha)
    with (release/'v10_joint_jscc_arch_v1_publication_payload.json').open('x',encoding='utf-8') as stream: json.dump(publication,stream,ensure_ascii=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
