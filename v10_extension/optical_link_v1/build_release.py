"""Deterministic source-only archive, restoration and additive-install validation."""
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
    folder=Path(__file__).resolve().parent
    project=folder.parents[1]; releases=project/'releases'
    releases.mkdir(exist_ok=True)
    archive=releases/'v10_optical_link_v1.tar.gz'
    if archive.exists() or (folder/'MANIFEST.json').exists(): raise FileExistsError('Release exists; use a new version')
    files=sorted(p for p in folder.iterdir() if p.suffix in ('.py','.md','.json'))
    for p in files:
        if p.suffix=='.py': ast.parse(p.read_text(),filename=str(p),feature_version=(3,8))
    required=['v10_extension/'+name for name in (
        'physical_link/digital.py','physical_link/waveform.py','business_link/engine.py',
        'business_link/metrics.py','token_guidance/wire.py','token_guidance/model.py','token_guidance/spec.py',
        'v10/model.py','v10/channels.py','v10/protocol.py','v10/data.py','v10/runtime.py','v10/io.py','v10/paths.py',
        'models/joint_image_text_model.py','models/fusion.py','eval_v10.py',
        'configs/v10_multichannel.json','configs/v9a_feedback_generation.yaml')]
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(schema='optical-link-v1-source',source_only=True,files={p.name:digest(p) for p in files},
                  required_sources={name:digest(project/name) for name in required})
    with (folder/'MANIFEST.json').open('x') as stream: json.dump(manifest,stream,indent=2); stream.write('\n')
    with archive.open('xb') as stream:
        with gzip.GzipFile(fileobj=stream,mode='wb',filename='',mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode='w') as tar:
                for path in files+[folder/'MANIFEST.json']:
                    data=path.read_bytes(); info=tarfile.TarInfo('_v10_optical_link_v1/'+path.name)
                    info.size,info.mode=len(data),0o644; tar.addfile(info,io.BytesIO(data))
    restore=Path(tempfile.mkdtemp(prefix='optical_v1_restore_',dir=str(releases)))
    with tarfile.open(archive,'r:gz') as tar:
        expected={'_v10_optical_link_v1/'+p.name for p in files+[folder/'MANIFEST.json']}
        if set(tar.getnames())!=expected or any(not m.isfile() for m in tar.getmembers()): raise ValueError('Unexpected archive member')
        tar.extractall(str(restore))
    source=restore/'_v10_optical_link_v1'
    for name,sha in manifest['files'].items():
        if digest(source/name)!=sha: raise ValueError('Restored source mismatch')
    def synthetic(name):
        root=restore/name; root.mkdir()
        for path in required:
            target=root/path; target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream: stream.write((project/path).read_bytes())
        handoff=root/'v10_extension/docs/HANDOFF.md'; handoff.parent.mkdir(parents=True,exist_ok=True)
        with handoff.open('xb') as stream: stream.write(b'# Preserved synthetic history\n')
        return root
    root=synthetic('synthetic_multimodel')
    install(source,root); install(source,root,verify_only=True)
    backups=list((root/'v10_extension/deployment_records').glob('HANDOFF_before_optical_v1_*.md'))
    if len(backups)!=1 or backups[0].read_bytes()!=b'# Preserved synthetic history\n':
        raise ValueError('Shared handoff original bytes were not backed up')
    if not (root/'v10_extension/docs/HANDOFF.md').read_text().startswith('# Preserved synthetic history\n'):
        raise ValueError('Shared history not retained')
    blocked=synthetic('conflict_multimodel'); target=blocked/'v10_extension/optical_link_v1'
    target.mkdir(); preserved=target/'run.py'
    with preserved.open('xb') as stream: stream.write(b'existing-user-code\n')
    try: install(source,blocked)
    except FileExistsError: pass
    else: raise ValueError('Conflicting installation not rejected')
    if preserved.read_bytes()!=b'existing-user-code\n': raise ValueError('User file was overwritten')
    # A distinct synthetic root lacks a dependency; no user file is modified.
    wrong=restore/'wrong_dependency_multimodel'; wrong.mkdir(); (wrong/'v10_extension').mkdir()
    try: install(source,wrong)
    except ValueError: pass
    else: raise ValueError('Missing dependency not rejected')
    sha=digest(archive)
    with archive.with_name(archive.name+'.sha256').open('x') as stream: stream.write(sha+'  '+archive.name+'\n')
    result=dict(archive=str(archive),sha256=sha,source_files=len(files),
        restore_install_reinstall_conflict_missing_dependency_checks=True,python38_parse=True,
        restore_directory=str(restore),server_gpu_effectiveness='not_executed')
    with (releases/'v10_optical_link_v1_build_verification.json').open('x') as stream: json.dump(result,stream,indent=2)
    # Explicit source-only payload for atomic connector publication, not secrets,
    # weights, outputs, caches or an implicit git-add of the whole workspace.
    def blob_sha(p):
        content=p.read_bytes()
        return hashlib.sha1(('blob '+str(len(content))+'\0').encode()+content).hexdigest()
    publication_files=files+[folder/'MANIFEST.json',project/'v10_extension/docs/HANDOFF.md']
    publication=dict(files=[dict(path=str(p.relative_to(project)),content=p.read_text(encoding='utf-8'),git_blob_sha1=blob_sha(p)) for p in publication_files],
        prerequisite_git_blob_sha1={name:blob_sha(project/name) for name in required},
        archive_sha256=sha,archive_bytes=archive.stat().st_size)
    with (releases/'v10_optical_link_v1_publication_payload.json').open('x',encoding='utf-8') as stream:
        json.dump(publication,stream,ensure_ascii=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
