"""Deterministic source-only release + restore/install/conflict verification."""
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
    releases=project/'releases'; releases.mkdir(exist_ok=True)
    archive=releases/'v10_optical_realism_v3.tar.gz'
    if archive.exists() or (folder/'MANIFEST.json').exists(): raise FileExistsError('Release exists; use a new version')
    files=sorted(p for p in folder.iterdir() if p.suffix in ('.py','.json','.md'))
    for p in files:
        if p.suffix=='.py': ast.parse(p.read_text(encoding='utf-8'),filename=str(p),feature_version=(3,8))
    required=['v10_extension/'+name for name in (
        'physical_link/digital.py','physical_link/waveform.py','business_link/engine.py','business_link/metrics.py',
        'token_guidance/wire.py','token_guidance/model.py','token_guidance/spec.py',
        'v10/model.py','v10/channels.py','v10/protocol.py','v10/data.py','v10/runtime.py','v10/io.py','v10/paths.py',
        'models/joint_image_text_model.py','models/fusion.py','eval_v10.py',
        'utils/checkpoint.py','utils/metrics.py','utils/config.py','utils/experiment.py',
        'configs/v10_multichannel.json','configs/v9a_feedback_generation.yaml')]
    required += [str(p.relative_to(project)) for p in sorted((project/'v10_extension/optical_link_v1').iterdir())
                 if p.suffix in ('.py','.json') and p.name!='MANIFEST.json']
    required += [str(p.relative_to(project)) for p in sorted((project/'v10_extension/final_compare_v1').iterdir())
                 if p.suffix in ('.py','.json') and p.name!='MANIFEST.json']
    required += [str(p.relative_to(project)) for p in sorted((project/'v10_extension/optical_realism_v2').iterdir())
                 if p.suffix in ('.py','.json') and p.name!='MANIFEST.json']
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(schema='optical-realism-v3-source',source_only=True,python_min='3.8',
        files={p.name:digest(p) for p in files},required_sources={name:digest(project/name) for name in required})
    with (folder/'MANIFEST.json').open('x',encoding='utf-8') as stream: json.dump(manifest,stream,indent=2); stream.write('\n')
    with archive.open('xb') as stream:
        with gzip.GzipFile(fileobj=stream,mode='wb',filename='',mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
                for p in files+[folder/'MANIFEST.json']:
                    data=p.read_bytes(); info=tarfile.TarInfo('_v10_optical_realism_v3/'+p.name)
                    info.size,info.mode=len(data),0o644; tar.addfile(info,io.BytesIO(data))
    restore=Path(tempfile.mkdtemp(prefix='optical_realism_v3_restore_',dir=str(releases)))
    with tarfile.open(archive,'r:gz') as tar:
        expected={'_v10_optical_realism_v3/'+p.name for p in files+[folder/'MANIFEST.json']}
        if set(tar.getnames())!=expected or any(not m.isfile() for m in tar.getmembers()): raise ValueError('Unexpected archive member')
        tar.extractall(str(restore))
    source=restore/'_v10_optical_realism_v3'
    for name,sha in manifest['files'].items():
        if digest(source/name)!=sha: raise ValueError('Restored source differs')
    def synthetic(name):
        root=restore/name; root.mkdir()
        for path in required:
            target=root/path; target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream: stream.write((project/path).read_bytes())
        handoff=root/'v10_extension/docs/HANDOFF.md'; handoff.parent.mkdir(parents=True,exist_ok=True)
        with handoff.open('xb') as stream: stream.write(b'# Retained synthetic history\n')
        return root
    root=synthetic('synthetic_multimodel')
    install(source,root); install(source,root,verify_only=True)
    backups=list((root/'v10_extension/deployment_records').glob('HANDOFF_before_optical_realism_v3_*.md'))
    if len(backups)!=1 or backups[0].read_bytes()!=b'# Retained synthetic history\n': raise ValueError('Original shared history not backed up exactly')
    blocked=synthetic('conflict_multimodel'); target=blocked/'v10_extension/optical_realism_v3'; target.mkdir()
    with (target/'run.py').open('xb') as stream: stream.write(b'existing-user-source\n')
    try: install(source,blocked)
    except FileExistsError: pass
    else: raise ValueError('Conflicting install was not rejected')
    if (target/'run.py').read_bytes()!=b'existing-user-source\n': raise ValueError('User source overwritten')
    wrong=restore/'missing_multimodel'; wrong.mkdir(); (wrong/'v10_extension').mkdir()
    try: install(source,wrong)
    except ValueError: pass
    else: raise ValueError('Missing dependency was not rejected')
    sha=digest(archive)
    with archive.with_name(archive.name+'.sha256').open('x') as stream: stream.write(sha+'  '+archive.name+'\n')
    result=dict(archive=str(archive),sha256=sha,source_files=len(files),source_only=True,python38_parse=True,
        restore_install_reinstall_conflict_missing_dependency_checks=True,restore_directory=str(restore),
        gpu_COCO_checkpoint_effectiveness='NOT_EXECUTED_LOCALLY')
    with (releases/'v10_optical_realism_v3_build_verification.json').open('x') as stream: json.dump(result,stream,indent=2)
    def blob(p):
        data=p.read_bytes(); return hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()
    publication=dict(files=[dict(path=str(p.relative_to(project)),content=p.read_text(encoding='utf-8'),git_blob_sha1=blob(p))
        for p in files+[folder/'MANIFEST.json']],prerequisite_git_blob_sha1={name:blob(project/name) for name in required},
        shared_handoff_entry=(folder/'SHARED_HANDOFF_ENTRY.md').read_text(encoding='utf-8'),archive_sha256=sha)
    with (releases/'v10_optical_realism_v3_publication_payload.json').open('x',encoding='utf-8') as stream: json.dump(publication,stream,ensure_ascii=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
