"""Build a tiny additive, source-only patch; no old source/manifest edits."""
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile


def sha(data): return hashlib.sha256(data).hexdigest()


def main():
    folder=Path(__file__).resolve().parent; project=folder.parents[1]
    release=project/'releases'; release.mkdir(exist_ok=True)
    archive=release/'v10_docci_semantic_probe_v12_collect_fix1.tar.gz'
    if archive.exists() or (folder/'MANIFEST.json').exists(): raise FileExistsError('Existing release preserved')
    files=sorted(p for p in folder.iterdir() if p.is_file() and p.suffix in ('.py','.md'))
    for file in files:
        if file.suffix=='.py': ast.parse(file.read_text(encoding='utf-8'),feature_version=(3,8))
    legacy=project/'v10_extension/docci_semantic_probe_v12'
    before={p.name:sha(p.read_bytes()) for p in legacy.iterdir() if p.is_file()}
    manifest=dict(schema='docci-semantic-probe-v12-collect-fix1-source',source_only=True,
        original_V12_manifest_sha256=sha((legacy/'MANIFEST.json').read_bytes()),
        files={p.name:sha(p.read_bytes()) for p in files},python_min='3.8',training_or_quality_changes=False)
    with (folder/'MANIFEST.json').open('x',encoding='utf-8') as handle:
        json.dump(manifest,handle,indent=2); handle.write('\n')
    subprocess.run([sys.executable,'-B',str(folder/'test_collect.py')],cwd=str(project),check=True)
    for test in ('test_core.py','test_report.py'):
        subprocess.run([sys.executable,'-B',str(legacy/test)],cwd=str(project),check=True)
    if before!={p.name:sha(p.read_bytes()) for p in legacy.iterdir() if p.is_file()}:
        raise ValueError('Original V12 source changed')
    published=files+[folder/'MANIFEST.json']; prefix='v10_extension/'+folder.name+'/'
    with archive.open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tar:
        for file in published:
            data=file.read_bytes(); entry=tarfile.TarInfo(prefix+file.name); entry.size=len(data); entry.mode=0o644
            tar.addfile(entry,io.BytesIO(data))
    with tarfile.open(archive) as tar:
        if set(tar.getnames())!={prefix+p.name for p in published} or any(not x.isfile() for x in tar.getmembers()):
            raise ValueError('Unsafe archive inventory')
        for member in tar.getmembers():
            if sha(tar.extractfile(member).read())!=manifest['files'].get(Path(member.name).name,sha((folder/'MANIFEST.json').read_bytes())):
                raise ValueError('Archive bytes differ')
    checksum=archive.with_name(archive.name+'.sha256')
    with checksum.open('x') as handle: handle.write(sha(archive.read_bytes())+'  '+archive.name+'\n')
    payload=dict(files=[dict(path=str(p.relative_to(project)),content=p.read_text(encoding='utf-8')) for p in published],
        archive_sha256=sha(archive.read_bytes()))
    with (release/'v10_docci_semantic_probe_v12_collect_fix1_publication_payload.json').open('x',encoding='utf-8') as handle:
        json.dump(payload,handle,ensure_ascii=False)
    verification=dict(python38_parse=True,new_synthetic_test_methods=6,original_CPU_tests=16,
        original_V12_all_source_bytes_unchanged=True,original_manifest_sha256=manifest['original_V12_manifest_sha256'],
        server_actual_V12_result_collection='NOT_EXECUTED',GPU_training='NOT_EXECUTED',
        archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size)
    with (release/'v10_docci_semantic_probe_v12_collect_fix1_verification.json').open('x') as handle:
        json.dump(verification,handle,indent=2)
    print(json.dumps(verification,indent=2))


if __name__=='__main__': main()
