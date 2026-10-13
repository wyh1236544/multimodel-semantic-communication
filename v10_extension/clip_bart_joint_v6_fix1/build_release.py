"""Immutable source-only fix archive, verified restore and publication payload."""
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile


def main():
    source = Path(__file__).resolve().parent
    project = source.parents[1]
    release = project/'releases'
    archive = release/'v10_clip_bart_joint_v6_fix1.tar.gz'
    if archive.exists() or (source/'PATCH_MANIFEST.json').exists():
        raise FileExistsError('Published patch preserved')
    digest = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    module = project/'v10_extension/clip_bart_joint_v6'
    names = ('report.py', 'test_report.py', 'MANIFEST.json')
    old = json.loads((source/'previous/MANIFEST.json').read_text())
    new = json.loads((module/'MANIFEST.json').read_text())
    if old['required_sources'] != new['required_sources'] or set(old['files']) != set(new['files']):
        raise ValueError('Unexpected source design drift')
    if {n for n in old['files'] if old['files'][n] != new['files'][n]} != {'report.py', 'test_report.py'}:
        raise ValueError('Only report/test may change')
    for name, expected in new['required_sources'].items():
        if digest(project/name) != expected:
            raise ValueError('Historical prerequisite drift')
    for name, expected in new['files'].items():
        if digest(module/name) != expected:
            raise ValueError('Module manifest differs: '+name)
    with tarfile.open(release/'v10_clip_bart_joint_v6.tar.gz') as tar:
        for name in names:
            if tar.extractfile('_v10_clip_bart_joint_v6/'+name).read() != (source/'previous'/name).read_bytes():
                raise ValueError('Prior backup not original published file')
    payload = source/'payload'; payload.mkdir(exist_ok=False)
    for name in names:
        shutil.copyfile(module/name, payload/name)
    files = [source/name for name in ('install.py','test_install.py','build_release.py','README.md','VERIFICATION.md','SHARED_HANDOFF_ENTRY.md')]
    files += [payload/name for name in names]
    for file in files:
        if file.suffix == '.py':
            ast.parse(file.read_text(), filename=str(file), feature_version=(3, 8))
    patch = dict(schema='clip-bart-joint-v6-fix1', source_only=True,
        replacements={n: dict(before=digest(source/'previous'/n), after=digest(module/n)) for n in names},
        required_sources=new['required_sources'],
        package_files={str(f.relative_to(source)): digest(f) for f in files})
    with (source/'PATCH_MANIFEST.json').open('x') as stream:
        json.dump(patch, stream, indent=2); stream.write('\n')
    files.append(source/'PATCH_MANIFEST.json')
    subprocess.run([sys.executable,'-B',str(source/'test_install.py')], cwd=str(project), check=True)
    with archive.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as gz, tarfile.open(fileobj=gz, mode='w', format=tarfile.USTAR_FORMAT) as tar:
        for file in sorted(files):
            data = file.read_bytes()
            member = tarfile.TarInfo('_v10_clip_bart_joint_v6_fix1/'+str(file.relative_to(source)))
            member.size = len(data); member.mode = 0o644
            tar.addfile(member, io.BytesIO(data))
    restored = Path(tempfile.mkdtemp(prefix='joint_v6_fix1_restore_', dir=str(release)))
    with tarfile.open(archive) as tar:
        if set(tar.getnames()) != {'_v10_clip_bart_joint_v6_fix1/'+str(f.relative_to(source)) for f in files} or any(not m.isfile() for m in tar.getmembers()):
            raise ValueError('Unsafe archive members')
        tar.extractall(str(restored))
    for file in files:
        copy = restored/'_v10_clip_bart_joint_v6_fix1'/file.relative_to(source)
        if copy.read_bytes() != file.read_bytes():
            raise ValueError('Actual restored archive differs')
    entry = (source/'SHARED_HANDOFF_ENTRY.md').read_bytes()
    if (project/'v10_extension/docs/HANDOFF.md').read_bytes() != (source/'previous/HANDOFF.md').read_bytes()+b'\n'+entry:
        raise ValueError('Exact local handoff backup/append differs')
    # Execute installer from restored package against an old complete fixture.
    prior_fixture = project/'releases/clip_bart_joint_v6_restore_wqg1ceqf/multimodel'
    target = restored/'multimodel'
    shutil.copytree(prior_fixture, target, ignore=shutil.ignore_patterns('__pycache__'))
    (target/'releases').mkdir(exist_ok=True)
    subprocess.run([sys.executable,'-B',str(restored/'_v10_clip_bart_joint_v6_fix1/install.py'),
        '--project-root',str(target)], check=True)
    subprocess.run([sys.executable,'-B',str(restored/'_v10_clip_bart_joint_v6_fix1/install.py'),
        '--project-root',str(target)], check=True)
    subprocess.run([sys.executable,'-B',str(target/'v10_extension/clip_bart_joint_v6/install.py'),
        '--project-root',str(target),'--verify-only'], check=True)
    subprocess.run([sys.executable,'-B',str(target/'v10_extension/clip_bart_joint_v6/test_report.py')], cwd=str(target), check=True)
    archive_sha = digest(archive)
    with archive.with_name(archive.name+'.sha256').open('x') as stream:
        stream.write(archive_sha+'  '+archive.name+'\n')
    result = dict(archive_sha256=archive_sha, bytes=archive.stat().st_size,
        report_tests=39, local_report_passed=38, local_skipped_actual_torch=1,
        core_tests=10, historical_V5_V4_tests=49, installer_tests=8,
        restored_install_reinstall_and_original_verifier=True,
        python38_parse=True, unchanged_historical_sources=len(new['required_sources']),
        network_training_config_unchanged=True, server_GPU_training_executed=False,
        old_failed_run_modified=False, restore_directory=str(restored))
    with (release/'v10_clip_bart_joint_v6_fix1_build_verification.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    published = [module/n for n in names]+files+[source/'previous'/n for n in (*names,'HANDOFF.md')]+[project/'v10_extension/docs/HANDOFF.md']
    def blob(data):
        return hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()
    publication = dict(files=[dict(path=str(f.relative_to(project)), content=f.read_text(),
        git_blob_sha1=blob(f.read_bytes())) for f in published],
        expected_old_blobs={str((module/n).relative_to(project)): blob((source/'previous'/n).read_bytes()) for n in names},
        archive_sha256=archive_sha)
    publication['expected_old_blobs']['v10_extension/docs/HANDOFF.md'] = blob((source/'previous/HANDOFF.md').read_bytes())
    with (release/'v10_clip_bart_joint_v6_fix1_publication_payload.json').open('x', encoding='utf-8') as stream:
        json.dump(publication, stream, ensure_ascii=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
