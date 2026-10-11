"""Deterministic source archive, restore/additive-install/conflict checks, publication payload."""
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
    folder = Path(__file__).resolve().parent; project = folder.parents[1]
    release = project/'releases'; release.mkdir(exist_ok=True)
    archive = release/'v10_fso_jscc_v1.tar.gz'
    if archive.exists() or (folder/'MANIFEST.json').exists():
        raise FileExistsError('Existing immutable release preserved; create a new version')
    files = sorted(f for f in folder.iterdir() if f.suffix in ('.py', '.json', '.md'))
    for f in files:
        if f.suffix == '.py':
            ast.parse(f.read_text(encoding='utf-8'), filename=str(f), feature_version=(3, 8))
    required = []
    for name in ('v10', 'models', 'utils', 'data', 'configs', 'token_guidance', 'business_link',
                 'physical_link', 'optical_link_v1', 'final_compare_v1', 'optical_realism_v2', 'optical_realism_v3'):
        # Pin all executed package source/configs, not old manifests or documents.
        required += [str(f.relative_to(project)) for f in sorted((project/'v10_extension'/name).rglob('*'))
                     if f.suffix in ('.py', '.json', '.yaml') and f.name != 'MANIFEST.json' and '__pycache__' not in f.parts]
    required += ['v10_extension/eval_v10.py']
    digest = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    manifest = dict(schema='fso-jscc-v1-source', source_only=True, python_min='3.8',
        files={f.name: digest(f) for f in files}, required_sources={n: digest(project/n) for n in required})
    with (folder/'MANIFEST.json').open('x') as stream:
        json.dump(manifest, stream, indent=2); stream.write('\n')
    with archive.open('xb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w', format=tarfile.USTAR_FORMAT) as tar:
                for f in files+[folder/'MANIFEST.json']:
                    data = f.read_bytes(); info = tarfile.TarInfo('_v10_fso_jscc_v1/'+f.name)
                    info.size, info.mode = len(data), 0o644; tar.addfile(info, io.BytesIO(data))
    restore = Path(tempfile.mkdtemp(prefix='fso_jscc_v1_restore_', dir=str(release)))
    with tarfile.open(archive, 'r:gz') as tar:
        expected = {'_v10_fso_jscc_v1/'+f.name for f in files+[folder/'MANIFEST.json']}
        if set(tar.getnames()) != expected or any(not m.isfile() for m in tar.getmembers()):
            raise ValueError('Wrong archive members')
        tar.extractall(str(restore))
    source = restore/'_v10_fso_jscc_v1'
    def synthetic(name):
        root = restore/name; root.mkdir()
        for n in required:
            target = root/n; target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write((project/n).read_bytes())
        handoff = root/'v10_extension/docs/HANDOFF.md'; handoff.parent.mkdir(parents=True, exist_ok=True)
        with handoff.open('xb') as stream:
            stream.write(b'# Preserved synthetic history\n')
        return root
    root = synthetic('multimodel')
    install(source, root); install(source, root, verify_only=True); install(source, root)
    backups = list((root/'v10_extension/deployment_records').glob('HANDOFF_before_fso_jscc_v1_*.md'))
    if len(backups) != 1 or backups[0].read_bytes() != b'# Preserved synthetic history\n':
        raise ValueError('Shared handoff backup/reinstall failed')
    blocked = synthetic('conflict'); target = blocked/'v10_extension/fso_jscc_v1'; target.mkdir()
    with (target/'run.py').open('xb') as stream:
        stream.write(b'user-source\n')
    try:
        install(source, blocked)
    except FileExistsError:
        pass
    else:
        raise ValueError('Conflicting source not refused')
    if (target/'run.py').read_bytes() != b'user-source\n':
        raise ValueError('User source overwritten')
    wrong = synthetic('wrong_dependency')
    with (wrong/required[0]).open('ab') as stream:
        stream.write(b'changed')
    try:
        install(source, wrong)
    except ValueError:
        pass
    else:
        raise ValueError('Wrong dependency not refused')
    if (wrong/'v10_extension/fso_jscc_v1').exists():
        raise ValueError('Source written before dependency preflight')
    sha = digest(archive)
    with archive.with_name(archive.name+'.sha256').open('x') as stream:
        stream.write(sha+'  '+archive.name+'\n')
    result = dict(archive=str(archive), sha256=sha, source_files=len(files)+1,
        python38_parse=True, restore_install_reinstall_conflict_dependency_backup_checks=True,
        restore_directory=str(restore), Torch_real_A_COCO_GPU='NOT_EXECUTED_LOCALLY')
    with (release/'v10_fso_jscc_v1_build_verification.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    def blob(f):
        data = f.read_bytes(); return hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()
    publication = dict(files=[dict(path=str(f.relative_to(project)), content=f.read_text(encoding='utf-8'), git_blob_sha1=blob(f))
        for f in files+[folder/'MANIFEST.json']], prerequisite_git_blob_sha1={n: blob(project/n) for n in required},
        shared_handoff_entry=(folder/'SHARED_HANDOFF_ENTRY.md').read_text(encoding='utf-8'), archive_sha256=sha)
    with (release/'v10_fso_jscc_v1_publication_payload.json').open('x', encoding='utf-8') as stream:
        json.dump(publication, stream, ensure_ascii=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
