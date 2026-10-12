"""Deterministic source tar/checksum plus real restore/install negative checks."""
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
    folder = Path(__file__).resolve().parent; project = folder.parents[1]
    release = project/'releases'; release.mkdir(exist_ok=True)
    archive = release/'v10_clip_bart_joint_v5.tar.gz'
    if archive.exists() or (folder/'MANIFEST.json').exists(): raise FileExistsError('Published source preserved; use new version')
    files = sorted(f for f in folder.iterdir() if f.suffix in ('.py', '.json', '.md'))
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    for f in files:
        if f.suffix == '.py': ast.parse(f.read_text(), filename=str(f), feature_version=(3, 8))
    parent = project/'v10_extension/bart_latent_codec_v4'
    required = dict(json.loads((parent/'MANIFEST.json').read_text())['required_sources'])
    for f in parent.iterdir():
        if f.suffix in ('.py', '.json', '.md'): required[str(f.relative_to(project))] = sha(f)
    if any(sha(project/n) != h for n, h in required.items()): raise ValueError('Old prerequisite drift')
    manifest = dict(schema='clip-bart-joint-v5-source', source_only=True, python_min='3.8',
        files={f.name: sha(f) for f in files}, required_sources=required)
    with (folder/'MANIFEST.json').open('x') as f: json.dump(manifest, f, indent=2); f.write('\n')
    published = files+[folder/'MANIFEST.json']
    with archive.open('xb') as f, gzip.GzipFile(fileobj=f, mode='wb', filename='', mtime=0) as gz, tarfile.open(fileobj=gz, mode='w', format=tarfile.USTAR_FORMAT) as tar:
        for file in published:
            data = file.read_bytes(); member = tarfile.TarInfo('_v10_clip_bart_joint_v5/'+file.name)
            member.size = len(data); member.mode = 0o644; tar.addfile(member, io.BytesIO(data))
    restore = Path(tempfile.mkdtemp(prefix='clip_bart_joint_v5_restore_', dir=str(release)))
    with tarfile.open(archive) as tar:
        if set(tar.getnames()) != {'_v10_clip_bart_joint_v5/'+f.name for f in published} or any(not m.isfile() for m in tar.getmembers()):
            raise ValueError('Unexpected source tar members')
        tar.extractall(str(restore))
    source = restore/'_v10_clip_bart_joint_v5'
    if any(sha(source/f.name) != sha(f) for f in published): raise ValueError('Archive restore differs')
    def fixture(name):
        root = restore/name; root.mkdir()
        for n in required:
            dest = root/n; dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as f: f.write((project/n).read_bytes())
        handoff = root/'v10_extension/docs/HANDOFF.md'; handoff.parent.mkdir(parents=True, exist_ok=True)
        with handoff.open('xb') as f: f.write(b'# Exact prior handoff\n')
        return root
    target = fixture('multimodel'); install(source, target); install(source, target, True); install(source, target)
    backups = list((target/'v10_extension/deployment_records').glob('HANDOFF_before_clip_bart_joint_v5_*.md'))
    if len(backups) != 1 or backups[0].read_bytes() != b'# Exact prior handoff\n': raise ValueError('Handoff backup/reinstall differs')
    if any(sha(target/n) != h for n, h in required.items()): raise ValueError('Historical source changed')
    conflict = fixture('conflict'); dest = conflict/'v10_extension/clip_bart_joint_v5'; dest.mkdir()
    with (dest/'run.py').open('xb') as f: f.write(b'user owned\n')
    try: install(source, conflict)
    except FileExistsError: pass
    else: raise ValueError('User source conflict not rejected')
    if (dest/'run.py').read_bytes() != b'user owned\n' or (conflict/'v10_extension/docs/HANDOFF.md').read_bytes() != b'# Exact prior handoff\n':
        raise ValueError('Conflict changed old files')
    drift = fixture('drift'); key = next(iter(required))
    with (drift/key).open('ab') as f: f.write(b'drift')
    try: install(source, drift)
    except ValueError: pass
    else: raise ValueError('Prerequisite drift accepted')
    if (drift/'v10_extension/clip_bart_joint_v5').exists(): raise ValueError('Drift wrote payload')
    link = fixture('symlink'); empty = restore/'empty'; empty.mkdir()
    (link/'v10_extension/clip_bart_joint_v5').symlink_to(empty, target_is_directory=True)
    try: install(source, link)
    except ValueError: pass
    else: raise ValueError('Symlink target accepted')
    if list(empty.iterdir()): raise ValueError('Symlink destination changed')
    archive_sha = sha(archive)
    with archive.with_name(archive.name+'.sha256').open('x') as f: f.write(archive_sha+'  '+archive.name+'\n')
    result = dict(archive=str(archive), archive_sha256=archive_sha, bytes=archive.stat().st_size,
        source_files=len(published), prerequisite_files=len(required), python38_parse=True,
        restore_install_reinstall_conflict_drift_symlink_handoff_backup=True,
        local_Torch_pretrained_GPU='NOT_EXECUTED; unavailable', restore_directory=str(restore))
    with (release/'v10_clip_bart_joint_v5_build_verification.json').open('x') as f: json.dump(result, f, indent=2)
    backup = project/'v10_extension/deployment_records/HANDOFF_before_clip_bart_joint_v5_20261013_local.md'
    handoff = project/'v10_extension/docs/HANDOFF.md'
    if not backup.is_file() or handoff.read_bytes() != backup.read_bytes()+b'\n'+(folder/'SHARED_HANDOFF_ENTRY.md').read_bytes():
        raise ValueError('Exact local handoff backup/append missing')
    def blob(file):
        data = file.read_bytes(); return hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()
    publication = dict(files=[dict(path=str(f.relative_to(project)), content=f.read_text(), git_blob_sha1=blob(f))
        for f in published+[backup, handoff]], prerequisite_git_blob_sha1={n: blob(project/n) for n in required},
        previous_handoff_git_blob_sha1=blob(backup), archive_sha256=archive_sha)
    with (release/'v10_clip_bart_joint_v5_publication_payload.json').open('x', encoding='utf-8') as f:
        json.dump(publication, f, ensure_ascii=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
