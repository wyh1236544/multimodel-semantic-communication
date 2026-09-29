"""Build a checksummed patch archive from the reviewed local source snapshot."""
import hashlib
import io
import json
import tarfile
from pathlib import Path

STAGE_NAME = 'v10_extension/deployment_patches/eval_fix1_20260929'
RELEASE = 'v10-eval-fix1-20260929'
FILES = (
    'run_v10_formal_suite.sh',
    'v10_extension/eval_v10.py',
    'v10_extension/models/joint_image_text_model.py',
    'v10_extension/v10/model.py',
    'v10_extension/v10/shuffle.py',
    'v10_extension/verify_shuffled_v10.py',
    'v10_extension/tests/test_v10_shuffle.py',
    'v10_extension/tests/test_v10_shuffled_integration.py',
    'v10_extension/docs/FORMAL_EXPERIMENTS.md',
    'v10_extension/docs/HANDOFF.md',
)


def add_bytes(archive, name, data, mode=0o644):
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = len(data), mode, 0
    archive.addfile(info, io.BytesIO(data))


def main():
    stage = Path(__file__).resolve().parent
    root = stage.parents[2]
    old_archive = root/'v10_extension/verification/v10_eval_fix1_preedit_20260929.tar.gz'
    package = stage/'multimodel_v10_eval_fix1_20260929.tar.gz'
    if package.exists():
        raise FileExistsError('Do not overwrite an existing release archive')
    entries, payload = [], {}
    with tarfile.open(old_archive, 'r:gz') as old:
        old_files = {member.name: old.extractfile(member).read()
                     for member in old if member.isfile()}
    for name in FILES:
        data = (root/name).read_bytes()
        prior = old_files.get(name)
        if prior == data:
            raise ValueError('Patch file was not changed: '+name)
        payload[name] = data
        entries.append({'path': name, 'size': len(data),
                        'old_sha256': hashlib.sha256(prior).hexdigest() if prior is not None else None,
                        'new_sha256': hashlib.sha256(data).hexdigest()})
    manifest = json.dumps({'release': RELEASE, 'files': entries}, indent=2).encode('utf-8')
    with tarfile.open(package, 'w:gz') as archive:
        add_bytes(archive, STAGE_NAME+'/install.py', (stage/'install.py').read_bytes())
        add_bytes(archive, STAGE_NAME+'/README_SERVER.md', (stage/'README_SERVER.md').read_bytes())
        add_bytes(archive, STAGE_NAME+'/manifest.json', manifest)
        for name in FILES:
            add_bytes(archive, STAGE_NAME+'/payload/'+name, payload[name])
    digest = hashlib.sha256(package.read_bytes()).hexdigest()
    (stage/(package.name+'.sha256')).write_text(digest+'  '+package.name+'\n', encoding='utf-8')
    print(json.dumps({'archive': str(package), 'sha256': digest, 'files': len(FILES)}, indent=2))


if __name__ == '__main__':
    main()
