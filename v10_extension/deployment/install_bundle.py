"""Stdlib-only, additive installer. This file also serves as the .pyz __main__."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import zipfile

RELEASE = 'multimodel_v10_inplace_20260920'
MANIFEST = 'v10_extension/PACKAGE_MANIFEST.json'


def safe_target(root, relative):
    p = PurePosixPath(relative)
    if (not relative or p.is_absolute() or '..' in p.parts or '\\' in relative
            or ':' in relative or p.as_posix() != relative
            or not (relative.startswith('v10_extension/')
                    or relative in ('run_v10.sh', 'README_V10_INPLACE.md'))):
        raise ValueError('Unsafe or non-additive payload path: '+relative)
    target = root.joinpath(*p.parts)
    try:
        target.resolve().relative_to(root)
    except ValueError:
        raise ValueError('Payload escapes project: '+relative)
    current = root
    for part in p.parts:
        current = current/part
        if current.is_symlink():
            raise ValueError('Refusing symlink in installation target: '+str(current))
        if current.exists() and current != target and not current.is_dir():
            raise ValueError('Parent is not a directory: '+str(current))
    return target


def install_archive(archive, project_root, dry_run=False):
    root, archive = Path(project_root).resolve(), Path(archive).resolve()
    if not root.is_dir() or root.name != 'multimodel':
        raise ValueError('Run in the existing project directory named multimodel')
    if archive.parent != root:
        raise ValueError('Upload the .pyz bundle directly into multimodel before running it')
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or len(names) > 500:
            raise ValueError('Duplicate or excessive archive members')
        if sum(i.file_size for i in bundle.infolist()) > 10*1024*1024:
            raise ValueError('Unexpected payload size; this is a small code-only bundle')
        raw_manifest = bundle.read('payload/'+MANIFEST)
        manifest = json.loads(raw_manifest.decode('utf-8'))
        if manifest.get('release') != RELEASE:
            raise ValueError('Unexpected release identity')
        records = manifest['files']
        expected = {'payload/'+r['path'] for r in records}
        if len(expected) != len(records) or 'payload/'+MANIFEST in expected:
            raise ValueError('Duplicate/recursive manifest entries')
        if set(names) != expected | {'__main__.py', 'payload/'+MANIFEST}:
            raise ValueError('Unlisted or missing archive member')
        contents = {}
        for row in records:
            safe_target(root, row['path'])
            data = bundle.read('payload/'+row['path'])
            if len(data) != row['size'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                raise ValueError('Corrupt payload: '+row['path'])
            contents[row['path']] = data
        safe_target(root, MANIFEST)
        contents[MANIFEST] = raw_manifest
    # Check every collision BEFORE the first filesystem mutation.
    conflicts, identical = [], []
    for relative, data in contents.items():
        target = safe_target(root, relative)
        if target.exists():
            if not target.is_file() or target.read_bytes() != data:
                conflicts.append(relative)
            else:
                identical.append(relative)
    if conflicts:
        raise FileExistsError('Nothing installed. Existing files preserved; resolve these conflicts without overwriting:\n'+'\n'.join(conflicts))
    new = [p for p in contents if p not in identical]
    report = {'release': RELEASE, 'project_root': str(root), 'dry_run': dry_run,
              'new_files': len(new), 'identical_files_left_untouched': len(identical),
              'overwritten_files': 0, 'bundle_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    if dry_run:
        return report
    for relative in new:
        target = safe_target(root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target = safe_target(root, relative)
        with target.open('xb') as f:
            f.write(contents[relative])
            f.flush()
            os.fsync(f.fileno())
        if target.read_bytes() != contents[relative]:
            raise OSError('Read-back mismatch: '+relative)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    receipt = safe_target(root, 'v10_extension/deployment_records/install_'+stamp+'_'+str(os.getpid())+'.json')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    report['receipt'] = str(receipt)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--dry-run', action='store_true', help='Read-only collision/path/hash inspection')
    args = parser.parse_args()
    report = install_archive(sys.argv[0], args.project_root, args.dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.dry_run:
        print('Installed inside multimodel only. Next: bash run_v10.sh package && bash run_v10.sh check')


if __name__ == '__main__':
    main()
