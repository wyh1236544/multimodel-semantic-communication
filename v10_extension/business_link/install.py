"""Stdlib additive install/verify; never replace old source, weights or results."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(root, name):
    parts = Path(name).parts
    if not parts or Path(name).is_absolute() or any(p in ('.', '..') for p in parts):
        raise ValueError('Unsafe installation path')
    path = root
    for part in parts:
        path = path/part
        if path.is_symlink():
            raise ValueError('Refusing linked installation path: '+str(path))
    return path


def install(source, root, verify_only=False):
    source, root = Path(source).resolve(strict=True), Path(root).resolve(strict=True)
    manifest = json.loads((source/'MANIFEST.json').read_text(encoding='utf-8'))
    ext = safe(root, 'v10_extension')
    if not ext.is_dir():
        raise ValueError('Need existing V10 multimodel root')
    for name, expected in manifest['required_sources'].items():
        path = safe(root, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Missing/changed prerequisite; preserved: '+str(path))
    files = manifest['files']
    for name, expected in files.items():
        if Path(name).name != name:
            raise ValueError('Flat source filenames required')
        path = safe(source, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Package checksum mismatch: '+name)
    target = safe(root, 'v10_extension/business_link')
    if target.exists():
        for name, expected in files.items():
            path = safe(target, name)
            if not path.is_file() or digest(path) != expected:
                raise FileExistsError('Existing V3 differs; do not overwrite: '+str(path))
        if digest(safe(target, 'MANIFEST.json')) != digest(source/'MANIFEST.json'):
            raise FileExistsError('Existing V3 manifest differs')
        print('Existing identical V3 source verified, unchanged: '+str(target))
        return
    if verify_only:
        raise FileNotFoundError('V3 target not installed')
    target.mkdir(exist_ok=False)
    for name in list(files)+['MANIFEST.json']:
        with (target/name).open('xb') as stream:
            stream.write((source/name).read_bytes())
    for name, expected in files.items():
        if digest(target/name) != expected:
            raise ValueError('Post-install checksum mismatch: '+name)
    print('Added V3 source; old files untouched: '+str(target))
    print('Source verified; server CPU/GPU runtime verification still required.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', required=True)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    install(Path(__file__).resolve().parent, Path(args.project_root), args.verify_only)
