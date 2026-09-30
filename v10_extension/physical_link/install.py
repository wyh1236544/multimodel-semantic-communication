#!/usr/bin/env python3
"""Additive installer, stdlib only; no existing code target may be replaced."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

NOTE = ('## 2026-09-30 coherent waveform physical-link V2\n\n'
        'New additive channel: see [../physical_link/HANDOFF.md](../physical_link/HANDOFF.md) '
        'and [../physical_link/README.md](../physical_link/README.md). '
        'No KK; actual bitstream/CRC/optional convolutional FEC/Gray QAM/RRC/64GSa/s. '
        'Old code and weights unchanged; real server/GPU validation pending.\n\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', required=True)
    args = p.parse_args()
    root = Path(args.project_root).resolve(strict=True)
    extension = root / 'v10_extension'
    target = extension / 'physical_link'
    for directory in (extension, extension / 'link_compare', extension / 'docs'):
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError('Missing or linked prerequisite: ' + str(directory))
    if not (extension / 'link_compare/coherent.py').is_file():
        raise ValueError('Install V1 first')
    if target.exists() or target.is_symlink():
        raise FileExistsError('Preserving existing target: ' + str(target))
    source = Path(__file__).resolve().parent
    manifest = json.loads((source / 'MANIFEST.json').read_text(encoding='utf-8'))
    files = manifest['files']
    for name, expected in files.items():
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Unsafe manifest filename')
        path = source / name
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Source checksum mismatch: ' + name)
    handoff = extension / 'docs/HANDOFF.md'
    if handoff.is_symlink():
        raise ValueError('Linked shared handoff refused')
    records = extension / 'deployment_records'
    if records.is_symlink():
        raise ValueError('Linked deployment_records refused')
    record = records / ('physical_link_v2_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + str(os.getpid()))
    record.mkdir(parents=True, exist_ok=False)
    previous = handoff.read_bytes() if handoff.exists() else b''
    if handoff.exists():
        shutil.copy2(str(handoff), str(record / 'HANDOFF.before.md'))
    target.mkdir(exist_ok=False)
    for name in list(files) + ['MANIFEST.json']:
        with (target / name).open('xb') as stream:
            stream.write((source / name).read_bytes())
    if NOTE.splitlines()[0].encode() not in previous:
        temporary = record / 'HANDOFF.new.md'
        temporary.write_bytes(NOTE.encode('utf-8') + previous)
        os.replace(str(temporary), str(handoff))
    evidence = {'status': 'installed_not_runtime_verified', 'target': str(target),
                'files': files, 'shared_handoff_backup': str(record / 'HANDOFF.before.md') if previous else None}
    (record / 'installation.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    for name, expected in files.items():
        if hashlib.sha256((target / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Installed checksum mismatch: ' + name)
    print('Installed and source checksums verified: ' + str(target))
    print('Record: ' + str(record))


if __name__ == '__main__':
    main()
