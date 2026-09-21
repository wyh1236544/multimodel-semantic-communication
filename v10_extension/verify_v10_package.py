"""Verify listed release files. Unlisted experiment outputs are left untouched."""
import hashlib
import json
from pathlib import Path
from v10.paths import PROJECT_ROOT, write_path

def main():
    root = PROJECT_ROOT
    manifest = json.loads((Path(__file__).resolve().parent/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    if manifest.get('release') != 'multimodel_v10_inplace_20260920':
        raise ValueError('Unexpected release identity')
    for record in manifest['files']:
        target = write_path(root/record['path'])
        if root not in target.parents or not target.is_file():
            raise ValueError('Missing or unsafe package file: '+record['path'])
        content = target.read_bytes()
        if len(content) != record['size'] or hashlib.sha256(content).hexdigest() != record['sha256']:
            raise ValueError('Hash mismatch: '+record['path'])
    print('Verified {} release files; model performance not certified.'.format(len(manifest['files'])))

if __name__ == '__main__':
    main()
