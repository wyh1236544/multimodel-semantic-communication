"""Additive USTAR source release. Runtime import closure pinned; no data/weights."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parents[1]
sys.path[:0] = [str(FOLDER.parent), str(PROJECT)]
from docci_direct_token_v28.core import read, sha, check_source


def dependency_closure():
    result = {}; seen = set(); queue = list(FOLDER.glob('*.py'))
    def resolve(name):
        for root in (FOLDER.parent, PROJECT):
            stem = root.joinpath(*name.split('.'))
            for p in (stem.with_suffix('.py'), stem/'__init__.py'):
                if p.is_file(): return p
        return None
    while queue:
        file = queue.pop()
        if file in seen: continue
        seen.add(file)
        if FOLDER not in file.parents:
            result[file.relative_to(PROJECT).as_posix()] = sha(file)
        tree = ast.parse(file.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import): names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package = file.parent.relative_to(FOLDER.parent).parts
                    package = package[:len(package)-node.level+1]
                    prefix = '.'.join(package+tuple((node.module or '').split('.'))).strip('.')
                else: prefix = node.module or ''
                names = [prefix]+[prefix+'.'+a.name for a in node.names if a.name != '*']
            for name in names:
                found = resolve(name)
                if found is not None and found not in seen: queue.append(found)
    return dict(sorted(result.items()))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--manifest-only', action='store_true'); a = p.parse_args()
    dest = PROJECT/'releases/v10_docci_direct_token_v28.tar.gz'
    if dest.exists() or dest.with_name(dest.name+'.sha256').exists(): raise FileExistsError('Published release preserved')
    files = {p.name: sha(p) for p in sorted(FOLDER.iterdir()) if p.is_file() and p.name != 'MANIFEST.json'}
    for name in files:
        if name.endswith('.py'): ast.parse((FOLDER/name).read_text(), filename=name, feature_version=(3, 8))
    deps = dependency_closure()
    # Check known historic pins; no silent edits to old modules for this branch.
    for version in ('docci_baseline_v14', 'docci_text_channel_v18', 'docci_text_guided_image_v20'):
        old = read(FOLDER.parent/version/'MANIFEST.json')
        expected = dict(old['dependencies'])
        expected.update({'v10_extension/'+version+'/'+n: h for n, h in old['files'].items()})
        for name in set(expected) & set(deps):
            if deps[name] != expected[name]: raise ValueError('Historical pin differs: '+name)
    value = dict(schema='v28-source-only', files=files, dependencies=deps,
        contains_weights_or_data=False, source_branch='experiment/direct-token-low-snr-v28',
        server_runtime='Python3.8 / Torch1.12 / Transformers4.30.2; existing environment only')
    (FOLDER/'MANIFEST.json').write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    check_source()
    if a.manifest_only:
        print('{} source files / {} pinned historical dependency files'.format(len(files), len(deps))); return
    dest.parent.mkdir(exist_ok=True)
    mapping = {'v10_extension/docci_direct_token_v28/'+n: FOLDER/n for n in list(files)+['MANIFEST.json']}
    with dest.open('xb') as out, tarfile.open(fileobj=out, mode='w:gz', format=tarfile.USTAR_FORMAT) as t:
        for name, file in mapping.items():
            value = file.read_bytes(); info = tarfile.TarInfo(name)
            info.mode = 0o644; info.size = len(value); info.mtime = 0
            t.addfile(info, io.BytesIO(value))
    with tarfile.open(dest) as t:
        members = t.getmembers()
        if len(members) != len(mapping) or set(mapping) != {m.name for m in members}: raise ValueError('Archive member matrix differs')
        for m in members:
            if not m.isfile() or m.pax_headers or hashlib.sha256(t.extractfile(m).read()).hexdigest() != sha(mapping[m.name]):
                raise ValueError('Source archive content differs')
    with dest.with_name(dest.name+'.sha256').open('x', encoding='utf-8') as f:
        f.write(sha(dest)+'  '+dest.name+'\n')
    print(str(dest)); print(sha(dest))


if __name__ == '__main__': main()
