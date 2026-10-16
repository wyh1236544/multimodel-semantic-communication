"""Create source-only USTAR, pin transitive local sources, verify isolated restore."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_baseline_v14.core import FOLDER, PROJECT, sha, safe, atomic


def local_module(module, base=None):
    roots = [base] if base is not None else [EXT, PROJECT]
    for root in roots:
        p = root.joinpath(*module.split('.')) if module else root
        for candidate in (p.with_suffix('.py'), p/'__init__.py'):
            if candidate.is_file() and not candidate.is_symlink():
                return candidate
    return None


def dependencies():
    pending = list(FOLDER.glob('*.py'))
    seen = set()
    while pending:
        file = pending.pop().resolve()
        if file in seen:
            continue
        seen.add(file)
        tree = ast.parse(file.read_text(encoding='utf-8'), filename=str(file), feature_version=8)
        for node in ast.walk(tree):
            mods, base = [], None
            if isinstance(node, ast.Import):
                mods = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or '']
                if node.level:
                    base = file.parent
                    for _ in range(node.level-1):
                        base = base.parent
            for mod in mods:
                candidate = local_module(mod, base)
                if candidate is not None and candidate.resolve() not in seen:
                    pending.append(candidate)
    result = {p.relative_to(PROJECT).as_posix():sha(p) for p in seen if FOLDER not in p.parents}
    # Dynamic data reads not discoverable from imports.
    for name in ('v10_extension/docci_semantic_probe_v12/parents.json',
                 'v10_extension/docci_reconstruction_v10/config.json'):
        result[name] = sha(PROJECT/name)
    return dict(sorted(result.items()))


def build(destination):
    out = safe(destination)
    if out.exists() or out.with_name(out.name+'.sha256').exists():
        raise FileExistsError('Release exists; preserve and choose a NEW filename')
    source = {p.relative_to(FOLDER).as_posix():sha(p) for p in FOLDER.rglob('*')
              if p.is_file() and p.suffix in ('.py','.md','.json') and p.name != 'MANIFEST.json'
              and '__pycache__' not in p.parts}
    for p in FOLDER.rglob('*'):
        if p.is_symlink():
            raise ValueError('Symlink source rejected')
    manifest = dict(schema='docci-baseline-v14-source', files=dict(sorted(source.items())), dependencies=dependencies(),
                    contains_weights_or_data=False, local_Torch_GPU_execution='NOT_EXECUTED',
                    upstream_LD4LG_commit='0bf9381e049ff288e5e79edc38e4a952a371bee2',
                    full_official_replication=False)
    atomic(FOLDER/'MANIFEST.json', manifest)
    names = sorted(source)+['MANIFEST.json']
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz', format=tarfile.USTAR_FORMAT) as tar:
            for name in names:
                data = (FOLDER/name).read_bytes()
                info = tarfile.TarInfo('v10_extension/docci_baseline_v14/'+name)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                tar.addfile(info, io.BytesIO(data))
    with tempfile.TemporaryDirectory(prefix='v14_source_restore_', dir=str(PROJECT/'releases')) as tmp:
        restored = Path(tmp)
        with tarfile.open(out) as tar:
            if len(tar.getmembers()) != len(names):
                raise ValueError('Release member count differs')
            for member in tar.getmembers():
                if not member.isfile() or not member.name.startswith('v10_extension/docci_baseline_v14/') or '..' in Path(member.name).parts:
                    raise ValueError('Unsafe archive member')
                target = restored/member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as f:
                    f.write(tar.extractfile(member).read())
                relative = Path(member.name).relative_to('v10_extension/docci_baseline_v14').as_posix()
                if sha(target) != sha(FOLDER/relative):
                    raise ValueError('Independent source restore differs')
    with out.with_name(out.name+'.sha256').open('x', encoding='utf-8') as f:
        f.write(sha(out)+'  '+out.name+'\n')
    print(json.dumps(dict(archive=str(out), bytes=out.stat().st_size, sha256=sha(out),
                         source_files=len(names), pinned_dependencies=len(manifest['dependencies']),
                         isolated_restore='PASSED', real_GPU='NOT_EXECUTED'), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('destination')
    a = p.parse_args()
    build(a.destination)
