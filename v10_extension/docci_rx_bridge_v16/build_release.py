"""Source-only USTAR, transitive local import pins, independent restore verification."""
import argparse
import ast
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
EXT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXT), str(EXT.parent)]
from docci_rx_bridge_v16.core import FOLDER, PROJECT, safe, sha, atomic


def dependencies():
    pending, seen = list(FOLDER.glob('*.py')), set()
    while pending:
        file = pending.pop().resolve()
        if file in seen:
            continue
        seen.add(file)
        tree = ast.parse(file.read_text(encoding='utf-8'), filename=str(file), feature_version=8)
        for node in ast.walk(tree):
            names = ([x.name for x in node.names] if isinstance(node, ast.Import)
                     else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                roots = (EXT, PROJECT)
                if isinstance(node, ast.ImportFrom) and node.level:
                    base = file.parent
                    for _ in range(node.level-1):
                        base = base.parent
                    roots = (base,)
                for root in roots:
                    base = root.joinpath(*name.split('.'))
                    choices = (base.with_suffix('.py'), base/'__init__.py')
                    candidate = next((p for p in choices if p.is_file()), None)
                    if candidate is not None:
                        if candidate.is_symlink():
                            raise ValueError('Symlink dependency rejected')
                        pending.append(candidate)
                        break
    pinned = {p.relative_to(PROJECT).as_posix(): sha(p) for p in seen if FOLDER not in p.parents}
    for name in ('v10_extension/docci_baseline_v14/MANIFEST.json',
                 'v10_extension/docci_reconstruction_v10/config.json',
                 'v10_extension/docci_modal_v9/config.json',
                 'v10_extension/docci_longclip_joint_v8/config.json'):
        pinned[name] = sha(PROJECT/name)
    return dict(sorted(pinned.items()))


def build(destination):
    out = safe(destination)
    if out.exists() or out.with_name(out.name+'.sha256').exists():
        raise FileExistsError('Release/checksum exists; choose new version, never overwrite')
    files = {}
    for p in sorted(FOLDER.rglob('*')):
        if p.is_symlink():
            raise ValueError('Symlink source rejected')
        if p.is_file() and p.suffix in ('.py', '.md', '.json') and '__pycache__' not in p.parts and p.name != 'MANIFEST.json':
            if p.suffix == '.py':
                ast.parse(p.read_text(encoding='utf-8'), filename=str(p), feature_version=8)
            files[p.relative_to(FOLDER).as_posix()] = sha(p)
    atomic(FOLDER/'MANIFEST.json', dict(schema='docci-rx-bridge-v16-source', files=files,
           dependencies=dependencies(), weights_or_raw_data=False, real_GPU_validation='NOT_EXECUTED'))
    out.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(files)+['MANIFEST.json']
    with out.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz', format=tarfile.USTAR_FORMAT) as tar:
            for name in names:
                data = (FOLDER/name).read_bytes()
                info = tarfile.TarInfo('v10_extension/docci_rx_bridge_v16/'+name)
                info.mode, info.mtime, info.size = 0o644, 0, len(data)
                tar.addfile(info, io.BytesIO(data))
    with tempfile.TemporaryDirectory(prefix='v16_restore_', dir=str(PROJECT/'releases')) as tmp:
        root = Path(tmp)
        with tarfile.open(out) as tar:
            if len(tar.getmembers()) != len(names):
                raise ValueError('Source archive count differs')
            for member in tar.getmembers():
                rel = Path(member.name)
                if not member.isfile() or rel.parts[:2] != ('v10_extension', 'docci_rx_bridge_v16') or '..' in rel.parts:
                    raise ValueError('Unsafe source archive member')
                target = root/rel
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as f:
                    f.write(tar.extractfile(member).read())
                if sha(target) != sha(FOLDER/Path(*rel.parts[2:])):
                    raise ValueError('Independent source restoration differs')
        # Same exclusive directory claim as documented deployment, no old writes.
        try:
            (root/'v10_extension/docci_rx_bridge_v16').mkdir(exist_ok=False)
        except FileExistsError:
            pass
        else:
            raise ValueError('Existing-directory conflict was not rejected')
    with out.with_name(out.name+'.sha256').open('x', encoding='utf-8') as f:
        f.write(sha(out)+'  '+out.name+'\n')
    result = dict(archive=str(out), bytes=out.stat().st_size, sha256=sha(out),
                  source_files=len(names), dependencies=len(read_manifest()['dependencies']),
                  isolated_restore='PASSED', no_overwrite_conflict='PASSED', Torch_GPU='NOT_EXECUTED')
    atomic(out.with_name(out.name+'.build.json'), result)
    print(json.dumps(result, indent=2))


def read_manifest():
    return json.loads((FOLDER/'MANIFEST.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('destination')
    build(p.parse_args().destination)
