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
from docci_rx_bridge_v16.core import PROJECT, safe, atomic, sha, check_source
from docci_rx_bridge_v16_fix1.compat import FOLDER, ORIGINAL, ORIGINAL_MANIFEST


def build(destination):
    check_source()
    if sha(ORIGINAL/'MANIFEST.json') != ORIGINAL_MANIFEST:
        raise ValueError('Original V16 source differs')
    out = safe(destination)
    if out.exists() or out.with_name(out.name+'.sha256').exists():
        raise FileExistsError('Archive/checksum exists; preserve')
    files = {}
    for p in sorted(FOLDER.iterdir()):
        if p.is_symlink():
            raise ValueError('Symlink source rejected')
        if p.is_file() and p.suffix in ('.py', '.md'):
            if p.suffix == '.py':
                ast.parse(p.read_text(), filename=str(p), feature_version=8)
            files[p.name] = sha(p)
    atomic(FOLDER/'MANIFEST.json', dict(schema='docci-rx-bridge-v16-fix1-source', files=files,
           original_manifest_sha256=ORIGINAL_MANIFEST, model_or_quality_changes=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(files)+['MANIFEST.json']
    with out.open('xb') as f:
        with tarfile.open(fileobj=f, mode='w:gz', format=tarfile.USTAR_FORMAT) as tar:
            for name in names:
                data = (FOLDER/name).read_bytes()
                info = tarfile.TarInfo('v10_extension/docci_rx_bridge_v16_fix1/'+name)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                tar.addfile(info, io.BytesIO(data))
    with tempfile.TemporaryDirectory(prefix='v16_fix1_restore_', dir=str(PROJECT/'releases')) as d:
        with tarfile.open(out) as tar:
            if len(tar.getmembers()) != len(names):
                raise ValueError('Archive count differs')
            for member in tar.getmembers():
                if not member.isfile() or '..' in Path(member.name).parts or Path(member.name).parts[:2] != ('v10_extension', 'docci_rx_bridge_v16_fix1'):
                    raise ValueError('Unsafe archive member')
                p = Path(d)/member.name
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open('xb') as f:
                    f.write(tar.extractfile(member).read())
                if sha(p) != sha(FOLDER/p.name):
                    raise ValueError('Independent restored bytes differ')
    with out.with_name(out.name+'.sha256').open('x', encoding='utf-8') as f:
        f.write(sha(out)+'  '+out.name+'\n')
    print(json.dumps(dict(archive=str(out), sha256=sha(out), bytes=out.stat().st_size,
                         members=len(names), isolated_restore='PASSED', GPU='NOT_EXECUTED'), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('destination')
    build(p.parse_args().destination)
