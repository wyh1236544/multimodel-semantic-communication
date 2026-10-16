"""Pin transitive historical inputs, source-only USTAR and independent restoration."""
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
from docci_segment_bridge_v17.core import FOLDER, PROJECT, safe, sha, atomic, read


def manifest():
    files = {}
    for p in sorted(FOLDER.iterdir()):
        if p.is_symlink():raise ValueError('Symlink source rejected')
        if p.is_file() and p.suffix in ('.py', '.md', '.json') and p.name != 'MANIFEST.json':
            if p.suffix == '.py':ast.parse(p.read_text(), filename=str(p), feature_version=8)
            files[p.name] = sha(p)
    dependencies = {}
    for folder in ('docci_rx_bridge_v16', 'docci_rx_bridge_v16_fix1'):
        base = EXT/folder; m = read(base/'MANIFEST.json')
        dependencies['v10_extension/'+folder+'/MANIFEST.json'] = sha(base/'MANIFEST.json')
        for name, digest in m['files'].items():dependencies['v10_extension/'+folder+'/'+name] = digest
        dependencies.update(m.get('dependencies', {}))
    for name, digest in dependencies.items():
        if sha(PROJECT/name) != digest:raise ValueError('Historical dependency changed: '+name)
    value = dict(schema='docci-segment-bridge-v17-source', files=files, dependencies=dict(sorted(dependencies.items())),
                 source_only=True, real_GPU_validation='NOT_EXECUTED', no_weights_or_DOCCI_text=True)
    atomic(FOLDER/'MANIFEST.json', value)
    return value


def build(destination):
    out = safe(destination)
    if out.exists() or out.with_name(out.name+'.sha256').exists():raise FileExistsError('Release preserved; choose a new version')
    m = manifest()
    names = sorted(m['files'])+['MANIFEST.json']; out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz', format=tarfile.USTAR_FORMAT) as tar:
            for name in names:
                data = (FOLDER/name).read_bytes(); member = tarfile.TarInfo('v10_extension/docci_segment_bridge_v17/'+name)
                member.mode, member.mtime, member.size = 0o644, 0, len(data); tar.addfile(member, io.BytesIO(data))
    with tempfile.TemporaryDirectory(prefix='v17_restore_', dir=str(PROJECT/'releases')) as d:
        root = Path(d)
        with tarfile.open(out) as tar:
            if len(tar.getmembers()) != len(names):raise ValueError('Source archive member count')
            for mbr in tar.getmembers():
                rel = Path(mbr.name)
                if not mbr.isfile() or '..' in rel.parts or rel.parts[:2] != ('v10_extension', 'docci_segment_bridge_v17'):
                    raise ValueError('Unsafe release member')
                target = root/rel; target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as f:f.write(tar.extractfile(mbr).read())
                if sha(target) != sha(FOLDER/rel.name):raise ValueError('Restored source hash differs')
        try:(root/'v10_extension/docci_segment_bridge_v17').mkdir(exist_ok=False)
        except FileExistsError:pass
        else:raise ValueError('Existing module overwrite not rejected')
    with out.with_name(out.name+'.sha256').open('x', encoding='utf-8') as f:f.write(sha(out)+'  '+out.name+'\n')
    result = dict(archive=str(out), sha256=sha(out), bytes=out.stat().st_size, members=len(names),
                  historical_dependencies=len(m['dependencies']), isolated_restore='PASSED', Python38_parse='PASSED', GPU='NOT_EXECUTED')
    atomic(out.with_name(out.name+'.build.json'), result); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('destination', nargs='?'); p.add_argument('--manifest-only', action='store_true'); a = p.parse_args()
    if a.manifest_only:manifest()
    elif a.destination:build(a.destination)
    else:p.error('destination or --manifest-only required')
