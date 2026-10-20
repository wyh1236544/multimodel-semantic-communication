"""Build a source-only additive release. Never overwrite any published source archive."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tarfile
from core import FOLDER, PROJECT, atomic, read, sha, write_text


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest-only',action='store_true')
    p.add_argument('--refresh-unreleased',action='store_true');a=p.parse_args()
    files={x.name:sha(x) for x in sorted(FOLDER.iterdir()) if x.is_file() and x.name!='MANIFEST.json'}
    for n in files:
        if n.endswith('.py'):ast.parse((FOLDER/n).read_text(),filename=n,feature_version=(3,8))
    value=dict(schema='v27-source',files=files,dependencies={},runtime='Python>=3.8 stdlib only')
    dest=PROJECT/'releases/v10_docci_fact_bridge_v27.tar.gz';manifest=FOLDER/'MANIFEST.json'
    if manifest.exists() and read(manifest)!=value:
        if not a.refresh_unreleased or dest.exists():raise ValueError('Frozen source/release preserved')
        # Explicit development-only manifest refresh, forbidden once the archive exists.
        manifest.write_text(json.dumps(value,indent=2)+'\n')
    elif not manifest.exists():atomic(manifest,value)
    if a.manifest_only:print('{} source files, zero runtime dependencies'.format(len(files)));return
    checksum=dest.with_name(dest.name+'.sha256')
    if dest.exists() or checksum.exists():raise FileExistsError('Published release preserved')
    mapping={'v10_extension/docci_fact_bridge_v27/'+n:FOLDER/n for n in list(files)+['MANIFEST.json']}
    with dest.open('xb') as f,tarfile.open(fileobj=f,mode='w:gz',format=tarfile.USTAR_FORMAT) as t:
        for n,path in mapping.items():
            info=tarfile.TarInfo(n);info.mode=0o644;info.size=path.stat().st_size
            with path.open('rb') as src:t.addfile(info,src)
    with tarfile.open(dest) as t:
        members=t.getmembers()
        if len(members)!=len(mapping) or {m.name for m in members}!=set(mapping):raise ValueError('Archive member mismatch')
        for m in members:
            if not m.isfile() or m.pax_headers or hashlib.sha256(t.extractfile(m).read()).hexdigest()!=sha(mapping[m.name]):
                raise ValueError('Actual source archive bytes differ')
    write_text(checksum,sha(dest)+'  '+dest.name+'\n')
    print(str(dest));print(sha(dest))


if __name__=='__main__':main()
