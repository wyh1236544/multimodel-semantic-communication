"""Source-only additive V26 release. Historical manifests/releases stay frozen."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tarfile
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_blip_feature_rx_v25 import core as old
FOLDER=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest-only',action='store_true');p.add_argument('--refresh-unreleased',action='store_true');a=p.parse_args()
    previous=old.source_check();deps=dict(previous['dependencies'])
    for n in list(previous['files'])+['MANIFEST.json']:deps[(old.FOLDER/n).relative_to(old.PROJECT).as_posix()]=old.sha(old.FOLDER/n)
    files={p.name:old.sha(p) for p in sorted(FOLDER.iterdir()) if p.is_file() and p.name!='MANIFEST.json'}
    for n in files:
        if n.endswith('.py'):ast.parse((FOLDER/n).read_text(),feature_version=(3,8))
    value=dict(schema='v26-source',files=files,dependencies=deps)
    manifest=FOLDER/'MANIFEST.json';dest=old.PROJECT/'releases/v10_docci_rx_diagnose_v26.tar.gz'
    if manifest.exists() and old.read(manifest)!=value:
        if not a.refresh_unreleased or dest.exists():raise ValueError('Frozen source/release preserved')
        manifest.write_text(json.dumps(value,indent=2)+'\n')
    elif not manifest.exists():old.atomic(manifest,value)
    if a.manifest_only:print('{} new sources, {} unchanged dependencies'.format(len(files),len(deps)));return
    if dest.exists() or dest.with_name(dest.name+'.sha256').exists():raise FileExistsError('Existing release preserved')
    mapping={'v10_extension/docci_rx_diagnose_v26/'+n:FOLDER/n for n in list(files)+['MANIFEST.json']}
    with dest.open('xb') as f,tarfile.open(fileobj=f,mode='w:gz',format=tarfile.USTAR_FORMAT) as tf:
        for n,path in mapping.items():
            info=tarfile.TarInfo(n);info.mode=0o644;info.size=path.stat().st_size
            with path.open('rb') as src:tf.addfile(info,src)
    with tarfile.open(dest) as tf:
        if {m.name for m in tf.getmembers()}!=set(mapping):raise ValueError('Source archive members differ')
        for m in tf.getmembers():
            if not m.isfile() or m.pax_headers or hashlib.sha256(tf.extractfile(m).read()).hexdigest()!=old.sha(mapping[m.name]):
                raise ValueError('Source archive bytes differ')
    with dest.with_name(dest.name+'.sha256').open('x') as f:f.write(old.sha(dest)+'  '+dest.name+'\n')
    print(dest);print(old.sha(dest))
if __name__=='__main__':main()
