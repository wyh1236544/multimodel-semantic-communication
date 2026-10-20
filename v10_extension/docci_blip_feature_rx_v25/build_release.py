"""Source-only V25 tar; immutable transitive V24 Fix2 dependencies."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tarfile
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_caption_calibration_v24 import core as c
from docci_caption_calibration_v24_fix2 import fix as previous
FOLDER=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest-only',action='store_true')
    p.add_argument('--refresh-unreleased',action='store_true');a=p.parse_args()
    old=previous.source_check();deps=dict(old['dependencies'])
    for n in list(old['files'])+['MANIFEST.json']:deps[(previous.FOLDER/n).relative_to(c.PROJECT).as_posix()]=c.sha(previous.FOLDER/n)
    files={p.name:c.sha(p) for p in sorted(FOLDER.iterdir()) if p.is_file() and p.name!='MANIFEST.json'}
    for n in files:
        if n.endswith('.py'):ast.parse((FOLDER/n).read_text(),feature_version=(3,8))
    value=dict(schema='v25-source',files=files,dependencies=deps)
    manifest=FOLDER/'MANIFEST.json';dest=c.PROJECT/'releases/v10_docci_blip_feature_rx_v25.tar.gz'
    if manifest.exists() and c.read(manifest)!=value:
        if not a.refresh_unreleased or dest.exists():raise ValueError('Frozen repair preserved')
        manifest.write_text(json.dumps(value,indent=2)+'\n')
    elif not manifest.exists():c.atomic(manifest,value)
    if a.manifest_only:
        print('{} new sources; {} frozen dependencies'.format(len(files),len(deps)));return
    if dest.exists() or dest.with_name(dest.name+'.sha256').exists():raise FileExistsError('Existing release preserved')
    mapping={'v10_extension/docci_blip_feature_rx_v25/'+n:FOLDER/n for n in list(files)+['MANIFEST.json']}
    with dest.open('xb') as f,tarfile.open(fileobj=f,mode='w:gz',format=tarfile.USTAR_FORMAT) as tf:
        for n,path in mapping.items():
            info=tarfile.TarInfo(n);info.size=path.stat().st_size;info.mode=0o644
            with path.open('rb') as src:tf.addfile(info,src)
    with tarfile.open(dest,'r:gz') as tf:
        if {m.name for m in tf.getmembers()}!=set(mapping):raise ValueError('Archive member matrix differs')
        for m in tf.getmembers():
            if not m.isfile() or hashlib.sha256(tf.extractfile(m).read()).hexdigest()!=c.sha(mapping[m.name]):raise ValueError('Archive bytes differ')
    with dest.with_name(dest.name+'.sha256').open('x') as f:f.write(c.sha(dest)+'  '+dest.name+'\n')
    print(dest);print(c.sha(dest))

if __name__=='__main__':main()
