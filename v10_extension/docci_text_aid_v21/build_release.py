"""Source-only USTAR release with dependency pins, hashes and byte verification."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tarfile
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_text_aid_v21.core import FOLDER,PROJECT,sha
from docci_text_guided_image_v20_fix1.compat import source_check as old_check,FOLDER as FIX
from docci_text_guided_image_v20.core import FOLDER as OLD,check_source

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest-only',action='store_true');a=p.parse_args()
    fix=old_check();old=check_source()
    files={p.name:sha(p) for p in sorted(FOLDER.iterdir()) if p.is_file() and p.name!='MANIFEST.json'}
    deps=dict(old['dependencies'])
    for base,names in ((OLD,list(old['files'])+['MANIFEST.json']),
                       (FIX,list(fix['files'])+['MANIFEST.json'])):
        for n in names:deps[(base/n).relative_to(PROJECT).as_posix()]=sha(base/n)
    for n in files:
        if n.endswith('.py'):ast.parse((FOLDER/n).read_text(),feature_version=(3,8))
    m=dict(schema='v21-source-release',files=files,dependencies=deps)
    manifest=FOLDER/'MANIFEST.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=m:raise ValueError('Different old manifest preserved')
    else:
        with manifest.open('x',encoding='utf-8') as f:json.dump(m,f,indent=2);f.write('\n')
    if a.manifest_only:return
    dest=PROJECT/'releases/v10_docci_text_aid_v21.tar.gz'
    if dest.exists() or dest.with_name(dest.name+'.sha256').exists():raise FileExistsError('Old release preserved')
    mapping={'v10_extension/docci_text_aid_v21/'+n:FOLDER/n for n in list(files)+['MANIFEST.json']}
    with dest.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w:gz',format=tarfile.USTAR_FORMAT) as tf:
            for n,file in mapping.items():
                info=tarfile.TarInfo(n);info.size=file.stat().st_size;info.mode=0o644
                with file.open('rb') as f:tf.addfile(info,f)
    with tarfile.open(dest,'r:gz') as tf:
        members=tf.getmembers()
        if len(members)!=len(mapping) or {m.name for m in members}!=set(mapping):raise ValueError('Release member matrix differs')
        for m in members:
            if not m.isfile() or hashlib.sha256(tf.extractfile(m).read()).hexdigest()!=sha(mapping[m.name]):
                raise ValueError('Release byte mismatch')
    with dest.with_name(dest.name+'.sha256').open('x',encoding='utf-8') as f:f.write(sha(dest)+'  '+dest.name+'\n')
    print(str(dest));print(sha(dest))

if __name__=='__main__':main()
