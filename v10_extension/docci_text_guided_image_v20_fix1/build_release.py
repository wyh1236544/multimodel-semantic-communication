"""Additive source-only release; never rewrite an old manifest/archive."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tarfile
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_text_guided_image_v20.core import sha,check_source,PROJECT
from docci_text_guided_image_v20_fix1.compat import FOLDER,ORIGINAL,ORIGINAL_MANIFEST,source_check

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest-only',action='store_true');a=p.parse_args()
    check_source()
    if sha(ORIGINAL/'MANIFEST.json')!=ORIGINAL_MANIFEST:
        raise ValueError('Original V20 release differs')
    files={p.name:sha(p) for p in FOLDER.iterdir() if p.is_file() and p.name!='MANIFEST.json'}
    for n in files:
        if n.endswith('.py'): ast.parse((FOLDER/n).read_text(),feature_version=(3,8))
    value=dict(schema='v20-pooling-fix1-source',original_manifest_sha256=ORIGINAL_MANIFEST,files=files)
    manifest=FOLDER/'MANIFEST.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=value:
            raise ValueError('Existing different fix manifest preserved; freeze only after QA')
    else:
        with manifest.open('x',encoding='utf-8') as f: json.dump(value,f,indent=2);f.write('\n')
    source_check()
    if a.manifest_only: return
    dest=PROJECT/'releases/v10_docci_text_guided_image_v20_fix1.tar.gz'
    if dest.exists() or dest.with_name(dest.name+'.sha256').exists():
        raise FileExistsError('Delivered archive preserved')
    names=sorted(files)+['MANIFEST.json']
    mapping={'v10_extension/docci_text_guided_image_v20_fix1/'+n:FOLDER/n for n in names}
    with dest.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w:gz',format=tarfile.USTAR_FORMAT) as tf:
            for n,file in mapping.items():
                info=tarfile.TarInfo(n);info.mode=0o644;info.size=file.stat().st_size;info.mtime=0
                with file.open('rb') as f: tf.addfile(info,f)
    with tarfile.open(dest,'r:gz') as tf:
        members=tf.getmembers()
        if len(members)!=len(mapping) or {m.name for m in members}!=set(mapping) or any(not m.isfile() for m in members):
            raise ValueError('Archive member matrix differs')
        for m in members:
            if hashlib.sha256(tf.extractfile(m).read()).hexdigest()!=sha(mapping[m.name]):
                raise ValueError('Archived source differs')
    with dest.with_name(dest.name+'.sha256').open('x',encoding='utf-8') as f: f.write(sha(dest)+'  '+dest.name+'\n')
    print(str(dest));print(sha(dest))

if __name__=='__main__': main()
