"""Source-only USTAR; immutable old dependencies, no overwrite of a delivered archive."""
import argparse
import ast
import io
import json
from pathlib import Path
import sys
import tarfile
EXT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_digital_compare_v19.common import FOLDER,PROJECT,sha,read,check_source

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest-only',action='store_true');a=p.parse_args()
    out=PROJECT/'releases/v10_docci_digital_compare_v19.tar.gz'
    if out.exists() or out.with_name(out.name+'.sha256').exists():raise FileExistsError('Existing release preserved')
    old=read(EXT/'docci_text_channel_v18/MANIFEST.json');deps=dict(old['dependencies'])
    for n in list(old['files'])+['MANIFEST.json']:deps['v10_extension/docci_text_channel_v18/'+n]=sha(EXT/'docci_text_channel_v18'/n)
    for n,h in deps.items():
        if sha(PROJECT/n)!=h:raise ValueError('Historical dependency changed: '+n)
    files={f.name:sha(f) for f in sorted(FOLDER.iterdir()) if f.is_file() and f.name!='MANIFEST.json'}
    for n in files:
        if n.endswith('.py'):ast.parse((FOLDER/n).read_text(),feature_version=(3,8))
    manifest=dict(schema='v19-source-release',files=files,dependencies=deps)
    (FOLDER/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');check_source()
    if a.manifest_only:print('V19 manifest prepared');return
    out.parent.mkdir(exist_ok=True)
    with out.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w:gz',format=tarfile.USTAR_FORMAT) as tf:
            for n in sorted(files)+['MANIFEST.json']:
                data=(FOLDER/n).read_bytes();entry=tarfile.TarInfo('v10_extension/docci_digital_compare_v19/'+n)
                entry.mode=0o644;entry.size=len(data);entry.mtime=0;tf.addfile(entry,io.BytesIO(data))
    with out.with_name(out.name+'.sha256').open('x',encoding='utf-8') as f:f.write(sha(out)+'  '+out.name+'\n')
    with out.with_name(out.name+'.build.json').open('x',encoding='utf-8') as f:json.dump(dict(sha256=sha(out),members=len(files)+1,dependencies=len(deps),neural_inference='NOT_EXECUTED_LOCALLY'),f,indent=2)
    print(out);print(sha(out))
if __name__=='__main__':main()
