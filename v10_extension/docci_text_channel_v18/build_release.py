"""Source-only USTAR release. Refuses existing archive and preserves historical files."""
import ast
import io
import json
from pathlib import Path
import tarfile
import sys
FOLDER=Path(__file__).resolve().parent
PROJECT=FOLDER.parents[1]
sys.path[:0]=[str(FOLDER.parent),str(PROJECT)]
from docci_text_channel_v18.core import sha,read,atomic,check_source

def main():
    out=PROJECT/'releases/v10_docci_text_channel_v18.tar.gz'
    if out.exists() or out.with_name(out.name+'.sha256').exists():raise FileExistsError('Published archive preserved')
    files={p.name:sha(p) for p in sorted(FOLDER.iterdir()) if p.is_file() and p.name!='MANIFEST.json'}
    for name in files:
        if name.endswith('.py'):ast.parse((FOLDER/name).read_text(),feature_version=(3,8))
    # Existing V17 transitively pins the historical V14/BART/codecs/metrics.
    deps=read(FOLDER.parent/'docci_segment_bridge_v17/MANIFEST.json')['dependencies']
    for name,h in deps.items():
        if sha(PROJECT/name)!=h:raise ValueError('Historical source changed: '+name)
    atomic(FOLDER/'MANIFEST.json',dict(schema='v18-source-release',files=files,dependencies=deps))
    check_source();names=sorted(files)+['MANIFEST.json'];out.parent.mkdir(exist_ok=True)
    with out.open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz',format=tarfile.USTAR_FORMAT) as tf:
            for name in names:
                data=(FOLDER/name).read_bytes();info=tarfile.TarInfo('v10_extension/docci_text_channel_v18/'+name)
                info.mode=0o644;info.mtime=0;info.size=len(data);tf.addfile(info,io.BytesIO(data))
    with out.with_name(out.name+'.sha256').open('x',encoding='utf-8') as f:f.write(sha(out)+'  '+out.name+'\n')
    atomic(out.with_name(out.name+'.build.json'),dict(archive_sha256=sha(out),members=len(names),old_dependencies=len(deps),neural_tests='NOT_EXECUTED_LOCALLY'))
    print(str(out));print(sha(out))
if __name__=='__main__':main()
