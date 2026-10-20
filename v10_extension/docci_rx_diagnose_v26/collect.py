"""PRIVATE lightweight records and blind factual-review templates; no weights."""
import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import tarfile
EXT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EXT),str(EXT.parent)]
from docci_rx_diagnose_v26.core import safe,sha,atomic,source_check
from docci_rx_diagnose_v26.report import pipeline,render,facts

def collect(root,destination):
    root=Path(root).resolve(strict=True);dest=safe(destination);staging=dest.with_name(dest.name+'.contents')
    if any(p.exists() for p in (dest,staging,dest.with_name(dest.name+'.sha256'))):raise FileExistsError('Existing collection preserved')
    if root in dest.parents:raise ValueError('Collection must be outside run')
    result=pipeline(root)
    if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('No linked inputs')
    before={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p.relative_to(root).as_posix()!='collect.log'}
    staging.mkdir(parents=True);data=staging/'data';data.mkdir()
    for n,h in before.items():
        p=root/n
        if p.suffix not in ('.json','.jsonl','.log','.md','.py','.csv') or p.stat().st_size>64*1024**2:
            raise ValueError('Unexpected/large collection input '+n)
        q=data/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
        if sha(q)!=h:raise ValueError('Copied diagnostic bytes differ')
    copied=pipeline(data)
    atomic(staging/'SUMMARY.json',dict(copied,private=True,weights_or_NPZ_included=False))
    with (staging/'index.html').open('x',encoding='utf-8') as f:f.write(render(copied['stages']['diagnose']))
    facts(data/'diagnose',data/'inputs',staging)
    if any(sha(root/n)!=h for n,h in before.items()):raise ValueError('Original results changed during collection')
    files={p.relative_to(staging).as_posix():sha(p) for p in staging.rglob('*') if p.is_file()}
    atomic(staging/'COLLECTION.json',dict(files=files,private=True,training_updates=0,weights_or_NPZ_included=False))
    with dest.open('xb') as f,tarfile.open(fileobj=f,mode='w:gz',format=tarfile.USTAR_FORMAT) as tf:
        for p in sorted(staging.rglob('*')):
            if p.is_file():
                info=tarfile.TarInfo('v26_records/'+p.relative_to(staging).as_posix());info.mode=0o644;info.size=p.stat().st_size
                with p.open('rb') as src:tf.addfile(info,src)
    with tarfile.open(dest) as tf:
        for m in tf.getmembers():
            p=staging/Path(m.name).relative_to('v26_records')
            if not m.isfile() or hashlib.sha256(tf.extractfile(m).read()).hexdigest()!=sha(p):raise ValueError('Actual archive differs')
    with dest.with_name(dest.name+'.sha256').open('x') as f:f.write(sha(dest)+'  '+dest.name+'\n')
    print('PRIVATE V26 records: '+str(dest),flush=True)
    return dest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('destination');a=p.parse_args();source_check();collect(a.root,a.destination)
